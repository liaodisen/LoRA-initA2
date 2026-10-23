"""Fine-tune a causal language model with the paper's LoRA initializers."""

import os
import hashlib
import json
import sys
import importlib.metadata
import inspect
from pathlib import Path
import torch
import socket
import random
import argparse
import numpy as np
import transformers
from datetime import datetime
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from initialization import (
    initialize_lora,
    initialize_orthogonal_lora_a2,
    rescale_lora_a_to_a2,
    summarize_lora_initialization,
)
from logger import get_log
from trainer import LoraPlusTrainer, LoraPlusTrainingArguments
from coherence_logging import LoraCoherenceCallback
from feature_decomposition import LoraFeatureDecompositionCallback
from training_dynamics import LoraTrainingDynamicsCallback
from width_alignment import LoraWidthAlignmentCallback
from peft import (
    EvaConfig,
    LoraConfig,
    get_peft_model,
    initialize_lora_eva_weights,
    prepare_model_for_kbit_training,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y", "on"):
        return True
    if value in ("no", "false", "f", "0", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}.")


def get_torch_dtype(dtype_name):
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype_name = dtype_name.lower()
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtype_map[dtype_name]


def ensure_qlora_dependencies():
    try:
        return importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "QLoRA requires `bitsandbytes`, but it is not installed in the current environment. "
            "Activate the training environment and install it with `pip install bitsandbytes` "
            "(or `conda install -c conda-forge bitsandbytes`)."
        ) from exc


def safe_path_component(value):
    value = str(value)
    safe = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    return safe.strip("-") or "unknown"


def resolve_data_path(path):
    path = os.path.expanduser(path)
    candidates = [Path(path)]
    if not Path(path).is_absolute():
        candidates.append(REPO_ROOT / path)
        cleaned = path
        while cleaned.startswith("../"):
            cleaned = cleaned[3:]
        candidates.append(REPO_ROOT / cleaned)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return path


def make_loraplus_training_args(**kwargs):
    signature = inspect.signature(LoraPlusTrainingArguments.__init__).parameters
    if "eval_strategy" not in signature and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return LoraPlusTrainingArguments(**kwargs)


def normalise_init_method(init_method):
    return (init_method or "").strip().lower().replace("-", "_")


def peft_init_method(init_method):
    """Map experiment aliases to the initializer PEFT should run first."""
    method = normalise_init_method(init_method)
    if method in {"kaiming", "orthogonal_a2"}:
        return True
    if method == "pissa_a2":
        return "pissa"
    if method == "eva_a2":
        return "eva"
    return init_method


def build_eva_dataloader(dataset, data_collator, batch_size, max_samples):
    if max_samples > 0 and len(dataset) > max_samples:
        if hasattr(dataset, "select"):
            dataset = dataset.select(range(max_samples))
        else:
            dataset = [dataset[index] for index in range(max_samples)]

    # Cached tokenized datasets created by older versions of this script can
    # retain the original text columns (instruction/input/output). Trainer
    # removes those columns before collating, but EVA consumes this dataloader
    # directly, so project each example to tensor-valued model inputs here.
    model_input_columns = {
        "input_ids",
        "attention_mask",
        "labels",
        "token_type_ids",
        "position_ids",
    }
    if hasattr(dataset, "column_names") and hasattr(dataset, "remove_columns"):
        extra_columns = [
            column
            for column in dataset.column_names
            if column not in model_input_columns
        ]
        if extra_columns:
            dataset = dataset.remove_columns(extra_columns)
    else:
        dataset = [
            {
                key: value
                for key, value in dataset[index].items()
                if key in model_input_columns
            }
            for index in range(len(dataset))
        ]

    if len(dataset) == 0 or "input_ids" not in dataset[0]:
        raise ValueError(
            "EVA initialization requires a non-empty tokenized dataset with "
            "an input_ids column."
        )

    class EvaModelInput(dict):
        """Dict with attribute access for PEFT 0.18 EVA label masking."""

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    def collate_as_eva_input(examples):
        # PEFT 0.18 requires isinstance(inputs, dict), but also checks
        # inputs.labels when applying the configured language-model label mask.
        return EvaModelInput(data_collator(examples))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_as_eva_input,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LoRA adapters")
    # model/data params
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--base_model",
        type=str,
        default="NousResearch/Meta-Llama-3-8B",
        help="base model name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/math_10k.json",
        help="path to the training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./log/debug",
        help="output directory for logs and models",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="lora",
        choices=["lora", "qlora"],
        help="adapter name",
    )
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    parser.add_argument(
        "--cache_dir", default=None, help="Hugging Face cache directory"
    )
    parser.add_argument(
        "--tokenized_cache_dir", default=str(REPO_ROOT / ".cache" / "tokenized")
    )
    parser.add_argument("--report_to", default="none", choices=["none", "wandb"])
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Limit optimizer steps for validation runs",
    )
    parser.add_argument("--warmup_steps", type=int, default=100)
    # training params
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=16, help="micro batch size"
    )
    parser.add_argument(
        "--num_epochs", type=float, default=3, help="number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--cutoff_len", type=int, default=256, help="cutoff length for input sequences"
    )
    parser.add_argument(
        "--val_set_size", type=int, default=0, help="validation set size"
    )
    parser.add_argument(
        "--eval_step", type=int, default=80, help="steps between evaluation"
    )
    parser.add_argument(
        "--save_step", type=int, default=80, help="steps between saving checkpoints"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="optimizer steps between training-loss logs",
    )
    parser.add_argument(
        "--skip_save_adapter",
        action="store_true",
        default=False,
        help="skip saving intermediate checkpoints and the final adapter",
    )
    parser.add_argument(
        "--skip_save_init_ckpt",
        action="store_true",
        default=False,
        help="skip saving output_dir/init_ckpt after LoRA initialization",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=False,
        help="reduce memory usage",
    )
    parser.add_argument(
        "--train_on_inputs",
        action="store_true",
        default=False,
        help="masks out inputs in loss if False",
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        default=False,
        help="faster, but produces an odd training loss curve",
    )
    parser.add_argument(
        "--disable_dataset_cache",
        action="store_true",
        default=False,
        help="Tokenize in memory and do not load or save a tokenized dataset cache",
    )
    parser.add_argument(
        "--overwrite_dataset_cache",
        action="store_true",
        default=False,
        help="Rebuild the tokenized dataset cache even if it already exists",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="Compute dtype used for QLoRA 4-bit quantization",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="4-bit quantization type used for QLoRA",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to use nested quantization for QLoRA",
    )
    # lora params
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--init_para",
        type=str,
        default="A",
        help=(
            "Custom/Kaiming LoRA init selector: A, B, AB, AA, or a reset "
            "variant such as AB_reset. Any value containing 'reset' subtracts "
            "the initialized LoRA update from the base weight after init."
        ),
    )
    parser.add_argument("--lora_alpha", type=float, default=8, help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout rate"
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        help="list of modules",
    )
    parser.add_argument("--lora_A_std", type=float, default=1.0, help="LoRA A std")
    parser.add_argument("--lora_B_std", type=float, default=1.0, help="LoRA B std")
    parser.add_argument("--lora_A_scale", type=float, default=0.5, help="LoRA A scale")
    parser.add_argument("--lora_B_scale", type=float, default=0.5, help="LoRA B scale")
    parser.add_argument(
        "--lora_normal",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to use normal initialization for LoRA",
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="Kaiming",
        help=(
            "LoRA initializer: Kaiming, orthogonal_a2, eva_a2, pissa_a2, "
            "or a PEFT init_lora_weights value such as pissa/olora/eva"
        ),
    )
    parser.add_argument(
        "--eva_max_samples",
        type=int,
        default=0,
        help="maximum training examples used to estimate EVA directions; <=0 uses the full training set",
    )
    parser.add_argument(
        "--eva_rho",
        type=float,
        default=1.0,
        help="EVA maximum rank multiplier; 1 disables rank redistribution for matched-rank comparisons",
    )
    parser.add_argument(
        "--loraplus_lr_ratio",
        type=float,
        default=None,
        help="LoRA+ learning rate ratio lr_B / lr_A.",
    )
    parser.add_argument(
        "--loraplus_lr_embedding",
        type=float,
        default=1e-5,
        help="LoRA+ learning rate for lora embedding layers.",
    )
    # rank-coherence diagnostics
    parser.add_argument(
        "--log_coherence_stats",
        action="store_true",
        default=False,
        help="log LoRA rank-coherence statistics during training",
    )
    parser.add_argument(
        "--coherence_output_dir",
        type=str,
        default=None,
        help="directory for coherence JSONL/PT outputs; defaults to output_dir/coherence_stats",
    )
    parser.add_argument(
        "--coherence_probe_size",
        type=int,
        default=1,
        help="number of fixed tokenized examples in the coherence probe batch",
    )
    parser.add_argument(
        "--coherence_probe_seed",
        type=int,
        default=1234,
        help="seed for selecting the fixed coherence probe examples",
    )
    parser.add_argument(
        "--coherence_max_modules",
        type=int,
        default=8,
        help="maximum adapted modules to log; use <=0 for all",
    )
    parser.add_argument(
        "--coherence_max_tokens",
        type=int,
        default=64,
        help="maximum probe token positions per module",
    )
    parser.add_argument(
        "--coherence_max_rows",
        type=int,
        default=64,
        help="maximum LoRA B output rows per module",
    )
    parser.add_argument(
        "--coherence_log_fractions",
        type=str,
        default="0.01 0.05 0.25 0.5 1.0",
        help="first-epoch fractions to log, space or comma separated",
    )
    parser.add_argument(
        "--coherence_log_steps",
        type=str,
        default="",
        help="explicit global steps to log, space or comma separated",
    )
    parser.add_argument(
        "--coherence_every_n_steps",
        type=int,
        default=0,
        help="also log coherence every N optimizer steps when >0",
    )
    parser.add_argument(
        "--log_training_dynamics",
        action="store_true",
        default=False,
        help="log LoRA activations and gradient norms during training",
    )
    parser.add_argument(
        "--dynamics_output_dir",
        type=str,
        default=None,
        help="directory for LoRA training-dynamics JSONL output; defaults to output_dir/training_dynamics",
    )
    parser.add_argument(
        "--dynamics_max_modules",
        type=int,
        default=8,
        help="maximum LoRA modules to log for training dynamics; use <=0 for all",
    )
    parser.add_argument(
        "--dynamics_log_fractions",
        type=str,
        default="0.01 0.05 0.1 0.25 0.5 1.0",
        help="first-epoch fractions to log for training dynamics, space or comma separated",
    )
    parser.add_argument(
        "--dynamics_log_steps",
        type=str,
        default="",
        help="explicit global steps to log for training dynamics, space or comma separated",
    )
    parser.add_argument(
        "--dynamics_every_n_steps",
        type=int,
        default=10,
        help="also log training dynamics every N optimizer steps when >0",
    )
    parser.add_argument(
        "--log_feature_decomposition",
        action="store_true",
        default=False,
        help="log the exact LoRA feature noise, signal, and cross-energy decomposition",
    )
    parser.add_argument(
        "--decomposition_output_dir",
        type=str,
        default=None,
        help="directory for feature-decomposition JSONL output; defaults to output_dir/feature_decomposition",
    )
    parser.add_argument(
        "--decomposition_probe_size",
        type=int,
        default=1,
        help="number of fixed tokenized examples in the decomposition probe batch",
    )
    parser.add_argument(
        "--decomposition_probe_seed",
        type=int,
        default=1234,
        help="seed for selecting fixed decomposition probe examples",
    )
    parser.add_argument(
        "--decomposition_max_modules",
        type=int,
        default=8,
        help="maximum LoRA modules to log for feature decomposition; use <=0 for all",
    )
    parser.add_argument(
        "--decomposition_max_tokens",
        type=int,
        default=64,
        help="maximum fixed-probe token positions per decomposed module",
    )
    parser.add_argument(
        "--decomposition_log_fractions",
        type=str,
        default="0.01 0.05 0.1 0.25 0.5 1.0",
        help="first-epoch fractions to log for feature decomposition",
    )
    parser.add_argument(
        "--decomposition_log_steps",
        type=str,
        default="",
        help="explicit global steps to log for feature decomposition",
    )
    parser.add_argument(
        "--decomposition_every_n_steps",
        type=int,
        default=100,
        help="also log feature decomposition every N optimizer steps when >0",
    )
    parser.add_argument(
        "--log_width_alignment",
        action="store_true",
        default=False,
        help="log minibatch alignment between the processed LoRA A update and its layer input",
    )
    parser.add_argument(
        "--alignment_output_dir",
        type=str,
        default=None,
        help="directory for width-alignment JSONL output; defaults to output_dir/width_alignment",
    )
    parser.add_argument(
        "--alignment_module_pattern",
        type=str,
        default="q_proj",
        help="substring selecting adapted modules for width-alignment logging; use 'all' or '*' for every LoRA module",
    )
    parser.add_argument(
        "--alignment_max_modules",
        type=int,
        default=4,
        help="maximum adapted modules to log for width alignment; use <=0 for all",
    )
    parser.add_argument(
        "--alignment_max_tokens",
        type=int,
        default=32,
        help="maximum same-minibatch token inputs retained per aligned module",
    )
    parser.add_argument(
        "--alignment_log_fractions",
        type=str,
        default="0.25 0.5 1.0",
        help="first-epoch fractions to log for width alignment",
    )
    parser.add_argument(
        "--alignment_log_steps",
        type=str,
        default="2 5 10 25 50 100",
        help="explicit optimizer steps to log for width alignment",
    )
    parser.add_argument(
        "--alignment_every_n_steps",
        type=int,
        default=0,
        help="also log width alignment every N optimizer steps when >0",
    )
    parser.add_argument(
        "--disable_alignment_shuffled_control",
        action="store_true",
        default=False,
        help="disable the coordinate-shuffled negative control",
    )
    return parser.parse_args()


def parse_number_list(raw_value, value_type):
    values = []
    for item in str(raw_value or "").replace(",", " ").split():
        values.append(value_type(item))
    return values


def set_global_seed(seed=-1):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"]
        and result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < args.cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        if "chatglm" not in args.base_model:
            result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    if "chatglm" in args.base_model:
        return {"input_ids": result["input_ids"], "labels": result["labels"]}
    else:
        return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    if not args.train_on_inputs:
        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt


def tokenize_and_save_dataset(data_path, output_path, cutoff_len, train_on_inputs):
    """Run this once to pre-tokenize your dataset"""
    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path, cache_dir=args.cache_dir)
    else:
        data = load_dataset(data_path, cache_dir=args.cache_dir)

    tokenized = data["train"].map(
        generate_and_tokenize_prompt,
        remove_columns=data["train"].column_names,
        desc="Tokenizing dataset",
    )

    # Save to disk
    tokenized.save_to_disk(output_path)
    print(f"Tokenized dataset saved to {output_path}")


def get_tokenized_dataset_path(data_path):
    source = Path(data_path)
    source_hash = (
        hashlib.blake2b(source.read_bytes()).hexdigest()
        if source.is_file()
        else data_path
    )
    settings = {
        "source": source_hash,
        "model": args.base_model,
        "vocab": tokenizer.get_vocab(),
        "special_tokens": tokenizer.special_tokens_map,
        "cutoff": args.cutoff_len,
        "train_on_inputs": args.train_on_inputs,
        "prompt_version": 1,
    }
    digest = hashlib.blake2b(json.dumps(settings, sort_keys=True).encode()).hexdigest()[
        :24
    ]
    return str(Path(args.tokenized_cache_dir) / digest)


def validate_tokenized_dataset(data, tokenized_path=None):
    vocab_size = len(tokenizer)
    for idx, example in enumerate(data):
        input_ids = example.get("input_ids", [])
        labels = [label for label in example.get("labels", []) if label != -100]
        bad_input = input_ids and (min(input_ids) < 0 or max(input_ids) >= vocab_size)
        bad_label = labels and (min(labels) < 0 or max(labels) >= vocab_size)
        if bad_input or bad_label:
            source = f" from {tokenized_path}" if tokenized_path else ""
            input_range = (
                f"({min(input_ids)}, {max(input_ids)})" if input_ids else "(empty)"
            )
            label_range = f"({min(labels)}, {max(labels)})" if labels else "(empty)"
            raise ValueError(
                f"Tokenized dataset{source} contains token ids outside the "
                f"current tokenizer vocabulary. Example index {idx}; "
                f"vocab_size={vocab_size}; "
                f"input_id_range={input_range}; label_range={label_range}"
                + ". Delete the stale cache or rerun with a model-aware cache."
            )


def get_data(data_path, val_set_size, return_tokenized_data=False):
    data_path = resolve_data_path(data_path)
    tokenized_path = get_tokenized_dataset_path(data_path)
    legacy_tokenized_path = (
        data_path.replace(".json", f"_tokenized_{args.cutoff_len}")
        if data_path.endswith(".json")
        else None
    )
    if (
        legacy_tokenized_path
        and legacy_tokenized_path != tokenized_path
        and os.path.exists(legacy_tokenized_path)
    ):
        print(f"Ignoring legacy tokenizer-agnostic cache at {legacy_tokenized_path}")

    loaded_cache_path = None
    if (
        not args.disable_dataset_cache
        and not args.overwrite_dataset_cache
        and os.path.exists(tokenized_path)
    ):
        print(f"Loading pre-tokenized dataset from {tokenized_path}")
        data = load_from_disk(tokenized_path)
        loaded_cache_path = tokenized_path
    else:
        print("Tokenizing dataset (this may take a while)...")
        # Original tokenization code
        if data_path.endswith(".json"):
            data = load_dataset("json", data_files=data_path, cache_dir=args.cache_dir)
        else:
            data = load_dataset(data_path, cache_dir=args.cache_dir)
        data = data["train"].map(
            generate_and_tokenize_prompt, remove_columns=data["train"].column_names
        )
        if args.disable_dataset_cache:
            print("Dataset cache disabled; tokenized dataset will not be saved")
        else:
            data.save_to_disk(tokenized_path)
            print(f"Tokenized dataset saved to {tokenized_path}")
    validate_tokenized_dataset(data, loaded_cache_path)

    if val_set_size > 0:
        train_val = data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle(seed=args.seed)
        val_data = train_val["test"].shuffle(seed=args.seed)
    else:
        train_data = data.shuffle(seed=args.seed)
        val_data = None

    if return_tokenized_data:
        return train_data, val_data, data
    return train_data, val_data


def build_coherence_probe_batch(tokenized_data, data_collator, probe_size, probe_seed):
    probe_size = min(max(int(probe_size), 1), len(tokenized_data))
    rng = random.Random(probe_seed)
    indices = sorted(rng.sample(range(len(tokenized_data)), probe_size))
    model_keys = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    examples = [
        {
            key: value
            for key, value in tokenized_data[int(index)].items()
            if key in model_keys
        }
        for index in indices
    ]
    batch = data_collator(examples)
    return {key: value for key, value in batch.items() if torch.is_tensor(value)}


def model_dtype(args):
    if args.dtype != "auto":
        return get_torch_dtype(args.dtype)
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def get_model_tokenizer(args):
    base_model = args.base_model
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": False,
        "cache_dir": args.cache_dir,
    }
    if args.adapter_name == "qlora":
        ensure_qlora_dependencies()
        compute_dtype = get_torch_dtype(args.bnb_4bit_compute_dtype)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        model_kwargs["torch_dtype"] = compute_dtype
    else:
        model_kwargs["torch_dtype"] = model_dtype(args)

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=args.cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return model, tokenizer


def get_peft_config(args):
    if args.adapter_name in {"lora", "qlora"}:
        init_lora_weights = peft_init_method(args.init_method)
        eva_config = (
            EvaConfig(rho=args.eva_rho)
            if normalise_init_method(args.init_method) in {"eva", "eva_a2"}
            else None
        )
        return LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=init_lora_weights,
            eva_config=eva_config,
        )
    raise ValueError(f"Unsupported adapter_name: {args.adapter_name}")


def get_precision_flags(args):
    if args.adapter_name == "qlora":
        compute_dtype = get_torch_dtype(args.bnb_4bit_compute_dtype)
        return compute_dtype == torch.float16, compute_dtype == torch.bfloat16
    dtype = model_dtype(args)
    return dtype == torch.float16, dtype == torch.bfloat16


def get_last_logged_train_loss(log_history):
    for entry in reversed(log_history):
        if "loss" in entry and "eval_loss" not in entry:
            return float(entry["loss"])
    return None


if __name__ == "__main__":
    args = parse_args()
    replay_required = (
        "reset" in args.init_para.lower()
        or normalise_init_method(args.init_method).startswith("pissa")
        or normalise_init_method(args.init_method) == "olora"
    )
    if args.micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if (
        args.batch_size < args.micro_batch_size
        or args.batch_size % args.micro_batch_size
    ):
        raise ValueError("batch_size must be a positive multiple of micro_batch_size")
    if args.adapter_name == "qlora" and not torch.cuda.is_available():
        raise ValueError("QLoRA requires a CUDA GPU")
    if replay_required and args.adapter_name == "qlora":
        raise ValueError("Residual-weight initializers require full-precision LoRA")
    if args.skip_save_init_ckpt and replay_required:
        raise ValueError(
            "--skip_save_init_ckpt cannot be used with reset-style init_para values "
            "because evaluation reloads output_dir/init_ckpt for those modes."
        )
    set_global_seed(args.seed)
    server_name = socket.gethostname()
    logger = get_log(args.output_dir, "setting")
    logger.info(f"server_name: {server_name}")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    with open(Path(args.output_dir) / "run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2)
    start_time = datetime.now()

    model, tokenizer = get_model_tokenizer(args)
    if args.log_coherence_stats or args.log_feature_decomposition:
        train_data, val_data, tokenized_data = get_data(
            args.data_path,
            args.val_set_size,
            return_tokenized_data=True,
        )
    else:
        train_data, val_data = get_data(args.data_path, args.val_set_size)
        tokenized_data = None

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    if args.adapter_name == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
        )

    config = get_peft_config(args)
    model = get_peft_model(model, config)

    init_method = normalise_init_method(args.init_method)
    initialization_transform = []
    if init_method == "kaiming":
        initialize_lora(
            model,
            init_para=args.init_para,
            lora_A_std=args.lora_A_std,
            lora_B_std=args.lora_B_std,
            lora_A_scale=args.lora_A_scale,
            lora_B_scale=args.lora_B_scale,
            normal=args.lora_normal,
            freeze_A=False,
            freeze_B=False,
        )
    elif init_method == "orthogonal_a2":
        initialization_transform = initialize_orthogonal_lora_a2(
            model,
            beta=args.lora_A_std,
        )
    elif init_method in {"eva", "eva_a2"}:
        eva_dataloader = build_eva_dataloader(
            train_data,
            data_collator,
            args.micro_batch_size,
            args.eva_max_samples,
        )
        logger.info(
            "Computing EVA initialization from %s examples.",
            len(eva_dataloader.dataset),
        )
        initialize_lora_eva_weights(model, dataloader=eva_dataloader)
        if init_method == "eva_a2":
            initialization_transform = rescale_lora_a_to_a2(
                model,
                beta=args.lora_A_std,
                preserve_product=False,
            )
    elif init_method == "pissa_a2":
        initialization_transform = rescale_lora_a_to_a2(
            model,
            beta=args.lora_A_std,
            preserve_product=True,
        )

    initialization_payload = {
        "init_method": args.init_method,
        "normalised_init_method": init_method,
        "beta": (
            float(args.lora_A_std)
            if init_method in {"kaiming", "orthogonal_a2", "eva_a2", "pissa_a2"}
            else None
        ),
        "lora_alpha": float(args.lora_alpha),
        "lora_rank": int(args.lora_r),
        "lora_dropout": float(args.lora_dropout),
        "eva_rho": float(args.eva_rho) if init_method in {"eva", "eva_a2"} else None,
        "eva_max_samples": (
            int(args.eva_max_samples) if init_method in {"eva", "eva_a2"} else None
        ),
        "transform": initialization_transform,
        "adapters": summarize_lora_initialization(model),
    }
    initialization_stats_path = os.path.join(
        args.output_dir,
        "initialization_stats.json",
    )
    with open(initialization_stats_path, "w") as handle:
        json.dump(initialization_payload, handle, indent=2)
    logger.info(f"Saved initialization diagnostics to {initialization_stats_path}")

    if args.skip_save_init_ckpt:
        logger.info(
            "Skipping init checkpoint save because --skip_save_init_ckpt was set."
        )
    else:
        model.save_pretrained(os.path.join(args.output_dir, "init_ckpt"))
    model.print_trainable_parameters()

    # Debug: Check which parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {trainable_params}")
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    logger.info(model)

    fp16, bf16 = get_precision_flags(args)
    save_strategy = "no" if args.skip_save_adapter else "steps"
    training_args = make_loraplus_training_args(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        seed=args.seed,
        data_seed=args.seed,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=fp16,
        bf16=bf16,
        logging_steps=args.logging_steps,
        optim="adamw_torch",
        eval_strategy="steps" if args.val_set_size > 0 else "no",
        save_strategy=save_strategy,
        eval_steps=args.eval_step if args.val_set_size > 0 else None,
        save_steps=args.save_step if not args.skip_save_adapter else None,
        output_dir=args.output_dir,
        save_total_limit=1 if not args.skip_save_adapter else 0,
        load_best_model_at_end=True
        if args.val_set_size > 0 and not args.skip_save_adapter
        else False,
        report_to=args.report_to,
        run_name=None,
        gradient_checkpointing=args.use_gradient_checkpointing,
        loraplus_lr_ratio=args.loraplus_lr_ratio,
        loraplus_lr_embedding=args.loraplus_lr_embedding,
    )

    callbacks = []
    if args.log_coherence_stats:
        coherence_output_dir = args.coherence_output_dir or os.path.join(
            args.output_dir,
            "coherence_stats",
        )
        probe_batch = build_coherence_probe_batch(
            tokenized_data,
            data_collator,
            args.coherence_probe_size,
            args.coherence_probe_seed,
        )
        callbacks.append(
            LoraCoherenceCallback(
                model=model,
                probe_batch=probe_batch,
                output_dir=coherence_output_dir,
                seed=args.seed,
                rank=args.lora_r,
                init_para=args.init_para,
                init_std=args.lora_A_std,
                lr_a=args.learning_rate,
                lr_ratio=args.loraplus_lr_ratio,
                max_modules=args.coherence_max_modules,
                max_tokens=args.coherence_max_tokens,
                max_rows=args.coherence_max_rows,
                log_fractions=parse_number_list(args.coherence_log_fractions, float),
                log_steps=parse_number_list(args.coherence_log_steps, int),
                every_n_steps=args.coherence_every_n_steps,
            )
        )
        logger.info(f"Logging coherence statistics to {coherence_output_dir}")

    if args.log_training_dynamics:
        dynamics_output_dir = args.dynamics_output_dir or os.path.join(
            args.output_dir,
            "training_dynamics",
        )
        callbacks.append(
            LoraTrainingDynamicsCallback(
                model=model,
                output_dir=dynamics_output_dir,
                seed=args.seed,
                rank=args.lora_r,
                init_para=args.init_para,
                init_std=args.lora_A_std,
                lr_a=args.learning_rate,
                lr_ratio=args.loraplus_lr_ratio,
                max_modules=args.dynamics_max_modules,
                log_fractions=parse_number_list(args.dynamics_log_fractions, float),
                log_steps=parse_number_list(args.dynamics_log_steps, int),
                every_n_steps=args.dynamics_every_n_steps,
            )
        )
        logger.info(f"Logging LoRA training dynamics to {dynamics_output_dir}")

    if args.log_feature_decomposition:
        decomposition_output_dir = args.decomposition_output_dir or os.path.join(
            args.output_dir,
            "feature_decomposition",
        )
        decomposition_probe_batch = build_coherence_probe_batch(
            tokenized_data,
            data_collator,
            args.decomposition_probe_size,
            args.decomposition_probe_seed,
        )
        callbacks.append(
            LoraFeatureDecompositionCallback(
                model=model,
                probe_batch=decomposition_probe_batch,
                output_dir=decomposition_output_dir,
                seed=args.seed,
                rank=args.lora_r,
                init_para=args.init_para,
                init_std=args.lora_A_std,
                lr_a=args.learning_rate,
                lr_ratio=args.loraplus_lr_ratio,
                max_modules=args.decomposition_max_modules,
                max_tokens=args.decomposition_max_tokens,
                log_fractions=parse_number_list(
                    args.decomposition_log_fractions, float
                ),
                log_steps=parse_number_list(args.decomposition_log_steps, int),
                every_n_steps=args.decomposition_every_n_steps,
            )
        )
        logger.info(f"Logging LoRA feature decomposition to {decomposition_output_dir}")

    if args.log_width_alignment:
        alignment_output_dir = args.alignment_output_dir or os.path.join(
            args.output_dir,
            "width_alignment",
        )
        callbacks.append(
            LoraWidthAlignmentCallback(
                model=model,
                output_dir=alignment_output_dir,
                model_name=Path(args.base_model.rstrip("/")).name,
                seed=args.seed,
                rank=args.lora_r,
                init_para=args.init_para,
                init_std=args.lora_A_std,
                lr_a=args.learning_rate,
                lr_ratio=args.loraplus_lr_ratio,
                module_pattern=args.alignment_module_pattern,
                max_modules=args.alignment_max_modules,
                max_tokens=args.alignment_max_tokens,
                log_fractions=parse_number_list(args.alignment_log_fractions, float),
                log_steps=parse_number_list(args.alignment_log_steps, int),
                every_n_steps=args.alignment_every_n_steps,
                shuffled_control=not args.disable_alignment_shuffled_control,
            )
        )
        logger.info(f"Logging minibatch width alignment to {alignment_output_dir}")

    trainer = LoraPlusTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    model.config.use_cache = False
    train_result = trainer.train()
    if args.skip_save_adapter:
        logger.info("Skipping adapter save because --skip_save_adapter was set.")
    else:
        model.save_pretrained(args.output_dir)

    final_metrics = dict(train_result.metrics)
    final_metrics["average_training_loss"] = float(train_result.training_loss)
    final_metrics["last_logged_train_loss"] = get_last_logged_train_loss(
        trainer.state.log_history
    )
    final_metrics["final_training_loss"] = final_metrics["last_logged_train_loss"]
    final_metrics["num_train_epochs"] = float(args.num_epochs)
    final_metrics["global_step"] = int(trainer.state.global_step)
    final_metrics["epoch"] = (
        float(trainer.state.epoch) if trainer.state.epoch is not None else None
    )
    final_metrics["output_dir"] = args.output_dir
    final_metrics["base_model"] = args.base_model
    final_metrics["data_path"] = args.data_path
    final_metrics["init_method"] = args.init_method
    final_metrics["init_para"] = args.init_para
    final_metrics["initialization_beta"] = initialization_payload["beta"]
    final_metrics["learning_rate_a"] = float(args.learning_rate)
    final_metrics["learning_rate_ratio_b_over_a"] = (
        float(args.loraplus_lr_ratio) if args.loraplus_lr_ratio is not None else 1.0
    )
    final_metrics["learning_rate_b"] = (
        final_metrics["learning_rate_a"] * final_metrics["learning_rate_ratio_b_over_a"]
    )
    final_metrics["lora_rank"] = int(args.lora_r)
    final_metrics["lora_alpha"] = float(args.lora_alpha)
    final_metrics["lora_dropout"] = float(args.lora_dropout)
    final_metrics["seed"] = int(args.seed)

    train_log_history_path = os.path.join(args.output_dir, "train_log_history.json")
    with open(train_log_history_path, "w") as handle:
        json.dump(trainer.state.log_history, handle, indent=2)
    logger.info(f"Saved training log history to {train_log_history_path}")

    metrics_path = os.path.join(args.output_dir, "final_train_metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(final_metrics, handle, indent=2)
    logger.info(f"Saved final train metrics to {metrics_path}")
    logger.info(f"Final training loss: {final_metrics['final_training_loss']}")
    if final_metrics["last_logged_train_loss"] is not None:
        logger.info(
            f"Last logged train loss: {final_metrics['last_logged_train_loss']}"
        )

    end_time = datetime.now()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"total time: {int(hours)} hours, {int(minutes)} minutes")
