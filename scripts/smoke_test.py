#!/usr/bin/env python3
"""Run training, diagnostics, checkpoint reload, and evaluation without downloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def create_fixtures(directory):
    from datasets import ClassLabel, Dataset, DatasetDict
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
        RobertaConfig,
        RobertaForSequenceClassification,
    )
    import torch

    torch.manual_seed(42)
    words = [
        "<pad>",
        "<unk>",
        "<eos>",
        "<bos>",
        "Below",
        "is",
        "an",
        "instruction",
        "that",
        "describes",
        "a",
        "task",
        "Write",
        "response",
        "appropriately",
        "completes",
        "the",
        "request",
        "Instruction",
        "Response",
        "#",
        ".",
        ":",
        "Add",
        "1",
        "and",
        "2",
        "3",
        "good",
        "bad",
        "movie",
    ]
    vocab = {word: index for index, word in enumerate(words)}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<bos>",
        model_max_length=256,
    )
    llm_dir, glue_dir = directory / "tiny_llama", directory / "tiny_roberta"
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            pad_token_id=0,
            bos_token_id=3,
            eos_token_id=2,
        )
    ).save_pretrained(llm_dir)
    tokenizer.save_pretrained(llm_dir)
    RobertaForSequenceClassification(
        RobertaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=258,
            pad_token_id=0,
            bos_token_id=3,
            eos_token_id=2,
            num_labels=2,
        )
    ).save_pretrained(glue_dir)
    tokenizer.save_pretrained(glue_dir)
    train_file = directory / "math.json"
    train_file.write_text(
        json.dumps([{"instruction": "Add 1 and 2", "input": "", "output": "3"}] * 8)
    )
    test_file = directory / "test.json"
    test_file.write_text(json.dumps([{"instruction": "Add 1 and 2", "answer": "3"}]))
    data = Dataset.from_dict(
        {"sentence": ["good movie", "bad movie"] * 4, "label": [1, 0] * 4}
    )
    data = data.cast_column("label", ClassLabel(names=["negative", "positive"]))
    glue_data = directory / "data" / "glue" / "sst2"
    glue_data.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict(train=data, validation=data).save_to_disk(glue_data)
    return llm_dir, glue_dir, train_file, test_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache" / "smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--qlora", action="store_true", help="Also validate NF4 training on CUDA"
    )
    parser.add_argument(
        "--all-initializers",
        action="store_true",
        help="Train and reload every supported paper initializer",
    )
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        import torch

        if not torch.cuda.is_available():
            parser.error("--device cuda requires an available CUDA GPU")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    llm, roberta, train_file, test_file = create_fixtures(args.output_dir)
    adapter = args.output_dir / "adapter"
    python = sys.executable
    commands = [
        [
            python,
            str(ROOT / "scripts" / "finetune.py"),
            "--base_model",
            str(llm),
            "--data_path",
            str(train_file),
            "--output_dir",
            str(adapter),
            "--dtype",
            "float32",
            "--batch_size",
            "4",
            "--micro_batch_size",
            "2",
            "--max_steps",
            "3",
            "--warmup_steps",
            "0",
            "--logging_steps",
            "1",
            "--lora_r",
            "4",
            "--lora_alpha",
            "4",
            "--init_para",
            "AA",
            "--lora_A_scale",
            "1",
            "--lora_dropout",
            "0",
            "--loraplus_lr_ratio",
            "20",
            "--disable_dataset_cache",
            "--log_coherence_stats",
            "--coherence_max_modules",
            "2",
            "--coherence_max_tokens",
            "4",
            "--coherence_max_rows",
            "4",
            "--coherence_every_n_steps",
            "1",
            "--log_training_dynamics",
            "--dynamics_max_modules",
            "2",
            "--dynamics_every_n_steps",
            "1",
            "--log_feature_decomposition",
            "--decomposition_max_modules",
            "2",
            "--decomposition_max_tokens",
            "4",
            "--decomposition_every_n_steps",
            "1",
            "--log_width_alignment",
            "--alignment_max_modules",
            "2",
            "--alignment_max_tokens",
            "4",
            "--alignment_every_n_steps",
            "1",
        ],
        [
            python,
            str(ROOT / "scripts" / "eval_llama.py"),
            "--base_model",
            str(llm),
            "--lora_weights",
            str(adapter),
            "--data_file",
            str(test_file),
            "--dataset",
            "gsm8k",
            "--output_dir",
            str(args.output_dir / "evaluation"),
            "--max_new_tokens",
            "4",
            "--num_beams",
            "1",
            "--dtype",
            "float32",
        ],
        [
            python,
            str(ROOT / "scripts" / "run_glue.py"),
            "--model_name_or_path",
            str(roberta),
            "--task_name",
            "sst2",
            "--use_local",
            "--data_dir",
            str(args.output_dir / "data"),
            "--output_dir",
            str(args.output_dir / "glue"),
            "--do_train",
            "--do_eval",
            "--max_steps",
            "2",
            "--per_device_train_batch_size",
            "4",
            "--per_device_eval_batch_size",
            "4",
            "--max_seq_length",
            "64",
            "--use_lora",
            "--target_modules",
            "query,value",
            "--lora_rank",
            "4",
            "--lora_alpha",
            "4",
            "--lora_init_method",
            "AA",
            "--lora_A_scale",
            "1",
            "--loraplus_lr_ratio",
            "20",
            "--eval_strategy",
            "no",
            "--save_strategy",
            "no",
            "--logging_steps",
            "1",
            "--report_to",
            "none",
            "--log_coherence_stats",
            "--coherence_max_modules",
            "2",
            "--coherence_max_tokens",
            "4",
            "--coherence_max_rows",
            "4",
            "--coherence_every_n_steps",
            "1",
            "--overwrite_output_dir",
        ],
        [
            python,
            str(ROOT / "scripts" / "train_toy.py"),
            "--output_dir",
            str(args.output_dir / "toy"),
            "--width",
            "16",
            "--rank",
            "4",
            "--max_steps",
            "2",
            "--synthetic",
        ],
    ]
    if args.qlora:
        if args.device != "cuda":
            parser.error("--qlora requires --device cuda")
        commands.append(
            [
                python,
                str(ROOT / "scripts" / "finetune.py"),
                "--base_model",
                str(llm),
                "--data_path",
                str(train_file),
                "--output_dir",
                str(args.output_dir / "qlora"),
                "--adapter_name",
                "qlora",
                "--max_steps",
                "2",
                "--warmup_steps",
                "0",
                "--batch_size",
                "4",
                "--micro_batch_size",
                "2",
                "--lora_r",
                "4",
                "--lora_alpha",
                "4",
                "--loraplus_lr_ratio",
                "20",
                "--disable_dataset_cache",
            ]
        )
    if args.all_initializers:
        for method in [
            "A",
            "AA",
            "AB_reset",
            "orthogonal",
            "orthogonal_a2",
            "eva",
            "eva_a2",
            "pissa",
            "pissa_a2",
            "olora",
        ]:
            directory = args.output_dir / ("initializer_" + method)
            custom = method in {"A", "AA", "AB_reset"}
            commands.append(
                [
                    python,
                    str(ROOT / "scripts" / "finetune.py"),
                    "--base_model",
                    str(llm),
                    "--data_path",
                    str(train_file),
                    "--output_dir",
                    str(directory),
                    "--dtype",
                    "float32",
                    "--max_steps",
                    "2",
                    "--warmup_steps",
                    "0",
                    "--batch_size",
                    "4",
                    "--micro_batch_size",
                    "2",
                    "--lora_r",
                    "4",
                    "--lora_alpha",
                    "4",
                    "--lora_dropout",
                    "0",
                    "--loraplus_lr_ratio",
                    "20",
                    "--init_method",
                    "Kaiming" if custom else method,
                    "--init_para",
                    method if custom else "A",
                    "--lora_A_scale",
                    "1" if method == "AA" else "0.5",
                    "--eva_max_samples",
                    "8",
                    "--disable_dataset_cache",
                    "--logging_steps",
                    "1",
                ]
            )
            commands.append(
                [
                    python,
                    str(ROOT / "scripts" / "eval_llama.py"),
                    "--base_model",
                    str(llm),
                    "--lora_weights",
                    str(directory),
                    "--data_file",
                    str(test_file),
                    "--dataset",
                    "gsm8k",
                    "--output_dir",
                    str(directory / "evaluation"),
                    "--max_new_tokens",
                    "2",
                    "--num_beams",
                    "1",
                    "--dtype",
                    "float32",
                ]
            )
    for index, command in enumerate(commands):
        log = args.output_dir / f"smoke_{index}.log"
        print("Running", Path(command[1]).name, "→", log, flush=True)
        with log.open("w") as handle:
            result = subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
            )
        if result.returncode:
            print(log.read_text(), file=sys.stderr)
            raise SystemExit(result.returncode)
    expected = [
        adapter / "adapter_config.json",
        adapter / "final_train_metrics.json",
        adapter / "coherence_stats" / "coherence_stats.jsonl",
        adapter / "training_dynamics" / "training_dynamics.jsonl",
        adapter / "feature_decomposition" / "feature_decomposition.jsonl",
        adapter / "width_alignment" / "width_alignment.jsonl",
        args.output_dir / "evaluation" / "accuracy.json",
        args.output_dir / "glue" / "eval_results.json",
        args.output_dir / "toy" / "toy_metrics.json",
    ]
    for path in expected:
        if not path.exists() or not path.stat().st_size:
            raise AssertionError(f"Missing smoke-test output: {path}")
    metrics = json.loads((adapter / "final_train_metrics.json").read_text())
    if metrics["global_step"] != 3:
        raise AssertionError(
            "The LLM training loop did not finish three optimizer steps"
        )
    for path in adapter.glob("*/*.jsonl"):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len({record["step"] for record in records}) != len(records):
            raise AssertionError(f"Duplicate diagnostic checkpoints in {path}")
        if any(not record.get("n_modules_logged") for record in records):
            raise AssertionError(f"No modules measured in {path}")
        if path.parent.name == "feature_decomposition":
            for record in records:
                error = record["aggregate"][
                    "reconstruction_relative_error_across_modules_mean"
                ]
                if error > 1e-5:
                    raise AssertionError(f"Feature reconstruction error: {error}")
    print(
        "Training, all diagnostics, adapter reload, generation evaluation, GLUE, and toy smoke checks passed."
    )


if __name__ == "__main__":
    main()
