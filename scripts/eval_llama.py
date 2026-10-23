"""Evaluate saved LoRA adapters on the LLM-Adapters generation benchmarks."""

import os
import re
import sys
import copy
import json
from pathlib import Path
import torch
import socket
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from peft import PeftConfig, PeftModel
from initialization import minus_init_lora
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
)

server_name = socket.gethostname()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
math_datasets = ["AddSub", "MultiArith", "SingleEq", "gsm8k", "AQuA", "SVAMP"]
commonsense_datasets = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--dataset", choices=math_datasets + commonsense_datasets, default="boolq"
    )
    parser.add_argument(
        "--init_para",
        type=str,
        default=None,
        help=(
            "Pass the same init_para used for training. Any value containing "
            "'reset' reloads lora_weights/init_ckpt and subtracts the initialized "
            "LoRA update before the finetuned adapter is loaded; use AB_reset "
            "for non-zero A/B init plus subtraction."
        ),
    )
    parser.add_argument("--adapter", type=str, default="lora")
    parser.add_argument("--base_model", default="NousResearch/Meta-Llama-3-8B")
    parser.add_argument("--lora_weights", default="log/finetuned_result")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--data_root", type=Path, default=REPO_ROOT / "LLM-Adapters-main"
    )
    parser.add_argument(
        "--data_file", type=Path, default=None, help="Override dataset test.json"
    )
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit examples for validation runs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./log/debug",
        help="output directory for logs and models",
    )
    return parser.parse_args()


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


def load_data(args):
    file_path = (
        args.data_file or args.data_root / "dataset" / args.dataset / "test.json"
    )
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Missing {file_path}. Run scripts/prepare_data.py first."
        )
    with file_path.open() as handle:
        json_data = json.load(handle)
    if args.max_samples is not None:
        json_data = json_data[: args.max_samples]
    if not json_data:
        raise ValueError(f"Evaluation dataset is empty: {file_path}")
    return json_data


def create_batch(dataset, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches = []
    num_batch = (
        len(dataset) // batch_size
        if len(dataset) % batch_size == 0
        else len(dataset) // batch_size + 1
    )
    for i in range(num_batch):
        batch = dataset[i * batch_size : min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def extract_answer(dataset, sentence: str) -> float:
    if dataset == "boolq":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"true|false", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == "piqa":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"solution1|solution2", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == "hellaswag":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"ending1|ending2|ending3|ending4", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == "winogrande":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"option1|option2", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


def extract_answer_number(dataset, sentence: str) -> float:
    dataset = dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
        if not pred:
            return float("inf")
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(" not support dataset: {}".format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError:
            pred_answer = float("inf")
    return pred_answer


def extract_answer_letter(dataset, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        if not pred_answers:
            return ""
        return pred_answers[0]
    else:
        return ""


def generate_predict_from_answer(dataset, output, label, miss):
    flag = False
    if dataset in commonsense_datasets:
        predict = extract_answer(dataset, output)
        if label == predict:
            flag = True
    elif dataset == "AQuA":
        predict = extract_answer_letter(dataset, output)
        if label == predict:
            flag = True
    else:
        if isinstance(label, str):
            label = float(label)
        predict = extract_answer_number(dataset, output)
        if abs(label - predict) <= miss:
            flag = True
    return predict, flag


def load_model(args) -> tuple:
    base_model = args.base_model
    lora_weights = args.lora_weights

    dtype = (
        torch.float32
        if not torch.cuda.is_available()
        else torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    if args.dtype != "auto":
        dtype = getattr(torch, args.dtype)
    metadata_path = Path(lora_weights) / "run_config.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    init_para = args.init_para or metadata.get("init_para", "A")
    adapter_config = PeftConfig.from_pretrained(lora_weights)
    init_method = metadata.get("init_method", str(adapter_config.init_lora_weights))
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
    )
    model = minus_init_lora(model, init_para, lora_weights, init_method=init_method)
    adapter_config.init_lora_weights = False
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        config=adapter_config,
        torch_device="cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    model.eval()
    return tokenizer, model


def write_eval_result(output_dir, dataset, output_data, correct, total, start_time):
    end_time = datetime.now()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"total time: {int(hours)} hours, {int(minutes)} minutes")

    os.makedirs(output_dir, exist_ok=True)
    save_file = f"{output_dir}/{dataset}.json"
    with open(save_file, "w+") as f:
        json.dump(output_data, f, indent=4)
    acc_file = f"{output_dir}/accuracy.json"
    if os.path.exists(acc_file):
        acc = json.load(open(acc_file, "r"))
    else:
        acc = {"all": {"avg_accuracy": 0}}
    acc[dataset] = {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "time": f"{int(hours)}:{int(minutes)}",
        "server_name": server_name,
    }
    dataset_list = []
    accuracy_list = []
    for k, v in acc.items():
        if k != "all":
            dataset_list.append(k)
            accuracy_list.append(v["accuracy"])
    acc["all"] = {
        "avg_accuracy": sum(accuracy_list) / len(accuracy_list),
        "accuracy_list": accuracy_list,
        "dataset_list": dataset_list,
        "time": f"{end_time.strftime('%Y-%m-%d %H:%M:%S')}",
    }
    with open(acc_file, "w+") as f:
        json.dump(acc, f, indent=4)


def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    set_global_seed(args.seed)
    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)
    start_time = datetime.now()

    def evaluate(instructions):
        prompts = [generate_prompt(instruction) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        model_device = model.get_input_embeddings().weight.device
        input_ids = inputs["input_ids"].to(model_device)
        max_new_tokens = args.max_new_tokens or (
            32 if args.dataset in commonsense_datasets else 256
        )
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"].to(model_device),
                do_sample=False,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        generated = sequences[:, input_ids.shape[1] :]
        outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [output.split("### Response:")[-1].strip() for output in outputs]

    total = len(batches)
    correct, current, output_data = 0, 0, []
    miss = 0.001
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get("instruction") for data in batch]
        outputs = evaluate(instructions)
        for data, output in zip(batch, outputs):
            label = data.get("answer")
            predict, flag = generate_predict_from_answer(
                args.dataset, output, label, miss
            )
            if flag == True:
                correct += 1
            new_data = copy.deepcopy(data)
            new_data["output_pred"] = output
            new_data["pred"] = (
                None
                if isinstance(predict, float) and not np.isfinite(predict)
                else predict
            )
            new_data["flag"] = flag
            output_data.append(new_data)
            print(new_data)
        print(f"\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}")
        pbar.update(1)
    pbar.close()
    print("\n test finished")

    write_eval_result(
        args.output_dir, args.dataset, output_data, correct, current, start_time
    )


if __name__ == "__main__":
    main()
