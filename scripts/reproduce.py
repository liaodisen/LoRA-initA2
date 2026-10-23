#!/usr/bin/env python3
"""Expand the paper's explicit experiment grids and run them locally."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MATH = ["AddSub", "MultiArith", "SingleEq", "SVAMP", "gsm8k", "AQuA"]
COMMONSENSE = [
    "openbookqa",
    "ARC-Challenge",
    "winogrande",
    "piqa",
    "social_i_qa",
    "ARC-Easy",
    "boolq",
    "hellaswag",
]
SCALES = {"A": 0.5, "AA": 1.0, "A3": 1.5, "A4": 2.0, "AB_reset": 0.5}


def load_recipes(path=ROOT / "experiments" / "paper.json"):
    payload = json.loads(Path(path).read_text())
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported recipe schema")
    return payload["experiments"]


def expand_recipe(recipe):
    for grid in [recipe["grid"], *recipe.get("scenarios", [])]:
        keys = list(grid)
        for values in itertools.product(*(grid[key] for key in keys)):
            cell = {**recipe["defaults"], **dict(zip(keys, values))}
            cell.pop("methods", None)
            if "lr_b" in cell:
                cell["ratio"] = cell["lr_b"] / cell["lr_a"]
            cell["alpha"] = cell["rank"] * cell.get("multiplier", 1)
            if cell["kind"] == "llm" and "epochs" not in grid:
                cell["epochs"] = (
                    1
                    if cell["dataset"] == "commonsense_170k"
                    else recipe["defaults"]["epochs"]
                )
            if cell["method"].startswith("pissa") or cell["method"] in {
                "olora",
                "AB_reset",
            }:
                cell["dropout"] = 0
            yield cell


def flags(**kwargs):
    return [
        item
        for key, value in kwargs.items()
        if value is not None
        for item in ("--" + key, str(value))
    ]


def build_commands(
    cell, recipe, run_dir, data_root, python=sys.executable, skip_eval=False
):
    c = cell
    init_method = "Kaiming" if c["method"] in SCALES else c["method"]
    scale = c.get("a_scale", SCALES.get(c["method"], 1.0))
    b_scale = c.get("b_scale", scale)
    if c["kind"] == "toy":
        return [
            [
                python,
                str(ROOT / "scripts" / "train_toy.py"),
                *flags(
                    output_dir=run_dir,
                    width=c["width"],
                    rank=c["rank"],
                    beta=c["beta"],
                    init=c["method"],
                    lr_a=c["lr_a"],
                    epochs=c["epochs"],
                    seed=c["seed"],
                    max_steps=recipe.get("max_steps", -1),
                ),
            ]
        ]
    if c["kind"] == "glue":
        cmd = [
            python,
            str(ROOT / "scripts" / "run_glue.py"),
            *flags(
                model_name_or_path=c["model"],
                task_name=c["task"],
                output_dir=run_dir,
                target_modules="query,value",
                lora_init_method=c["method"],
                lora_A_scale=scale,
                lora_B_scale=b_scale,
                lora_A_std=c["beta"],
                lora_B_std=c["beta"],
                lora_normal=False,
                freeze_lora_A=c.get("freeze_a", False),
                freeze_lora_B=False,
                lora_rank=c["rank"],
                lora_alpha=c["alpha"],
                lora_dropout=c["dropout"],
                learning_rate=c["lr_a"],
                loraplus_lr_ratio=c["ratio"],
                num_train_epochs=c["epochs"],
                per_device_train_batch_size=32,
                per_device_eval_batch_size=128,
                max_seq_length=128,
                adam_beta1=0.9,
                adam_beta2=0.99,
                adam_epsilon=1e-8,
                weight_decay=0,
                warmup_steps=0,
                lr_scheduler_type="linear",
                optim="adamw_torch",
                eval_strategy="steps",
                eval_steps=2000,
                save_strategy="no",
                logging_steps=10,
                report_to="none",
                seed=c["seed"],
                data_seed=c["seed"],
                fp16=c.get("fp16", True),
            ),
            "--do_train",
            "--do_eval",
            "--use_lora",
        ]
        if c.get("local_glue"):
            cmd += ["--use_local", *flags(data_dir=c["local_glue"])]
        return [cmd]
    train_dir = run_dir / "finetuned_result"
    cmd = [
        python,
        str(ROOT / "scripts" / "finetune.py"),
        *flags(
            base_model=c["model"],
            data_path=data_root / "ft-training_set" / (c["dataset"] + ".json"),
            output_dir=train_dir,
            init_method=init_method,
            init_para=c["method"] if init_method == "Kaiming" else "A",
            lora_A_scale=scale,
            lora_B_scale=b_scale,
            lora_A_std=c["beta"],
            lora_B_std=c["beta"],
            lora_normal=False,
            lora_r=c["rank"],
            lora_alpha=c["alpha"],
            lora_dropout=c["dropout"],
            adapter_name=c["adapter"],
            learning_rate=c["lr_a"],
            loraplus_lr_ratio=c["ratio"],
            num_epochs=c["epochs"],
            cutoff_len=c["cutoff_len"],
            batch_size=16,
            micro_batch_size=8,
            weight_decay=0,
            warmup_steps=100,
            val_set_size=0,
            logging_steps=10,
            save_step=500,
            report_to="none",
            seed=c["seed"],
            dtype=c["dtype"],
            eva_max_samples=recipe.get("eva_max_samples", 256),
            eva_rho=recipe.get("eva_rho", 1),
        ),
        "--target_modules",
        "q_proj",
        "k_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "--use_gradient_checkpointing",
    ]
    for diagnostic in recipe.get("diagnostics", []):
        cmd.append(
            {
                "coherence": "--log_coherence_stats",
                "dynamics": "--log_training_dynamics",
                "decomposition": "--log_feature_decomposition",
                "alignment": "--log_width_alignment",
            }[diagnostic]
        )
        if diagnostic == "decomposition" and "decomposition_max_tokens" in recipe:
            cmd.extend(
                ["--decomposition_max_tokens", str(recipe["decomposition_max_tokens"])]
            )
    commands = [cmd]
    if not skip_eval and recipe.get("evaluate", True):
        benchmarks = MATH if c["dataset"] == "math_10k" else COMMONSENSE
        for benchmark in benchmarks:
            commands.append(
                [
                    python,
                    str(ROOT / "scripts" / "eval_llama.py"),
                    *flags(
                        base_model=c["model"],
                        lora_weights=train_dir,
                        dataset=benchmark,
                        data_root=data_root,
                        output_dir=run_dir / "evaluated_result",
                        batch_size=10 if benchmark == "hellaswag" else 16,
                        seed=c["seed"],
                        dtype=c["dtype"],
                        adapter=c["adapter"],
                    ),
                ]
            )
    return commands


def validate_cell(cell):
    if (
        cell["rank"] <= 0
        or cell["seed"] < 0
        or cell["lr_a"] <= 0
        or cell.get("ratio", 1) <= 0
    ):
        raise ValueError(
            "Rank, learning rates, and ratio must be positive; seed must be nonnegative"
        )
    if cell["kind"] != "toy" and cell.get("alpha", 1) <= 0:
        raise ValueError("LoRA alpha must be positive")


def code_fingerprint():
    digest = hashlib.blake2b(digest_size=16)
    for path in sorted(
        [*ROOT.joinpath("src").glob("*.py"), *ROOT.joinpath("scripts").glob("*.py")]
    ):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def data_fingerprint(cell, data_root, cache):
    if cell["kind"] != "llm":
        return {}
    dataset = cell["dataset"]
    if dataset not in cache:
        paths = [data_root / "ft-training_set" / (dataset + ".json")]
        paths += [
            data_root / "dataset" / name / "test.json"
            for name in (MATH if dataset == "math_10k" else COMMONSENSE)
        ]
        cache[dataset] = {
            str(path.relative_to(data_root)): hashlib.blake2b(
                path.read_bytes()
            ).hexdigest()
            for path in paths
            if path.is_file()
        }
    return cache[dataset]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=list(load_recipes()))
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--run", action="store_true", help="Execute the printed commands"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands (the default)"
    )
    parser.add_argument(
        "--limit", type=int, help="Take this many cells after filtering"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start at this zero-based grid index"
    )
    parser.add_argument(
        "--model", help="Override backbone (a local path or Hugging Face ID)"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--method", help="Filter initializer")
    parser.add_argument("--task", help="Filter task or training dataset")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Override a cell value",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--data-root", type=Path, default=ROOT / "LLM-Adapters-main")
    parser.add_argument("--local-glue", type=Path)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    recipes = load_recipes()
    if args.list:
        for name, recipe in recipes.items():
            print(
                f"{name:26} {len(list(expand_recipe(recipe))):5} runs  {recipe['description']}"
            )
        return
    if not args.experiment:
        parser.error("Choose --experiment or --list")
    if args.run and args.dry_run:
        parser.error("Use either --run or --dry-run")
    if args.limit is not None and args.limit < 1 or args.start < 0:
        parser.error("--limit must be positive; --start must be nonnegative")
    recipe = recipes[args.experiment]
    if recipe.get("notes"):
        print("Recipe note:", recipe["notes"])
    overrides = {}
    for value in args.set:
        key, raw = value.split("=", 1)
        overrides[key] = json.loads(raw)
    cells = list(expand_recipe(recipe))
    if args.seeds:
        cells = [
            {**c, "seed": seed}
            for c in cells
            if c["seed"] == min(x["seed"] for x in cells)
            for seed in args.seeds
        ]
    cells = [
        c
        for c in cells
        if (args.method is None or c["method"] == args.method)
        and (args.task is None or c.get("task", c.get("dataset")) == args.task)
    ]
    cells = cells[args.start : args.start + args.limit if args.limit else None]
    if not cells:
        parser.error("No runs match the filters")
    fingerprint = code_fingerprint()
    data_hash_cache = {}
    for cell in cells:
        unknown = (
            set(overrides) - set(cell) - {"fp16", "local_glue", "a_scale", "b_scale"}
        )
        if unknown:
            parser.error("Unknown overrides: " + ", ".join(sorted(unknown)))
        cell.update(overrides)
        if args.model:
            cell["model"] = args.model
        if args.local_glue:
            cell["local_glue"] = str(args.local_glue.resolve())
        cell["alpha"] = cell["rank"] * cell.get("multiplier", 1)
        if "lr_b" in cell:
            cell["ratio"] = cell["lr_b"] / cell["lr_a"]
        validate_cell(cell)
        data_hashes = data_fingerprint(cell, args.data_root.resolve(), data_hash_cache)
        identity = {
            "cell": cell,
            "code": fingerprint,
            "recipe": recipe,
            "data_root": str(args.data_root.resolve()),
            "skip_eval": args.skip_eval,
            "data_fingerprints": data_hashes,
        }
        digest = hashlib.blake2b(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()[:12]
        run_dir = (
            args.output_root.resolve()
            / args.experiment
            / f"{cell['method']}_s{cell['seed']}_{digest}"
        )
        commands = build_commands(
            cell, recipe, run_dir, args.data_root.resolve(), skip_eval=args.skip_eval
        )
        print(f"\n{run_dir}")
        for command in commands:
            print(shlex.join(command))
        if not args.run:
            continue
        if (run_dir / "completed.json").exists():
            print("Completed; skipping")
            continue
        if cell["kind"] == "llm":
            required = [
                args.data_root / "ft-training_set" / (cell["dataset"] + ".json")
            ]
            if len(commands) > 1:
                required += [
                    args.data_root
                    / "dataset"
                    / command[command.index("--dataset") + 1]
                    / "test.json"
                    for command in commands[1:]
                ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                parser.error(
                    "Missing data. Run scripts/prepare_data.py first: "
                    + ", ".join(missing)
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        versions = {}
        for package in [
            "torch",
            "transformers",
            "peft",
            "datasets",
            "accelerate",
            "numpy",
        ]:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        manifest = {
            "experiment": args.experiment,
            "cell": cell,
            "commands": commands,
            "versions": versions,
            "recipe": recipe,
        }
        (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        env = {
            **os.environ,
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for index, command in enumerate(commands):
            log_path = run_dir / f"command_{index:02d}.log"
            print("Running; log:", log_path, flush=True)
            with log_path.open("w") as handle:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
                )
            if result.returncode:
                print(
                    "\n".join(log_path.read_text().splitlines()[-30:]), file=sys.stderr
                )
                raise SystemExit(result.returncode)
        (run_dir / "completed.json").write_text(
            json.dumps({"commands_completed": len(commands)}) + "\n"
        )


if __name__ == "__main__":
    main()
