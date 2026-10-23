#!/usr/bin/env python3

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


INIT_ORDER = ["A", "AA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize final training loss from experiments/run_llama_test.sh outputs."
        )
    )
    parser.add_argument(
        "--input_root", type=Path, default=Path("./log_rank_ratio_robustness2")
    )
    parser.add_argument("--dataset", type=str, default="math_10k")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--summary_name",
        type=str,
        default="rank_ratio_robustness_train_loss_summary.json",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="rank_ratio_robustness_train_loss_runs2.csv",
    )
    return parser.parse_args()


def canonical_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def parse_init_dir_name(name: str) -> Optional[Dict[str, str]]:
    if "_Kaiming_" not in name:
        return None

    prefix, suffix = name.split("_Kaiming_", 1)
    if "_" not in prefix:
        return None
    model_tag, adapter_name = prefix.rsplit("_", 1)
    match = re.fullmatch(
        r"(?P<init_para>.+?)_A(?P<a_scale>[^_]+)_B(?P<b_scale>[^_]+)_normal(?P<normal>[^_]+)_init_std(?P<init_std>.+)",
        suffix,
    )
    if not match:
        return None

    values = match.groupdict()
    values["model_tag"] = model_tag
    values["adapter_name"] = adapter_name
    return values


def parse_run_dir_name(name: str) -> Optional[Dict[str, str]]:
    match = re.search(
        r"_r(?P<rank>\d+)_a(?P<alpha>[^_]+)_lr(?P<lr>[^_]+)_ratio(?P<ratio>[^_]+)_s(?P<seed>[^_]+)$",
        name,
    )
    if not match:
        return None
    return match.groupdict()


def parse_float(value: str) -> float:
    return float(value)


def make_null_matrix(rows: int, cols: int) -> List[List[Optional[float]]]:
    return [[None for _ in range(cols)] for _ in range(rows)]


def make_path_grid(rows: int, cols: int) -> List[List[Optional[str]]]:
    return [[None for _ in range(cols)] for _ in range(rows)]


def round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def sort_init_key(value: str) -> tuple[int, str]:
    if value in INIT_ORDER:
        return (INIT_ORDER.index(value), value)
    return (len(INIT_ORDER), value)


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset_root = args.input_root / args.dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    records: List[Dict[str, Any]] = []
    for json_path in dataset_root.glob("*/*/finetuned_result/final_train_metrics.json"):
        init_dir_name = json_path.parent.parent.parent.name
        run_dir_name = json_path.parent.parent.name
        init_info = parse_init_dir_name(init_dir_name)
        run_info = parse_run_dir_name(run_dir_name)
        if init_info is None or run_info is None:
            continue

        try:
            with json_path.open() as handle:
                payload = json.load(handle)
            final_training_loss = payload.get(
                "final_training_loss", payload.get("train_loss")
            )
            if final_training_loss is None:
                raise KeyError("final_training_loss")
            loss = float(final_training_loss)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"Skipping malformed result file: {json_path} ({exc})", file=sys.stderr
            )
            continue

        records.append(
            {
                "metrics_path": str(json_path),
                "model_tag": init_info["model_tag"],
                "adapter_name": init_info["adapter_name"],
                "init_para": init_info["init_para"],
                "init_std": float(init_info["init_std"]),
                "a_scale": float(init_info["a_scale"]),
                "b_scale": float(init_info["b_scale"]),
                "rank": int(run_info["rank"]),
                "alpha": float(run_info["alpha"]),
                "lr": run_info["lr"],
                "ratio": float(run_info["ratio"]),
                "seed": run_info["seed"],
                "final_training_loss": loss,
                "train_runtime": payload.get("train_runtime"),
                "num_train_epochs": payload.get("num_train_epochs"),
                "global_step": payload.get("global_step"),
            }
        )

    records.sort(
        key=lambda record: (
            sort_init_key(str(record["init_para"])),
            float(record["init_std"]),
            int(record["rank"]),
            parse_float(str(record["lr"])),
            float(record["ratio"]),
        )
    )
    return records


def build_nested_runs(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for record in records:
        init_para = str(record["init_para"])
        std_key = canonical_number(float(record["init_std"]))
        rank_key = str(int(record["rank"]))
        lr_key = str(record["lr"])
        ratio_key = canonical_number(float(record["ratio"]))

        nested.setdefault(init_para, {})
        nested[init_para].setdefault(std_key, {})
        nested[init_para][std_key].setdefault(rank_key, {})
        nested[init_para][std_key][rank_key].setdefault(lr_key, {})
        nested[init_para][std_key][rank_key][lr_key][ratio_key] = {
            "final_training_loss": round(float(record["final_training_loss"]), 6),
            "metrics_path": str(record["metrics_path"]),
        }
    return nested


def build_table(
    records: List[Dict[str, Any]],
    row_values: List[int],
    col_values: List[Any],
    row_field: str,
    col_field: str,
    filter_field: str,
    filter_value: Any,
) -> Dict[str, Any]:
    row_index = {int(value): idx for idx, value in enumerate(row_values)}
    col_index = {col_values[idx]: idx for idx in range(len(col_values))}
    table: Dict[str, Any] = {}

    init_values = sorted(
        {str(record["init_para"]) for record in records}, key=sort_init_key
    )
    std_values = sorted({float(record["init_std"]) for record in records})

    for init_para in init_values:
        table[init_para] = {}
        for init_std in std_values:
            values = make_null_matrix(len(row_values), len(col_values))
            paths = make_path_grid(len(row_values), len(col_values))

            for record in records:
                if str(record["init_para"]) != init_para:
                    continue
                if float(record["init_std"]) != float(init_std):
                    continue
                if record[filter_field] != filter_value:
                    continue

                row = row_index[int(record[row_field])]
                col = col_index[record[col_field]]
                values[row][col] = round(float(record["final_training_loss"]), 6)
                paths[row][col] = str(record["metrics_path"])

            table[init_para][canonical_number(init_std)] = {
                "Ranks": list(row_values),
                "Columns": [
                    canonical_number(float(value))
                    if isinstance(value, (int, float))
                    else str(value)
                    for value in col_values
                ],
                "Values": values,
                "Paths": paths,
            }

    return table


def build_summary(
    args: argparse.Namespace, records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    init_values = sorted(
        {str(record["init_para"]) for record in records}, key=sort_init_key
    )
    std_values = sorted({float(record["init_std"]) for record in records})
    rank_values = sorted({int(record["rank"]) for record in records})
    lr_values = sorted({str(record["lr"]) for record in records}, key=parse_float)
    ratio_values = sorted({float(record["ratio"]) for record in records})

    summary: Dict[str, Any] = {
        "_meta": {
            "input_root": str(args.input_root),
            "dataset": args.dataset,
            "metric": "final_training_loss",
            "matched_runs": len(records),
            "init_paras": init_values,
            "init_stds": [canonical_number(value) for value in std_values],
            "ranks": rank_values,
            "lrs": lr_values,
            "ratios": [canonical_number(value) for value in ratio_values],
        },
        "records": [
            {
                "init_para": str(record["init_para"]),
                "init_std": canonical_number(float(record["init_std"])),
                "rank": int(record["rank"]),
                "lr": str(record["lr"]),
                "ratio": canonical_number(float(record["ratio"])),
                "final_training_loss": round(float(record["final_training_loss"]), 6),
                "metrics_path": str(record["metrics_path"]),
            }
            for record in records
        ],
        "by_init": build_nested_runs(records),
        "tables": {
            "lr_sweep_at_ratio_1": {
                "_constraint": {"ratio": "1"},
                **build_table(
                    records=records,
                    row_values=rank_values,
                    col_values=lr_values,
                    row_field="rank",
                    col_field="lr",
                    filter_field="ratio",
                    filter_value=1.0,
                ),
            },
            "ratio_sweep_at_lr_1e-5": {
                "_constraint": {"lr": "1e-5"},
                **build_table(
                    records=records,
                    row_values=rank_values,
                    col_values=ratio_values,
                    row_field="rank",
                    col_field="ratio",
                    filter_field="lr",
                    filter_value="1e-5",
                ),
            },
        },
    }
    return summary


def write_csv(csv_path: Path, records: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "init_para",
                "init_std",
                "rank",
                "lr",
                "ratio",
                "final_training_loss",
                "metrics_path",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    str(record["init_para"]),
                    canonical_number(float(record["init_std"])),
                    int(record["rank"]),
                    str(record["lr"]),
                    canonical_number(float(record["ratio"])),
                    round(float(record["final_training_loss"]), 6),
                    str(record["metrics_path"]),
                ]
            )


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_root / args.dataset

    records = load_records(args)
    summary = build_summary(args, records)

    summary_path = args.output_dir / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    csv_path = args.output_dir / args.csv_name
    write_csv(csv_path, records)

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()
