#!/usr/bin/env python3
"""Collect completed reproduction runs and aggregate matched cells over seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def collect_records(root):
    records = []
    for manifest_path in sorted(root.rglob("run.json")):
        run_dir = manifest_path.parent
        if not (run_dir / "completed.json").is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        cell = manifest["cell"]
        row = {"experiment": manifest["experiment"], **cell, "run_dir": str(run_dir)}
        if cell["kind"] == "toy":
            metrics = json.loads((run_dir / "toy_metrics.json").read_text())
            row.update(
                accuracy=100 * metrics["accuracy"],
                loss=metrics["train_loss"],
                validation_loss=metrics["best_validation_loss"],
            )
        elif cell["kind"] == "glue":
            metrics = json.loads((run_dir / "eval_results.json").read_text())
            row.update(
                accuracy=100 * metrics["eval_accuracy"], loss=metrics["eval_loss"]
            )
            if "eval_accuracy_mm" in metrics:
                row["accuracy_mm"] = 100 * metrics["eval_accuracy_mm"]
        else:
            metrics = json.loads(
                (run_dir / "finetuned_result" / "final_train_metrics.json").read_text()
            )
            row["loss"] = metrics.get("last_logged_train_loss")
            row["average_train_loss"] = metrics["train_loss"]
            accuracy_path = run_dir / "evaluated_result" / "accuracy.json"
            if accuracy_path.exists():
                accuracy = json.loads(accuracy_path.read_text())
                expected = 6 if cell["dataset"] == "math_10k" else 8
                scores = [
                    info["accuracy"] for name, info in accuracy.items() if name != "all"
                ]
                names = {name for name in accuracy if name != "all"}
                expected_names = (
                    {"AddSub", "MultiArith", "SingleEq", "SVAMP", "gsm8k", "AQuA"}
                    if expected == 6
                    else {
                        "boolq",
                        "piqa",
                        "social_i_qa",
                        "hellaswag",
                        "winogrande",
                        "ARC-Challenge",
                        "ARC-Easy",
                        "openbookqa",
                    }
                )
                if names != expected_names:
                    raise ValueError(
                        f"Expected {expected} evaluation datasets in {accuracy_path}"
                    )
                row["accuracy"] = 100 * mean(scores)
                for name, info in accuracy.items():
                    if name != "all":
                        row["accuracy_" + name] = 100 * info["accuracy"]
        for key in ["accuracy", "loss", "validation_loss", "average_train_loss"]:
            value = row.get(key)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Nonfinite {key} in {run_dir}")
        records.append(row)
    return records


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        identity = {
            key: value
            for key, value in row.items()
            if key
            not in {"seed", "run_dir", "loss", "validation_loss", "average_train_loss"}
            and not key.startswith("accuracy")
        }
        groups[json.dumps(identity, sort_keys=True)].append(row)
    summaries = []
    for identity, rows in sorted(groups.items()):
        seeds = [r["seed"] for r in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                "Multiple completed runs for the same cell and seed; select one output tree"
            )
        summary = {
            **json.loads(identity),
            "n_seeds": len(rows),
            "seeds": ",".join(map(str, sorted(seeds))),
        }
        for metric in [
            "accuracy",
            "accuracy_mm",
            "loss",
            "validation_loss",
            "average_train_loss",
        ]:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            if values:
                summary[metric + "_mean"] = mean(values)
                summary[metric + "_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    records = collect_records(args.input_root)
    if not records:
        parser.error("No completed reproduction runs found")
    summaries = aggregate(records)
    output = args.output_dir or args.input_root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "records.csv", records)
    write_csv(output / "summary.csv", summaries)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Summarized {len(records)} runs into {len(summaries)} cells under {output}")


if __name__ == "__main__":
    main()
