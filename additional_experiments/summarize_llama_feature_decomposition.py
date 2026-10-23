#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


RUN_RE = re.compile(
    r"^(?P<model>.+?)_(?P<adapter>lora|qlora)_"
    r"(?P<init_para>A|AA)_r(?P<rank>\d+)_a(?P<alpha>[^_]+)_"
    r"lr(?P<lr>[^_]+)_ratio(?P<ratio>[^_]+)_s(?P<seed>\d+)$"
)
INIT_LABELS = {"A": "Init[A]", "AA": "Init[A^2]"}
METRICS = [
    "noise_energy_across_modules_mean",
    "signal_energy_across_modules_mean",
    "cross_energy_across_modules_mean",
    "total_energy_across_modules_mean",
    "noise_fraction",
    "signal_fraction",
    "cross_fraction",
    "noise_signal_cosine_across_modules_mean",
    "reconstruction_relative_error_across_modules_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize LoRA feature noise/signal/cross dynamics."
    )
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def run_metadata(path: Path) -> dict[str, object] | None:
    for parent in path.parents:
        match = RUN_RE.match(parent.name)
        if match is None:
            continue
        rank = int(match.group("rank"))
        alpha = float(match.group("alpha"))
        return {
            "model": match.group("model"),
            "adapter": match.group("adapter"),
            "init_para": match.group("init_para"),
            "init_label": INIT_LABELS[match.group("init_para")],
            "rank": rank,
            "alpha": alpha,
            "multiplier": alpha / rank,
            "lr_a": float(match.group("lr")),
            "lr_ratio": float(match.group("ratio")),
            "seed": int(match.group("seed")),
        }
    return None


def discover_records(input_root: Path) -> list[dict[str, object]]:
    records = []
    for path in input_root.rglob("feature_decomposition.jsonl"):
        metadata = run_metadata(path)
        if metadata is None:
            continue
        for line in path.read_text().splitlines():
            payload = json.loads(line)
            aggregate = payload.get("aggregate")
            if not aggregate:
                continue
            record = {
                **metadata,
                "step": int(payload["step"]),
                "epoch": float(payload["epoch"]),
                "max_steps": int(payload["max_steps"]),
                "path": str(path),
            }
            for metric in METRICS:
                record[metric] = aggregate.get(metric)
            records.append(record)
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["model"],
            record["init_para"],
            record["rank"],
            record["multiplier"],
            record["lr_a"],
            record["lr_ratio"],
            record["step"],
        )
        grouped[key].append(record)

    summaries = []
    for key, group in sorted(grouped.items()):
        model, init_para, rank, multiplier, lr_a, lr_ratio, step = key
        row = {
            "model": model,
            "init_para": init_para,
            "init_label": INIT_LABELS[str(init_para)],
            "rank": rank,
            "multiplier": multiplier,
            "lr_a": lr_a,
            "lr_ratio": lr_ratio,
            "step": step,
            "epoch_mean": mean(float(record["epoch"]) for record in group),
            "n": len(group),
            "seeds": " ".join(
                str(seed) for seed in sorted(int(record["seed"]) for record in group)
            ),
        }
        for metric in METRICS:
            values = [
                float(record[metric]) for record in group if record[metric] is not None
            ]
            row[f"{metric}_mean"] = mean(values) if values else None
            row[f"{metric}_std"] = (
                stdev(values) if len(values) > 1 else 0.0 if values else None
            )
        summaries.append(row)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}g}"


def write_markdown(path: Path, summaries: list[dict[str, object]]) -> None:
    selected = []
    grouped = defaultdict(list)
    for row in summaries:
        grouped[(row["model"], row["init_para"], row["multiplier"])].append(row)
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        selected.extend([ordered[0], ordered[-1]] if len(ordered) > 1 else ordered)
    lines = [
        "# LoRA Feature Decomposition Summary",
        "",
        "The signed fractions satisfy `noise + signal + cross = total` up to numerical error.",
        "",
        "| Model | Init | alpha/r | Step | Epoch | n | Noise fraction | Signal fraction | Cross fraction | Noise-signal cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        selected,
        key=lambda item: (
            str(item["model"]),
            float(item["multiplier"]),
            str(item["init_para"]),
            int(item["step"]),
        ),
    ):
        lines.append(
            "| {model} | {init} | {multiplier:g} | {step} | {epoch} | {n} | {noise} | {signal} | {cross} | {cosine} |".format(
                model=row["model"],
                init=row["init_label"],
                multiplier=float(row["multiplier"]),
                step=row["step"],
                epoch=fmt(row["epoch_mean"], 3),
                n=row["n"],
                noise=fmt(row["noise_fraction_mean"]),
                signal=fmt(row["signal_fraction_mean"]),
                cross=fmt(row["cross_fraction_mean"]),
                cosine=fmt(row["noise_signal_cosine_across_modules_mean_mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    records = discover_records(args.input_root)
    if not records:
        raise SystemExit(
            f"No feature_decomposition.jsonl records found under {args.input_root}"
        )
    summaries = summarize(records)
    output_dir = args.output_dir or (args.input_root / "summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "feature_decomposition_records.csv", records)
    write_csv(output_dir / "feature_decomposition_summary.csv", summaries)
    write_markdown(output_dir / "feature_decomposition_summary.md", summaries)
    print(f"Wrote decomposition summaries to {output_dir}")


if __name__ == "__main__":
    main()
