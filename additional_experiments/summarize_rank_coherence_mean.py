#!/usr/bin/env python3
"""Compute mean-based rank-coherence scaling from saved JSONL summaries.

The coherence logger stores the mean S value and observation count for every
module.  This is sufficient to recover the mean over sampled token-row
contractions within each run, but not their second moment/RMS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_SPECS = (
    (
        "Llama-3.2-1B",
        Path("additional_outputs/rank_coherence_sweep"),
    ),
    (
        "Qwen3-0.6B",
        Path("additional_outputs/qwen3_0p6b_rank_coherence_math10k"),
    ),
    (
        "RoBERTa-base",
        Path("additional_outputs/roberta_rank_coherence_sst2"),
    ),
)
INIT_LABELS = {"A": "Init[A]", "AA": "Init[A²]"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-base",
        type=Path,
        default=Path("additional_outputs"),
        help="Parent of the Llama, Qwen, and RoBERTa coherence output trees",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("additional_experiments/results/rank_coherence_mean_scaling.csv"),
    )
    return parser.parse_args()


def fit_log_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    log_x = [math.log(value) for value in xs]
    log_y = [math.log(value) for value in ys]
    mean_x = mean(log_x)
    mean_y = mean(log_y)
    denominator = sum((value - mean_x) ** 2 for value in log_x)
    slope = (
        sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(log_x, log_y)
        )
        / denominator
    )
    intercept = mean_y - slope * mean_x
    residual = sum(
        (y_value - (slope * x_value + intercept)) ** 2
        for x_value, y_value in zip(log_x, log_y)
    )
    total = sum((value - mean_y) ** 2 for value in log_y)
    return slope, 1.0 - residual / total if total > 0 else 1.0


def valid_record(payload: dict[str, Any]) -> bool:
    aggregate = payload.get("aggregate", {})
    try:
        s_median = float(aggregate["current_s_median_across_modules_median"])
        c_median = float(aggregate["current_c_median_across_modules_median"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(s_median)
        and math.isfinite(c_median)
        and s_median > 1e-8
        and c_median > 1e-8
    )


def run_mean(payload: dict[str, Any]) -> tuple[float, int]:
    weighted_sum = 0.0
    observations = 0
    for module in payload.get("modules", []):
        count = int(module["n_tokens"]) * int(module["n_rows"])
        weighted_sum += float(module["current_s_mean"]) * count
        observations += count
    if observations == 0:
        raise ValueError("coherence record contains no sampled observations")
    return weighted_sum / observations, observations


def load_records(model: str, root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/coherence_stats.jsonl")):
        by_step: dict[int, dict[str, Any]] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                payload = json.loads(line)
                by_step[int(payload["step"])] = payload
        for step, payload in sorted(by_step.items()):
            if not valid_record(payload):
                continue
            s_mean, observations = run_mean(payload)
            records.append(
                {
                    "model": model,
                    "init_para": str(payload["init_para"]),
                    "rank": int(payload["rank"]),
                    "step": step,
                    "epoch": float(payload["epoch"]),
                    "seed": int(payload["seed"]),
                    "lr_ratio": float(payload["lr_ratio"]),
                    "s_mean": s_mean,
                    "observations": observations,
                }
            )
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["init_para"], record["step"])].append(record)

    summaries: list[dict[str, Any]] = []
    for (model, init_para, step), group in sorted(grouped.items()):
        by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in group:
            by_rank[int(record["rank"])].append(record)
        ranks = sorted(by_rank)
        if len(ranks) < 3:
            continue
        means_by_rank = [
            mean(float(record["s_mean"]) for record in by_rank[rank]) for rank in ranks
        ]
        gamma, r_squared = fit_log_slope(
            [float(rank) for rank in ranks],
            means_by_rank,
        )
        summaries.append(
            {
                "model": model,
                "init_para": init_para,
                "init_label": INIT_LABELS.get(init_para, init_para),
                "step": step,
                "epoch": mean(float(record["epoch"]) for record in group),
                "min_runs_per_rank": min(len(by_rank[rank]) for rank in ranks),
                "max_runs_per_rank": max(len(by_rank[rank]) for rank in ranks),
                "gamma_mean_S": gamma,
                "r2_mean_S": r_squared,
                "ranks": " ".join(str(rank) for rank in ranks),
                "mean_S_by_rank": " ".join(f"{value:.6f}" for value in means_by_rank),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    records = [
        record
        for model, root in DEFAULT_SPECS
        for record in load_records(model, args.input_base / root.name)
    ]
    summaries = summarize(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(
        f"Wrote {len(summaries)} rows from {len(records)} valid run-checkpoint records to {args.output}"
    )


if __name__ == "__main__":
    main()
