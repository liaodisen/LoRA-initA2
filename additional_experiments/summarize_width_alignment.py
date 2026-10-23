#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


INIT_LABELS = {"A": "Init[A]", "AA": "Init[A^2]"}
FIT_METRICS = {
    "processed_dot": "processed_dot_abs_median_across_modules_median",
    "processed_cosine": "processed_cosine_abs_median_across_modules_median",
    "shuffled_dot": "shuffled_dot_abs_median_across_modules_median",
    "shuffled_cosine": "shuffled_cosine_abs_median_across_modules_median",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit minibatch update-input width exponents."
    )
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=1234)
    return parser.parse_args()


def discover_records(input_root: Path) -> list[dict[str, object]]:
    deduplicated: dict[tuple[str, int], dict[str, object]] = {}
    for path in sorted(input_root.rglob("width_alignment.jsonl")):
        for line in path.read_text().splitlines():
            payload = json.loads(line)
            aggregate = payload.get("aggregate")
            if not aggregate:
                continue
            record: dict[str, object] = {
                "model": payload["model_name"],
                "width": int(payload["model_width"]),
                "init_para": payload["init_para"],
                "init_label": INIT_LABELS.get(
                    payload["init_para"], payload["init_para"]
                ),
                "seed": int(payload["seed"]),
                "rank": int(payload["rank"]),
                "step": int(payload["step"]),
                "epoch": float(payload["epoch"]),
                "max_steps": int(payload["max_steps"]),
                "n_modules_logged": int(payload["n_modules_logged"]),
                "path": str(path),
            }
            for output_name, aggregate_name in FIT_METRICS.items():
                record[output_name] = aggregate.get(aggregate_name)
            deduplicated[(str(path), int(payload["step"]))] = record
    return list(deduplicated.values())


def summarize_widths(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            record["model"],
            record["width"],
            record["init_para"],
            record["rank"],
            record["step"],
        )
        grouped[key].append(record)

    summaries = []
    for key, group in sorted(
        grouped.items(),
        key=lambda item: (int(item[0][4]), str(item[0][2]), int(item[0][1])),
    ):
        model, width, init_para, rank, step = key
        row: dict[str, object] = {
            "model": model,
            "width": width,
            "init_para": init_para,
            "init_label": INIT_LABELS.get(str(init_para), str(init_para)),
            "rank": rank,
            "step": step,
            "epoch_mean": mean(float(record["epoch"]) for record in group),
            "max_steps": max(int(record["max_steps"]) for record in group),
            "n": len(group),
            "seeds": " ".join(
                str(value) for value in sorted(int(record["seed"]) for record in group)
            ),
        }
        for metric in FIT_METRICS:
            values = [
                float(record[metric])
                for record in group
                if record.get(metric) is not None
            ]
            row[f"{metric}_mean"] = mean(values) if values else None
            row[f"{metric}_std"] = (
                stdev(values) if len(values) > 1 else 0.0 if values else None
            )
        summaries.append(row)
    return summaries


def linear_fit(widths: list[float], values: list[float]) -> tuple[float, float, float]:
    log_widths = [math.log(value) for value in widths]
    log_values = [math.log(value) for value in values]
    width_mean = mean(log_widths)
    value_mean = mean(log_values)
    denominator = sum((value - width_mean) ** 2 for value in log_widths)
    if denominator <= 0:
        raise ValueError("Width fit requires at least two distinct widths")
    slope = (
        sum(
            (width - width_mean) * (value - value_mean)
            for width, value in zip(log_widths, log_values)
        )
        / denominator
    )
    intercept = value_mean - slope * width_mean
    predictions = [intercept + slope * width for width in log_widths]
    residual_sum = sum(
        (value - prediction) ** 2 for value, prediction in zip(log_values, predictions)
    )
    total_sum = sum((value - value_mean) ** 2 for value in log_values)
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    return slope, intercept, r_squared


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_slope(
    width_groups: dict[int, list[float]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        return math.nan, math.nan
    generator = random.Random(seed)
    slopes = []
    widths = sorted(width_groups)
    for _ in range(samples):
        sampled_values = []
        for width in widths:
            values = width_groups[width]
            sampled_values.append(
                mean(generator.choice(values) for _ in range(len(values)))
            )
        if all(value > 0 for value in sampled_values):
            slopes.append(
                linear_fit([float(width) for width in widths], sampled_values)[0]
            )
    return percentile(slopes, 0.025), percentile(slopes, 0.975)


def fit_slopes(
    records: list[dict[str, object]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[
            (str(record["init_para"]), int(record["step"]), str(record["init_label"]))
        ].append(record)

    slopes = []
    for (init_para, step, init_label), group in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        row: dict[str, object] = {
            "init_para": init_para,
            "init_label": init_label,
            "step": step,
            "epoch_mean": mean(float(record["epoch"]) for record in group),
            "n_widths": len({int(record["width"]) for record in group}),
            "n_runs": len(group),
        }
        for metric_index, metric in enumerate(FIT_METRICS):
            width_groups: dict[int, list[float]] = defaultdict(list)
            for record in group:
                value = record.get(metric)
                if value is not None and float(value) > 0:
                    width_groups[int(record["width"])].append(float(value))
            widths = sorted(width_groups)
            if len(widths) < 3:
                row[f"{metric}_slope"] = None
                row[f"{metric}_ci_low"] = None
                row[f"{metric}_ci_high"] = None
                row[f"{metric}_r2"] = None
                continue
            averaged_values = [mean(width_groups[width]) for width in widths]
            slope, _intercept, r_squared = linear_fit(
                [float(width) for width in widths],
                averaged_values,
            )
            confidence_low, confidence_high = bootstrap_slope(
                width_groups,
                samples=bootstrap_samples,
                seed=bootstrap_seed
                + 1000 * step
                + 100 * metric_index
                + (0 if init_para == "A" else 1),
            )
            row[f"{metric}_slope"] = slope
            row[f"{metric}_ci_low"] = confidence_low
            row[f"{metric}_ci_high"] = confidence_high
            row[f"{metric}_r2"] = r_squared
        slopes.append(row)
    return slopes


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def write_markdown(path: Path, slopes: list[dict[str, object]]) -> None:
    preserved_follow_up = ""
    follow_up_marker = "\n## All-module follow-up: down_proj\n"
    if path.exists():
        existing = path.read_text()
        marker_position = existing.find(follow_up_marker)
        if marker_position >= 0:
            preserved_follow_up = existing[marker_position:]
    lines = [
        "# Minibatch Width-Alignment Summary",
        "",
        "## Experiment setup",
        "",
        "- **Models and widths:** Qwen2.5-0.5B-Instruct (`n=896`), "
        "Qwen2.5-1.5B-Instruct (`n=1536`), Qwen2.5-3B-Instruct (`n=2048`), "
        "and Qwen2.5-7B-Instruct (`n=3584`).",
        "- **Initialization:** Init[A] and Init[A²], each run with seeds 42, 43, "
        "and 44, giving 24 runs in total.",
        "- **LoRA configuration:** rank `r=8`, scaling `alpha=8` (`alpha/r=1`), "
        "Kaiming initialization with `init_std=0.5`, `eta_A=1e-5`, and asymmetric "
        "learning-rate ratio `eta_B/eta_A=200`.",
        "- **Training configuration:** `math_10k`, one epoch (620 optimizer steps), "
        "effective batch size 16, micro-batch size 2, sequence cutoff 256, and no "
        "validation or checkpoint saving.",
        "- **Alignment measurement:** at steps 2, 5, 10, 25, 50, 100, 155, 310, "
        "and 620, measure `|(U_A^t)_{i,:}^T Z|` and its cosine-normalized form on "
        "32 tokens from the same minibatch that produced the update. Four evenly "
        "spaced `q_proj` modules are recorded per model.",
        "- **Control:** deterministically permute the coordinates of `Z` before "
        "computing the same dot-product and cosine statistics. Metrics are first "
        "summarized across the four modules and then averaged across the three "
        "seeds at each width.",
        "",
        "## Original q_proj width-scaling fits",
        "",
        "For each checkpoint we fit `log M(n) = c + kappa log n` across model widths.",
        "The theory predicts `kappa = 1` for the realized update-input dot product and",
        "`kappa = 0` for its cosine normalization. The coordinate-shuffled control",
        "should instead approach `kappa = 0.5` for the dot product and `-0.5` for cosine.",
        "",
        "| Init | Step | Epoch | Widths | Runs | Actual dot kappa [95% CI] | R2 | Actual cosine slope | Shuffled dot kappa | Shuffled cosine slope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in slopes:
        dot_interval = "{slope} [{low}, {high}]".format(
            slope=format_value(row.get("processed_dot_slope")),
            low=format_value(row.get("processed_dot_ci_low")),
            high=format_value(row.get("processed_dot_ci_high")),
        )
        lines.append(
            "| {init} | {step} | {epoch} | {widths} | {runs} | {dot_interval} | {r2} | {cosine} | {shuffled_dot} | {shuffled_cosine} |".format(
                init=row["init_label"],
                step=row["step"],
                epoch=format_value(row["epoch_mean"]),
                widths=row["n_widths"],
                runs=row["n_runs"],
                dot_interval=dot_interval,
                r2=format_value(row.get("processed_dot_r2")),
                cosine=format_value(row.get("processed_cosine_slope")),
                shuffled_dot=format_value(row.get("shuffled_dot_slope")),
                shuffled_cosine=format_value(row.get("shuffled_cosine_slope")),
            )
        )
    path.write_text("\n".join(lines) + "\n" + preserved_follow_up)


def main() -> None:
    args = parse_args()
    records = discover_records(args.input_root)
    if not records:
        raise SystemExit(
            f"No width_alignment.jsonl records found under {args.input_root}"
        )
    width_summaries = summarize_widths(records)
    slopes = fit_slopes(
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_dir = args.output_dir or (args.input_root / "summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "width_alignment_records.csv", records)
    write_csv(output_dir / "width_alignment_by_width.csv", width_summaries)
    write_csv(output_dir / "width_alignment_slopes.csv", slopes)
    write_markdown(output_dir / "width_alignment_summary.md", slopes)
    print(f"Wrote width-alignment summaries to {output_dir}")


if __name__ == "__main__":
    main()
