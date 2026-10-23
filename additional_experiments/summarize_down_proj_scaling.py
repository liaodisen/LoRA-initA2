#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


METRICS = (
    "processed_dot",
    "processed_cosine",
    "shuffled_dot",
    "shuffled_cosine",
    "cosine_enrichment",
)
INIT_LABELS = {"A": "Init[A]", "AA": "Init[A^2]"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit width scaling for fixed-rule down_proj alignment statistics."
    )
    parser.add_argument("--records_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260726)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, object]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    records: list[dict[str, object]] = []
    for row in rows:
        if row["valid_update"] != "True" or row["target"] != "down_proj":
            continue
        records.append(
            {
                "model": row["model"],
                "init_para": row["init_para"],
                "seed": int(row["seed"]),
                "step": int(row["step"]),
                "epoch": float(row["epoch"]),
                "layer": int(row["layer"]),
                "module": row["module"],
                "input_width": int(row["input_width"]),
                "processed_dot": float(row["processed_dot"]),
                "processed_cosine": float(row["processed_cosine"]),
                "shuffled_dot": float(row["shuffled_dot"]),
                "shuffled_cosine": float(row["shuffled_cosine"]),
            }
        )
    return records


def aggregate_runs(
    records: list[dict[str, object]], rule: str
) -> list[dict[str, object]]:
    max_layers: dict[str, int] = defaultdict(int)
    for record in records:
        max_layers[str(record["model"])] = max(
            max_layers[str(record["model"])], int(record["layer"])
        )

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if (
            rule == "last_layer"
            and int(record["layer"]) != max_layers[str(record["model"])]
        ):
            continue
        key = (
            record["model"],
            record["input_width"],
            record["init_para"],
            record["seed"],
            record["step"],
            record["epoch"],
        )
        grouped[key].append(record)

    summaries: list[dict[str, object]] = []
    for key, group in grouped.items():
        model, input_width, init_para, seed, step, epoch = key
        summary: dict[str, object] = {
            "rule": rule,
            "model": model,
            "input_width": input_width,
            "init_para": init_para,
            "init_label": INIT_LABELS.get(str(init_para), str(init_para)),
            "seed": seed,
            "step": step,
            "epoch": epoch,
            "n_layers": len(group),
            "modules": " ".join(sorted(str(record["module"]) for record in group)),
        }
        for metric in METRICS[:-1]:
            summary[metric] = median(float(record[metric]) for record in group)
        summary["cosine_enrichment"] = float(summary["processed_cosine"]) / float(
            summary["shuffled_cosine"]
        )
        summaries.append(summary)
    return summaries


def linear_fit(widths: list[float], values: list[float]) -> tuple[float, float]:
    log_widths = [math.log(value) for value in widths]
    log_values = [math.log(value) for value in values]
    width_mean = mean(log_widths)
    value_mean = mean(log_values)
    denominator = sum((value - width_mean) ** 2 for value in log_widths)
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
    return slope, r_squared


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_slope(
    width_groups: dict[int, list[float]], samples: int, seed: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    widths = sorted(width_groups)
    slopes = []
    for _ in range(samples):
        values = [
            mean(
                generator.choice(width_groups[width])
                for _ in range(len(width_groups[width]))
            )
            for width in widths
        ]
        if all(value > 0 for value in values):
            slopes.append(
                linear_fit(
                    [float(width) for width in widths],
                    values,
                )[0]
            )
    return percentile(slopes, 0.025), percentile(slopes, 0.975)


def fit_scaling(
    summaries: list[dict[str, object]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        grouped[(str(row["rule"]), str(row["init_para"]), int(row["step"]))].append(row)

    fits: list[dict[str, object]] = []
    for (rule, init_para, step), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])
    ):
        fit: dict[str, object] = {
            "rule": rule,
            "init_para": init_para,
            "init_label": INIT_LABELS.get(init_para, init_para),
            "step": step,
            "epoch_mean": mean(float(row["epoch"]) for row in group),
            "n_widths": len({int(row["input_width"]) for row in group}),
            "n_runs": len(group),
        }
        for metric_index, metric in enumerate(METRICS):
            width_groups: dict[int, list[float]] = defaultdict(list)
            for row in group:
                value = float(row[metric])
                if value > 0:
                    width_groups[int(row["input_width"])].append(value)
            widths = sorted(width_groups)
            averaged = [mean(width_groups[width]) for width in widths]
            slope, r_squared = linear_fit([float(width) for width in widths], averaged)
            low, high = bootstrap_slope(
                width_groups,
                bootstrap_samples,
                bootstrap_seed
                + 1000 * step
                + 100 * metric_index
                + (0 if init_para == "A" else 1),
            )
            fit[f"{metric}_slope"] = slope
            fit[f"{metric}_ci_low"] = low
            fit[f"{metric}_ci_high"] = high
            fit[f"{metric}_r2"] = r_squared
        fits.append(fit)
    return fits


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, fits: list[dict[str, object]]) -> None:
    rows = [row for row in fits if row["rule"] == "last_layer"]
    lines = [
        "# Last-Layer down_proj Width Scaling",
        "",
        "The same structural module—the `down_proj` in the final transformer "
        "block—is used in every model. Width `n` is that module's input width "
        "(4864, 8960, 11008, and 18944).",
        "",
        "| Init | Step | Dot kappa [95% CI] | R2 | Cosine slope | Enrichment slope | Shuffled cosine slope |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {init} | {step} | {slope} [{low}, {high}] | {r2} | "
            "{cosine} | {enrichment} | {shuffled} |".format(
                init=row["init_label"],
                step=row["step"],
                slope=format_value(row["processed_dot_slope"]),
                low=format_value(row["processed_dot_ci_low"]),
                high=format_value(row["processed_dot_ci_high"]),
                r2=format_value(row["processed_dot_r2"]),
                cosine=format_value(row["processed_cosine_slope"]),
                enrichment=format_value(row["cosine_enrichment_slope"]),
                shuffled=format_value(row["shuffled_cosine_slope"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    records = read_records(args.records_csv)
    if not records:
        raise SystemExit("No valid down_proj records found")
    summaries = []
    for rule in ("last_layer", "median_all_layers"):
        summaries.extend(aggregate_runs(records, rule))
    fits = fit_scaling(
        summaries,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "down_proj_run_summaries.csv", summaries)
    write_csv(args.output_dir / "down_proj_width_scaling.csv", fits)
    write_markdown(args.output_dir / "down_proj_width_scaling.md", fits)
    print(f"Wrote down_proj width-scaling results to {args.output_dir}")


if __name__ == "__main__":
    main()
