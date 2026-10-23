#!/usr/bin/env python3
"""Summarize the Llama-3.1-8B initialization/asymmetric-LR screening grid."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def method_label(metrics):
    method = str(metrics.get("init_method", "unknown")).lower().replace("-", "_")
    if method == "kaiming" and str(metrics.get("init_para", "")).lower() == "aa":
        return "Init[A^2]"
    labels = {
        "orthogonal": "Orthogonal",
        "orthogonal_a2": "Orthogonal-A^2",
        "eva": "EVA",
        "eva_a2": "EVA-A^2",
        "pissa": "PiSSA",
        "pissa_a2": "PiSSA-A^2",
    }
    return labels.get(method, metrics.get("init_method", "unknown"))


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def mean(values):
    values = [value for value in values if math.isfinite(value)]
    return statistics.fmean(values) if values else math.nan


def sample_std(values):
    values = [value for value in values if math.isfinite(value)]
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else math.nan


def fmt(value, digits=2):
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def initialization_scale_diagnostics(stats):
    transform = stats.get("transform", []) if stats else []
    ratios = []
    gauges = []
    product_errors = []
    for row in transform:
        target = finite_float(row.get("target_a_rms"))
        actual = finite_float(row.get("a_rms"))
        gauge = finite_float(row.get("gauge_scale"))
        product_error = finite_float(row.get("product_relative_error"))
        if math.isfinite(target) and target > 0 and math.isfinite(actual):
            ratios.append(actual / target)
        if math.isfinite(gauge):
            gauges.append(gauge)
        if math.isfinite(product_error):
            product_errors.append(product_error)
    return (
        mean(ratios),
        statistics.median(gauges) if gauges else math.nan,
        max(product_errors) if product_errors else math.nan,
    )


def collect_records(input_root):
    records = []
    for metrics_path in sorted(
        input_root.glob("**/finetuned_result/final_train_metrics.json")
    ):
        metrics = read_json(metrics_path)
        if not metrics:
            continue

        run_dir = metrics_path.parent.parent
        accuracy = read_json(run_dir / "evaluated_result" / "accuracy.json")
        avg_accuracy = math.nan
        if accuracy:
            avg_accuracy = finite_float(accuracy.get("all", {}).get("avg_accuracy"))
            if math.isfinite(avg_accuracy):
                avg_accuracy *= 100.0

        stats = read_json(metrics_path.parent / "initialization_stats.json")
        target_ratio, median_gauge, max_product_error = (
            initialization_scale_diagnostics(stats)
        )
        beta = metrics.get("initialization_beta")
        records.append(
            {
                "method": method_label(metrics),
                "beta": "" if beta is None else float(beta),
                "seed": int(metrics.get("seed", -1)),
                "lr_a": finite_float(metrics.get("learning_rate_a")),
                "lr_b": finite_float(metrics.get("learning_rate_b")),
                "lr_ratio": finite_float(metrics.get("learning_rate_ratio_b_over_a")),
                "epochs": finite_float(metrics.get("num_train_epochs")),
                "accuracy_mean_six_tasks": avg_accuracy,
                "final_training_loss": finite_float(metrics.get("final_training_loss")),
                "a_rms_over_target": target_ratio,
                "median_pissa_gauge_scale": median_gauge,
                "max_pissa_product_relative_error": max_product_error,
                "evaluation_complete": math.isfinite(avg_accuracy),
                "run_dir": str(run_dir),
            }
        )
    return records


def aggregate_records(records):
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["method"],
            record["beta"],
            record["lr_a"],
            record["lr_ratio"],
            record["epochs"],
        )
        grouped[key].append(record)

    rows = []
    for key, group in grouped.items():
        method, beta, lr_a, lr_ratio, epochs = key
        accuracies = [record["accuracy_mean_six_tasks"] for record in group]
        rows.append(
            {
                "method": method,
                "beta": beta,
                "lr_a": lr_a,
                "lr_b": lr_a * lr_ratio,
                "lr_ratio": lr_ratio,
                "epochs": epochs,
                "n_runs": len(group),
                "n_evaluated": sum(record["evaluation_complete"] for record in group),
                "accuracy_mean": mean(accuracies),
                "accuracy_std": sample_std(accuracies),
                "final_training_loss_mean": mean(
                    [record["final_training_loss"] for record in group]
                ),
                "a_rms_over_target_mean": mean(
                    [record["a_rms_over_target"] for record in group]
                ),
                "median_pissa_gauge_scale": mean(
                    [record["median_pissa_gauge_scale"] for record in group]
                ),
                "max_pissa_product_relative_error": max(
                    (
                        record["max_pissa_product_relative_error"]
                        for record in group
                        if math.isfinite(record["max_pissa_product_relative_error"])
                    ),
                    default=math.nan,
                ),
            }
        )

    baselines = {}
    for row in rows:
        if row["lr_ratio"] == 1:
            baseline_key = (row["method"], row["beta"], row["lr_a"], row["epochs"])
            baselines[baseline_key] = row["accuracy_mean"]
    for row in rows:
        baseline_key = (row["method"], row["beta"], row["lr_a"], row["epochs"])
        baseline = baselines.get(baseline_key, math.nan)
        row["accuracy_delta_vs_ratio1"] = (
            row["accuracy_mean"] - baseline
            if math.isfinite(row["accuracy_mean"]) and math.isfinite(baseline)
            else math.nan
        )

    return sorted(
        rows,
        key=lambda row: (
            row["method"],
            str(row["beta"]),
            row["lr_a"],
            row["lr_ratio"],
        ),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, records, summary):
    lines = [
        "# Llama-3.1-8B initialization and asymmetric-LR summary",
        "",
        f"- Discovered runs: {len(records)}",
        f"- Runs with six-task generation evaluation: {sum(row['evaluation_complete'] for row in records)}",
        "- Accuracy is the mean percentage over AddSub, MultiArith, SingleEq, SVAMP, GSM8K, and AQuA.",
        "- `Delta vs ratio 1` compares each asymmetric cell with the matched initialization, beta, eta_A, and epoch count.",
        "",
        "| Method | Beta | eta_A | eta_B/eta_A | n eval | Accuracy | Delta vs ratio 1 | Train loss | A RMS / target | Median A^2 scale c | Max product error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        beta = row["beta"] if row["beta"] != "" else "NA"
        lines.append(
            "| {method} | {beta} | {lr_a:.2g} | {ratio:g} | {n_eval}/{n_runs} | "
            "{accuracy} +/- {std} | {delta} | {loss} | {target} | {gauge} | {product_error} |".format(
                method=row["method"],
                beta=beta,
                lr_a=row["lr_a"],
                ratio=row["lr_ratio"],
                n_eval=row["n_evaluated"],
                n_runs=row["n_runs"],
                accuracy=fmt(row["accuracy_mean"]),
                std=fmt(row["accuracy_std"]),
                delta=fmt(row["accuracy_delta_vs_ratio1"]),
                loss=fmt(row["final_training_loss_mean"], 4),
                target=fmt(row["a_rms_over_target_mean"], 4),
                gauge=fmt(row["median_pissa_gauge_scale"], 4),
                product_error=fmt(
                    row["max_pissa_product_relative_error"],
                    3,
                ),
            )
        )

    lines.extend(
        [
            "",
            "Interpretation rule: a variant supports asymmetric LR when at least one ratio greater than 1 improves accuracy or preserves it while remaining finite, and the result repeats across confirmation seeds. Treat single-seed screening differences as cell selection only.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    output_dir = args.output_dir or args.input_root / "summary"
    records = collect_records(args.input_root)
    summary = aggregate_records(records)
    write_csv(output_dir / "run_records.csv", records)
    write_csv(output_dir / "cell_summary.csv", summary)
    write_markdown(output_dir / "summary.md", records, summary)
    print(f"Wrote {len(records)} run records and {len(summary)} cells to {output_dir}")


if __name__ == "__main__":
    main()
