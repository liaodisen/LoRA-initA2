#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


GROUP_RE = re.compile(r"^(?P<init_label>.+)_beta(?P<init_std>[^_]+)_seed(?P<seed>\d+)$")
RUN_RE = re.compile(r"^lr-(?P<lr>.+)_ratio-(?P<ratio>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate one RoBERTa heatmap over 5 seeds."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("additional_outputs/roberta_heatmap_5seeds"),
    )
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--metric", type=str, default="eval_accuracy")
    return parser.parse_args()


def metric_to_percent(metric: str, value: float) -> float:
    if "loss" in metric.lower():
        return value
    return value * 100.0 if abs(value) <= 1.0 else value


def discover_records(args: argparse.Namespace) -> list[dict[str, object]]:
    task_root = args.input_root / args.task
    records: list[dict[str, object]] = []
    for result_path in task_root.glob("*/*/eval_results.json"):
        group_match = GROUP_RE.match(result_path.parent.parent.name)
        run_match = RUN_RE.match(result_path.parent.name)
        if group_match is None or run_match is None:
            continue
        try:
            payload = json.loads(result_path.read_text())
            metric_value = float(payload[args.metric])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

        records.append(
            {
                "path": str(result_path),
                "init_label": group_match.group("init_label"),
                "init_std": float(group_match.group("init_std")),
                "seed": int(group_match.group("seed")),
                "lr": run_match.group("lr"),
                "lr_value": float(run_match.group("lr")),
                "ratio": float(run_match.group("ratio")),
                "metric": metric_to_percent(args.metric, metric_value),
            }
        )
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str, float], list[dict[str, object]]] = defaultdict(
        list
    )
    for record in records:
        grouped[
            (
                str(record["init_label"]),
                float(record["init_std"]),
                str(record["lr"]),
                float(record["ratio"]),
            )
        ].append(record)

    rows: list[dict[str, object]] = []
    for (init_label, init_std, lr, ratio), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], float(item[0][2]), item[0][3]),
    ):
        values = [float(record["metric"]) for record in group]
        rows.append(
            {
                "init_label": init_label,
                "init_std": init_std,
                "lr": lr,
                "lr_value": float(lr),
                "ratio": ratio,
                "mean_metric": mean(values),
                "std_metric": stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
                "seeds": sorted(int(record["seed"]) for record in group),
            }
        )
    return rows


def write_outputs(
    output_dir: Path,
    records: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "roberta_heatmap_seed_records.csv"
    with records_path.open("w", newline="") as handle:
        fieldnames = ["init_label", "init_std", "seed", "lr", "ratio", "metric", "path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fieldnames})

    summary_path = output_dir / "roberta_heatmap_seed_summary.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "init_label",
            "init_std",
            "lr",
            "ratio",
            "mean_metric",
            "std_metric",
            "n",
            "seeds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    field: (
                        " ".join(str(seed) for seed in row["seeds"])
                        if field == "seeds"
                        else row[field]
                    )
                    for field in fieldnames
                }
            )

    json_path = output_dir / "roberta_heatmap_seed_summary.json"
    json_path.write_text(
        json.dumps({"records": records, "summary": summary_rows}, indent=2)
    )


def build_matrix(
    rows: list[dict[str, object]],
    lr_values: list[str],
    ratio_values: list[float],
) -> list[list[float]]:
    lookup = {
        (str(row["lr"]), float(row["ratio"])): float(row["mean_metric"]) for row in rows
    }
    matrix: list[list[float]] = []
    for lr in lr_values:
        matrix.append([lookup.get((lr, ratio), math.nan) for ratio in ratio_values])
    return matrix


def plot(
    args: argparse.Namespace, output_dir: Path, summary_rows: list[dict[str, object]]
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "matplotlib and numpy are required for plotting this heatmap."
        ) from exc

    init_labels = sorted({str(row["init_label"]) for row in summary_rows})
    init_stds = sorted({float(row["init_std"]) for row in summary_rows})
    lr_values = sorted({str(row["lr"]) for row in summary_rows}, key=float)
    ratio_values = sorted({float(row["ratio"]) for row in summary_rows})

    nrows = len(init_stds)
    ncols = len(init_labels)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.4 * ncols + 1.0, 3.2 * nrows + 0.8),
        constrained_layout=True,
        squeeze=False,
    )

    all_values = [float(row["mean_metric"]) for row in summary_rows]
    vmin = min(all_values)
    vmax = max(all_values)
    image = None
    for row_idx, init_std in enumerate(init_stds):
        for col_idx, init_label in enumerate(init_labels):
            ax = axes[row_idx][col_idx]
            panel_rows = [
                row
                for row in summary_rows
                if str(row["init_label"]) == init_label
                and math.isclose(float(row["init_std"]), init_std)
            ]
            matrix = np.array(
                build_matrix(panel_rows, lr_values, ratio_values), dtype=float
            )
            masked = np.ma.masked_invalid(matrix)
            image = ax.imshow(
                masked,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(f"{init_label}, beta={init_std:g}")
            ax.set_xticks(range(len(ratio_values)))
            ax.set_xticklabels(
                [f"{ratio:g}" for ratio in ratio_values], rotation=45, ha="right"
            )
            ax.set_yticks(range(len(lr_values)))
            ax.set_yticklabels(lr_values)
            ax.set_xlabel("LR ratio eta_B / eta_A")
            ax.set_ylabel("LR eta_A")

    if image is not None:
        flat_axes = [axis for row in axes for axis in row]
        label = args.metric if "loss" in args.metric.lower() else f"{args.metric} (%)"
        fig.colorbar(image, ax=flat_axes, label=label)

    fig.suptitle(f"RoBERTa {args.task}: 5-seed heatmap")
    png_path = output_dir / "roberta_heatmap_5seed_mean.png"
    pdf_path = output_dir / "roberta_heatmap_5seed_mean.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    args = parse_args()
    records = discover_records(args)
    if not records:
        raise SystemExit(
            f"No matching RoBERTa records found under {args.input_root / args.task}"
        )

    output_dir = args.output_dir or (
        args.input_root / args.task / "roberta_heatmap_5seed_summary"
    )
    summary_rows = summarize(records)
    write_outputs(output_dir, records, summary_rows)
    plot(args, output_dir, summary_rows)
    print(f"Wrote CSV/JSON summaries to {output_dir}")


if __name__ == "__main__":
    main()
