#!/usr/bin/env python3
"""Export paper experiment curves or LR heatmaps from aggregated CSV results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#B64342", "#0F4D92", "#42949E", "#9A4D8E", "#767676"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="summary/summary.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--x", default="ratio")
    parser.add_argument("--y", help="Second grid coordinate for heatmaps")
    parser.add_argument(
        "--series", default="method", help="Comma-separated curve grouping fields"
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "loss", "validation_loss", "average_train_loss"],
        default="accuracy",
    )
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--log-x", action="store_true")
    args = parser.parse_args()
    with args.input.open() as handle:
        rows = list(csv.DictReader(handle))
    for value in args.filter:
        key, expected = value.split("=", 1)
        rows = [row for row in rows if row.get(key) == expected]
    if not rows:
        parser.error("No matching results")
    metric = args.metric + "_mean"
    if any(not row.get(args.x) or not row.get(metric) for row in rows):
        parser.error(f"Selected results require numeric {args.x} and {metric}")
    series_keys = args.series.split(",")
    excluded = {args.x, args.y, "seed", "seeds", "n_seeds", "alpha", *series_keys}
    if args.x == "rank" or args.y == "rank":
        excluded.add("alpha")
    if "lr_b" in {args.x, args.y}:
        excluded.add("ratio")
    fields = [
        key
        for key in rows[0]
        if key not in excluded and not key.endswith(("_mean", "_std"))
    ]
    if args.y:
        fields += series_keys
    groups = defaultdict(list)
    for row in rows:
        groups[tuple((key, row.get(key, "")) for key in fields)].append(row)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.linewidth": 2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for identity, group in groups.items():
        fig, ax = plt.subplots(figsize=(7, 4.8))
        if args.y:
            xs = sorted({float(row[args.x]) for row in group})
            ys = sorted({float(row[args.y]) for row in group})
            matrix = np.full((len(ys), len(xs)), np.nan)
            for row in group:
                iy, ix = ys.index(float(row[args.y])), xs.index(float(row[args.x]))
                if np.isfinite(matrix[iy, ix]):
                    raise ValueError("Duplicate heatmap cell; select more filters")
                matrix[iy, ix] = float(row[metric])
            artist = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
            ax.set_xticks(range(len(xs)), [f"{value:g}" for value in xs], rotation=45)
            ax.set_yticks(range(len(ys)), [f"{value:g}" for value in ys])
            ax.set_ylabel(args.y)
            fig.colorbar(artist, ax=ax, label=args.metric.replace("_", " "))
        else:
            curves = defaultdict(list)
            for row in group:
                curves[tuple(row[key] for key in series_keys)].append(row)
            for index, (label, curve) in enumerate(sorted(curves.items())):
                curve.sort(key=lambda row: float(row[args.x]))
                x = np.array([float(row[args.x]) for row in curve])
                y = np.array([float(row[metric]) for row in curve])
                error = np.array(
                    [float(row.get(args.metric + "_std") or 0) for row in curve]
                )
                ax.errorbar(
                    x,
                    y,
                    yerr=error,
                    marker="o",
                    linewidth=2,
                    capsize=3,
                    color=COLORS[index % len(COLORS)],
                    label=", ".join(label),
                )
            ax.set_ylabel(
                "Accuracy (%)"
                if args.metric == "accuracy"
                else args.metric.replace("_", " ")
            )
            if args.log_x:
                ax.set_xscale("log")
            ax.legend(fontsize=10)
        ax.set_xlabel(args.x.replace("_", " "))
        title = "; ".join(
            f"{key}={value}"
            for key, value in identity
            if key
            in {
                "experiment",
                "dataset",
                "task",
                "beta",
                "method",
                "width",
                "multiplier",
            }
        )
        ax.set_title(title, fontsize=11, wrap=True)
        fig.tight_layout(pad=1)
        digest = hashlib.blake2b(json.dumps(identity).encode()).hexdigest()[:10]
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(
                args.output_dir / f"{rows[0]['experiment']}_{digest}.{extension}",
                dpi=300,
            )
        plt.close(fig)
    print(f"Exported {len(groups)} plots as PDF, SVG, and PNG to {args.output_dir}")


if __name__ == "__main__":
    main()
