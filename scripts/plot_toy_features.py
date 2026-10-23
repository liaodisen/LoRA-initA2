#!/usr/bin/env python3
"""Export Figure 1 feature trajectories from completed reconstructed toy runs."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("figures/figure1"))
    args = parser.parse_args()
    curves = defaultdict(list)
    widths = set()
    for manifest in args.input_root.rglob("run.json"):
        if not (manifest.parent / "completed.json").exists():
            continue
        cell = json.loads(manifest.read_text())["cell"]
        widths.add(cell["width"])
        curves[(cell["method"], cell["beta"])].append(
            json.loads((manifest.parent / "toy_features.json").read_text())
        )
    if not curves:
        parser.error("No completed toy trajectories found")
    if len(widths) != 1:
        parser.error("Select a single width to compare normalized feature trajectories")
    methods = [
        method
        for method in ["A", "AB_reset", "AA"]
        if any(key[0] == method for key in curves)
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.linewidth": 2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2, len(methods), squeeze=False, figsize=(4 * len(methods), 6)
    )
    labels = {"A": "Init[A]", "AB_reset": "Init[AB]", "AA": "Init[A²]"}
    colors = ["#0F4D92", "#9A4D8E", "#B64342"]
    for column, method in enumerate(methods):
        for index, beta in enumerate(
            sorted(key[1] for key in curves if key[0] == method)
        ):
            histories = curves[(method, beta)]
            steps = [record["step"] for record in histories[0]]
            if any(
                [record["step"] for record in history] != steps for history in histories
            ):
                raise ValueError("Seed trajectories have different checkpoint grids")
            for row, field in enumerate(
                ["za_norm_squared_over_width", "zb_norm_squared_over_width"]
            ):
                values = np.array(
                    [[record[field] for record in history] for history in histories]
                )
                average = values.mean(0)
                axes[row, column].plot(
                    steps,
                    average,
                    color=colors[index % len(colors)],
                    linewidth=2,
                    label=f"β={beta:g}",
                )
                if len(histories) > 1:
                    deviation = values.std(0, ddof=1)
                    axes[row, column].fill_between(
                        steps,
                        average - deviation,
                        average + deviation,
                        color=colors[index % len(colors)],
                        alpha=0.15,
                    )
        axes[0, column].set_title(labels[method])
        axes[1, column].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel(r"$\mathbb{E}\|Z_A\|^2/n$")
    axes[1, 0].set_ylabel(r"$\mathbb{E}\|Z_B\|^2/n$")
    axes[0, -1].legend(fontsize=10)
    fig.tight_layout(pad=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(args.output.with_suffix("." + extension), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
