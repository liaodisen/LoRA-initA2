#!/usr/bin/env python3
"""Plot Z_A and scaled Z_B RMS trajectories for two Llama models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from plot_llama_parameter_gradient_norms import (
    INITS,
    INIT_LABELS,
    MODELS,
    MODEL_LABELS,
    PALETTE,
    aggregate_records,
    apply_publication_style,
    load_records,
)


METRICS = (
    ("z_a_rms", r"$\|Z_A^t\|_{\mathrm{RMS}}$", PALETTE["paper_purple"], "s"),
    ("lora_delta_rms", r"$\|Z_B^t\|_{\mathrm{RMS}}$", PALETTE["paper_magenta"], "^"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot seed-aggregated Z_A and scaled Z_B RMS trajectories."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("additional_outputs/llama_feature_decomposition"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("additional_outputs/llama_za_zb_rms_track"),
        help="Output basename; .png and .pdf are written.",
    )
    return parser.parse_args()


def plot(
    aggregated: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    output: Path,
) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.3), sharex=True, sharey=True)

    legend_handles = []
    panel_index = 0
    for model in MODELS:
        for init_para in INITS:
            ax = axes[panel_index]
            for metric, label, color, marker in METRICS:
                steps, mean, std = aggregated[(model, init_para, metric)]
                (line,) = ax.plot(
                    steps,
                    mean,
                    color=color,
                    linewidth=1.9,
                    marker=marker,
                    markersize=4.2,
                    markevery=20,
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    label=label,
                    zorder=3,
                )
                ax.fill_between(
                    steps,
                    np.maximum(mean - std, 0.0),
                    mean + std,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                    zorder=2,
                )
                if panel_index == 0:
                    legend_handles.append(line)

            ax.set_title(
                f"{MODEL_LABELS[model]}\n{INIT_LABELS[init_para]}",
                pad=8,
                fontsize=13,
            )
            ax.set_xlim(left=0)
            ax.set_ylim(0, 0.2)
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.tick_params(direction="out", length=4.5, width=1.4)
            ax.grid(axis="y", color="#CFCECE", linewidth=0.7, alpha=0.5, zorder=0)
            if panel_index == 0:
                ax.set_ylabel("Normalized feature norm")
            panel_index += 1

    axes[3].legend(
        legend_handles,
        [label for _, label, _, _ in METRICS],
        loc="upper left",
        handlelength=2.7,
        fontsize=14,
        markerscale=1.3,
        labelspacing=0.8,
    )
    fig.supxlabel("Optimizer step", y=0.025)
    fig.tight_layout(rect=(0.01, 0.07, 1.0, 1.0), pad=1.15)

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output.with_suffix(suffix),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_records(args.input_root, METRICS)
    aggregated = aggregate_records(records, METRICS)
    plot(aggregated, args.output)
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
