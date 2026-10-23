#!/usr/bin/env python3
"""Plot rank-scaling exponents for Llama, Qwen, and RoBERTa coherence runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "paper_purple": "#281456",
    "paper_magenta": "#B73779",
    "neutral": "#777777",
}

INIT_ORDER = ("A", "AA")
INIT_LABELS = {"A": r"Init[$A$]", "AA": r"Init[$A^2$]"}
INIT_STYLES = {
    "A": (PALETTE["paper_purple"], "s"),
    "AA": (PALETTE["paper_magenta"], "^"),
}
MODEL_ORDER = ("Llama-3.2-1B", "Qwen3-0.6B", "RoBERTa-base")
MODEL_TITLES = {
    "Llama-3.2-1B": "(a) Llama-3.2-1B\nMath-10K",
    "Qwen3-0.6B": "(b) Qwen3-0.6B\nMath-10K",
    "RoBERTa-base": "(c) RoBERTa-base\nSST-2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the ratio-aggregated rank exponent across three models."
    )
    parser.add_argument(
        "--llama-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cross-model-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("additional_outputs/three_model_rank_scaling_coherence"),
        help="Output basename; .png and .pdf are written.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_series(
    llama_csv: Path,
    cross_model_csv: Path,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows: list[dict[str, str]] = []
    for row in read_rows(llama_csv):
        rows.append({**row, "model": "Llama-3.2-1B"})
    rows.extend(read_rows(cross_model_csv))

    series = {}
    for model in MODEL_ORDER:
        for init in INIT_ORDER:
            selected = [
                row
                for row in rows
                if row["model"] == model
                and row["init_para"] == init
                and row.get("lr_ratio", "all") == "all"
            ]
            selected.sort(key=lambda row: float(row["epoch"]))
            if len(selected) != 6:
                raise ValueError(
                    f"Expected six checkpoints for {model}, init={init}; found {len(selected)}"
                )
            series[(model, init)] = (
                np.asarray([float(row["epoch"]) for row in selected]),
                np.asarray([float(row["gamma_S"]) for row in selected]),
                np.asarray([float(row["r2_S"]) for row in selected]),
            )
    return series


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.linewidth": 1.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(
    series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    output: Path,
) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.75), sharex=True, sharey=True)
    legend_handles = []

    for panel_index, model in enumerate(MODEL_ORDER):
        ax = axes[panel_index]
        for init in INIT_ORDER:
            color, marker = INIT_STYLES[init]
            epochs, gamma, _r2 = series[(model, init)]
            (line,) = ax.plot(
                epochs,
                gamma,
                color=color,
                marker=marker,
                markersize=6.5,
                markerfacecolor="white",
                markeredgewidth=1.3,
                linewidth=2.5,
                label=INIT_LABELS[init],
                zorder=3,
            )
            if panel_index == 0:
                legend_handles.append(line)
        ax.axhline(
            1.0, color=PALETTE["neutral"], linewidth=1.5, linestyle="--", zorder=1
        )
        ax.axhline(
            0.5, color=PALETTE["neutral"], linewidth=1.5, linestyle=":", zorder=1
        )
        ax.set_title(MODEL_TITLES[model], pad=9, fontsize=14)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0.4, 1.25)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(direction="out", length=4.5, width=1.4)
        ax.grid(axis="y", color="#CFCECE", linewidth=0.7, alpha=0.5, zorder=0)

    axes[0].set_ylabel(r"Rank exponent $\gamma_t$ in $S_t(r)\propto r^{\gamma_t}$")
    axes[2].text(
        0.99,
        1.015,
        "LLN",
        color=PALETTE["neutral"],
        ha="right",
        va="bottom",
        fontsize=10.5,
    )
    axes[2].text(
        0.03,
        0.515,
        "CLT",
        color=PALETTE["neutral"],
        ha="left",
        va="bottom",
        fontsize=10.5,
    )
    axes[2].legend(
        legend_handles,
        [INIT_LABELS[init] for init in INIT_ORDER],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.20),
        fontsize=13,
        handlelength=2.6,
        markerscale=1.2,
        labelspacing=0.7,
    )
    fig.supxlabel("Training progress (epoch)", fontsize=11.5, y=0.01)
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 1.0), pad=0.55, w_pad=1.7)

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
    series = load_series(args.llama_csv, args.cross_model_csv)
    plot(series, args.output)
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
