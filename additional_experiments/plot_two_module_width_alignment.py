#!/usr/bin/env python3
"""Plot width-alignment scaling for q_proj and final-block down_proj modules."""

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
    "grid": "#CFCECE",
}

INIT_ORDER = ("A", "AA")
INIT_LABELS = {"A": r"Init[$A$]", "AA": r"Init[$A^2$]"}
INIT_STYLES = {
    "A": (PALETTE["paper_purple"], "s"),
    "AA": (PALETTE["paper_magenta"], "^"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the actual dot-product width exponent for two module rules."
    )
    parser.add_argument(
        "--q-proj-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--down-proj-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("additional_outputs/two_module_width_alignment"),
        help="Output basename; .png and .pdf are written.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def load_series(
    q_proj_csv: Path,
    down_proj_csv: Path,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    source_rows = {
        "q_proj": read_rows(q_proj_csv),
        "down_proj": [
            row for row in read_rows(down_proj_csv) if row.get("rule") == "last_layer"
        ],
    }
    series = {}
    for module, rows in source_rows.items():
        for init in INIT_ORDER:
            selected = [
                row
                for row in rows
                if row["init_para"] == init
                and finite_float(row, "processed_dot_slope") is not None
            ]
            selected.sort(key=lambda row: int(row["step"]))
            if len(selected) != 8:
                raise ValueError(
                    f"Expected eight finite checkpoints for {module}, init={init}; "
                    f"found {len(selected)}"
                )
            series[(module, init)] = (
                np.asarray([int(row["step"]) for row in selected]),
                np.asarray([float(row["processed_dot_slope"]) for row in selected]),
                np.asarray([float(row["processed_dot_ci_low"]) for row in selected]),
                np.asarray([float(row["processed_dot_ci_high"]) for row in selected]),
            )
    return series


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12.5,
            "axes.linewidth": 1.7,
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
    series: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
    output: Path,
) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.65), sharex=True, sharey=True)
    panel_specs = (
        ("q_proj", r"$q_{\mathrm{proj}}$"),
        ("down_proj", r"$\mathrm{down}_{\mathrm{proj}}$"),
    )
    legend_handles = []

    for panel_index, (module, title) in enumerate(panel_specs):
        ax = axes[panel_index]
        for init in INIT_ORDER:
            color, marker = INIT_STYLES[init]
            steps, slope, ci_low, ci_high = series[(module, init)]
            ax.fill_between(
                steps,
                ci_low,
                ci_high,
                color=color,
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )
            (line,) = ax.plot(
                steps,
                slope,
                color=color,
                marker=marker,
                markersize=6.2,
                markerfacecolor="white",
                markeredgewidth=1.25,
                linewidth=2.35,
                label=INIT_LABELS[init],
                zorder=3,
            )
            if panel_index == 0:
                legend_handles.append(line)

        ax.axhline(
            1.0,
            color=PALETTE["neutral"],
            linewidth=1.45,
            linestyle="--",
            zorder=1,
        )
        ax.axhline(
            0.0,
            color=PALETTE["neutral"],
            linewidth=1.35,
            linestyle=":",
            zorder=1,
        )
        ax.set_title(title, fontsize=14, pad=8)
        ax.set_xscale("log")
        ax.set_xlim(4.5, 700)
        ax.set_ylim(-0.75, 2.45)
        ax.set_xticks([5, 25, 100, 620])
        ax.set_xticklabels(["5", "25", "100", "620"])
        ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        ax.minorticks_off()
        ax.tick_params(direction="out", length=4.2, width=1.35)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7, alpha=0.5, zorder=0)

    axes[0].set_ylabel(r"Width-scaling exponent $\kappa$")
    axes[1].text(
        0.98,
        1.025,
        r"Theory: $\kappa=1$",
        transform=axes[1].get_yaxis_transform(),
        color=PALETTE["neutral"],
        fontsize=10,
        ha="right",
        va="bottom",
    )
    axes[1].text(
        0.98,
        0.025,
        "No width scaling",
        transform=axes[1].get_yaxis_transform(),
        color=PALETTE["neutral"],
        fontsize=10,
        ha="right",
        va="bottom",
    )
    axes[1].legend(
        legend_handles,
        [INIT_LABELS[init] for init in INIT_ORDER],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.26),
        fontsize=12.5,
        handlelength=2.5,
        markerscale=1.15,
        labelspacing=0.65,
    )

    fig.supxlabel("Optimizer Step", fontsize=11.5, y=0.012)
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 1.0), pad=0.55, w_pad=1.6)

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
    series = load_series(args.q_proj_csv, args.down_proj_csv)
    plot(series, args.output)
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
