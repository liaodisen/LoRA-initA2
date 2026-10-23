#!/usr/bin/env python3
"""Plot LoRA A/B parameter-gradient trajectories for two Llama models."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "paper_purple": "#281456",
    "paper_magenta": "#B73779",
}

MODELS = ("Llama-3.2-1B", "Meta-Llama-3-8B")
MODEL_LABELS = {
    "Llama-3.2-1B": "Llama-3.2-1B",
    "Meta-Llama-3-8B": "Llama-3-8B",
}
INITS = ("A", "AA")
INIT_LABELS = {"A": r"Init[$A$]", "AA": r"Init[$A^2$]"}
METRICS = (
    ("grad_a_l2", r"$\|\nabla_A L\|_F$", PALETTE["paper_purple"], "s"),
    ("grad_b_l2", r"$\|\nabla_B L\|_F$", PALETTE["paper_magenta"], "^"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot seed-aggregated LoRA A/B pre-optimizer gradient norms."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("additional_outputs/llama_feature_decomposition"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("additional_outputs/llama_parameter_gradient_norms_track"),
        help="Output basename; .png and .pdf are written.",
    )
    return parser.parse_args()


def identify_model(path: Path) -> str:
    matches = [model for model in MODELS if model in str(path)]
    if len(matches) != 1:
        raise ValueError(f"Could not identify exactly one model from {path}")
    return matches[0]


def load_records(
    input_root: Path,
    metrics: tuple[tuple[str, str, str, str], ...] = METRICS,
) -> dict[tuple[str, str, int, int, str], float]:
    paths = sorted(input_root.rglob("training_dynamics.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No training_dynamics.jsonl files found under {input_root}"
        )

    records: dict[tuple[str, str, int, int, str], float] = {}
    for path in paths:
        model = identify_model(path)
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                init_para = payload["init_para"]
                if init_para not in INITS:
                    continue
                modules = payload["modules"]
                if not modules:
                    raise ValueError(f"No module records in {path}:{line_number}")
                for metric, _, _, _ in metrics:
                    values = np.asarray(
                        [module[metric] for module in modules], dtype=float
                    )
                    key = (
                        model,
                        init_para,
                        int(payload["seed"]),
                        int(payload["step"]),
                        metric,
                    )
                    if key in records:
                        raise ValueError(f"Duplicate record for {key}")
                    records[key] = float(values.mean())
    return records


def aggregate_records(
    records: dict[tuple[str, str, int, int, str], float],
    metrics: tuple[tuple[str, str, str, str], ...] = METRICS,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for (model, init_para, _seed, step, metric), value in records.items():
        grouped[(model, init_para, metric, step)].append(value)

    aggregated = {}
    for model in MODELS:
        for init_para in INITS:
            for metric, _, _, _ in metrics:
                steps = sorted(
                    step
                    for group_model, group_init, group_metric, step in grouped
                    if (group_model, group_init, group_metric)
                    == (model, init_para, metric)
                )
                if not steps:
                    raise ValueError(f"No data for {model}, {init_para}, {metric}")
                seed_values = [
                    grouped[(model, init_para, metric, step)] for step in steps
                ]
                seed_counts = {len(values) for values in seed_values}
                if seed_counts != {3}:
                    raise ValueError(
                        f"Expected three seeds for {model}, {init_para}, {metric}; "
                        f"observed counts {sorted(seed_counts)}"
                    )
                matrix = np.asarray(seed_values, dtype=float)
                aggregated[(model, init_para, metric)] = (
                    np.asarray(steps),
                    matrix.mean(axis=1),
                    matrix.std(axis=1, ddof=1),
                )
    return aggregated


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
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
            ax.set_yscale("symlog", linthresh=1e-3, linscale=0.7, base=10)
            ax.tick_params(direction="out", length=4.5, width=1.4)
            ax.grid(axis="y", color="#CFCECE", linewidth=0.7, alpha=0.5, zorder=0)
            if panel_index == 0:
                ax.set_ylabel("Pre-optimizer gradient norm")
            panel_index += 1

    axes[3].legend(
        legend_handles,
        [label for _, label, _, _ in METRICS],
        loc="center right",
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
    records = load_records(args.input_root)
    aggregated = aggregate_records(records)
    plot(aggregated, args.output)
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
