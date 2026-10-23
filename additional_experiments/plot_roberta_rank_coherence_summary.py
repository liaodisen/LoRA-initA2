#!/usr/bin/env python3
"""Summarize the RoBERTa SST-2 rank-coherence sweep in one paper figure."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "paper_purple": "#281456",
    "paper_magenta": "#B73779",
    "paper_peach": "#FEB77E",
    "neutral": "#777777",
}

INIT_ORDER = ("A", "AA")
INIT_LABELS = {"A": r"Init[$A$]", "AA": r"Init[$A^2$]"}
INIT_STYLES = {
    "A": (PALETTE["paper_purple"], "s"),
    "AA": (PALETTE["paper_magenta"], "^"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final coherence, rank exponent, and accuracy for RoBERTa/SST-2."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("additional_outputs/roberta_rank_coherence_sst2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("additional_outputs/roberta_rank_coherence_summary"),
        help="Output basename; .png and .pdf are written.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def discover_records(input_root: Path) -> list[dict[str, float | int | str]]:
    paths = sorted(input_root.rglob("coherence_stats.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No coherence_stats.jsonl files found under {input_root}"
        )

    records: list[dict[str, float | int | str]] = []
    for path in paths:
        rows_by_step: dict[int, dict[str, Any]] = {}
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    rows_by_step[int(payload["step"])] = payload
        if not rows_by_step:
            continue

        run_dir = path.parent.parent
        eval_payload = read_json(run_dir / "eval_results.json")
        eval_accuracy = float(eval_payload["eval_accuracy"]) * 100.0
        for step in sorted(rows_by_step):
            payload = rows_by_step[step]
            aggregate = payload["aggregate"]
            records.append(
                {
                    "init": str(payload["init_para"]),
                    "seed": int(payload["seed"]),
                    "rank": int(payload["rank"]),
                    "ratio": float(payload["lr_ratio"]),
                    "step": step,
                    "epoch": float(payload["epoch"]),
                    "max_steps": int(payload["max_steps"]),
                    "s": float(aggregate["current_s_median_across_modules_median"]),
                    "c": float(aggregate["current_c_median_across_modules_median"]),
                    "accuracy": eval_accuracy,
                }
            )

    expected_runs = 2 * 6 * 3 * 3
    run_keys = {
        (record["init"], record["seed"], record["rank"], record["ratio"])
        for record in records
    }
    if len(run_keys) != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} completed runs, found {len(run_keys)}"
        )
    return records


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def final_by_rank(
    records: list[dict[str, float | int | str]],
) -> dict[str, dict[str, np.ndarray]]:
    final_records = [
        record for record in records if record["step"] == record["max_steps"]
    ]
    result: dict[str, dict[str, np.ndarray]] = {}
    for init in INIT_ORDER:
        ranks = sorted(
            {int(record["rank"]) for record in final_records if record["init"] == init}
        )
        c_mean, c_std, acc_mean, acc_std = [], [], [], []
        for rank in ranks:
            group = [
                record
                for record in final_records
                if record["init"] == init and int(record["rank"]) == rank
            ]
            if len(group) != 9:
                raise ValueError(
                    f"Expected 9 final records for init={init}, rank={rank}; found {len(group)}"
                )
            c_stats = mean_std([float(record["c"]) for record in group])
            acc_stats = mean_std([float(record["accuracy"]) for record in group])
            c_mean.append(c_stats[0])
            c_std.append(c_stats[1])
            acc_mean.append(acc_stats[0])
            acc_std.append(acc_stats[1])
        result[init] = {
            "rank": np.asarray(ranks, dtype=float),
            "c_mean": np.asarray(c_mean),
            "c_std": np.asarray(c_std),
            "acc_mean": np.asarray(acc_mean),
            "acc_std": np.asarray(acc_std),
        }
    return result


def fit_gamma_over_time(
    records: list[dict[str, float | int | str]],
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for init in INIT_ORDER:
        steps = sorted(
            {int(record["step"]) for record in records if record["init"] == init}
        )
        epochs, gammas, r_squared = [], [], []
        for step in steps:
            step_group = [
                record
                for record in records
                if record["init"] == init and int(record["step"]) == step
            ]
            by_rank: dict[int, list[float]] = defaultdict(list)
            for record in step_group:
                by_rank[int(record["rank"])].append(float(record["s"]))
            ranks = sorted(by_rank)
            if any(len(by_rank[rank]) != 9 for rank in ranks):
                raise ValueError(
                    f"Incomplete rank aggregation for init={init}, step={step}"
                )
            s_values = np.asarray([np.mean(by_rank[rank]) for rank in ranks])
            log_rank = np.log(np.asarray(ranks, dtype=float))
            log_s = np.log(s_values)
            slope, intercept = np.polyfit(log_rank, log_s, 1)
            predictions = slope * log_rank + intercept
            residual = float(np.square(log_s - predictions).sum())
            total = float(np.square(log_s - log_s.mean()).sum())
            epochs.append(
                float(np.mean([float(record["epoch"]) for record in step_group]))
            )
            gammas.append(float(slope))
            r_squared.append(1.0 - residual / total if total > 0 else math.nan)
        result[init] = {
            "epoch": np.asarray(epochs),
            "gamma": np.asarray(gammas),
            "r2": np.asarray(r_squared),
        }
    return result


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
    rank_summary: dict[str, dict[str, np.ndarray]],
    gamma_summary: dict[str, dict[str, np.ndarray]],
    output: Path,
) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.25))
    legend_handles = []

    for init in INIT_ORDER:
        color, marker = INIT_STYLES[init]
        data = rank_summary[init]
        (line,) = axes[0].plot(
            data["rank"],
            data["c_mean"],
            color=color,
            marker=marker,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=2.4,
            label=INIT_LABELS[init],
            zorder=3,
        )
        axes[0].fill_between(
            data["rank"],
            np.maximum(data["c_mean"] - data["c_std"], 0.0),
            np.minimum(data["c_mean"] + data["c_std"], 1.0),
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )
        legend_handles.append(line)

        gamma = gamma_summary[init]
        axes[1].plot(
            gamma["epoch"],
            gamma["gamma"],
            color=color,
            marker=marker,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=2.4,
            zorder=3,
        )

        axes[2].plot(
            data["rank"],
            data["acc_mean"],
            color=color,
            marker=marker,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=2.4,
            zorder=3,
        )
        axes[2].fill_between(
            data["rank"],
            data["acc_mean"] - data["acc_std"],
            data["acc_mean"] + data["acc_std"],
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )

    axes[1].axhline(
        1.0, color=PALETTE["neutral"], linewidth=1.5, linestyle="--", zorder=1
    )
    axes[1].axhline(
        0.5, color=PALETTE["neutral"], linewidth=1.5, linestyle=":", zorder=1
    )
    axes[1].text(
        1.0,
        1.015,
        "LLN",
        color=PALETTE["neutral"],
        ha="right",
        va="bottom",
        fontsize=10,
    )
    axes[1].text(
        1.0,
        0.515,
        "CLT",
        color=PALETTE["neutral"],
        ha="right",
        va="bottom",
        fontsize=10,
    )

    axes[0].set_title("(a) Final normalized coherence", pad=9)
    axes[0].set_ylabel(r"$C_T(r)$")
    axes[0].set_ylim(0.5, 1.02)

    axes[1].set_title("(b) Rank-scaling exponent", pad=9)
    axes[1].set_xlabel("Training progress (epoch)")
    axes[1].set_ylabel(r"$\gamma_t$ in $S_t(r)\propto r^{\gamma_t}$")
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0.4, 1.25)

    axes[2].set_title("(c) SST-2 accuracy", pad=9)
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_ylim(45, 100)
    axes[2].legend(
        legend_handles,
        [INIT_LABELS[init] for init in INIT_ORDER],
        loc="upper right",
        fontsize=13,
        handlelength=2.5,
        markerscale=1.2,
        labelspacing=0.7,
    )

    for ax in (axes[0], axes[2]):
        ax.set_xscale("log", base=2)
        ax.set_xlim(3.5, 150)
        ax.set_xticks([4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(["4", "8", "16", "32", "64", "128"])
        ax.set_xlabel("LoRA rank $r$")

    for ax in axes:
        ax.tick_params(direction="out", length=4.5, width=1.4)
        ax.grid(axis="y", color="#CFCECE", linewidth=0.7, alpha=0.5, zorder=0)

    fig.tight_layout(pad=1.25, w_pad=2.2)
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
    records = discover_records(args.input_root)
    rank_summary = final_by_rank(records)
    gamma_summary = fit_gamma_over_time(records)
    plot(rank_summary, gamma_summary, args.output)
    print(f"Parsed {len(records)} checkpoint records from 108 completed runs")
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
