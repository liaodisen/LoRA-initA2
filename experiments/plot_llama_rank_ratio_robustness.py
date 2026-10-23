#!/usr/bin/env python3

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it in the active environment, "
        "for example with `pip install matplotlib`."
    ) from exc


INIT_LABELS = {
    "A": "Init[A]",
    "AA": "Init[A^2]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize and plot the Llama rank-ratio robustness sweep. "
            "This expects results written by experiments/run_llama_rank_ratio_robustness.sh."
        )
    )
    parser.add_argument(
        "--input_root", type=Path, default=Path("./log_rank_ratio_robustness")
    )
    parser.add_argument("--dataset", type=str, default="math_10k")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--summary_name", type=str, default="rank_ratio_robustness_summary.json"
    )
    parser.add_argument(
        "--figure_name", type=str, default="rank_ratio_robustness_heatmap.png"
    )
    parser.add_argument("--init_paras", nargs="+", default=["A", "AA"])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.5, 4.0, 8.0])
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[1, 10, 200, 500, 1000, 2000, 4000]
    )
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=100.0)
    return parser.parse_args()


def canonical_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def canonical_ratio(value: float) -> str:
    return f"{float(value):.3f}"


def make_nan_matrix(rows: int, cols: int) -> List[List[float]]:
    return [[math.nan for _ in range(cols)] for _ in range(rows)]


def row_all_missing(row: List[float]) -> bool:
    return all(math.isnan(value) for value in row)


def row_max(row: List[float]) -> Optional[float]:
    present = [value for value in row if not math.isnan(value)]
    if not present:
        return None
    return max(present)


def row_argmax(row: List[float]) -> Optional[int]:
    best_value = row_max(row)
    if best_value is None:
        return None
    for index, value in enumerate(row):
        if not math.isnan(value) and value == best_value:
            return index
    return None


def parse_init_dir_name(name: str) -> Optional[Dict[str, str]]:
    if "_Kaiming_" not in name:
        return None

    prefix, suffix = name.split("_Kaiming_", 1)
    if "_" not in prefix:
        return None
    model_tag, adapter_name = prefix.rsplit("_", 1)
    match = re.fullmatch(
        r"(?P<init_para>.+?)_A(?P<a_scale>[^_]+)_B(?P<b_scale>[^_]+)_normal(?P<normal>[^_]+)_init_std(?P<init_std>.+)",
        suffix,
    )
    if not match:
        return None

    values = match.groupdict()
    values["model_tag"] = model_tag
    values["adapter_name"] = adapter_name
    return values


def parse_run_dir_name(name: str) -> Optional[Dict[str, str]]:
    match = re.search(
        r"_r(?P<rank>\d+)_a(?P<alpha>[^_]+)_lr(?P<lr>[^_]+)_ratio(?P<ratio>[^_]+)_s(?P<seed>[^_]+)$",
        name,
    )
    if not match:
        return None
    return match.groupdict()


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset_root = args.input_root / args.dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    records: List[Dict[str, Any]] = []
    for json_path in dataset_root.glob("*/*/evaluated_result/accuracy.json"):
        init_dir_name = json_path.parent.parent.parent.name
        run_dir_name = json_path.parent.parent.name
        init_info = parse_init_dir_name(init_dir_name)
        run_info = parse_run_dir_name(run_dir_name)
        if init_info is None or run_info is None:
            continue

        try:
            with json_path.open() as handle:
                payload = json.load(handle)
            accuracy = float(payload["all"]["avg_accuracy"]) * 100.0
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"Skipping malformed result file: {json_path} ({exc})", file=sys.stderr
            )
            continue

        record = {
            "json_path": str(json_path),
            "model_tag": init_info["model_tag"],
            "adapter_name": init_info["adapter_name"],
            "init_para": init_info["init_para"],
            "beta": float(init_info["init_std"]),
            "a_scale": float(init_info["a_scale"]),
            "b_scale": float(init_info["b_scale"]),
            "rank": int(run_info["rank"]),
            "alpha": float(run_info["alpha"]),
            "lr": run_info["lr"],
            "ratio": float(run_info["ratio"]),
            "seed": run_info["seed"],
            "accuracy": accuracy,
        }
        records.append(record)

    return records


def best_ratio_per_rank(
    matrix: List[List[float]], ratios: List[float]
) -> List[Optional[float]]:
    best: List[Optional[float]] = []
    for row in matrix:
        if row_all_missing(row):
            best.append(None)
            continue
        col = row_argmax(row)
        best.append(None if col is None else ratios[col])
    return best


def build_summary(
    args: argparse.Namespace, records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    init_set = set(args.init_paras)
    beta_set = {float(beta) for beta in args.betas}
    rank_index = {rank: idx for idx, rank in enumerate(args.ranks)}
    ratio_index = {float(ratio): idx for idx, ratio in enumerate(args.ratios)}

    summary: Dict[str, Any] = {
        "_meta": {
            "input_root": str(args.input_root),
            "dataset": args.dataset,
            "ranks": list(args.ranks),
            "ratios": list(args.ratios),
            "betas": list(args.betas),
            "beta_note": "beta corresponds to init_std in this repository",
        }
    }

    for init_para in args.init_paras:
        summary[init_para] = {}
        for beta in args.betas:
            matrix = make_nan_matrix(len(args.ranks), len(args.ratios))
            path_grid: List[List[Optional[str]]] = [
                [None for _ in args.ratios] for _ in args.ranks
            ]

            for record in records:
                if record["init_para"] != init_para:
                    continue
                if float(record["beta"]) != float(beta):
                    continue
                if (
                    record["rank"] not in rank_index
                    or float(record["ratio"]) not in ratio_index
                ):
                    continue

                row = rank_index[int(record["rank"])]
                col = ratio_index[float(record["ratio"])]
                matrix[row][col] = float(record["accuracy"])
                path_grid[row][col] = str(record["json_path"])

            best_ratios = best_ratio_per_rank(matrix, list(args.ratios))
            best_accuracies: List[Optional[float]] = []
            for row in matrix:
                best_accuracies.append(row_max(row))

            beta_key = canonical_number(beta)
            summary[init_para][beta_key] = {
                "Ranks": list(args.ranks),
                "Ratios": list(args.ratios),
                "Values": [
                    [
                        None if math.isnan(value) else round(float(value), 4)
                        for value in row
                    ]
                    for row in matrix
                ],
                "BestRatioByRank": best_ratios,
                "BestAccuracyByRank": best_accuracies,
                "Paths": path_grid,
            }

    missing = [
        (init_para, beta, rank, ratio)
        for init_para in args.init_paras
        for beta in args.betas
        for rank in args.ranks
        for ratio in args.ratios
        if summary[init_para][canonical_number(beta)]["Paths"][args.ranks.index(rank)][
            args.ratios.index(ratio)
        ]
        is None
    ]
    if missing:
        print(
            f"Missing {len(missing)} grid points; heatmap will contain blanks.",
            file=sys.stderr,
        )

    used_records = [
        record
        for record in records
        if record["init_para"] in init_set and float(record["beta"]) in beta_set
    ]
    summary["_meta"]["matched_runs"] = len(used_records)
    summary["_meta"]["missing_grid_points"] = len(missing)
    return summary


def plot_summary(
    args: argparse.Namespace, summary: Dict[str, Any], output_path: Path
) -> None:
    init_paras = list(args.init_paras)
    betas = list(args.betas)
    rows = len(init_paras)
    cols = len(betas)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.4 * cols, 3.6 * rows),
        constrained_layout=True,
        squeeze=False,
    )

    color_image = None
    for row_idx, init_para in enumerate(init_paras):
        for col_idx, beta in enumerate(betas):
            axis = axes[row_idx][col_idx]
            beta_key = canonical_number(beta)
            panel = summary[init_para][beta_key]
            matrix = [
                [math.nan if value is None else float(value) for value in row]
                for row in panel["Values"]
            ]

            color_image = axis.imshow(
                matrix,
                aspect="auto",
                origin="upper",
                interpolation="nearest",
                cmap="viridis",
                vmin=args.vmin,
                vmax=args.vmax,
            )

            axis.set_title(f"{INIT_LABELS.get(init_para, init_para)}, beta={beta_key}")
            axis.set_xticks(range(len(args.ratios)))
            axis.set_xticklabels(
                [canonical_number(ratio) for ratio in args.ratios],
                rotation=45,
                ha="right",
            )
            axis.set_yticks(range(len(args.ranks)))
            axis.set_yticklabels([str(rank) for rank in args.ranks])

            if row_idx == rows - 1:
                axis.set_xlabel("LR ratio eta_B / eta_A")
            if col_idx == 0:
                axis.set_ylabel("Rank r")

            best_ratios = panel["BestRatioByRank"]
            for matrix_row, ratio in enumerate(best_ratios):
                if ratio is None:
                    continue
                matrix_col = args.ratios.index(ratio)
                axis.add_patch(
                    Rectangle(
                        (matrix_col - 0.5, matrix_row - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="white",
                        linewidth=2.0,
                    )
                )
                axis.plot(
                    matrix_col, matrix_row, marker="*", color="white", markersize=8
                )

            axis.set_xticks(
                [index - 0.5 for index in range(len(args.ratios) + 1)], minor=True
            )
            axis.set_yticks(
                [index - 0.5 for index in range(len(args.ranks) + 1)], minor=True
            )
            axis.grid(which="minor", color="white", linewidth=0.35, alpha=0.25)
            axis.tick_params(which="minor", bottom=False, left=False)

    if color_image is not None:
        flat_axes = [axis for row in axes for axis in row]
        fig.colorbar(color_image, ax=flat_axes, label="Average accuracy (%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_root / args.dataset

    records = load_records(args)
    summary = build_summary(args, records)

    summary_path = args.output_dir / args.summary_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    figure_path = args.output_dir / args.figure_name
    plot_summary(args, summary, figure_path)

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote heatmap to {figure_path}")


if __name__ == "__main__":
    main()
