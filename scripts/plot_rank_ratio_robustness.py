#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


INIT_DIR_RE = re.compile(
    r"^(?P<model>.+?)_(?P<adapter>lora|qlora)_Kaiming_"
    r"(?P<init_para>A|AA|AB(?:_reset)?|B(?:_reset)?)_"
    r"A(?P<a_scale>[^_]+)_B(?P<b_scale>[^_]+)_normal(?P<normal>[^_]+)_init_std(?P<init_std>.+)$"
)
RUN_DIR_RE = re.compile(
    r"^(?P<model>.+?)_(?P<adapter>lora|qlora)_"
    r"(?P<init_para>A|AA|AB(?:_reset)?|B(?:_reset)?)_"
    r"r(?P<rank>\d+)_a(?P<alpha>[^_]+)_lr(?P<lr>[^_]+)_ratio(?P<ratio>[^_]+)_s(?P<seed>\d+)$"
)
INIT_LABELS = {
    "A": "Init[A]",
    "AA": "Init[A^2]",
}


@dataclass(frozen=True)
class Record:
    model_tag: str
    adapter_name: str
    init_para: str
    init_std: float
    lora_a_scale: float
    lora_b_scale: float
    lora_normal: str
    rank: int
    alpha: float
    lr_a: str
    ratio: float
    seed: int
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build rank-vs-ratio summaries and heatmaps from "
            "experiments/run_llama_test.sh outputs."
        )
    )
    parser.add_argument("--input_root", type=Path, default=Path("log_rank_ratio"))
    parser.add_argument("--dataset", type=str, default="math_10k")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model_tag", type=str, default="Llama-3.2-1B")
    parser.add_argument("--adapter_name", type=str, default="lora")
    parser.add_argument("--lora_normal", type=str, default="False")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_a", type=str, default="1e-5")
    parser.add_argument("--betas", nargs="+", default=["0.5", "4", "8"])
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=["1", "10", "200", "500", "1000", "2000", "4000"],
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def canonical_float_str(value: float) -> str:
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return f"{value:g}"


def parse_float(text: str) -> float:
    return float(text)


def resolve_search_root(input_root: Path, dataset: str) -> Path:
    dataset_root = input_root / dataset
    if dataset_root.exists():
        return dataset_root
    return input_root


def discover_records(search_root: Path) -> list[Record]:
    records: list[Record] = []
    for accuracy_path in search_root.glob("*/*/evaluated_result/accuracy.json"):
        init_dir = accuracy_path.parent.parent.parent.name
        run_dir = accuracy_path.parent.parent.name
        init_match = INIT_DIR_RE.match(init_dir)
        run_match = RUN_DIR_RE.match(run_dir)
        if not init_match or not run_match:
            continue
        payload = json.loads(accuracy_path.read_text())
        avg_accuracy = payload.get("all", {}).get("avg_accuracy")
        if avg_accuracy is None:
            continue
        records.append(
            Record(
                model_tag=init_match.group("model"),
                adapter_name=init_match.group("adapter"),
                init_para=init_match.group("init_para"),
                init_std=parse_float(init_match.group("init_std")),
                lora_a_scale=parse_float(init_match.group("a_scale")),
                lora_b_scale=parse_float(init_match.group("b_scale")),
                lora_normal=init_match.group("normal"),
                rank=int(run_match.group("rank")),
                alpha=parse_float(run_match.group("alpha")),
                lr_a=run_match.group("lr"),
                ratio=parse_float(run_match.group("ratio")),
                seed=int(run_match.group("seed")),
                accuracy=float(avg_accuracy) * 100.0,
            )
        )
    return records


def filter_records(records: list[Record], args: argparse.Namespace) -> list[Record]:
    beta_set = {parse_float(beta) for beta in args.betas}
    rank_set = set(args.ranks)
    ratio_set = {parse_float(ratio) for ratio in args.ratios}
    filtered: list[Record] = []

    for record in records:
        if record.model_tag != args.model_tag:
            continue
        if record.adapter_name != args.adapter_name:
            continue
        if record.init_para not in INIT_LABELS:
            continue
        if record.rank not in rank_set:
            continue
        if record.seed != args.seed:
            continue
        if record.lora_normal.lower() != args.lora_normal.lower():
            continue
        if record.lr_a != args.lr_a:
            continue
        if record.init_std not in beta_set:
            continue
        if record.ratio not in ratio_set:
            continue
        if record.init_para == "A" and (
            not math.isclose(record.lora_a_scale, 0.5)
            or not math.isclose(record.lora_b_scale, 0.5)
        ):
            continue
        if record.init_para == "AA" and (
            not math.isclose(record.lora_a_scale, 1.0)
            or not math.isclose(record.lora_b_scale, 1.0)
        ):
            continue
        filtered.append(record)

    return filtered


def build_panels(
    records: list[Record],
    betas: list[float],
    ranks: list[int],
    ratios: list[float],
    strict: bool,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    panels: dict[str, dict[str, object]] = {}
    warnings: list[str] = []

    for init_para, label in INIT_LABELS.items():
        for beta in betas:
            panel_key = f"{init_para}_beta{canonical_float_str(beta)}"
            values = [[math.nan for _ in ratios] for _ in ranks]
            panel_records = [
                record
                for record in records
                if record.init_para == init_para and math.isclose(record.init_std, beta)
            ]

            for record in panel_records:
                row = ranks.index(record.rank)
                col = ratios.index(record.ratio)
                values[row][col] = record.accuracy

            missing = [
                (rank, ratio)
                for row, rank in enumerate(ranks)
                for col, ratio in enumerate(ratios)
                if math.isnan(values[row][col])
            ]
            if missing:
                message = (
                    f"{panel_key} is missing {len(missing)} cells: "
                    + ", ".join(
                        f"(r={rank}, ratio={canonical_float_str(ratio)})"
                        for rank, ratio in missing[:6]
                    )
                    + (" ..." if len(missing) > 6 else "")
                )
                if strict:
                    raise ValueError(message)
                warnings.append(message)

            if any(
                not math.isclose(record.alpha, float(record.rank))
                for record in panel_records
            ):
                warnings.append(f"{panel_key} includes runs where alpha != rank.")

            best_ratio_by_rank: dict[str, int | float | None] = {}
            for row, rank in enumerate(ranks):
                row_values = values[row]
                finite_entries = [
                    (col_index, value)
                    for col_index, value in enumerate(row_values)
                    if not math.isnan(value)
                ]
                if not finite_entries:
                    best_ratio_by_rank[str(rank)] = None
                    continue
                best_col, _ = max(finite_entries, key=lambda item: item[1])
                best_ratio = ratios[best_col]
                best_ratio_by_rank[str(rank)] = (
                    int(best_ratio) if float(best_ratio).is_integer() else best_ratio
                )

            panels[panel_key] = {
                "label": label,
                "init_para": init_para,
                "beta": beta,
                "lora_A_scale": 0.5 if init_para == "A" else 1.0,
                "lora_B_scale": 0.5 if init_para == "A" else 1.0,
                "Rank": ranks,
                "Ratio": [
                    int(ratio) if float(ratio).is_integer() else ratio
                    for ratio in ratios
                ],
                "Values": [
                    [
                        None if math.isnan(value) else round(float(value), 4)
                        for value in row
                    ]
                    for row in values
                ],
                "BestRatioByRank": best_ratio_by_rank,
            }

    return panels, warnings


def write_csv_panels(output_dir: Path, panels: dict[str, dict[str, object]]) -> None:
    for panel_key, panel in panels.items():
        csv_path = output_dir / f"{panel_key}.csv"
        ratios = panel["Ratio"]
        values = panel["Values"]
        ranks = panel["Rank"]
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", *ratios, "best_ratio"])
            best_ratio_by_rank = panel["BestRatioByRank"]
            for rank, row in zip(ranks, values):
                writer.writerow([rank, *row, best_ratio_by_rank[str(rank)]])


def render_heatmap(
    output_dir: Path,
    dataset: str,
    model_tag: str,
    panels: dict[str, dict[str, object]],
    show: bool,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np
    except ImportError:
        print(
            "matplotlib/numpy is not installed; wrote JSON/CSV summaries only.",
            file=sys.stderr,
        )
        return None

    panel_items = sorted(
        panels.items(),
        key=lambda item: (item[1]["beta"], item[1]["init_para"]),
    )
    nrows = len({panel["beta"] for _, panel in panel_items})
    ncols = len(INIT_LABELS)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols + 1.2, 2.7 * nrows + 1.0),
        constrained_layout=True,
        squeeze=False,
    )

    all_values = [
        value
        for _, panel in panel_items
        for row in panel["Values"]
        for value in row
        if value is not None
    ]
    vmin = min(all_values)
    vmax = max(all_values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#e5e7eb")

    image = None
    for index, (panel_key, panel) in enumerate(panel_items):
        row = index // ncols
        col = index % ncols
        ax = axes[row][col]
        matrix = np.array(
            [
                [np.nan if value is None else value for value in row_vals]
                for row_vals in panel["Values"]
            ],
            dtype=float,
        )
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ratios = [str(value) for value in panel["Ratio"]]
        ranks = [str(value) for value in panel["Rank"]]
        ax.set_xticks(np.arange(len(ratios)))
        ax.set_xticklabels(ratios, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(ranks)))
        ax.set_yticklabels(ranks)
        ax.set_xlabel("LR ratio ($\\eta_B / \\eta_A$)")
        ax.set_ylabel("Rank")
        ax.set_title(f"{panel['label']}, beta={canonical_float_str(panel['beta'])}")

        for rank_index, rank in enumerate(panel["Rank"]):
            best_ratio = panel["BestRatioByRank"][str(rank)]
            if best_ratio is None:
                continue
            best_col = panel["Ratio"].index(best_ratio)
            ax.add_patch(
                Rectangle(
                    (best_col - 0.5, rank_index - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="white",
                    linewidth=2.0,
                )
            )
            ax.scatter(best_col, rank_index, marker="o", s=18, c="black", linewidths=0)

    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.92, label="Average accuracy (%)")
    fig.suptitle(f"{model_tag} {dataset}: rank-ratio robustness", fontsize=13)

    png_path = output_dir / "rank_ratio_robustness_heatmap.png"
    pdf_path = output_dir / "rank_ratio_robustness_heatmap.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    search_root = resolve_search_root(args.input_root, args.dataset)
    if not search_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {search_root}")

    output_dir = args.output_dir or (search_root / "rank_ratio_robustness")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(search_root)
    filtered = filter_records(records, args)
    if not filtered:
        raise RuntimeError(f"No matching accuracy files found under {search_root}")

    betas = [parse_float(beta) for beta in args.betas]
    ratios = [parse_float(ratio) for ratio in args.ratios]
    panels, warnings = build_panels(
        records=filtered,
        betas=betas,
        ranks=args.ranks,
        ratios=ratios,
        strict=args.strict,
    )

    summary = {
        "dataset": args.dataset,
        "model_tag": args.model_tag,
        "adapter_name": args.adapter_name,
        "seed": args.seed,
        "lr_a": args.lr_a,
        "panels": panels,
    }
    if warnings:
        summary["warnings"] = warnings

    summary_path = output_dir / "rank_ratio_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_csv_panels(output_dir, panels)
    heatmap_path = render_heatmap(
        output_dir=output_dir,
        dataset=args.dataset,
        model_tag=args.model_tag,
        panels=panels,
        show=args.show,
    )

    print(f"Wrote summary JSON to {summary_path}")
    print(f"Wrote panel CSV files to {output_dir}")
    if heatmap_path is not None:
        print(f"Wrote heatmap to {heatmap_path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
