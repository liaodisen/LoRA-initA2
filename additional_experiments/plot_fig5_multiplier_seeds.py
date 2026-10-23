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
INIT_LABELS = {"A": "Init[A]", "AA": "Init[A^2]"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Figure 5 multiplier runs over seeds."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("additional_outputs/fig5_multiplier_seeds"),
    )
    parser.add_argument("--dataset", type=str, default="math_10k")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model_tag", type=str, default="Meta-Llama-3-8B")
    parser.add_argument("--adapter_name", type=str, default="lora")
    parser.add_argument("--init_std", type=float, default=0.5)
    parser.add_argument("--lr_a", type=str, default="1e-5")
    parser.add_argument("--lr_ratio", type=float, default=200.0)
    return parser.parse_args()


def parse_accuracy(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text())
        return float(payload["all"]["avg_accuracy"]) * 100.0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def discover_records(args: argparse.Namespace) -> list[dict[str, object]]:
    dataset_root = args.input_root / args.dataset
    records: list[dict[str, object]] = []
    for accuracy_path in dataset_root.glob("*/*/evaluated_result/accuracy.json"):
        init_match = INIT_DIR_RE.match(accuracy_path.parent.parent.parent.name)
        run_match = RUN_DIR_RE.match(accuracy_path.parent.parent.name)
        if init_match is None or run_match is None:
            continue

        accuracy = parse_accuracy(accuracy_path)
        if accuracy is None:
            continue

        init_para = init_match.group("init_para")
        rank = int(run_match.group("rank"))
        alpha = float(run_match.group("alpha"))
        multiplier = alpha / rank
        record = {
            "path": str(accuracy_path),
            "model_tag": init_match.group("model"),
            "adapter_name": init_match.group("adapter"),
            "init_para": init_para,
            "init_label": INIT_LABELS.get(init_para, init_para),
            "init_std": float(init_match.group("init_std")),
            "rank": rank,
            "alpha": alpha,
            "multiplier": multiplier,
            "lr_a": run_match.group("lr"),
            "lr_ratio": float(run_match.group("ratio")),
            "seed": int(run_match.group("seed")),
            "accuracy": accuracy,
        }
        records.append(record)
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["init_para"]), float(record["multiplier"]))].append(record)

    rows: list[dict[str, object]] = []
    for (init_para, multiplier), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        values = [float(record["accuracy"]) for record in group]
        seeds = sorted(int(record["seed"]) for record in group)
        rows.append(
            {
                "init_para": init_para,
                "init_label": INIT_LABELS.get(init_para, init_para),
                "multiplier": multiplier,
                "mean_accuracy": mean(values),
                "std_accuracy": stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
                "seeds": seeds,
            }
        )
    return rows


def write_outputs(
    output_dir: Path,
    records: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "fig5_multiplier_seed_records.csv"
    with records_path.open("w", newline="") as handle:
        fieldnames = [
            "init_para",
            "init_label",
            "multiplier",
            "rank",
            "alpha",
            "seed",
            "accuracy",
            "lr_a",
            "lr_ratio",
            "path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fieldnames})

    summary_path = output_dir / "fig5_multiplier_seed_summary.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "init_para",
            "init_label",
            "multiplier",
            "mean_accuracy",
            "std_accuracy",
            "n",
            "seeds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {**row, "seeds": " ".join(str(seed) for seed in row["seeds"])}
            )

    json_path = output_dir / "fig5_multiplier_seed_summary.json"
    json_path.write_text(
        json.dumps({"records": records, "summary": summary_rows}, indent=2)
    )


def plot(output_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting this summary.") from exc

    by_init: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_init[str(row["init_para"])].append(row)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    for init_para in ["A", "AA"]:
        rows = sorted(
            by_init.get(init_para, []), key=lambda row: float(row["multiplier"])
        )
        if not rows:
            continue
        x = [float(row["multiplier"]) for row in rows]
        y = [float(row["mean_accuracy"]) for row in rows]
        err = [float(row["std_accuracy"]) for row in rows]
        ax.errorbar(
            x,
            y,
            yerr=err,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=INIT_LABELS.get(init_para, init_para),
        )

    ax.set_xlabel("LoRA multiplier alpha/r")
    ax.set_ylabel("Arithmetic average accuracy (%)")
    ax.set_title("Llama 3-8B Figure 5 seed aggregation")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()

    png_path = output_dir / "fig5_multiplier_seed_accuracy.png"
    pdf_path = output_dir / "fig5_multiplier_seed_accuracy.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    args = parse_args()
    records = [
        record
        for record in discover_records(args)
        if record["model_tag"] == args.model_tag
        and record["adapter_name"] == args.adapter_name
        and math.isclose(float(record["init_std"]), args.init_std)
        and record["lr_a"] == args.lr_a
        and math.isclose(float(record["lr_ratio"]), args.lr_ratio)
        and record["init_para"] in INIT_LABELS
    ]
    if not records:
        raise SystemExit(
            f"No matching Figure 5 records found under {args.input_root / args.dataset}"
        )

    output_dir = args.output_dir or (
        args.input_root / args.dataset / "fig5_multiplier_seed_summary"
    )
    summary_rows = summarize(records)
    write_outputs(output_dir, records, summary_rows)
    plot(output_dir, summary_rows)
    print(f"Wrote CSV/JSON summaries to {output_dir}")


if __name__ == "__main__":
    main()
