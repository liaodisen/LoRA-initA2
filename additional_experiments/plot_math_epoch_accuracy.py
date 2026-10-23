#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
        description="Plot math generation accuracy over longer fine-tuning epochs."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("additional_outputs/math_epoch_curve"),
    )
    parser.add_argument("--dataset", type=str, default="math_10k")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model_tag", type=str, default="Meta-Llama-3-8B")
    parser.add_argument("--adapter_name", type=str, default="lora")
    return parser.parse_args()


def parse_accuracy(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text())
        return float(payload["all"]["avg_accuracy"]) * 100.0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def discover_records(args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for epoch_dir in args.input_root.glob("epoch_*"):
        if not epoch_dir.is_dir():
            continue
        try:
            epoch = int(epoch_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        dataset_root = epoch_dir / args.dataset
        for accuracy_path in dataset_root.glob("*/*/evaluated_result/accuracy.json"):
            init_match = INIT_DIR_RE.match(accuracy_path.parent.parent.parent.name)
            run_match = RUN_DIR_RE.match(accuracy_path.parent.parent.name)
            if init_match is None or run_match is None:
                continue
            accuracy = parse_accuracy(accuracy_path)
            if accuracy is None:
                continue
            init_para = init_match.group("init_para")
            records.append(
                {
                    "path": str(accuracy_path),
                    "epoch": epoch,
                    "model_tag": init_match.group("model"),
                    "adapter_name": init_match.group("adapter"),
                    "init_para": init_para,
                    "init_label": INIT_LABELS.get(init_para, init_para),
                    "init_std": float(init_match.group("init_std")),
                    "rank": int(run_match.group("rank")),
                    "alpha": float(run_match.group("alpha")),
                    "lr_a": run_match.group("lr"),
                    "lr_ratio": float(run_match.group("ratio")),
                    "seed": int(run_match.group("seed")),
                    "accuracy": accuracy,
                }
            )
    return records


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["init_para"]), int(record["epoch"]))].append(record)

    rows: list[dict[str, object]] = []
    for (init_para, epoch), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        values = [float(record["accuracy"]) for record in group]
        rows.append(
            {
                "init_para": init_para,
                "init_label": INIT_LABELS.get(init_para, init_para),
                "epoch": epoch,
                "mean_accuracy": mean(values),
                "std_accuracy": stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
                "seeds": sorted(int(record["seed"]) for record in group),
            }
        )
    return rows


def write_outputs(
    output_dir: Path,
    records: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "math_epoch_accuracy_records.csv"
    with records_path.open("w", newline="") as handle:
        fieldnames = [
            "epoch",
            "init_para",
            "init_label",
            "seed",
            "accuracy",
            "rank",
            "alpha",
            "lr_a",
            "lr_ratio",
            "path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fieldnames})

    summary_path = output_dir / "math_epoch_accuracy_summary.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "epoch",
            "init_para",
            "init_label",
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

    json_path = output_dir / "math_epoch_accuracy_summary.json"
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
        rows = sorted(by_init.get(init_para, []), key=lambda row: int(row["epoch"]))
        if not rows:
            continue
        x = [int(row["epoch"]) for row in rows]
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

    ax.axvline(3, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_xlabel("Fine-tuning epochs")
    ax.set_ylabel("Arithmetic average accuracy (%)")
    ax.set_title("Llama 3-8B arithmetic accuracy over epochs")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()

    png_path = output_dir / "math_accuracy_over_epochs.png"
    pdf_path = output_dir / "math_accuracy_over_epochs.pdf"
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
        and record["init_para"] in INIT_LABELS
    ]
    if not records:
        raise SystemExit(
            f"No matching epoch accuracy records found under {args.input_root}"
        )

    output_dir = args.output_dir or (
        args.input_root / args.dataset / "math_epoch_accuracy_summary"
    )
    summary_rows = summarize(records)
    write_outputs(output_dir, records, summary_rows)
    plot(output_dir, summary_rows)
    print(f"Wrote CSV/JSON summaries to {output_dir}")


if __name__ == "__main__":
    main()
