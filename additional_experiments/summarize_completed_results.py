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
from typing import Any, Iterable


DATE_TAG = "release"
INIT_LABELS = {"A": "InitA", "AA": "InitA2"}
HEATMAP_GROUP_RE = re.compile(
    r"^(?P<init_label>.+)_beta(?P<init_std>[^_]+)_seed(?P<seed>\d+)$"
)
HEATMAP_RUN_RE = re.compile(r"^lr-(?P<lr>.+)_ratio-(?P<ratio>.+)$")
LLM_INIT_DIR_RE = re.compile(
    r"^(?P<model>.+?)_(?P<adapter>lora|qlora)_Kaiming_"
    r"(?P<init_para>A|AA|AB(?:_reset)?|B(?:_reset)?)_"
    r"A(?P<a_scale>[^_]+)_B(?P<b_scale>[^_]+)_normal(?P<normal>[^_]+)_init_std(?P<init_std>.+)$"
)
LLM_RUN_DIR_RE = re.compile(
    r"^(?P<model>.+?)_(?P<adapter>lora|qlora)_"
    r"(?P<init_para>A|AA|AB(?:_reset)?|B(?:_reset)?)_"
    r"r(?P<rank>\d+)_a(?P<alpha>[^_]+)_lr(?P<lr>[^_]+)_ratio(?P<ratio>[^_]+)_s(?P<seed>\d+)$"
)


def parse_args() -> argparse.Namespace:
    default_base = Path("additional_outputs")
    parser = argparse.ArgumentParser(
        description="Summarize completed additional experiment outputs."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("additional_experiments/results")
    )
    parser.add_argument(
        "--heatmap_root",
        type=Path,
        default=default_base / "roberta_heatmap_5seeds",
    )
    parser.add_argument(
        "--roberta_coherence_root",
        type=Path,
        default=default_base / "roberta_rank_coherence_sst2",
    )
    parser.add_argument(
        "--qwen_coherence_root",
        type=Path,
        default=default_base / "qwen3_0p6b_rank_coherence_math10k",
    )
    parser.add_argument(
        "--long_horizon_root",
        type=Path,
        default=default_base / "math_epoch_curve_llama31_8b_ratio4000_initstd1",
    )
    parser.add_argument("--heatmap_task", type=str, default="sst2")
    parser.add_argument("--long_horizon_dataset", type=str, default="math_10k")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def rows_to_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def safe_mean(values: Iterable[float]) -> float:
    values = [
        value for value in values if value is not None and math.isfinite(float(value))
    ]
    return mean(values) if values else math.nan


def safe_std(values: Iterable[float]) -> float:
    values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else math.nan
    return stdev(values)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def fit_log_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 2:
        return math.nan, math.nan
    log_x = [math.log(x) for x, _ in pairs]
    log_y = [math.log(y) for _, y in pairs]
    mean_x = mean(log_x)
    mean_y = mean(log_y)
    denom = sum((x - mean_x) ** 2 for x in log_x)
    if denom == 0:
        return math.nan, math.nan
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y)) / denom
    intercept = mean_y - slope * mean_x
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_x, log_y))
    total = sum((y - mean_y) ** 2 for y in log_y)
    r2 = 1.0 - residual / total if total > 0 else 1.0
    return slope, r2


def valid_slope_record(record: dict[str, Any], floor: float = 1e-8) -> bool:
    try:
        current_s = float(record["current_s_median"])
        current_c = float(record["current_c_median"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(current_s)
        and math.isfinite(current_c)
        and current_s > floor
        and current_c > floor
    )


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def parse_roberta_heatmap(
    root: Path, task: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for result_path in sorted((root / task).glob("*/*/eval_results.json")):
        group_match = HEATMAP_GROUP_RE.match(result_path.parent.parent.name)
        run_match = HEATMAP_RUN_RE.match(result_path.parent.name)
        payload = read_json(result_path)
        if group_match is None or run_match is None or payload is None:
            continue
        if "eval_accuracy" not in payload:
            continue
        records.append(
            {
                "init_label": group_match.group("init_label"),
                "init_std": float(group_match.group("init_std")),
                "seed": int(group_match.group("seed")),
                "lr_a": run_match.group("lr"),
                "lr_a_value": float(run_match.group("lr")),
                "lr_ratio": float(run_match.group("ratio")),
                "eval_accuracy": float(payload["eval_accuracy"]) * 100.0,
                "eval_loss": float(payload.get("eval_loss", math.nan)),
                "path": str(result_path),
            }
        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["init_label"],
                record["init_std"],
                record["lr_a"],
                record["lr_ratio"],
            )
        ].append(record)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], float(item[0][2]), item[0][3]),
    ):
        init_label, init_std, lr_a, lr_ratio = key
        values = [float(record["eval_accuracy"]) for record in group]
        losses = [float(record["eval_loss"]) for record in group]
        summary.append(
            {
                "init_label": init_label,
                "init_std": init_std,
                "lr_a": lr_a,
                "lr_ratio": lr_ratio,
                "n": len(group),
                "eval_accuracy_mean": safe_mean(values),
                "eval_accuracy_std": safe_std(values),
                "eval_loss_mean": safe_mean(losses),
                "seeds": " ".join(
                    str(record["seed"])
                    for record in sorted(group, key=lambda item: item["seed"])
                ),
            }
        )
    return records, summary


def parse_coherence_jsonl(path: Path) -> list[dict[str, Any]]:
    rows_by_step: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        aggregate = payload.get("aggregate", {})
        init_s = float(aggregate.get("init_s_median_across_modules_median", math.nan))
        current_s = float(
            aggregate.get("current_s_median_across_modules_median", math.nan)
        )
        init_c = float(aggregate.get("init_c_median_across_modules_median", math.nan))
        current_c = float(
            aggregate.get("current_c_median_across_modules_median", math.nan)
        )
        step = int(payload.get("step", 0))
        # Some reruns append to an existing JSONL. Keep the last record for a
        # step so each run contributes one value to each logged checkpoint.
        rows_by_step[step] = {
            "step": step,
            "epoch": float(payload.get("epoch", math.nan)),
            "max_steps": int(payload.get("max_steps", 0)),
            "seed": int(payload.get("seed", -1)),
            "rank": int(payload.get("rank", 0)),
            "init_para": str(payload.get("init_para", "")),
            "init_std": float(payload.get("init_std", math.nan)),
            "lr_a": float(payload.get("lr_a", math.nan)),
            "lr_ratio": float(payload.get("lr_ratio", math.nan)),
            "n_modules_logged": int(payload.get("n_modules_logged", 0)),
            "init_s_median": init_s,
            "current_s_median": current_s,
            "delta_s_median": current_s - init_s,
            "init_c_median": init_c,
            "current_c_median": current_c,
            "delta_c_median": current_c - init_c,
            "coherence_path": str(path),
        }
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def add_metrics_to_coherence_row(row: dict[str, Any], jsonl_path: Path) -> None:
    output_dir = jsonl_path.parent.parent
    if output_dir.name == "finetuned_result":
        train_metrics = read_json(output_dir / "final_train_metrics.json") or {}
        row["train_loss"] = train_metrics.get(
            "final_training_loss", train_metrics.get("train_loss", math.nan)
        )
        row["last_logged_train_loss"] = train_metrics.get(
            "last_logged_train_loss", math.nan
        )
        row["eval_accuracy"] = math.nan
        row["eval_loss"] = math.nan
        return

    train_metrics = read_json(output_dir / "train_results.json") or {}
    eval_metrics = read_json(output_dir / "eval_results.json") or {}
    row["train_loss"] = train_metrics.get("train_loss", math.nan)
    row["last_logged_train_loss"] = math.nan
    row["eval_accuracy"] = (
        float(eval_metrics["eval_accuracy"]) * 100.0
        if "eval_accuracy" in eval_metrics
        else math.nan
    )
    row["eval_loss"] = eval_metrics.get("eval_loss", math.nan)


def discover_coherence(
    root: Path, model_label: str, dataset: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for jsonl_path in sorted(root.glob("**/coherence_stats.jsonl")):
        step_rows = parse_coherence_jsonl(jsonl_path)
        for row in step_rows:
            row["model"] = model_label
            row["dataset"] = dataset
            row["init_label"] = INIT_LABELS.get(row["init_para"], row["init_para"])
            add_metrics_to_coherence_row(row, jsonl_path)
            row["is_final"] = int(
                row["step"] == row["max_steps"] and row["max_steps"] > 0
            )
            records.append(row)
    return records


def summarize_final_coherence(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_rows = [record for record in records if int(record["is_final"]) == 1]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in final_rows:
        grouped[
            (
                record["model"],
                record["dataset"],
                record["init_para"],
                record["rank"],
                record["lr_ratio"],
            )
        ].append(record)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][2], item[0][3], item[0][4]),
    ):
        model, dataset, init_para, rank, lr_ratio = key
        summary.append(
            {
                "model": model,
                "dataset": dataset,
                "init_para": init_para,
                "init_label": INIT_LABELS.get(init_para, init_para),
                "rank": rank,
                "lr_ratio": lr_ratio,
                "n": len(group),
                "seeds": " ".join(
                    str(record["seed"])
                    for record in sorted(group, key=lambda item: item["seed"])
                ),
                "final_step": int(group[0]["step"]),
                "final_c_median_mean": safe_mean(
                    record["current_c_median"] for record in group
                ),
                "final_c_median_std": safe_std(
                    record["current_c_median"] for record in group
                ),
                "delta_c_median_mean": safe_mean(
                    record["delta_c_median"] for record in group
                ),
                "final_s_median_mean": safe_mean(
                    record["current_s_median"] for record in group
                ),
                "delta_s_median_mean": safe_mean(
                    record["delta_s_median"] for record in group
                ),
                "train_loss_mean": safe_mean(record["train_loss"] for record in group),
                "eval_accuracy_mean": safe_mean(
                    record["eval_accuracy"] for record in group
                ),
            }
        )
    return summary


def summarize_final_by_rank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_rows = [record for record in records if int(record["is_final"]) == 1]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in final_rows:
        grouped[
            (record["model"], record["dataset"], record["init_para"], record["rank"])
        ].append(record)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][2], item[0][3])
    ):
        model, dataset, init_para, rank = key
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "init_para": init_para,
                "init_label": INIT_LABELS.get(init_para, init_para),
                "rank": rank,
                "n": len(group),
                "ratio_values": " ".join(
                    str(int(record["lr_ratio"]))
                    for record in sorted(
                        group, key=lambda item: (item["lr_ratio"], item["seed"])
                    )
                ),
                "final_c_median_mean": safe_mean(
                    record["current_c_median"] for record in group
                ),
                "final_c_median_std": safe_std(
                    record["current_c_median"] for record in group
                ),
                "final_s_median_mean": safe_mean(
                    record["current_s_median"] for record in group
                ),
                "train_loss_mean": safe_mean(record["train_loss"] for record in group),
                "eval_accuracy_mean": safe_mean(
                    record["eval_accuracy"] for record in group
                ),
            }
        )
    return rows


def make_slopes(
    records: list[dict[str, Any]], aggregate_ratios: bool
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not valid_slope_record(record):
            continue
        if aggregate_ratios:
            key = (
                record["model"],
                record["dataset"],
                record["init_para"],
                record["step"],
            )
        else:
            key = (
                record["model"],
                record["dataset"],
                record["init_para"],
                record["lr_ratio"],
                record["step"],
            )
        grouped[key].append(record)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in group:
            by_rank[int(record["rank"])].append(record)
        ranks = sorted(by_rank)
        if len(ranks) < 3:
            continue
        s_by_rank = [
            safe_mean(record["current_s_median"] for record in by_rank[rank])
            for rank in ranks
        ]
        c_by_rank = [
            safe_mean(record["current_c_median"] for record in by_rank[rank])
            for rank in ranks
        ]
        gamma_s, r2_s = fit_log_slope([float(rank) for rank in ranks], s_by_rank)
        slope_c, r2_c = fit_log_slope([float(rank) for rank in ranks], c_by_rank)
        epoch = safe_mean(record["epoch"] for record in group)
        if aggregate_ratios:
            model, dataset, init_para, step = key
            lr_ratio = "all"
        else:
            model, dataset, init_para, lr_ratio, step = key
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "init_para": init_para,
                "init_label": INIT_LABELS.get(init_para, init_para),
                "lr_ratio": lr_ratio,
                "step": step,
                "epoch": epoch,
                "n_values_per_rank": min(len(by_rank[rank]) for rank in ranks),
                "gamma_S": gamma_s,
                "r2_S": r2_s,
                "slope_C": slope_c,
                "r2_C": r2_c,
                "ranks": " ".join(str(rank) for rank in ranks),
                "S_by_rank": " ".join(fmt(value, 6) for value in s_by_rank),
                "C_by_rank": " ".join(fmt(value, 6) for value in c_by_rank),
            }
        )
    return rows


def parse_long_horizon(
    root: Path, dataset: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for accuracy_path in sorted(
        root.glob(f"epoch_*/{dataset}/*/*/evaluated_result/accuracy.json")
    ):
        epoch_match = re.search(r"epoch_(\d+)", str(accuracy_path))
        init_match = LLM_INIT_DIR_RE.match(accuracy_path.parent.parent.parent.name)
        run_match = LLM_RUN_DIR_RE.match(accuracy_path.parent.parent.name)
        payload = read_json(accuracy_path)
        if (
            epoch_match is None
            or init_match is None
            or run_match is None
            or payload is None
        ):
            continue
        init_para = init_match.group("init_para")
        if init_para not in INIT_LABELS:
            continue
        run_dir = accuracy_path.parent.parent
        train_metrics = (
            read_json(run_dir / "finetuned_result" / "final_train_metrics.json") or {}
        )
        row = {
            "epoch": int(epoch_match.group(1)),
            "model": init_match.group("model"),
            "dataset": dataset,
            "init_para": init_para,
            "init_label": INIT_LABELS[init_para],
            "seed": int(run_match.group("seed")),
            "rank": int(run_match.group("rank")),
            "alpha": float(run_match.group("alpha")),
            "lr_a": run_match.group("lr"),
            "lr_ratio": float(run_match.group("ratio")),
            "avg_accuracy": float(payload["all"]["avg_accuracy"]) * 100.0,
            "train_loss": train_metrics.get(
                "final_training_loss", train_metrics.get("train_loss", math.nan)
            ),
            "last_logged_train_loss": train_metrics.get(
                "last_logged_train_loss", math.nan
            ),
            "global_step": train_metrics.get("global_step", math.nan),
            "path": str(accuracy_path),
        }
        for dataset_name, accuracy in zip(
            payload["all"].get("dataset_list", []),
            payload["all"].get("accuracy_list", []),
        ):
            row[f"{dataset_name}_accuracy"] = float(accuracy) * 100.0
        records.append(row)

    by_epoch: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_epoch[int(record["epoch"])][str(record["init_para"])] = record

    gaps: list[dict[str, Any]] = []
    datasets = sorted(
        {
            field.removesuffix("_accuracy")
            for record in records
            for field in record
            if field.endswith("_accuracy") and field != "avg_accuracy"
        }
    )
    for epoch, init_rows in sorted(by_epoch.items()):
        a = init_rows.get("A")
        aa = init_rows.get("AA")
        if a is None or aa is None:
            continue
        row = {
            "epoch": epoch,
            "initA_avg_accuracy": a["avg_accuracy"],
            "initA2_avg_accuracy": aa["avg_accuracy"],
            "gap_initA2_minus_initA": aa["avg_accuracy"] - a["avg_accuracy"],
            "initA_train_loss": a["train_loss"],
            "initA2_train_loss": aa["train_loss"],
            "train_loss_gap_initA2_minus_initA": aa["train_loss"] - a["train_loss"],
        }
        for dataset_name in datasets:
            a_value = a.get(f"{dataset_name}_accuracy", math.nan)
            aa_value = aa.get(f"{dataset_name}_accuracy", math.nan)
            row[f"{dataset_name}_gap"] = aa_value - a_value
        gaps.append(row)
    return records, gaps


def write_long_horizon_plot(path: Path, gaps: list[dict[str, Any]]) -> None:
    if not gaps:
        return
    epochs = [int(row["epoch"]) for row in gaps]
    init_a = [float(row["initA_avg_accuracy"]) for row in gaps]
    init_aa = [float(row["initA2_avg_accuracy"]) for row in gaps]

    width, height = 720, 440
    left, right, top, bottom = 72, 28, 28, 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min = math.floor((min(init_a + init_aa) - 2) / 5) * 5
    y_max = math.ceil((max(init_a + init_aa) + 2) / 5) * 5
    x_min, x_max = min(epochs), max(epochs)

    def sx(epoch: int) -> float:
        return left + (epoch - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    def points(values: list[float]) -> str:
        return " ".join(
            f"{sx(epoch):.2f},{sy(value):.2f}" for epoch, value in zip(epochs, values)
        )

    tick_lines: list[str] = []
    for y in range(int(y_min), int(y_max) + 1, 5):
        y_pos = sy(float(y))
        tick_lines.append(
            f'<line x1="{left}" y1="{y_pos:.2f}" x2="{width - right}" y2="{y_pos:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        tick_lines.append(
            f'<text x="{left - 10}" y="{y_pos + 4:.2f}" text-anchor="end" font-size="12" fill="#475569">{y}</text>'
        )
    for epoch in epochs:
        x_pos = sx(epoch)
        tick_lines.append(
            f'<line x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{height - bottom}" stroke="#f1f5f9" stroke-width="1"/>'
        )
        tick_lines.append(
            f'<text x="{x_pos:.2f}" y="{height - bottom + 24}" text-anchor="middle" font-size="12" fill="#475569">{epoch}</text>'
        )

    marker_lines: list[str] = []
    for epoch, value in zip(epochs, init_a):
        marker_lines.append(
            f'<circle cx="{sx(epoch):.2f}" cy="{sy(value):.2f}" r="4" fill="#2563eb"/>'
        )
    for epoch, value in zip(epochs, init_aa):
        marker_lines.append(
            f'<circle cx="{sx(epoch):.2f}" cy="{sy(value):.2f}" r="4" fill="#dc2626"/>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.1f}" y="20" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#0f172a">Long-horizon arithmetic fine-tuning</text>
{"".join(tick_lines)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155" stroke-width="1.3"/>
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155" stroke-width="1.3"/>
<polyline points="{points(init_a)}" fill="none" stroke="#2563eb" stroke-width="2.2"/>
<polyline points="{points(init_aa)}" fill="none" stroke="#dc2626" stroke-width="2.2"/>
{"".join(marker_lines)}
<text x="{width / 2:.1f}" y="{height - 16}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#334155">Fine-tuning epochs</text>
<text x="16" y="{height / 2:.1f}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#334155" transform="rotate(-90 16 {height / 2:.1f})">Average accuracy (%)</text>
<rect x="{width - 150}" y="42" width="112" height="48" fill="white" stroke="#cbd5e1"/>
<line x1="{width - 138}" y1="58" x2="{width - 108}" y2="58" stroke="#2563eb" stroke-width="2.2"/>
<text x="{width - 100}" y="62" font-size="12" font-family="Arial, sans-serif" fill="#0f172a">InitA</text>
<line x1="{width - 138}" y1="78" x2="{width - 108}" y2="78" stroke="#dc2626" stroke-width="2.2"/>
<text x="{width - 100}" y="82" font-size="12" font-family="Arial, sans-serif" fill="#0f172a">InitA2</text>
</svg>
'''
    path.with_suffix(".svg").write_text(svg)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(epochs, init_a, marker="o", linewidth=1.8, label="InitA")
    ax.plot(epochs, init_aa, marker="o", linewidth=1.8, label="InitA2")
    ax.axvline(3, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_xlabel("Fine-tuning epochs")
    ax.set_ylabel("Arithmetic average accuracy (%)")
    ax.set_title("Long-horizon arithmetic fine-tuning")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def best_by_init(
    rows: list[dict[str, Any]], metric: str, reverse: bool = True
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for init_label in sorted({row["init_label"] for row in rows}):
        group = [row for row in rows if row["init_label"] == init_label]
        if not group:
            continue
        result.append(
            sorted(group, key=lambda row: float(row[metric]), reverse=reverse)[0]
        )
    return result


def final_aggregated_slopes(slopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model_init: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in slopes:
        by_model_init[(str(row["model"]), str(row["init_para"]))].append(row)
    result: list[dict[str, Any]] = []
    for key, group in sorted(by_model_init.items()):
        max_step = max(int(row["step"]) for row in group)
        result.extend(row for row in group if int(row["step"]) == max_step)
    return result


def slope_interpretation(gamma_s: Any) -> str:
    try:
        gamma = float(gamma_s)
    except (TypeError, ValueError):
        return ""
    if gamma < 0.75:
        return "partial/CLT-like"
    if gamma < 0.85:
        return "intermediate"
    return "near-coherent"


def write_markdown(
    path: Path,
    *,
    heatmap_records: list[dict[str, Any]],
    heatmap_summary: list[dict[str, Any]],
    coherence_records: list[dict[str, Any]],
    coherence_summary: list[dict[str, Any]],
    slopes_aggregated: list[dict[str, Any]],
    long_records: list[dict[str, Any]],
    long_gaps: list[dict[str, Any]],
) -> None:
    heatmap_best = best_by_init(heatmap_summary, "eval_accuracy_mean")
    final_slope_rows = final_aggregated_slopes(slopes_aggregated)
    long_final = long_gaps[-1] if long_gaps else {}
    long_best_gap = (
        max(long_gaps, key=lambda row: float(row["gap_initA2_minus_initA"]))
        if long_gaps
        else {}
    )
    long_min_gap = (
        min(long_gaps, key=lambda row: float(row["gap_initA2_minus_initA"]))
        if long_gaps
        else {}
    )
    roberta_paths = {
        row["coherence_path"]
        for row in coherence_records
        if row["model"] == "RoBERTa-base"
    }
    qwen_paths = {
        row["coherence_path"]
        for row in coherence_records
        if row["model"] == "Qwen3-0.6B"
    }
    roberta_logged = sum(
        1 for row in coherence_records if row["model"] == "RoBERTa-base"
    )
    qwen_logged = sum(1 for row in coherence_records if row["model"] == "Qwen3-0.6B")
    slope_excluded = [row for row in coherence_records if not valid_slope_record(row)]

    lines: list[str] = [
        f"# Completed Additional Results Summary ({DATE_TAG})",
        "",
        "## Completion",
        f"- RoBERTa SST-2 heatmap: {len(heatmap_records)}/250 eval files parsed.",
        f"- RoBERTa SST-2 rank coherence: {len(roberta_paths)}/108 runs parsed, {roberta_logged} deduplicated logged-step records.",
        f"- Qwen3-0.6B math_10K rank coherence: {len(qwen_paths)}/108 runs parsed, {qwen_logged} deduplicated logged-step records.",
        f"- Long-horizon Llama-3.1-8B arithmetic: {len(long_records)}/20 evaluated runs parsed.",
        f"- Rank-scaling fits exclude {len(slope_excluded)} non-finite or near-zero coherence records; these rows remain in the all-steps CSV.",
        "",
        "## RoBERTa Heatmap",
        "Best 5-seed SST-2 cells by initialization:",
        "",
        md_table(
            ["Init", "LR A", "Ratio", "n", "Accuracy Mean", "Accuracy Std"],
            [
                [
                    row["init_label"],
                    row["lr_a"],
                    fmt(row["lr_ratio"], 0),
                    row["n"],
                    fmt(row["eval_accuracy_mean"], 2),
                    fmt(row["eval_accuracy_std"], 2),
                ]
                for row in heatmap_best
            ],
        ),
        "",
        "Interpretation: the completed RoBERTa heatmap confirms that InitA2 remains competitive on SST-2 and prefers a nontrivial B/A LR ratio. The best InitA2 cell is slightly above the best InitA cell in this grid, while both methods are in the same accuracy band.",
        "",
        "## Long-Horizon Arithmetic",
        "Average arithmetic accuracy over independently trained N-epoch runs:",
        "",
        md_table(
            ["Epoch", "InitA", "InitA2", "InitA2 - InitA", "InitA Loss", "InitA2 Loss"],
            [
                [
                    row["epoch"],
                    fmt(row["initA_avg_accuracy"], 2),
                    fmt(row["initA2_avg_accuracy"], 2),
                    fmt(row["gap_initA2_minus_initA"], 2),
                    fmt(row["initA_train_loss"], 3),
                    fmt(row["initA2_train_loss"], 3),
                ]
                for row in long_gaps
            ],
        ),
        "",
        f"Interpretation: InitA does not catch up under the stress setting `ratio=4000`, `init_std=1`, `r=8`, `alpha/r=1`. The final 10-epoch gap is {fmt(long_final.get('gap_initA2_minus_initA'), 2)} accuracy points, and the gap remains positive at every evaluated horizon. The largest gap is {fmt(long_best_gap.get('gap_initA2_minus_initA'), 2)} points; the smallest is {fmt(long_min_gap.get('gap_initA2_minus_initA'), 2)} points.",
        "",
        "## Rank Coherence",
        "Definitions: `gamma_S` is the fitted slope of `log S_t(r)` vs `log r`; LLN-like rank aggregation would be near 1, while CLT-like aggregation is near 0.5. `slope_C` is the fitted log-log slope for normalized coherence `C_t(r)`.",
        "",
        "Final ratio-aggregated slopes:",
        "",
        md_table(
            [
                "Model",
                "Init",
                "Step",
                "Epoch",
                "gamma_S",
                "R2(S)",
                "slope_C",
                "R2(C)",
                "Interpretation",
            ],
            [
                [
                    row["model"],
                    row["init_label"],
                    row["step"],
                    fmt(row["epoch"], 3),
                    fmt(row["gamma_S"], 3),
                    fmt(row["r2_S"], 2),
                    fmt(row["slope_C"], 3),
                    fmt(row["r2_C"], 2),
                    slope_interpretation(row["gamma_S"]),
                ]
                for row in final_slope_rows
            ],
        ),
        "",
        "Interpretation: Qwen3-0.6B is the stronger cross-model confirmation that scaling remains below the LLN ideal: InitA2 is close to CLT at the end of training, while InitA is intermediate after excluding degenerate late high-ratio coherence records. RoBERTa behaves differently by the final SST-2 step: `S` scales closer to coherent/LLN behavior and normalized `C` decays only weakly with rank. CLT scaling is a conservative reference; some classification settings can exhibit more coherent behavior than the idealized CLT case.",
        "",
        "## Output Files",
        "- `roberta_sst2_heatmap_records_release.csv`",
        "- `roberta_sst2_heatmap_summary_release.csv`",
        "- `long_horizon_epoch_accuracy_records_release.csv`",
        "- `long_horizon_epoch_accuracy_gap_release.csv`",
        "- `long_horizon_epoch_accuracy_release.svg`",
        "- `roberta_qwen_rank_coherence_all_steps_release.csv`",
        "- `roberta_qwen_rank_coherence_final_summary_release.csv`",
        "- `roberta_qwen_rank_coherence_by_rank_final_release.csv`",
        "- `roberta_qwen_rank_coherence_scaling_slopes_release.csv`",
        "- `roberta_qwen_rank_coherence_scaling_slopes_aggregated_release.csv`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_records, heatmap_summary = parse_roberta_heatmap(
        args.heatmap_root, args.heatmap_task
    )
    rows_to_csv(
        args.output_dir / f"roberta_sst2_heatmap_records_{DATE_TAG}.csv",
        heatmap_records,
        [
            "init_label",
            "init_std",
            "seed",
            "lr_a",
            "lr_ratio",
            "eval_accuracy",
            "eval_loss",
            "path",
        ],
    )
    rows_to_csv(
        args.output_dir / f"roberta_sst2_heatmap_summary_{DATE_TAG}.csv",
        heatmap_summary,
        [
            "init_label",
            "init_std",
            "lr_a",
            "lr_ratio",
            "n",
            "eval_accuracy_mean",
            "eval_accuracy_std",
            "eval_loss_mean",
            "seeds",
        ],
    )

    roberta_records = discover_coherence(
        args.roberta_coherence_root, "RoBERTa-base", "sst2"
    )
    qwen_records = discover_coherence(
        args.qwen_coherence_root, "Qwen3-0.6B", "math_10k"
    )
    coherence_records = roberta_records + qwen_records
    coherence_summary = summarize_final_coherence(coherence_records)
    coherence_by_rank = summarize_final_by_rank(coherence_records)
    slopes = make_slopes(coherence_records, aggregate_ratios=False)
    slopes_aggregated = make_slopes(coherence_records, aggregate_ratios=True)

    coherence_fields = [
        "model",
        "dataset",
        "init_para",
        "init_label",
        "seed",
        "rank",
        "lr_a",
        "lr_ratio",
        "step",
        "epoch",
        "max_steps",
        "is_final",
        "current_s_median",
        "delta_s_median",
        "current_c_median",
        "delta_c_median",
        "train_loss",
        "eval_accuracy",
        "eval_loss",
        "n_modules_logged",
        "coherence_path",
    ]
    rows_to_csv(
        args.output_dir / f"roberta_qwen_rank_coherence_all_steps_{DATE_TAG}.csv",
        coherence_records,
        coherence_fields,
    )
    rows_to_csv(
        args.output_dir / f"roberta_qwen_rank_coherence_final_summary_{DATE_TAG}.csv",
        coherence_summary,
        [
            "model",
            "dataset",
            "init_para",
            "init_label",
            "rank",
            "lr_ratio",
            "n",
            "seeds",
            "final_step",
            "final_c_median_mean",
            "final_c_median_std",
            "delta_c_median_mean",
            "final_s_median_mean",
            "delta_s_median_mean",
            "train_loss_mean",
            "eval_accuracy_mean",
        ],
    )
    rows_to_csv(
        args.output_dir / f"roberta_qwen_rank_coherence_by_rank_final_{DATE_TAG}.csv",
        coherence_by_rank,
        [
            "model",
            "dataset",
            "init_para",
            "init_label",
            "rank",
            "n",
            "ratio_values",
            "final_c_median_mean",
            "final_c_median_std",
            "final_s_median_mean",
            "train_loss_mean",
            "eval_accuracy_mean",
        ],
    )
    slope_fields = [
        "model",
        "dataset",
        "init_para",
        "init_label",
        "lr_ratio",
        "step",
        "epoch",
        "n_values_per_rank",
        "gamma_S",
        "r2_S",
        "slope_C",
        "r2_C",
        "ranks",
        "S_by_rank",
        "C_by_rank",
    ]
    rows_to_csv(
        args.output_dir / f"roberta_qwen_rank_coherence_scaling_slopes_{DATE_TAG}.csv",
        slopes,
        slope_fields,
    )
    rows_to_csv(
        args.output_dir
        / f"roberta_qwen_rank_coherence_scaling_slopes_aggregated_{DATE_TAG}.csv",
        slopes_aggregated,
        slope_fields,
    )

    long_records, long_gaps = parse_long_horizon(
        args.long_horizon_root, args.long_horizon_dataset
    )
    long_fields = [
        "epoch",
        "model",
        "dataset",
        "init_para",
        "init_label",
        "seed",
        "rank",
        "alpha",
        "lr_a",
        "lr_ratio",
        "avg_accuracy",
        "AddSub_accuracy",
        "MultiArith_accuracy",
        "SingleEq_accuracy",
        "SVAMP_accuracy",
        "gsm8k_accuracy",
        "AQuA_accuracy",
        "train_loss",
        "last_logged_train_loss",
        "global_step",
        "path",
    ]
    rows_to_csv(
        args.output_dir / f"long_horizon_epoch_accuracy_records_{DATE_TAG}.csv",
        long_records,
        long_fields,
    )
    gap_fields = [
        "epoch",
        "initA_avg_accuracy",
        "initA2_avg_accuracy",
        "gap_initA2_minus_initA",
        "initA_train_loss",
        "initA2_train_loss",
        "train_loss_gap_initA2_minus_initA",
        "AQuA_gap",
        "AddSub_gap",
        "MultiArith_gap",
        "SVAMP_gap",
        "SingleEq_gap",
        "gsm8k_gap",
    ]
    rows_to_csv(
        args.output_dir / f"long_horizon_epoch_accuracy_gap_{DATE_TAG}.csv",
        long_gaps,
        gap_fields,
    )
    write_long_horizon_plot(
        args.output_dir / f"long_horizon_epoch_accuracy_{DATE_TAG}", long_gaps
    )

    write_markdown(
        args.output_dir / f"completed_results_summary_{DATE_TAG}.md",
        heatmap_records=heatmap_records,
        heatmap_summary=heatmap_summary,
        coherence_records=coherence_records,
        coherence_summary=coherence_summary,
        slopes_aggregated=slopes_aggregated,
        long_records=long_records,
        long_gaps=long_gaps,
    )
    print(f"Wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
