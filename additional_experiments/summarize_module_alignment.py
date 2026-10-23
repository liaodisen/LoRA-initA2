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


INIT_LABELS = {"A": "Init[A]", "AA": "Init[A^2]"}
LAYER_PATTERN = re.compile(r"(?:layers|layer)\.(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank LoRA modules by same-minibatch update-input alignment."
    )
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def stdev_or_zero(values: list[float]) -> float | None:
    if not values:
        return None
    return stdev(values) if len(values) > 1 else 0.0


def module_metadata(module_name: str) -> tuple[int | None, str]:
    match = LAYER_PATTERN.search(module_name)
    layer = int(match.group(1)) if match else None
    return layer, module_name.rsplit(".", 1)[-1]


def discover_module_records(input_root: Path) -> tuple[list[dict[str, object]], int]:
    records: dict[tuple[str, int, str], dict[str, object]] = {}
    paths = sorted(input_root.rglob("width_alignment.jsonl"))
    for path in paths:
        for line in path.read_text().splitlines():
            payload = json.loads(line)
            for module in payload.get("modules", []):
                module_name = str(module["module"])
                layer, target = module_metadata(module_name)
                processed_cosine = float(module.get("processed_cosine_abs_median", 0.0))
                shuffled_cosine = float(module.get("shuffled_cosine_abs_median", 0.0))
                processed_direction_rms = float(
                    module.get("processed_direction_entry_rms", 0.0)
                )
                record: dict[str, object] = {
                    "model": payload["model_name"],
                    "width": int(payload["model_width"]),
                    "init_para": payload["init_para"],
                    "init_label": INIT_LABELS.get(
                        payload["init_para"], payload["init_para"]
                    ),
                    "seed": int(payload["seed"]),
                    "rank": int(payload["rank"]),
                    "step": int(payload["step"]),
                    "epoch": float(payload["epoch"]),
                    "max_steps": int(payload["max_steps"]),
                    "module": module_name,
                    "layer": layer,
                    "target": target,
                    "input_width": int(module["input_width"]),
                    "n_tokens": int(module["n_tokens"]),
                    "n_lora_rows": int(module["n_lora_rows"]),
                    "processed_dot": float(module.get("processed_dot_abs_median", 0.0)),
                    "processed_cosine": processed_cosine,
                    "shuffled_dot": float(module.get("shuffled_dot_abs_median", 0.0)),
                    "shuffled_cosine": shuffled_cosine,
                    "cosine_excess": processed_cosine - shuffled_cosine,
                    "cosine_enrichment": safe_ratio(processed_cosine, shuffled_cosine),
                    "processed_direction_rms": processed_direction_rms,
                    "valid_update": processed_direction_rms > 0,
                    "path": str(path),
                }
                records[(str(path), int(payload["step"]), module_name)] = record
    return list(records.values()), len(paths)


def summarize_checkpoints(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if not record["valid_update"]:
            continue
        key = (
            record["model"],
            record["width"],
            record["init_para"],
            record["init_label"],
            record["step"],
            record["module"],
            record["layer"],
            record["target"],
            record["input_width"],
        )
        grouped[key].append(record)

    summaries: list[dict[str, object]] = []
    for key, group in grouped.items():
        (
            model,
            width,
            init_para,
            init_label,
            step,
            module,
            layer,
            target,
            input_width,
        ) = key
        processed_cosines = [float(record["processed_cosine"]) for record in group]
        shuffled_cosines = [float(record["shuffled_cosine"]) for record in group]
        processed_dots = [float(record["processed_dot"]) for record in group]
        shuffled_dots = [float(record["shuffled_dot"]) for record in group]
        processed_cosine_mean = mean(processed_cosines)
        shuffled_cosine_mean = mean(shuffled_cosines)
        summaries.append(
            {
                "model": model,
                "width": width,
                "init_para": init_para,
                "init_label": init_label,
                "step": step,
                "epoch_mean": mean(float(record["epoch"]) for record in group),
                "module": module,
                "layer": layer,
                "target": target,
                "input_width": input_width,
                "n_seeds": len({int(record["seed"]) for record in group}),
                "seeds": " ".join(
                    str(seed)
                    for seed in sorted({int(record["seed"]) for record in group})
                ),
                "processed_cosine_mean": processed_cosine_mean,
                "processed_cosine_std": stdev_or_zero(processed_cosines),
                "shuffled_cosine_mean": shuffled_cosine_mean,
                "shuffled_cosine_std": stdev_or_zero(shuffled_cosines),
                "cosine_excess": processed_cosine_mean - shuffled_cosine_mean,
                "cosine_enrichment": safe_ratio(
                    processed_cosine_mean, shuffled_cosine_mean
                ),
                "processed_dot_mean": mean(processed_dots),
                "processed_dot_std": stdev_or_zero(processed_dots),
                "shuffled_dot_mean": mean(shuffled_dots),
                "shuffled_dot_std": stdev_or_zero(shuffled_dots),
            }
        )

    rank_groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        rank_groups[
            (str(row["model"]), str(row["init_para"]), int(row["step"]))
        ].append(row)
    for group in rank_groups.values():
        group.sort(
            key=lambda row: (
                -float(row["cosine_enrichment"]),
                -float(row["processed_cosine_mean"]),
                str(row["module"]),
            )
        )
        for rank, row in enumerate(group, start=1):
            row["checkpoint_rank"] = rank
    return sorted(
        summaries,
        key=lambda row: (
            str(row["model"]),
            str(row["init_para"]),
            int(row["step"]),
            int(row["checkpoint_rank"]),
        ),
    )


def summarize_overall(
    records: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if not record["valid_update"]:
            continue
        key = (
            record["model"],
            record["width"],
            record["init_para"],
            record["init_label"],
            record["module"],
            record["layer"],
            record["target"],
            record["input_width"],
        )
        grouped[key].append(record)

    top_counts: dict[tuple[str, str, str], tuple[int, int]] = defaultdict(
        lambda: (0, 0)
    )
    for row in checkpoints:
        key = (str(row["model"]), str(row["init_para"]), str(row["module"]))
        top_one, top_five = top_counts[key]
        checkpoint_rank = int(row["checkpoint_rank"])
        top_counts[key] = (
            top_one + int(checkpoint_rank == 1),
            top_five + int(checkpoint_rank <= 5),
        )

    summaries: list[dict[str, object]] = []
    for key, group in grouped.items():
        (
            model,
            width,
            init_para,
            init_label,
            module,
            layer,
            target,
            input_width,
        ) = key
        processed_cosines = [float(record["processed_cosine"]) for record in group]
        shuffled_cosines = [float(record["shuffled_cosine"]) for record in group]
        processed_cosine_mean = mean(processed_cosines)
        shuffled_cosine_mean = mean(shuffled_cosines)
        top_one, top_five = top_counts[(str(model), str(init_para), str(module))]
        summaries.append(
            {
                "model": model,
                "width": width,
                "init_para": init_para,
                "init_label": init_label,
                "module": module,
                "layer": layer,
                "target": target,
                "input_width": input_width,
                "n_observations": len(group),
                "n_steps": len({int(record["step"]) for record in group}),
                "n_seeds": len({int(record["seed"]) for record in group}),
                "processed_cosine_mean": processed_cosine_mean,
                "processed_cosine_std": stdev_or_zero(processed_cosines),
                "shuffled_cosine_mean": shuffled_cosine_mean,
                "shuffled_cosine_std": stdev_or_zero(shuffled_cosines),
                "cosine_excess": processed_cosine_mean - shuffled_cosine_mean,
                "cosine_enrichment": safe_ratio(
                    processed_cosine_mean, shuffled_cosine_mean
                ),
                "processed_dot_mean": mean(
                    float(record["processed_dot"]) for record in group
                ),
                "shuffled_dot_mean": mean(
                    float(record["shuffled_dot"]) for record in group
                ),
                "checkpoint_top1_count": top_one,
                "checkpoint_top5_count": top_five,
            }
        )

    rank_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        rank_groups[(str(row["model"]), str(row["init_para"]))].append(row)
    for group in rank_groups.values():
        group.sort(
            key=lambda row: (
                -float(row["cosine_enrichment"]),
                -float(row["processed_cosine_mean"]),
                str(row["module"]),
            )
        )
        for rank, row in enumerate(group, start=1):
            row["overall_rank"] = rank
    return sorted(
        summaries,
        key=lambda row: (
            str(row["model"]),
            str(row["init_para"]),
            int(row["overall_rank"]),
        ),
    )


def summarize_targets(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for record in records:
        if not record["valid_update"]:
            continue
        key = (
            str(record["model"]),
            int(record["width"]),
            str(record["init_para"]),
            str(record["init_label"]),
            str(record["target"]),
        )
        grouped[key].append(record)

    summaries = []
    for key, group in grouped.items():
        model, width, init_para, init_label, target = key
        processed_cosine = mean(float(row["processed_cosine"]) for row in group)
        shuffled_cosine = mean(float(row["shuffled_cosine"]) for row in group)
        summaries.append(
            {
                "model": model,
                "width": width,
                "init_para": init_para,
                "init_label": init_label,
                "target": target,
                "n_modules": len({str(row["module"]) for row in group}),
                "n_observations": len(group),
                "processed_cosine_mean": processed_cosine,
                "shuffled_cosine_mean": shuffled_cosine,
                "cosine_excess": processed_cosine - shuffled_cosine,
                "cosine_enrichment": safe_ratio(processed_cosine, shuffled_cosine),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            str(row["model"]),
            str(row["init_para"]),
            -float(row["processed_cosine_mean"]),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def write_markdown(
    path: Path,
    *,
    records: list[dict[str, object]],
    path_count: int,
    overall: list[dict[str, object]],
    top_k: int,
) -> None:
    models = sorted({str(record["model"]) for record in records})
    valid_steps = sorted(
        {int(record["step"]) for record in records if bool(record["valid_update"])}
    )
    lines = [
        "# All-Module Alignment Ranking",
        "",
        f"- Run files: {path_count}",
        f"- Models: {len(models)} ({', '.join(models)})",
        f"- Valid nonzero-update checkpoints: {', '.join(map(str, valid_steps))}",
        "- Ranking metric: actual-to-shuffled cosine enrichment, averaged over "
        "valid checkpoints and seeds. This controls for different input widths "
        "across attention and MLP projections.",
        "- Actual cosine is the mean absolute cosine between each processed "
        "LoRA-A update row and its same-minibatch input; shuffled-coordinate "
        "cosine is the random-alignment baseline.",
        "",
    ]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in overall:
        grouped[(str(row["model"]), str(row["init_label"]))].append(row)
    for (model, init_label), group in sorted(grouped.items()):
        lines.extend(
            [
                f"## {model} — {init_label}",
                "",
                "| Rank | Module | Layer | Target | Input width | Actual cosine | Shuffled cosine | Enrichment | Checkpoint top-1 |",
                "|---:|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in group[:top_k]:
            lines.append(
                "| {rank} | `{module}` | {layer} | `{target}` | {input_width} | {actual} | "
                "{shuffled} | {enrichment}x | {top_one} |".format(
                    rank=row["overall_rank"],
                    module=row["module"],
                    layer=row["layer"] if row["layer"] is not None else "NA",
                    target=row["target"],
                    input_width=row["input_width"],
                    actual=format_value(row["processed_cosine_mean"]),
                    shuffled=format_value(row["shuffled_cosine_mean"]),
                    enrichment=format_value(row["cosine_enrichment"], 2),
                    top_one=row["checkpoint_top1_count"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    records, path_count = discover_module_records(args.input_root)
    if not records:
        raise SystemExit(f"No module records found under {args.input_root}")
    checkpoints = summarize_checkpoints(records)
    overall = summarize_overall(records, checkpoints)
    targets = summarize_targets(records)
    output_dir = args.output_dir or (args.input_root / "module_summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "module_alignment_records.csv", records)
    write_csv(output_dir / "module_alignment_by_checkpoint.csv", checkpoints)
    write_csv(output_dir / "module_alignment_overall.csv", overall)
    write_csv(output_dir / "module_type_summary.csv", targets)
    write_markdown(
        output_dir / "module_alignment_top.md",
        records=records,
        path_count=path_count,
        overall=overall,
        top_k=max(1, args.top_k),
    )
    print(f"Wrote module-alignment summaries to {output_dir}")


if __name__ == "__main__":
    main()
