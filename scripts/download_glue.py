#!/usr/bin/env python3
"""Save GLUE datasets for offline training."""

import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task_names", default="mnli,qqp,sst2,qnli", help="Comma-separated task names"
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    for task in args.task_names.replace(",", " ").split():
        if task not in {
            "cola",
            "mnli",
            "mrpc",
            "qnli",
            "qqp",
            "rte",
            "sst2",
            "stsb",
            "wnli",
        }:
            parser.error(f"Unknown GLUE task: {task}")
        path = args.data_dir / "glue" / task
        path.parent.mkdir(parents=True, exist_ok=True)
        load_dataset("nyu-mll/glue", task).save_to_disk(path)


if __name__ == "__main__":
    main()
