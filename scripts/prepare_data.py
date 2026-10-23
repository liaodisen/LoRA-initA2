#!/usr/bin/env python3
"""Download and validate benchmark JSON files from LLM-Adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

BASE = "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
BENCHMARKS = {
    "math_10k": ["AddSub", "MultiArith", "SingleEq", "SVAMP", "gsm8k", "AQuA"],
    "commonsense_170k": [
        "boolq",
        "piqa",
        "social_i_qa",
        "hellaswag",
        "winogrande",
        "ARC-Challenge",
        "ARC-Easy",
        "openbookqa",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Upstream branch or tag")
    parser.add_argument(
        "--datasets", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "LLM-Adapters-main",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check saved file sizes and JSON format",
    )
    args = parser.parse_args()
    manifest_path = args.output_dir / "download_manifest.json"
    if args.verify_only:
        manifest = json.loads(manifest_path.read_text())
        for name, info in manifest["files"].items():
            content = (args.output_dir / name).read_bytes()
            if not content or len(content) != info.get("size_bytes", len(content)):
                raise ValueError(f"File size mismatch: {name}")
            if name.endswith(".json"):
                records = json.loads(content)
                if not isinstance(records, list) or not records:
                    raise ValueError(f"Expected nonempty JSON examples in {name}")
                if len(records) != info.get("examples", len(records)):
                    raise ValueError(f"Example count mismatch: {name}")
        print("Downloaded file sizes, example counts, and JSON format verified")
        return
    manifest = {"revision": args.revision, "files": {}}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("revision") == args.revision:
            manifest["files"] = {
                name: info
                for name, info in previous["files"].items()
                if "size_bytes" in info
            }
    files = ["LICENSE", "DATA_LICENSE"]
    for dataset in args.datasets:
        files.append(f"ft-training_set/{dataset}.json")
        files.extend(
            f"dataset/{benchmark}/test.json" for benchmark in BENCHMARKS[dataset]
        )
    for name in files:
        path = args.output_dir / name
        url = f"{BASE}/{args.revision}/{name}"
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        if name.endswith(".json"):
            records = json.loads(content)
            if not isinstance(records, list) or not records:
                raise ValueError(f"Expected nonempty JSON examples in {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(content)
        temporary.replace(path)
        manifest["files"][name] = {
            "url": url,
            "size_bytes": len(content),
        }
        if name.endswith(".json"):
            manifest["files"][name]["examples"] = len(records)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("Downloaded", name)


if __name__ == "__main__":
    main()
