# Release validation

Initial checks: 2026-10-06. Source cleanup rechecked on 2026-10-22. This records executable checks of the release code; the full paper grids and numerical scores have **not** been reproduced.

## Environment

The validated dependency versions are recorded in `constraints.txt`. To install
the same versions and check compatibility:

```bash
python -m pip install -c constraints.txt -e '.[dev,qlora]'
uv pip check --python .venv/bin/python
```

Python 3.12.1; PyTorch 2.11.0+cu128 (CUDA 12.8); Transformers 5.3.0;
PEFT 0.18.1; Accelerate 1.13.0; Datasets 4.8.4; NumPy 2.4.3.
The dependency check passed for 96 installed packages. CPU checks explicitly
hide CUDA or use the smoke runner's CPU mode.

## Checks performed

| Check | Result / scope |
| --- | --- |
| `python -m pytest -q` | 16 passed. Tests cover initialization variance, LoRA+ optimizer coverage/rates, exact residual-weight adapter reloads, preserved PiSSA products, every recipe's command construction, matched method budgets, toy gradients, benchmark completeness, seed aggregation, and download validation. |
| `ruff check .` and `ruff format --check .` | Python lint and formatting checks pass. |
| Python entry points | All 30 training, evaluation, summary, and plotting CLIs accept `--help`. |
| Shell syntax | All 23 shell files pass `bash -n`. |
| Supplemental launchers | All 14 `submit_*.sh` launchers complete with `DRY_RUN=1`; no cluster jobs submitted. |
| Benchmark download | The two training files and fourteen test files matched the original local LLM-Adapters data byte for byte during initial validation. The downloader now records file sizes and example counts and validates JSON format. |
| Reproduction runner | A Figure 5 cell was executed with a tiny local Llama and fixture data, including training and all six arithmetic evaluation paths. The completed manifest, metrics, and completion marker were written. |
| Result aggregation | The completed runner output was collected and aggregated. Incomplete benchmark sets and duplicate cell/seed results are rejected by tests. |
| Plot export | Accuracy curves, an independent A/B LR heatmap, and toy feature trajectories export PDF, SVG, and 300-DPI PNG. Toy rendering was inspected. These are check fixtures, not paper results. |
| README figure update | Figure 1 was extracted from the verified source PDF, and the existing feature decomposition PNG/PDF were copied unchanged. Both images were inspected and their local Markdown links verified. The feature plotter exports PNG/PDF from twelve schema fixtures with hashed run directories and saved model metadata. |
| Real MNIST loading | A two-step CPU run using 256 real MNIST examples loaded the dataset, trained, and wrote metrics/history/features. Training loss fell from 2.3026 to 2.2688. This checks the reconstruction, not Figure 1/2 values. |

The offline smoke check is:

```bash
python scripts/smoke_test.py --device cpu --all-initializers
```

**Passed in the locked environment:** all 24 smoke subprocesses completed.
It creates tiny local Llama and RoBERTa models and exercises LLM training,
generation evaluation, GLUE training/evaluation, the synthetic toy path,
and all four diagnostic callbacks. It trains and reloads A, A², AB with
subtraction, Orthogonal, Orthogonal-A², EVA, EVA-A², PiSSA, PiSSA-A², and OLoRA.
The callback checks require populated module records, unique checkpoints,
and feature reconstruction relative error below `1e-5`.

## GPU verification and numerical limits

The available NVIDIA L40S (46,068 MiB reported by `nvidia-smi`) reports four
volatile uncorrectable ECC errors. A basic CUDA tensor operation in the locked
environment fails with `CUDA error: uncorrectable ECC error encountered`.
GPU training and NF4 QLoRA therefore remain unverified on this machine.
The GPU was not reset and no full grids were launched.

On a healthy CUDA machine, run:

```bash
python scripts/smoke_test.py --device cuda --qlora
```

Then execute the relevant recipes using `scripts/reproduce.py --run` and
compare their completed summaries to the paper. Large-checkpoint access,
memory requirements, and execution time depend on the selected grid.
The GitHub Actions CPU workflow is included as an inactive template at
`ci/reproduction.yml` and has not been run remotely. The release push credential
lacks workflow permission; enabling it requires copying the template to
`.github/workflows/ci.yml` and committing with the appropriate permission.

The original toy source, several original seeds, and some model/grid metadata
were absent or conflict with the final PDF. The public code records explicit
reference choices and supports historical overrides. See
[limits on exact numerical reproduction](EXPERIMENTS.md#limits-on-exact-numerical-reproduction)
before interpreting new runs as exact reproductions of the reported numbers.
