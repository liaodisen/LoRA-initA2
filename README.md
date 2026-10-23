# InitA²: LoRA initialization experiments

Code release for the NeurIPS paper **Rethinking LoRA Initialization for Robust
Asymmetric Learning Rates**.

LoRA+ uses a larger learning rate for B than for A. Under standard Init[A],
the random initial A can introduce noise that grows with the B learning rate,
initialization scale, and LoRA update multiplier. **Init[A²] suppresses this
initialization noise** by initializing A with variance `beta² / fan_in²` and B
with zeros. This preserves zero initial adapter output and widens the stable
region for asymmetric learning rates.

## Figure 1: initialization and feature stability

![Figure 1 from the paper: LoRA initialization schemes and normalized feature-norm trajectories for Init[A], Init[AB], and Init[A²].](assets/figure1.png)

**Figure 1 from the NeurIPS manuscript.** The left panel shows the LoRA branch
and initialization scalings. The right panels compare feature norms at width
`n=4096` for `beta=1,5,25`. Init[A²] keeps the trajectories much less sensitive
to the initialization scale. [Vector PDF](assets/figure1.pdf).

## Feature decomposition: suppressing initialization noise

For the zero-B initializations, the scaled LoRA output separates into a term
from the initial random projection and a term from learning A:

$$
Z_B^t = \underbrace{\frac{\alpha}{r} B_t A_0 Z}_{T_{\mathrm{noise}}^t}
      + \underbrace{\frac{\alpha}{r} B_t (A_t-A_0) Z}_{T_{\mathrm{signal}}^t}.
$$

![Archived feature decomposition: Init[A²] has much smaller initialization-noise curves than Init[A], while the signal remains visible.](assets/feature-decomposition.png)

**Init[A²] suppresses the noise component while retaining the learned signal.**
Peach shows the noise RMS, purple the signal RMS, and magenta the total LoRA
output RMS. Under Init[A], noise dominates these runs; with Init[A²], noise
stays small and the output follows the signal more closely. All four panels
share the same vertical scale.

These archived Math-10K runs use rank 8, `alpha/r=12`,
`beta=0.5`, `lr_A=1e-5`, and `lr_B=2e-3`, for three epochs. Lines show means
over seeds 42/43/44; shaded regions show ±1 sample standard deviation. Each
seed averages eight modules on a fixed probe. [Vector PDF](assets/feature-decomposition.pdf)
and [full settings and figure provenance](assets/README.md).

To run the same diagnostic configuration and plot its saved measurements:

```bash
python scripts/reproduce.py --experiment feature_decomposition --set beta=0.5 --set multiplier=12 --skip-eval --run
python additional_experiments/plot_llama_feature_decomposition_rms.py --input-root outputs/feature_decomposition --output figures/feature-decomposition
```

## Setup

Use Python 3.11 or 3.12. For Linux with a CUDA GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints.txt -e '.[dev]'
```

For CPU validation, replace the PyTorch index with
`https://download.pytorch.org/whl/cpu`. Install `'.[qlora]'` for the NF4 runs
and `'.[tracking]'` only if you want Weights & Biases. Experiment tracking is
disabled by default. `requirements.txt` installs the same core project dependencies.
`constraints.txt` records the validated dependency versions without artifact
checksums. Use `-c constraints.txt` when installing the project or its extras.

Download the benchmark data, then list the experiment grids:

```bash
python scripts/prepare_data.py
python scripts/reproduce.py --list
```

Llama checkpoints require upstream model access. Use `hf auth login` or pass a
local model directory using `--model`.

## Reproduce experiments

[Experiment setup](docs/EXPERIMENTS.md) describes every figure and table,
including models, datasets, initialization, learning-rate grids, training,
evaluation, seeds, commands, and known differences from historical launchers.
The machine-readable settings are in [experiments/paper.json](experiments/paper.json).

```bash
# Preview one Figure 5 cell, then execute its full grid.
python scripts/reproduce.py --experiment figure5 --dry-run --limit 1
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --experiment figure5 --run

# Aggregate matched cells across seeds and export the accuracy plot.
python scripts/summarize_results.py --input-root outputs/figure5
python scripts/plot_results.py --input outputs/figure5/summary/summary.csv --x multiplier --output-dir figures/figure5
```

The runner prints commands unless `--run` is supplied. `--method`, `--task`,
`--seeds`, `--start`, and `--limit` select subsets. `--data-root`, `--model`,
and `--output-root` configure local paths. Each run saves its exact settings,
software versions, command logs, adapters, and metrics.

## Validate the installation

These checks use tiny local models and synthetic fixtures and require no downloads:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
python scripts/smoke_test.py --device cpu --all-initializers
python scripts/smoke_test.py --device cuda --qlora
```

They exercise initialization, optimizer grouping, residual-weight checkpoint
replay, LLM training and generation evaluation, GLUE, the reconstructed MNIST
model, and all four diagnostic callbacks. See [validation status](docs/VALIDATION.md)
for the checks performed on this release. Full paper scores have not been rerun.
The original MNIST source was absent, and several historical settings conflict
with the final paper; [the experiment guide](docs/EXPERIMENTS.md#limits-on-exact-numerical-reproduction)
records these limits explicitly.

The CPU CI template is [ci/reproduction.yml](ci/reproduction.yml). To enable
GitHub Actions, copy it to `.github/workflows/ci.yml` and commit it using a
credential with workflow permission. The template is stored outside the active
workflow directory because the release push credential lacks that permission.

## Layout

- `assets/`: paper Figure 1 and the feature decomposition figure, with PNG/PDF exports and provenance.
- `src/`: initializers, LoRA+ trainer, argument definitions, diagnostic callbacks.
- `scripts/`: training, evaluation, data preparation, reproduction, summaries, plots, smoke checks.
- `experiments/paper.json`: Figure 1–10 and Table 1–4 grids plus diagnostic recipes.
- `additional_experiments/`: additional seed studies and diagnostics with portable Slurm/local launchers.
- `tests/`: numerical invariants and recipe checks.

Superseded `code/` wrappers and bundled third-party ZIP files were removed from
the release tree. Data, weights, caches, and training outputs are ignored by Git.
See [NOTICE.md](NOTICE.md) for upstream code attribution.

## Acknowledgments

This repository is built on **[LoRA[AB]](https://github.com/Leopold1423/non_zero_lora-icml25)**
(*Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization
on LoRA Fine-Tuning Dynamics*, ICML 2025) and
**[LoRA+](https://github.com/nikhilgsh/loraplus)**
(*LoRA+: Efficient Low Rank Adaptation of Large Models*, ICML 2024).
We thank their authors for releasing their code. LoRA[AB] provides the foundation
for the initialization comparisons and decoder experiment pipeline; LoRA+
provides the asymmetric-learning-rate optimizer, trainer, and GLUE utilities.
Additional upstream attribution and license notices are in [NOTICE.md](NOTICE.md).

## Publish a source snapshot

Export the committed source tree as a standalone snapshot:

```bash
git archive --format=tar.gz --output=../lora-inita2-public-release.tar.gz main
mkdir -p ../inita2-release
tar -xzf ../lora-inita2-public-release.tar.gz -C ../inita2-release
cd ../inita2-release
git init -b main
git add .
git commit -m "Release InitA² reproduction code"
```

The archive contains the committed release files. It omits Git history, data,
weights, local environments, and job outputs.
