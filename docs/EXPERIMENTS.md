# Experiment setup and reproduction

The release targets **Rethinking LoRA Initialization for Robust Asymmetric Learning Rates**, the author's `main-neurips.pdf`. [experiments/paper.json](../experiments/paper.json) contains every grid value, default, and provenance note used by the runner. The PDF, training data, model weights, and historical job outputs are downloaded or supplied separately.

## Installation and data

Use Python 3.11 or 3.12, install as described in [README.md](../README.md), and activate that environment. Each training command runs on one visible GPU. Set `CUDA_VISIBLE_DEVICES` before starting it. The paper reports NVIDIA L40S GPUs with 48 GB memory, roughly one hour per RoBERTa run and three hours per decoder run; larger ranks and longer horizons need more time. These estimates exclude grid expansion and generation evaluation.

Download the LLM benchmark files with:

```bash
python scripts/prepare_data.py
python scripts/prepare_data.py --verify-only
```

The downloader fetches only the two training files, fourteen test files, and upstream license files from LLM-Adapters. It defaults to the upstream `main` branch; use `--revision` to select a named branch or tag. It records download URLs, file sizes, and example counts in `LLM-Adapters-main/download_manifest.json`. Verification checks file sizes, example counts, and JSON format. An alternative location can be supplied with `--output-dir`; give the same path to the runner's `--data-root` and evaluator's `--data_root`. The upstream branch can change, so retain the downloaded dataset when comparing exact scores.

GLUE is loaded from Hugging Face. To prepare it for offline use:

```bash
python scripts/download_glue.py --task_names mnli,qqp,sst2,qnli --data_dir data
python scripts/reproduce.py --experiment figure6 --local-glue data --limit 1 --run
```

Base Llama checkpoints require access under their upstream model terms. Authenticate using `hf auth login`, or provide a local model directory with `--model`. Recipes use base checkpoints, except the supplemental Qwen width-alignment experiment, which uses the stated Instruct checkpoints. They do not substitute an Instruct checkpoint for a base model.

## Shared initialization

For a module with `A.shape = (r, fan_in)` and `B.shape = (fan_out, r)`:

```text
std(A) = beta / fan_in**s_A
std(B) = beta / fan_out**s_B
LoRA multiplier = alpha / r
lr_B = ratio * lr_A
```

Transformer experiments use uniform draws with bounds `±sqrt(3) * std`, giving the stated variance. The toy reconstruction uses Gaussian draws.

| Paper name | CLI selector | A standard deviation | Initial B |
| --- | --- | --- | --- |
| Init[A] | `A`, `s_A=0.5` | `beta / sqrt(fan_in)` | zero |
| Init[A²] | `AA`, `s_A=1` | `beta / fan_in` | zero |
| Init[A³] | `A3`, `s_A=1.5` | `beta / fan_in**1.5` | zero |
| Init[A⁴] | `A4`, `s_A=2` | `beta / fan_in**2` | zero |
| Init[AB] with preserved base function | `AB_reset`, both exponents `0.5` | `beta / sqrt(fan_in)` | nonzero, then subtract `alpha/r * B0 @ A0` from the base weight |

`AA` is a historical name for Init[A²]; it initializes one A matrix and zeroes B. Direct Python calls must also set `--lora_A_scale 1`; the selector alone does not change that exponent. The public runner sets both explicitly. `AB` without `reset` changes the initial model function and is not the paper's function-preserving comparison.

The LoRA+ optimizer assigns A the base LR and B the multiplied LR. One-dimensional trainable parameters follow the LoRA+ B-rate convention; classification head matrices follow the A-rate group. In the frozen-A experiment the backbone adapters' A parameters have no gradients; the supplied LR is used by B and the trainable classification head.

## Shared transformer training and evaluation

| Setting | RoBERTa / GLUE | Decoder models |
| --- | --- | --- |
| Optimizer | AdamW, betas `(0.9, 0.99)`, epsilon `1e-8` | AdamW, betas `(0.9, 0.999)`, epsilon `1e-8` |
| Weight decay | `0` | `0` |
| Schedule | linear, no warmup | linear, 100 warmup updates from the archived trainer |
| Batch size | 32 per training step | global 16, micro batch 8, accumulation 2 |
| Sequence length | 128 | 256 for both initializations |
| Epochs | 3 | arithmetic 3, commonsense 1, unless a recipe states otherwise |
| LoRA rank / alpha | `8 / 8` | `8 / 8`, except rank and multiplier sweeps |
| Targets | `query`, `value` | `q_proj`, `k_proj`, `v_proj`, `up_proj`, `down_proj` |
| Dropout | 0.1 | 0.05; 0 for residual-weight initializers and Table 4 |
| Precision | FP16 | BF16 for Llama, FP16 for Qwen/Mistral family sweep |
| Gradient clipping | norm 1 | norm 1 |
| Trainable parameters | A, B, classification head | A and B |

RoBERTa evaluates the validation split; MNLI also evaluates the mismatched validation split. The paper's heatmaps use matched accuracy. GLUE metric files are `eval_results.json` and `all_results.json`. Both LoRA matrices train by default. Figure 7 explicitly freezes A.

Decoder training uses the upstream instruction format, appends EOS when the cutoff permits, and masks instruction/input tokens from the loss. A seeded shuffle controls data order. Tokenized caches include a hash of the source file, tokenizer vocabulary and special tokens, model identifier, sequence length, and prompt settings.

Arithmetic evaluates **AddSub, MultiArith, SingleEq, SVAMP, GSM8K, AQuA**. Commonsense evaluates **OpenBookQA, ARC-Challenge, WinoGrande, PIQA, Social IQA, ARC-Easy, BoolQ, HellaSwag**. Generation is deterministic beam search with four beams, 256 new tokens for arithmetic and 32 for commonsense. Evaluation batches are 16, except HellaSwag uses 10. Numeric answers use the final generated number with tolerance `0.001`; AQuA and commonsense use the upstream answer extraction conventions. Reported aggregate accuracy is the unweighted mean of the six or eight dataset accuracies, in percentage points. QLoRA trains with NF4, double quantization, BF16 compute, then evaluates the saved adapter on the full-precision base as in the archived evaluator.

## Paper experiment catalog

The default seeds are explicit reference seeds: 42 for language-model cells, and 42/43/44 for the three-seed toy sweep. Most original language-model seed identities were not specified in the paper. Override with `--seeds 42 43 44` for a repeated comparison. Extra seeds multiply the run counts shown by `--list`; they do not establish that the historical results had those seeds.

| Result | Recipe | Setup / grid |
| --- | --- | --- |
| Figure 1 | `figure1` | MNIST toy, width 4096, rank 8; A, AB with subtraction, A²; beta `{1,5,25}`; first 100 updates. Reconstructed; LR A `2^-7` is a declared reference choice. |
| Figure 2 | `figure2` | MNIST toy; widths `{16,64,256,1024,4096}`; rank 8; beta `{1,5}`; A and A²; LR A `{2^-7,...,2^0}`, LR B = width × LR A; 20 epochs; 3 seeds. Best validation loss over epochs. |
| Figure 3(a) | `figure3_beta` | Llama 3-8B on both benchmark families; beta `{0.25,0.5,1,2,4,8}`; ratios `{1,10,200,500,1000,2000,4000}`; LR A `1e-5`; A and A². |
| Figure 3(b) | `figure3_rank` | Llama 3.2-1B on both families; beta 1; ranks `{4,8,16,32,64,128}`; alpha = rank; same ratio grid and LR A. |
| Figure 4(a) | `figure4_rank_equal`, `figure4_rank_ratio` | One arithmetic epoch; ranks `{4,16,64,256}`; beta 1; alpha = rank; equal LR powers `2^-19` through `2^-7`, or ratio powers `2^1` through `2^11` at LR A `1e-5`. |
| Figure 4(b) | `figure4_beta_equal`, `figure4_beta_ratio` | Same loss study, rank 8; beta `{1,4,8,32}`; equal-LR and ratio grids above. Backbone and intermediate axis points are reference choices; see limitations below. |
| Figure 5 | `figure5` | Llama 3-8B arithmetic; beta 0.5; rank 8; LR A `1e-5`, LR B `2e-3`; multipliers `{0.5,1,2,3,4,8,10,12}`. |
| C.3 fixed-ratio GLUE study | `glue_ratios` | RoBERTa-base on MNLI/QQP/SST-2/QNLI; beta `{0.25,0.5,1,2,4,8}`; LR A `1e-5`; ratios `{1,50,200,400,800}`. |
| Figure 6 | `figure6` | RoBERTa SST-2/MNLI; beta `{0.25,8}`; LR A `{5e-7,1e-6,3e-6,8e-6,1e-5,3e-5,8e-5,1e-4}`; independent LR B `{3e-4,8e-4,1e-3,3e-3,8e-3,1e-2,5e-2,1e-1}`; A and A². |
| Figure 7 | `figure7` | RoBERTa SST-2/MNLI; frozen A; beta `{0.25,0.5,1,2,4,8}`; LR B `{5e-5,1e-4,5e-4,1e-3,8e-3,1e-2}`. |
| Figure 8(a) | `figure8_power` | Llama 3.2-1B arithmetic; A/A²/A³/A⁴; beta `{4,8,12,16}`; ratios `{1,10,50,200,500,1000,2000,4000,8000,10000,20000}`. |
| Figure 8(b) | `figure8_qlora` | Llama 3-8B arithmetic; LoRA/QLoRA, A/A²; same beta and ratio grid as Figure 3(a). |
| Figure 9 | `figure9_llama1b`, `figure9_llama3b`, `figure9_qwen25`, `figure9_qwen3`, `figure9_mistral` | Llama 3.2-1B/3B, Qwen2.5-3B, Qwen3-0.6B, Mistral-7B-v0.3; beta `{0.5,1,2,4,8}`; ratios `{1,10,50,200,500,1000,2000}`. Qwen3 instead uses `{1,10,50,100,200,500,1000}` as labeled in its panel. |
| Figure 10 | `figure10` | RoBERTa SST-2/MNLI; A² versus AB with subtraction; beta `{0.25,0.5,1,2,4,8}`; equal LR `{1e-5,3e-5,8e-5,1e-4,3e-4,8e-4,1e-3}`. |
| Table 1 | `table1` | Llama arithmetic; beta 1; ratio 4000; rank/alpha 8; horizons `{1,3,6,10}`. |
| Table 2 | `table2` | Figure 5 learning rates and beta; multipliers `{4,12}`; horizons `{1,3,5,10}`. |
| Table 3 | `table3` | Arithmetic equal-LR comparison of EVA, OLoRA, PiSSA, A/A²/AB at beta `{1,8}`; LRs `{1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,8e-3}`. Llama 3.2-1B follows the historical launcher. |
| Table 4 | `table4` | Arithmetic; A², Orthogonal, Orthogonal-A², EVA, EVA-A², PiSSA, PiSSA-A²; ratios `{1,200,4000}`; beta `{0.5,1}` for compressed methods. Dropout 0; EVA uses 256 samples and rho 1. 33 cells. |

## Commands

Every recipe uses the same interface. Without `--run`, commands are printed and training is not started:

```bash
python scripts/reproduce.py --list
python scripts/reproduce.py --experiment figure5 --dry-run --limit 1
python scripts/reproduce.py --experiment figure5 --run
python scripts/reproduce.py --experiment figure3_rank --task math_10k --method AA --seeds 42 43 44 --run
python scripts/reproduce.py --experiment table4 --model /path/to/Llama-3.1-8B --run
```

Filter by `--method` or `--task`, choose a slice using `--start` and `--limit`, and override numeric settings with `--set KEY=JSON`. For example `--set epochs=1 --set beta=0.5`. `--skip-eval` is useful for the loss studies. Override the output location with `--output-root`. When overriding a setting across a grid, select a small slice first to avoid repeated identical configurations.

Run directories include a stable configuration identifier and contain `run.json` (the exact commands, recipe, settings, and software versions), per-command logs, model/metric outputs, and a completion marker. Repeating the same command skips completed cells. Incomplete cells restart from initialization; they do not silently resume a different checkpoint. Internal cache keys and run identifiers account for source and data changes; they are not published as provenance fields. Do not aggregate duplicate copies of the same setting and seed across release revisions.

Summarize and export figures after a grid completes:

```bash
python scripts/plot_toy_features.py --input-root outputs/figure1 --output figures/figure1
python scripts/summarize_results.py --input-root outputs/figure5
python scripts/plot_results.py --input outputs/figure5/summary/summary.csv --x multiplier --output-dir figures/figure5
python scripts/summarize_results.py --input-root outputs/figure6
python scripts/plot_results.py --input outputs/figure6/summary/summary.csv --x lr_a --y lr_b --output-dir figures/figure6
python scripts/summarize_results.py --input-root outputs/figure2
python scripts/plot_results.py --input outputs/figure2/summary/summary.csv --x lr_a --series width --metric validation_loss --log-x --output-dir figures/figure2
```

The collector includes completed runs only, checks benchmark completeness, and writes raw records plus means and sample standard deviations across seeds. A single seed has no measured uncertainty; its standard deviation column is zero. Plots export PDF/SVG and 300-DPI PNG, preserving missing heatmap cells as missing. Loss summaries distinguish the last logged training-loss window from the average loss over the entire training run. The plotting tool creates one panel per remaining setting combination; it does not silently average different models, betas, or horizons.

## Checkpoint handling

The final adapter is under `finetuned_result/`, with `run_config.json`, `initialization_stats.json`, `final_train_metrics.json`, and `train_log_history.json`. Residual-weight initializers (`AB_reset`, PiSSA, PiSSA-A², OLoRA) also require `init_ckpt/` inside that directory. During evaluation the loader subtracts the saved initial low-rank product from a fresh base and loads the trained adapter without rerunning its initializer. Keep both directories together. Newly trained adapters infer the replay mode from `run_config.json`; legacy AB checkpoints require `--init_para AB_reset` explicitly.

PiSSA-A² rescales `A <- c A, B <- B/c` to RMS(A) = beta/fan_in, preserving the initialized product. EVA-A² rescales the activation-informed A directions and zeroes B. Orthogonal-A² uses row-orthogonal A at the same RMS and zeroes B. These aliases use `--init_method`; beta is `--lora_A_std`. Residual-weight initializers use dropout zero so the cancellation remains exact. They are validated as full-precision LoRA paths; NF4 QLoRA uses A or A² initialization.

## Additional diagnostics and follow-ups

| Recipe / launcher | Setup | Outputs |
| --- | --- | --- |
| `rank_coherence` | Llama 3.2-1B, beta 1, ranks 4–128, ratios 200/1000/2000, 3 seeds; fixed probe | `coherence_stats/coherence_stats.jsonl`, initial A weights, probe/config metadata |
| `training_dynamics` | Llama 3.2-1B and Llama 3.1-8B, A/A², 3 seeds | A/B activations, scaled adapter output, A/B gradient and parameter norms |
| `feature_decomposition` | Same two models and seeds, fixed probe with at most 32 token positions per module; the README figure uses `--set beta=0.5 --set multiplier=12` | Exact noise/signal/signed cross energies and reconstruction error |
| `width_alignment` | Qwen2.5 Instruct 0.5B/1.5B/3B/7B, A/A², 3 seeds | Same-minibatch realized Adam A-update/input contractions and shuffled-coordinate control |
| `submit_fig5_multiplier_seeds.sh` | Figure 5 seed follow-up; default multipliers 1/3/10/12, 5 seeds | Seed accuracy aggregates |
| `submit_roberta_heatmap_5seeds.sh` | SST-2 beta 8; five A LRs, five ratios; 5 seeds | Supplemental ratio-based heatmap, distinct from Figure 6's independent A/B grid |
| `submit_math_epoch_curve.sh` | Llama 3.1-8B, beta 1, ratio 4000; independent horizons 1–10 | Epoch-budget accuracy/loss comparisons |
| `submit_fig5_tuning_long_horizon.sh` | Fixed-LR, matched-LR screen, and long-horizon follow-ups | Results in separate phase directories |
| `submit_roberta_qwen_rank_coherence.sh` | Additional RoBERTa/Qwen rank diagnostics | Same coherence JSONL schema |
| `submit_fig5_multiplier_4_12_eta_b_sweep.sh` | Multipliers 4/12, LR A `1e-5`, LR B `{5e-4,1e-3,2e-3,4e-3}` | Final arithmetic accuracy per LR pair |
| `submit_all_module_alignment.sh` | Width alignment across every adapted module | Per-module ranking and width fits |

The preserved supplemental launchers accept `DRY_RUN=1`. Set `RUN_LOCAL=1` to execute the same jobs sequentially on the current machine; otherwise they use Slurm. Supply `SBATCH_ACCOUNT` only if your cluster requires it. Resource requests can be overridden with the other `SBATCH_*` variables. `LLM_ADAPTERS_DIR` and `PYTHON_BIN` configure shared data and environment paths. See [additional_experiments/README.md](../additional_experiments/README.md) for their matching plotters.

Coherence measures contractions of the bias-corrected AdamW B direction with the fixed initial and current A features. Decomposition verifies `Z_B = noise + signal` and `E_total = E_noise + E_signal + E_cross`, with a signed cross term. Width alignment measures the realized update `-(A_next-A)/lr_A` on the actual training minibatch; it differs from a raw-gradient diagnostic. The callbacks save module names and dimensions so the selected modules and widths remain auditable.

## Limits on exact numerical reproduction

The original MNIST implementation and numerical outputs were absent from the repository. `train_toy.py` is a working reconstruction: frozen Gaussian Win, frozen Wout with standard deviation 1/width, zero W0, only A/B trained, images flattened and scaled to [0,1], official test split as validation, Adam with betas `(0,0)` as stated in Section 4.1. Its ReLU uses the valid subgradient **1 at zero** (`clamp_min(0)`). Using PyTorch's default ReLU subgradient 0 with W0=0 and B0=0 gives zero gradients on the first update, so that unspecified choice matters. Figure 1's LR and original seed identities are also missing. These runs test the described setup; they cannot certify the original plotted values.

The historical Table 1/2/4 launchers use Llama 3.1-8B while the final PDF calls these Llama 3-8B experiments. Public recipes follow the PDF; `--model meta-llama/Llama-3.1-8B` reproduces the historical backbone choice. Historical epoch curves train each horizon from scratch with a separate schedule; the Table 2 caption's “checkpoint trace” does not establish a single common 10-epoch trajectory. The Table 3 caption does not specify its backbone, so its recipe follows the archived Llama 3.2-1B launcher. Figure 4's backbone and intermediate grid points are likewise incompletely specified. Figure 8's old A³ launcher used a standard-deviation exponent of 1.73; the public recipe implements the named variance exponent 3 with s=1.5; `--set a_scale=1.73` selects the historical exponent. These discrepancies need the original run metadata to resolve exact score matching.

The release fixes settings that previously changed comparisons: method-dependent sequence lengths, accidentally frozen GLUE B, unseeded shuffles, and missing residual-weight replay. Those fixes make the specified comparisons runnable, but may change historical scores produced by the affected launchers. Full paper grids must be rerun on healthy GPUs before claiming numerical reproduction. See [VALIDATION.md](VALIDATION.md) for the checks actually performed for this release.
