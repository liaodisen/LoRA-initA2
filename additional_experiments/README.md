# Additional experiments

These scripts retain the output layout used by the additional seed studies,
training diagnostics, and initialization comparisons. The final paper settings
and the portable Python runner are documented in [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md).
The scripts here preserve historical experiment choices where they differ from
that final specification, including Llama 3.1-8B follow-ups and some 512-token diagnostics.

## Launch

Activate the release environment and prepare benchmark data first. Every
submission script supports `DRY_RUN=1` and sequential local execution with
`RUN_LOCAL=1`; otherwise it invokes Slurm. Cluster account names have no default.

```bash
DRY_RUN=1 bash additional_experiments/submit_fig5_multiplier_seeds.sh
RUN_LOCAL=1 SEEDS="42" MULTIPLIERS="1" bash additional_experiments/submit_fig5_multiplier_seeds.sh
SBATCH_ACCOUNT=your-account bash additional_experiments/submit_rank_coherence_sweep.sh
```

`MODEL` or `MODEL_PATHS` accept Hugging Face IDs or local paths. `PYTHON_BIN`
chooses the environment; its default is the checkout's `.venv/bin/python`.
`LLM_ADAPTERS_DIR` configures the benchmark root. `ADDITIONAL_OUTPUT_BASE` chooses
the output root; otherwise outputs use `${SCRATCH:-REPO_ROOT}/additional_outputs`.
`SBATCH_*` variables configure resources. Plots write into their input root or
an explicit `--output_dir`.

## Launchers and summaries

| Experiment | Launcher | Analysis |
| --- | --- | --- |
| Figure 5 seeds | `submit_fig5_multiplier_seeds.sh` | `plot_fig5_multiplier_seeds.py` |
| RoBERTa ratio heatmap seeds | `submit_roberta_heatmap_5seeds.sh` | `plot_roberta_heatmap_5seeds.py` |
| Missing heatmap cells | `submit_roberta_heatmap_missing.sh` | Same heatmap plotter |
| Independent epoch horizons | `submit_math_epoch_curve.sh` | `plot_math_epoch_accuracy.py` |
| Llama rank coherence | `submit_rank_coherence_sweep.sh` | `summarize_completed_results.py`, `summarize_rank_coherence_mean.py` |
| Fixed-LR / matched-LR / longer training | `submit_fig5_tuning_long_horizon.sh` | Phase output metrics and seed plotters |
| RoBERTa/Qwen coherence | `submit_roberta_qwen_rank_coherence.sh` | `summarize_completed_results.py`, `plot_roberta_rank_coherence_summary.py` |
| Multiplier 4/12 horizons | `submit_fig5_multiplier_4_12_10epochs.sh` | Epoch accuracy plotter |
| Multiplier 4/12 B-LR sweep | `submit_fig5_multiplier_4_12_eta_b_sweep.sh` | Per-run `accuracy.json` |
| Activation and gradient dynamics | `submit_llama_training_dynamics.sh` | `plot_llama_parameter_gradient_norms.py`, `plot_llama_za_zb_rms.py` |
| Exact feature decomposition | `submit_llama_feature_decomposition.sh` | `summarize_llama_feature_decomposition.py`, `plot_llama_feature_decomposition_rms.py` |
| Qwen width alignment | `submit_width_alignment.sh` | `summarize_width_alignment.py` |
| Every adapted module | `submit_all_module_alignment.sh` | `summarize_module_alignment.py`, `summarize_down_proj_scaling.py` |
| Structured initialization families | `submit_llama31_init_variants_beta_lr.sh` | `summarize_llama31_init_variants.py` |

Example analysis:

```bash
python additional_experiments/plot_fig5_multiplier_seeds.py --input_root additional_outputs/fig5_multiplier_seeds
python additional_experiments/plot_roberta_heatmap_5seeds.py --input_root additional_outputs/roberta_heatmap_5seeds --task sst2
python additional_experiments/summarize_llama31_init_variants.py --input_root additional_outputs/llama31_init_variants_beta_lr
python additional_experiments/summarize_width_alignment.py --input_root additional_outputs/width_alignment
```

Each Python tool exposes its accepted paths with `--help`. New results should be collected from completed runs under your chosen output root.
