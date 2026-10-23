#!/bin/bash
set -euo pipefail

# Repeat the width-alignment sweep while logging every adapted LoRA module.
# The underlying sweep defaults to four Qwen2.5 widths, Init[A]/Init[A^2],
# three seeds, one epoch, and the paper's LoRA/training configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_OUTPUT_BASE="${ADDITIONAL_OUTPUT_BASE:-${SCRATCH:-${REPO_ROOT}}/additional_outputs}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/width_alignment_all_modules}"
export ALIGNMENT_MODULE_PATTERN="${ALIGNMENT_MODULE_PATTERN:-all}"
export ALIGNMENT_MAX_MODULES="${ALIGNMENT_MAX_MODULES:-0}"
export SBATCH_OUTPUT="${SBATCH_OUTPUT:-slurm_logs/width_alignment_all_modules_%j.out}"
export SBATCH_ERROR="${SBATCH_ERROR:-slurm_logs/width_alignment_all_modules_%j.err}"

bash "${SCRIPT_DIR}/submit_width_alignment.sh"

echo "Rank every logged module after completion with:"
echo "python additional_experiments/summarize_module_alignment.py \\"
echo "  --input_root ${OUTPUT_ROOT} \\"
echo "  --output_dir additional_experiments/results/all_module_alignment"
