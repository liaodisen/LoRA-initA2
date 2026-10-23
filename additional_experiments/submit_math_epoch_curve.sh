#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
source "${REPO_ROOT}/experiments/runtime.sh"
DEFAULT_OUTPUT_BASE="${ADDITIONAL_OUTPUT_BASE:-${SCRATCH:-${REPO_ROOT}}/additional_outputs}"
mkdir -p slurm_logs "${DEFAULT_OUTPUT_BASE}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_NODES="${SBATCH_NODES:-1}"
SBATCH_NTASKS="${SBATCH_NTASKS:-1}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-4}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-slurm_logs/math_epoch_llama31_8b_ratio4000_initstd1_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-slurm_logs/math_epoch_llama31_8b_ratio4000_initstd1_%j.err}"

SBATCH_ARGS=(
    "--nodes=${SBATCH_NODES}"
    "--ntasks=${SBATCH_NTASKS}"
    "--cpus-per-task=${SBATCH_CPUS_PER_TASK}"
    "--mem=${SBATCH_MEM}"
    "--time=${SBATCH_TIME}"
    "--gres=${SBATCH_GRES}"
    "--output=${SBATCH_OUTPUT}"
    "--error=${SBATCH_ERROR}"
)
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then SBATCH_ARGS+=("--account=${SBATCH_ACCOUNT}"); fi

DATASET="${DATASET:-math_10k}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
ADAPTER_NAME="${ADAPTER_NAME:-lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/math_epoch_curve_llama31_8b_ratio4000_initstd1}"
SEEDS=(${SEEDS:-42})
EPOCHS=(${EPOCHS:-1 2 3 4 5 6 7 8 9 10})
LORA_R="${LORA_R:-8}"
MULTIPLIER="${MULTIPLIER:-1}"
INIT_STD="${INIT_STD:-1}"
LR_A="${LR_A:-1e-5}"
LR_RATIO="${LR_RATIO:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
CUTOFF_LEN="${CUTOFF_LEN:-256}"
VAL_SET_SIZE="${VAL_SET_SIZE:-0}"
LORA_NORMAL="${LORA_NORMAL:-False}"
DRY_RUN="${DRY_RUN:-0}"

INIT_CONFIGS=(
    "Init[A]:A:0.5:0.5"
    "Init[A^2]:AA:1:1"
)

export WANDB_PROJECT="${WANDB_PROJECT:-lora-init-additional-math-epochs-ratio4000-initstd1}"

compute_alpha() {
    local rank=$1
    local multiplier=$2
    awk -v r="${rank}" -v m="${multiplier}" 'BEGIN {
        value = r * m
        if (value == int(value)) {
            printf "%d", value
        } else {
            printf "%.10g", value
        }
    }'
}

submit_job() { submit_experiment "$@"; }

alpha="$(compute_alpha "${LORA_R}" "${MULTIPLIER}")"
job_count=0
for seed in "${SEEDS[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
        epoch_output_root="${OUTPUT_ROOT}/epoch_${epoch}"
        for config in "${INIT_CONFIGS[@]}"; do
            IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
            echo "Submitting math epoch run: init=${init_label}, seed=${seed}, epochs=${epoch}, init_std=${INIT_STD}, alpha/r=${MULTIPLIER}, lr_ratio=${LR_RATIO}"
            submit_job sbatch "${SBATCH_ARGS[@]}" "${REPO_ROOT}/experiments/run_llama_test.sh" \
                --dataset "${DATASET}" \
                --output_root "${epoch_output_root}" \
                --model "${MODEL}" \
                --adapter_name "${ADAPTER_NAME}" \
                --init_method "Kaiming" \
                --init_std "${INIT_STD}" \
                --lrs "${LR_A}" \
                --lr_ratios "${LR_RATIO}" \
                --init_para "${init_para}" \
                --lora_A_scale "${lora_A_scale}" \
                --lora_B_scale "${lora_B_scale}" \
                --lora_normal "${LORA_NORMAL}" \
                --alpha "${alpha}" \
                --lora_r "${LORA_R}" \
                --cutoff_len "${CUTOFF_LEN}" \
                --val_set_size "${VAL_SET_SIZE}" \
                --num_epochs "${epoch}" \
                --seed "${seed}" \
                --batch_size "${BATCH_SIZE}" \
                --micro_batch_size "${MICRO_BATCH_SIZE}"
            job_count=$((job_count + 1))
        done
    done
done

echo "Prepared ${job_count} math epoch-curve jobs under ${OUTPUT_ROOT}."
echo "Plot after completion with: python additional_experiments/plot_math_epoch_accuracy.py --input_root ${OUTPUT_ROOT}"
