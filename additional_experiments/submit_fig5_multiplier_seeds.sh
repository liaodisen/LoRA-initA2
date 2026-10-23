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
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-slurm_logs/fig5_multiplier_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-slurm_logs/fig5_multiplier_%j.err}"

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
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
ADAPTER_NAME="${ADAPTER_NAME:-lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/fig5_multiplier_seeds}"
SEEDS=(${SEEDS:-42 43 44 45 46})
MULTIPLIERS=(${MULTIPLIERS:-1 3 10 12})
LORA_R="${LORA_R:-8}"
INIT_STD="${INIT_STD:-0.5}"
LR_A="${LR_A:-1e-5}"
LR_RATIO="${LR_RATIO:-200}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
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

export WANDB_PROJECT="${WANDB_PROJECT:-lora-init-additional-fig5}"

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

job_count=0
for seed in "${SEEDS[@]}"; do
    for multiplier in "${MULTIPLIERS[@]}"; do
        alpha="$(compute_alpha "${LORA_R}" "${multiplier}")"
        for config in "${INIT_CONFIGS[@]}"; do
            IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
            echo "Submitting Figure 5 seed run: init=${init_label}, seed=${seed}, alpha/r=${multiplier}, alpha=${alpha}, lr_ratio=${LR_RATIO}"
            submit_job sbatch "${SBATCH_ARGS[@]}" "${REPO_ROOT}/experiments/run_llama_test.sh" \
                --dataset "${DATASET}" \
                --output_root "${OUTPUT_ROOT}" \
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
                --num_epochs "${NUM_EPOCHS}" \
                --seed "${seed}" \
                --batch_size "${BATCH_SIZE}" \
                --micro_batch_size "${MICRO_BATCH_SIZE}"
            job_count=$((job_count + 1))
        done
    done
done

echo "Prepared ${job_count} Figure 5 jobs under ${OUTPUT_ROOT}."
echo "Plot after completion with: python additional_experiments/plot_fig5_multiplier_seeds.py --input_root ${OUTPUT_ROOT}"
