#!/bin/bash
set -euo pipefail

# Figure 5 paper configuration, evaluated after each fixed training horizon.
# Each epoch point is an independent deterministic run from initialization.

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
SBATCH_TIME="${SBATCH_TIME:-48:00:00}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-slurm_logs/fig5_multiplier_4_12_10epochs_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-slurm_logs/fig5_multiplier_4_12_10epochs_%j.err}"

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/fig5_multiplier_4_12_epoch_curve_1seed}"
SEEDS=(${SEEDS:-42})
MULTIPLIERS=(${MULTIPLIERS:-4 12})
LORA_R="${LORA_R:-8}"
INIT_STD="${INIT_STD:-0.5}"
LR_A="${LR_A:-1e-5}"
LR_RATIO="${LR_RATIO:-200}"
EPOCHS=(${EPOCHS:-1 2 3 4 5 6 7 8 9 10})
LOGGING_STEPS="${LOGGING_STEPS:-10}"
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

export WANDB_PROJECT="${WANDB_PROJECT:-lora-init-additional-fig5-multiplier-4-12-epoch-curve}"

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
        for epoch in "${EPOCHS[@]}"; do
            epoch_output_root="${OUTPUT_ROOT}/epoch_${epoch}"
            for config in "${INIT_CONFIGS[@]}"; do
                IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
                echo "Submitting Figure 5 epoch point: init=${init_label}, seed=${seed}, epochs=${epoch}, alpha/r=${multiplier}, alpha=${alpha}, eta_A=${LR_A}, eta_B/eta_A=${LR_RATIO}, loss_every=${LOGGING_STEPS}"
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
                    --logging_steps "${LOGGING_STEPS}" \
                    --seed "${seed}" \
                    --batch_size "${BATCH_SIZE}" \
                    --micro_batch_size "${MICRO_BATCH_SIZE}"
                job_count=$((job_count + 1))
            done
        done
    done
done

echo "Prepared ${job_count} Figure 5 epoch-curve jobs under ${OUTPUT_ROOT}."
echo "Each epoch_N run writes that epoch's final accuracy to accuracy.json and loss records every ${LOGGING_STEPS} steps to finetuned_result/train_log_history.json."
