#!/bin/bash
set -euo pipefail

# Compare Init[A] and Init[A^2] training dynamics on Llama 3.2-1B and 3.1-8B.
# The diagnostics capture Z_A, Z_B, scaled LoRA output, and A/B gradient norms
# from the current training batch before each scheduled optimizer update.

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
SBATCH_TIME="${SBATCH_TIME:-8:00:00}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-slurm_logs/llama_training_dynamics_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-slurm_logs/llama_training_dynamics_%j.err}"

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
MODEL_PATHS=(${MODEL_PATHS:-meta-llama/Llama-3.2-1B meta-llama/Llama-3.1-8B})
ADAPTER_NAME="${ADAPTER_NAME:-lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/llama_training_dynamics}"
SEEDS=(${SEEDS:-42})
LORA_R="${LORA_R:-8}"
MULTIPLIER="${MULTIPLIER:-12}"
INIT_STD="${INIT_STD:-0.5}"
LR_A="${LR_A:-1e-5}"
LR_RATIO="${LR_RATIO:-200}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DYNAMICS_MAX_MODULES="${DYNAMICS_MAX_MODULES:-8}"
DYNAMICS_LOG_FRACTIONS="${DYNAMICS_LOG_FRACTIONS:-0.01 0.05 0.1 0.25 0.5 1.0}"
DYNAMICS_LOG_STEPS="${DYNAMICS_LOG_STEPS:-}"
DYNAMICS_EVERY_N_STEPS="${DYNAMICS_EVERY_N_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
CUTOFF_LEN="${CUTOFF_LEN:-256}"
VAL_SET_SIZE="${VAL_SET_SIZE:-0}"
LORA_NORMAL="${LORA_NORMAL:-False}"
SKIP_EVAL="${SKIP_EVAL:-1}"
DRY_RUN="${DRY_RUN:-0}"

INIT_CONFIGS=(
    "Init[A]:A:0.5:0.5"
    "Init[A^2]:AA:1:1"
)

export WANDB_PROJECT="${WANDB_PROJECT:-lora-init-additional-llama-training-dynamics}"

compute_alpha() {
    awk -v rank="${LORA_R}" -v multiplier="${MULTIPLIER}" 'BEGIN {
        value = rank * multiplier
        if (value == int(value)) {
            printf "%d", value
        } else {
            printf "%.10g", value
        }
    }'
}

submit_job() { submit_experiment "$@"; }

alpha="$(compute_alpha)"
job_count=0
for model in "${MODEL_PATHS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for config in "${INIT_CONFIGS[@]}"; do
            IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
            echo "Submitting training dynamics: model=${model}, init=${init_label}, seed=${seed}, epochs=${NUM_EPOCHS}, alpha/r=${MULTIPLIER}, eta_A=${LR_A}, eta_B/eta_A=${LR_RATIO}"
            cmd=(
                sbatch "${SBATCH_ARGS[@]}" "${REPO_ROOT}/experiments/run_llama_test.sh"
                --dataset "${DATASET}"
                --output_root "${OUTPUT_ROOT}"
                --model "${model}"
                --adapter_name "${ADAPTER_NAME}"
                --init_method Kaiming
                --init_std "${INIT_STD}"
                --lrs "${LR_A}"
                --lr_ratios "${LR_RATIO}"
                --init_para "${init_para}"
                --lora_A_scale "${lora_A_scale}"
                --lora_B_scale "${lora_B_scale}"
                --lora_normal "${LORA_NORMAL}"
                --alpha "${alpha}"
                --lora_r "${LORA_R}"
                --cutoff_len "${CUTOFF_LEN}"
                --val_set_size "${VAL_SET_SIZE}"
                --num_epochs "${NUM_EPOCHS}"
                --logging_steps "${LOGGING_STEPS}"
                --log_training_dynamics
                --dynamics_max_modules "${DYNAMICS_MAX_MODULES}"
                --dynamics_log_fractions "${DYNAMICS_LOG_FRACTIONS}"
                --dynamics_every_n_steps "${DYNAMICS_EVERY_N_STEPS}"
                --seed "${seed}"
                --batch_size "${BATCH_SIZE}"
                --micro_batch_size "${MICRO_BATCH_SIZE}"
            )
            if [[ -n "${DYNAMICS_LOG_STEPS}" ]]; then
                cmd+=(--dynamics_log_steps "${DYNAMICS_LOG_STEPS}")
            fi
            if [[ "${SKIP_EVAL}" == "1" ]]; then
                cmd+=(--skip_eval)
            fi
            submit_job "${cmd[@]}"
            job_count=$((job_count + 1))
        done
    done
done

echo "Prepared ${job_count} Llama training-dynamics jobs under ${OUTPUT_ROOT}."
echo "Each run writes finetuned_result/training_dynamics/training_dynamics.jsonl and training_dynamics_config.json."
