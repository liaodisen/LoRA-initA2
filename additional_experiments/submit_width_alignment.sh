#!/bin/bash
set -euo pipefail

# Measure <U_A^t, Z> across model width n
# during real minibatch fine-tuning. The default grid uses four Qwen2.5
# checkpoints from one model family, two LoRA initializations, and three seeds.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
source "${REPO_ROOT}/experiments/runtime.sh"

DEFAULT_OUTPUT_BASE="${ADDITIONAL_OUTPUT_BASE:-${SCRATCH:-${REPO_ROOT}}/additional_outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/width_alignment}"
mkdir -p slurm_logs "${OUTPUT_ROOT}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_NODES="${SBATCH_NODES:-1}"
SBATCH_NTASKS="${SBATCH_NTASKS:-1}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-4}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-slurm_logs/width_alignment_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-slurm_logs/width_alignment_%j.err}"

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

DEFAULT_MODEL_PATHS=(
    Qwen/Qwen2.5-0.5B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct
    Qwen/Qwen2.5-3B-Instruct
    Qwen/Qwen2.5-7B-Instruct
)
if [[ -n "${MODEL_PATHS:-}" ]]; then
    read -r -a MODEL_PATH_VALUES <<< "${MODEL_PATHS}"
else
    MODEL_PATH_VALUES=("${DEFAULT_MODEL_PATHS[@]}")
fi
read -r -a SEED_VALUES <<< "${SEEDS:-42 43 44}"
DATASET="${DATASET:-math_10k}"
ADAPTER_NAME="${ADAPTER_NAME:-lora}"
LORA_R="${LORA_R:-8}"
MULTIPLIER="${MULTIPLIER:-1}"
INIT_STD="${INIT_STD:-0.5}"
LR_A="${LR_A:-1e-5}"
LR_RATIO="${LR_RATIO:-200}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
CUTOFF_LEN="${CUTOFF_LEN:-256}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
ALIGNMENT_MODULE_PATTERN="${ALIGNMENT_MODULE_PATTERN:-q_proj}"
ALIGNMENT_MAX_MODULES="${ALIGNMENT_MAX_MODULES:-4}"
ALIGNMENT_MAX_TOKENS="${ALIGNMENT_MAX_TOKENS:-32}"
ALIGNMENT_LOG_STEPS="${ALIGNMENT_LOG_STEPS:-2 5 10 25 50 100}"
ALIGNMENT_LOG_FRACTIONS="${ALIGNMENT_LOG_FRACTIONS:-0.25 0.5 1.0}"
ALIGNMENT_EVERY_N_STEPS="${ALIGNMENT_EVERY_N_STEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"

INIT_CONFIGS=(
    "Init[A]:A:0.5:0.5"
    "Init[A^2]:AA:1:1"
)

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
for model in "${MODEL_PATH_VALUES[@]}"; do
    width="recorded-by-training-callback"
    for seed in "${SEED_VALUES[@]}"; do
        for config in "${INIT_CONFIGS[@]}"; do
            IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
            echo "Submitting alignment: model=${model}, width=${width}, init=${init_label}, seed=${seed}"
            command=(
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
                --lora_normal False
                --alpha "${alpha}"
                --lora_r "${LORA_R}"
                --cutoff_len "${CUTOFF_LEN}"
                --val_set_size 0
                --num_epochs "${NUM_EPOCHS}"
                --logging_steps "${LOGGING_STEPS}"
                --batch_size "${BATCH_SIZE}"
                --micro_batch_size "${MICRO_BATCH_SIZE}"
                --log_width_alignment
                --alignment_module_pattern "${ALIGNMENT_MODULE_PATTERN}"
                --alignment_max_modules "${ALIGNMENT_MAX_MODULES}"
                --alignment_max_tokens "${ALIGNMENT_MAX_TOKENS}"
                --alignment_log_steps "${ALIGNMENT_LOG_STEPS}"
                --alignment_log_fractions "${ALIGNMENT_LOG_FRACTIONS}"
                --alignment_every_n_steps "${ALIGNMENT_EVERY_N_STEPS}"
                --seed "${seed}"
                --skip_eval
                --skip_save_adapter
                --skip_save_init_ckpt
            )
            submit_job "${command[@]}"
            job_count=$((job_count + 1))
        done
    done
done

echo "Prepared ${job_count} width-alignment jobs under ${OUTPUT_ROOT}."
echo "Expected default grid: 4 widths x 2 initializations x 3 seeds = 24 jobs."
echo "Summarize after completion with:"
echo "python additional_experiments/summarize_width_alignment.py --input_root ${OUTPUT_ROOT}"
