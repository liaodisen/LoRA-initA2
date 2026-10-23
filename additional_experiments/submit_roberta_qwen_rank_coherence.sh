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
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"

ROBERTA_MEM="${ROBERTA_MEM:-64G}"
ROBERTA_TIME="${ROBERTA_TIME:-12:00:00}"
ROBERTA_OUTPUT_LOG="${ROBERTA_OUTPUT_LOG:-slurm_logs/roberta_rank_coherence_%j.out}"
ROBERTA_ERROR_LOG="${ROBERTA_ERROR_LOG:-slurm_logs/roberta_rank_coherence_%j.err}"

QWEN_MEM="${QWEN_MEM:-80G}"
QWEN_TIME="${QWEN_TIME:-12:00:00}"
QWEN_OUTPUT_LOG="${QWEN_OUTPUT_LOG:-slurm_logs/qwen_rank_coherence_%j.out}"
QWEN_ERROR_LOG="${QWEN_ERROR_LOG:-slurm_logs/qwen_rank_coherence_%j.err}"

RUN_ROBERTA="${RUN_ROBERTA:-1}"
RUN_QWEN="${RUN_QWEN:-1}"
DRY_RUN="${DRY_RUN:-0}"

SEEDS="${SEEDS:-42}"
RANKS="${RANKS:-4 8 16 32 64 128}"

ROBERTA_TASK="${ROBERTA_TASK:-sst2}"
ROBERTA_MODEL="${ROBERTA_MODEL:-roberta-base}"
ROBERTA_OUTPUT_ROOT="${ROBERTA_OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/roberta_rank_coherence_sst2}"
ROBERTA_LR_A="${ROBERTA_LR_A:-1e-5}"
ROBERTA_LR_RATIOS="${ROBERTA_LR_RATIOS:-50 200 800}"
ROBERTA_INIT_STD="${ROBERTA_INIT_STD:-8}"
ROBERTA_NUM_EPOCHS="${ROBERTA_NUM_EPOCHS:-1}"
ROBERTA_USE_LOCAL="${ROBERTA_USE_LOCAL:-0}"
ROBERTA_DATA_DIR="${ROBERTA_DATA_DIR:-data}"
ROBERTA_ISOLATE_HF_CACHE="${ROBERTA_ISOLATE_HF_CACHE:-1}"

QWEN_DATASET="${QWEN_DATASET:-math_10k}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-0.6B}"
QWEN_ADAPTER_NAME="${QWEN_ADAPTER_NAME:-lora}"
QWEN_OUTPUT_ROOT="${QWEN_OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/qwen3_0p6b_rank_coherence_math10k}"
QWEN_LR_A="${QWEN_LR_A:-1e-5}"
QWEN_LR_RATIOS="${QWEN_LR_RATIOS:-200 1000 2000}"
QWEN_INIT_STD="${QWEN_INIT_STD:-1}"
QWEN_NUM_EPOCHS="${QWEN_NUM_EPOCHS:-1}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-16}"
QWEN_MICRO_BATCH_SIZE="${QWEN_MICRO_BATCH_SIZE:-8}"
QWEN_CUTOFF_LEN="${QWEN_CUTOFF_LEN:-512}"
QWEN_VAL_SET_SIZE="${QWEN_VAL_SET_SIZE:-0}"
QWEN_TRAIN_ONLY="${QWEN_TRAIN_ONLY:-1}"

COHERENCE_PROBE_SIZE="${COHERENCE_PROBE_SIZE:-1}"
COHERENCE_PROBE_SEED="${COHERENCE_PROBE_SEED:-1234}"
COHERENCE_MAX_TOKENS="${COHERENCE_MAX_TOKENS:-64}"
COHERENCE_MAX_ROWS="${COHERENCE_MAX_ROWS:-64}"
COHERENCE_LOG_FRACTIONS="${COHERENCE_LOG_FRACTIONS:-0.01 0.05 0.25 0.5 1.0}"
COHERENCE_LOG_STEPS="${COHERENCE_LOG_STEPS:-1}"
COHERENCE_EVERY_N_STEPS="${COHERENCE_EVERY_N_STEPS:-0}"
ROBERTA_COHERENCE_MAX_MODULES="${ROBERTA_COHERENCE_MAX_MODULES:-24}"
QWEN_COHERENCE_MAX_MODULES="${QWEN_COHERENCE_MAX_MODULES:-12}"

submit_job() { submit_experiment "$@"; }

count_words() {
    local words
    read -r -a words <<< "$1"
    echo "${#words[@]}"
}

job_count=0
seed_count="$(count_words "${SEEDS}")"
rank_count="$(count_words "${RANKS}")"

if [[ "${RUN_ROBERTA}" == "1" ]]; then
    roberta_ratio_count="$(count_words "${ROBERTA_LR_RATIOS}")"
    roberta_run_count=$((seed_count * 2 * rank_count * roberta_ratio_count))
    roberta_args=(
            "--nodes=${SBATCH_NODES}"
        "--ntasks=${SBATCH_NTASKS}"
        "--cpus-per-task=${SBATCH_CPUS_PER_TASK}"
        "--mem=${ROBERTA_MEM}"
        "--time=${ROBERTA_TIME}"
        "--gres=${SBATCH_GRES}"
        "--job-name=roberta_rankcoh"
        "--output=${ROBERTA_OUTPUT_LOG}"
        "--error=${ROBERTA_ERROR_LOG}"
    )
    worker_args=(
        --task "${ROBERTA_TASK}"
        --model "${ROBERTA_MODEL}"
        --output_root "${ROBERTA_OUTPUT_ROOT}"
        --seeds "${SEEDS}"
        --ranks "${RANKS}"
        --lr_a "${ROBERTA_LR_A}"
        --lr_ratios "${ROBERTA_LR_RATIOS}"
        --init_std "${ROBERTA_INIT_STD}"
        --num_epochs "${ROBERTA_NUM_EPOCHS}"
        --use_local "${ROBERTA_USE_LOCAL}"
        --data_dir "${ROBERTA_DATA_DIR}"
        --skip_completed
        --coherence_probe_size "${COHERENCE_PROBE_SIZE}"
        --coherence_probe_seed "${COHERENCE_PROBE_SEED}"
        --coherence_max_modules "${ROBERTA_COHERENCE_MAX_MODULES}"
        --coherence_max_tokens "${COHERENCE_MAX_TOKENS}"
        --coherence_max_rows "${COHERENCE_MAX_ROWS}"
        --coherence_log_fractions "${COHERENCE_LOG_FRACTIONS}"
        --coherence_log_steps "${COHERENCE_LOG_STEPS}"
        --coherence_every_n_steps "${COHERENCE_EVERY_N_STEPS}"
    )
    if [[ "${ROBERTA_ISOLATE_HF_CACHE}" == "1" ]]; then
        worker_args+=(--isolate_hf_cache)
    fi

    echo "Submitting RoBERTa SST-2 rank-coherence job (${roberta_run_count} cells): output_root=${ROBERTA_OUTPUT_ROOT}"
    submit_job sbatch "${roberta_args[@]}" "${SCRIPT_DIR}/roberta_rank_coherence_grid_job.sh" "${worker_args[@]}"
    job_count=$((job_count + 1))
fi

if [[ "${RUN_QWEN}" == "1" ]]; then
    qwen_ratio_count="$(count_words "${QWEN_LR_RATIOS}")"
    qwen_run_count=$((seed_count * 2 * rank_count * qwen_ratio_count))
    qwen_args=(
            "--nodes=${SBATCH_NODES}"
        "--ntasks=${SBATCH_NTASKS}"
        "--cpus-per-task=${SBATCH_CPUS_PER_TASK}"
        "--mem=${QWEN_MEM}"
        "--time=${QWEN_TIME}"
        "--gres=${SBATCH_GRES}"
        "--job-name=qwen_rankcoh"
        "--output=${QWEN_OUTPUT_LOG}"
        "--error=${QWEN_ERROR_LOG}"
    )
    echo "Submitting Qwen3-0.6B math_10K rank-coherence job (${qwen_run_count} cells): output_root=${QWEN_OUTPUT_ROOT}"
    submit_job sbatch "${qwen_args[@]}" "${SCRIPT_DIR}/qwen_rank_coherence_grid_job.sh" \
        --dataset "${QWEN_DATASET}" \
        --model "${QWEN_MODEL}" \
        --adapter_name "${QWEN_ADAPTER_NAME}" \
        --output_root "${QWEN_OUTPUT_ROOT}" \
        --seeds "${SEEDS}" \
        --ranks "${RANKS}" \
        --lr_a "${QWEN_LR_A}" \
        --lr_ratios "${QWEN_LR_RATIOS}" \
        --init_std "${QWEN_INIT_STD}" \
        --num_epochs "${QWEN_NUM_EPOCHS}" \
        --batch_size "${QWEN_BATCH_SIZE}" \
        --micro_batch_size "${QWEN_MICRO_BATCH_SIZE}" \
        --cutoff_len "${QWEN_CUTOFF_LEN}" \
        --val_set_size "${QWEN_VAL_SET_SIZE}" \
        --train_only "${QWEN_TRAIN_ONLY}" \
        --skip_completed \
        --coherence_probe_size "${COHERENCE_PROBE_SIZE}" \
        --coherence_probe_seed "${COHERENCE_PROBE_SEED}" \
        --coherence_max_modules "${QWEN_COHERENCE_MAX_MODULES}" \
        --coherence_max_tokens "${COHERENCE_MAX_TOKENS}" \
        --coherence_max_rows "${COHERENCE_MAX_ROWS}" \
        --coherence_log_fractions "${COHERENCE_LOG_FRACTIONS}" \
        --coherence_log_steps "${COHERENCE_LOG_STEPS}" \
        --coherence_every_n_steps "${COHERENCE_EVERY_N_STEPS}"
    job_count=$((job_count + 1))
fi

echo "Prepared ${job_count} Slurm jobs."
echo "Default grids: seeds=${SEEDS}, ranks=${RANKS}; RoBERTa ratios=${ROBERTA_LR_RATIOS}; Qwen ratios=${QWEN_LR_RATIOS}."
echo "Use DRY_RUN=1 to print commands without submitting."
