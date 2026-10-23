#!/bin/bash
set -euo pipefail

if [[ -n "${LORA_REPO_ROOT:-}" && -d "${LORA_REPO_ROOT}/scripts" ]]; then
    REPO_ROOT="${LORA_REPO_ROOT}"
    SCRIPT_DIR="${REPO_ROOT}/additional_experiments"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/run_glue.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    SCRIPT_DIR="${REPO_ROOT}/additional_experiments"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}" || exit 1
source "${REPO_ROOT}/experiments/runtime.sh"

TASK="sst2"
MODEL="roberta-base"
DEFAULT_OUTPUT_BASE="${ADDITIONAL_OUTPUT_BASE:-${SCRATCH:-${REPO_ROOT}}/additional_outputs}"
OUTPUT_ROOT="${DEFAULT_OUTPUT_BASE}/roberta_rank_coherence_sst2"
SEEDS="42"
RANKS="4 8 16 32 64 128"
LR_A="1e-5"
LR_RATIOS="50 200 800"
INIT_STD="8"
NUM_EPOCHS="1"
USE_LOCAL="0"
DATA_DIR="data"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
LORA_NORMAL="${LORA_NORMAL:-False}"
FREEZE_LORA_A="${FREEZE_LORA_A:-False}"
FREEZE_LORA_B="${FREEZE_LORA_B:-False}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-0}"
ISOLATE_HF_CACHE="${ISOLATE_HF_CACHE:-0}"
COHERENCE_PROBE_SIZE="${COHERENCE_PROBE_SIZE:-1}"
COHERENCE_PROBE_SEED="${COHERENCE_PROBE_SEED:-1234}"
COHERENCE_MAX_MODULES="${COHERENCE_MAX_MODULES:-24}"
COHERENCE_MAX_TOKENS="${COHERENCE_MAX_TOKENS:-64}"
COHERENCE_MAX_ROWS="${COHERENCE_MAX_ROWS:-64}"
COHERENCE_LOG_FRACTIONS="${COHERENCE_LOG_FRACTIONS:-0.01 0.05 0.25 0.5 1.0}"
COHERENCE_LOG_STEPS="${COHERENCE_LOG_STEPS:-1}"
COHERENCE_EVERY_N_STEPS="${COHERENCE_EVERY_N_STEPS:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) shift; TASK="$1" ;;
        --model) shift; MODEL="$1" ;;
        --output_root) shift; OUTPUT_ROOT="$1" ;;
        --seeds) shift; SEEDS="$1" ;;
        --ranks) shift; RANKS="$1" ;;
        --lr_a) shift; LR_A="$1" ;;
        --lr_ratios) shift; LR_RATIOS="$1" ;;
        --init_std) shift; INIT_STD="$1" ;;
        --num_epochs) shift; NUM_EPOCHS="$1" ;;
        --use_local) shift; USE_LOCAL="$1" ;;
        --data_dir) shift; DATA_DIR="$1" ;;
        --skip_completed) SKIP_COMPLETED="1" ;;
        --no_skip_completed) SKIP_COMPLETED="0" ;;
        --clean_incomplete) CLEAN_INCOMPLETE="1" ;;
        --isolate_hf_cache) ISOLATE_HF_CACHE="1" ;;
        --coherence_probe_size) shift; COHERENCE_PROBE_SIZE="$1" ;;
        --coherence_probe_seed) shift; COHERENCE_PROBE_SEED="$1" ;;
        --coherence_max_modules) shift; COHERENCE_MAX_MODULES="$1" ;;
        --coherence_max_tokens) shift; COHERENCE_MAX_TOKENS="$1" ;;
        --coherence_max_rows) shift; COHERENCE_MAX_ROWS="$1" ;;
        --coherence_log_fractions) shift; COHERENCE_LOG_FRACTIONS="$1" ;;
        --coherence_log_steps) shift; COHERENCE_LOG_STEPS="$1" ;;
        --coherence_every_n_steps) shift; COHERENCE_EVERY_N_STEPS="$1" ;;
        --help|-h)
            echo "Usage: sbatch $0 --task sst2 --seeds \"42\" --ranks \"4 8 16\" --lr_ratios \"50 200 800\" [--isolate_hf_cache]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-lora-init-additional-roberta-rank-coherence}"

if [[ "${ISOLATE_HF_CACHE}" == "1" ]]; then
    CACHE_ROOT="${ADDITIONAL_CACHE_BASE:-${SCRATCH:-/tmp}/${USER}/lora-inita2/cache}"
    CACHE_RUN_ID="${SLURM_JOB_ID:-local}_${TASK}_rank_coherence"
    export HF_HOME="${CACHE_ROOT}/hf_home/${CACHE_RUN_ID}"
    export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets/${CACHE_RUN_ID}"
    export HF_HUB_CACHE="${CACHE_ROOT}/hub/${CACHE_RUN_ID}"
    mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
    echo "Using isolated HF cache under ${CACHE_ROOT} for ${CACHE_RUN_ID}"
fi

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a RANK_ARRAY <<< "${RANKS}"
read -r -a LR_RATIO_ARRAY <<< "${LR_RATIOS}"

local_data_args=()
if [[ "${USE_LOCAL}" == "1" ]]; then
    local_data_args+=(--use_local --data_dir "${DATA_DIR}")
fi

INIT_CONFIGS=(
    "InitA:A:0.5:0.5"
    "InitA2:AA:1:1"
)

echo "Job starting on $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "task=${TASK}, model=${MODEL}, output_root=${OUTPUT_ROOT}"
echo "seeds=${SEEDS}"
echo "ranks=${RANKS}"
echo "lr_A=${LR_A}, lr_ratios=${LR_RATIOS}, init_std=${INIT_STD}, epochs=${NUM_EPOCHS}"
echo "skip_completed=${SKIP_COMPLETED}, clean_incomplete=${CLEAN_INCOMPLETE}, isolate_hf_cache=${ISOLATE_HF_CACHE}"

for seed in "${SEED_ARRAY[@]}"; do
    for config in "${INIT_CONFIGS[@]}"; do
        IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
        for rank in "${RANK_ARRAY[@]}"; do
            alpha="${rank}"
            for lr_ratio in "${LR_RATIO_ARRAY[@]}"; do
                run_dir="${OUTPUT_ROOT}/${TASK}/${init_label}_beta${INIT_STD}_seed${seed}/rank-${rank}_lr-${LR_A}_ratio-${lr_ratio}"
                coherence_file="${run_dir}/coherence_stats/coherence_stats.jsonl"
                eval_file="${run_dir}/eval_results.json"

                if [[ "${SKIP_COMPLETED}" == "1" && -s "${coherence_file}" && -f "${eval_file}" ]]; then
                    echo "Skipping completed RoBERTa rank-coherence cell: ${run_dir}"
                    continue
                fi
                if [[ "${CLEAN_INCOMPLETE}" == "1" && -d "${run_dir}" && ! -s "${coherence_file}" ]]; then
                    echo "Removing incomplete RoBERTa rank-coherence cell before rerun: ${run_dir}"
                    rm -rf "${run_dir}"
                fi

                mkdir -p "${run_dir}"
                export TENSORBOARD_LOGGING_DIR="${run_dir}/logs"
                echo "Running RoBERTa rank-coherence cell: init=${init_label}, seed=${seed}, rank=${rank}, lr_A=${LR_A}, ratio=${lr_ratio}"

                "${PYTHON_BIN}" scripts/run_glue.py \
                    --model_name_or_path "${MODEL}" \
                    --task_name "${TASK}" \
                    "${local_data_args[@]}" \
                    --use_lora \
                    --target_modules "query, value" \
                    --do_train \
                    --do_eval \
                    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
                    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
                    --max_seq_length "${MAX_SEQ_LENGTH}" \
                    --eval_steps "${EVAL_STEPS}" \
                    --save_steps "${SAVE_STEPS}" \
                    --logging_steps "${LOGGING_STEPS}" \
                    --num_train_epochs "${NUM_EPOCHS}" \
                    --learning_rate "${LR_A}" \
                    --loraplus_lr_ratio "${lr_ratio}" \
                    --lora_rank "${rank}" \
                    --lora_alpha "${alpha}" \
                    --fp16 \
                    --lr_scheduler_type "linear" \
                    --adam_beta1 0.9 \
                    --adam_beta2 0.99 \
                    --adam_epsilon 1e-8 \
                    --output_dir "${run_dir}" \
                    --eval_strategy steps \
                    --save_strategy no \
                    --report_to none \
                    --save_total_limit 0 \
                    --ignore_mismatched_sizes \
                    --seed "${seed}" \
                    --lora_init_method "${init_para}" \
                    --lora_A_std "${INIT_STD}" \
                    --lora_B_std "${INIT_STD}" \
                    --lora_A_scale "${lora_A_scale}" \
                    --lora_B_scale "${lora_B_scale}" \
                    --lora_normal "${LORA_NORMAL}" \
                    --freeze_lora_A "${FREEZE_LORA_A}" \
                    --freeze_lora_B "${FREEZE_LORA_B}" \
                    --log_coherence_stats \
                    --coherence_output_dir "${run_dir}/coherence_stats" \
                    --coherence_probe_size "${COHERENCE_PROBE_SIZE}" \
                    --coherence_probe_seed "${COHERENCE_PROBE_SEED}" \
                    --coherence_max_modules "${COHERENCE_MAX_MODULES}" \
                    --coherence_max_tokens "${COHERENCE_MAX_TOKENS}" \
                    --coherence_max_rows "${COHERENCE_MAX_ROWS}" \
                    --coherence_log_fractions "${COHERENCE_LOG_FRACTIONS}" \
                    --coherence_log_steps "${COHERENCE_LOG_STEPS}" \
                    --coherence_every_n_steps "${COHERENCE_EVERY_N_STEPS}"
            done
        done
    done
done

echo "Completed RoBERTa rank-coherence sweep for task=${TASK}."
