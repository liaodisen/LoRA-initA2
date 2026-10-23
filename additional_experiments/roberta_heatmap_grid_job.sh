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
OUTPUT_ROOT="${DEFAULT_OUTPUT_BASE}/roberta_heatmap_5seeds"
SEED="42"
INIT_STD="8"
INIT_LABEL="InitA"
INIT_PARA="A"
LORA_A_SCALE="0.5"
LORA_B_SCALE="0.5"
LORA_NORMAL="False"
FREEZE_LORA_A="False"
FREEZE_LORA_B="False"
NUM_EPOCHS="3"
LR_VALUES="1e-6 5e-6 1e-5 5e-5 1e-4"
LR_RATIOS="1 50 200 400 800"
USE_LOCAL="0"
DATA_DIR="data"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
CLEAN_INCOMPLETE="${CLEAN_INCOMPLETE:-0}"
ISOLATE_HF_CACHE="${ISOLATE_HF_CACHE:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) shift; TASK="$1" ;;
        --model) shift; MODEL="$1" ;;
        --output_root) shift; OUTPUT_ROOT="$1" ;;
        --seed) shift; SEED="$1" ;;
        --init_std) shift; INIT_STD="$1" ;;
        --init_label) shift; INIT_LABEL="$1" ;;
        --init_para) shift; INIT_PARA="$1" ;;
        --lora_A_scale) shift; LORA_A_SCALE="$1" ;;
        --lora_B_scale) shift; LORA_B_SCALE="$1" ;;
        --lora_normal) shift; LORA_NORMAL="$1" ;;
        --freeze_lora_A) shift; FREEZE_LORA_A="$1" ;;
        --freeze_lora_B) shift; FREEZE_LORA_B="$1" ;;
        --num_epochs) shift; NUM_EPOCHS="$1" ;;
        --lr_values) shift; LR_VALUES="$1" ;;
        --lr_ratios) shift; LR_RATIOS="$1" ;;
        --use_local) shift; USE_LOCAL="$1" ;;
        --data_dir) shift; DATA_DIR="$1" ;;
        --skip_completed) SKIP_COMPLETED="1" ;;
        --clean_incomplete) CLEAN_INCOMPLETE="1" ;;
        --isolate_hf_cache) ISOLATE_HF_CACHE="1" ;;
        --help|-h)
            echo "Usage: sbatch $0 --task sst2 --seed 42 --init_label InitA --init_para A --lr_values \"1e-6 1e-5\" --lr_ratios \"1 50\" [--skip_completed] [--clean_incomplete] [--isolate_hf_cache]"
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
export WANDB_PROJECT="${WANDB_PROJECT:-lora-init-additional-roberta}"

if [[ "${ISOLATE_HF_CACHE}" == "1" ]]; then
    CACHE_ROOT="${ADDITIONAL_CACHE_BASE:-${SCRATCH:-/tmp}/${USER}/lora-inita2/cache}"
    CACHE_RUN_ID="${SLURM_JOB_ID:-local}_${TASK}_${INIT_LABEL}_seed${SEED}"
    export HF_HOME="${CACHE_ROOT}/hf_home/${CACHE_RUN_ID}"
    export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets/${CACHE_RUN_ID}"
    export HF_HUB_CACHE="${CACHE_ROOT}/hub/${CACHE_RUN_ID}"
    mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"
    echo "Using isolated HF cache under ${CACHE_ROOT} for ${CACHE_RUN_ID}"
fi

read -r -a LR_VALUE_ARRAY <<< "${LR_VALUES}"
read -r -a LR_RATIO_ARRAY <<< "${LR_RATIOS}"

local_data_args=()
if [[ "${USE_LOCAL}" == "1" ]]; then
    local_data_args+=(--use_local --data_dir "${DATA_DIR}")
fi

echo "Job starting on $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "task=${TASK}, init=${INIT_LABEL}, init_para=${INIT_PARA}, beta=${INIT_STD}, seed=${SEED}"
echo "lr_values=${LR_VALUES}"
echo "lr_ratios=${LR_RATIOS}"
echo "skip_completed=${SKIP_COMPLETED}, clean_incomplete=${CLEAN_INCOMPLETE}, isolate_hf_cache=${ISOLATE_HF_CACHE}"

for lr in "${LR_VALUE_ARRAY[@]}"; do
    for lr_ratio in "${LR_RATIO_ARRAY[@]}"; do
        run_dir="${OUTPUT_ROOT}/${TASK}/${INIT_LABEL}_beta${INIT_STD}_seed${SEED}/lr-${lr}_ratio-${lr_ratio}"
        if [[ "${SKIP_COMPLETED}" == "1" && -f "${run_dir}/eval_results.json" ]]; then
            echo "Skipping completed RoBERTa cell: ${run_dir}"
            continue
        fi
        if [[ "${CLEAN_INCOMPLETE}" == "1" && -d "${run_dir}" && ! -f "${run_dir}/eval_results.json" ]]; then
            echo "Removing incomplete RoBERTa cell before rerun: ${run_dir}"
            rm -rf "${run_dir}"
        fi
        mkdir -p "${run_dir}"
        export TENSORBOARD_LOGGING_DIR="${run_dir}/logs"
        echo "Running RoBERTa cell: init=${INIT_LABEL}, beta=${INIT_STD}, seed=${SEED}, lr_A=${lr}, ratio=${lr_ratio}"

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
            --learning_rate "${lr}" \
            --loraplus_lr_ratio "${lr_ratio}" \
            --lora_rank "${LORA_RANK}" \
            --lora_alpha "${LORA_ALPHA}" \
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
            --seed "${SEED}" \
            --lora_init_method "${INIT_PARA}" \
            --lora_A_std "${INIT_STD}" \
            --lora_B_std "${INIT_STD}" \
            --lora_A_scale "${LORA_A_SCALE}" \
            --lora_B_scale "${LORA_B_SCALE}" \
            --lora_normal "${LORA_NORMAL}" \
            --freeze_lora_A "${FREEZE_LORA_A}" \
            --freeze_lora_B "${FREEZE_LORA_B}"
    done
done

echo "Completed RoBERTa heatmap grid for task=${TASK}, init=${INIT_LABEL}, beta=${INIT_STD}, seed=${SEED}."
