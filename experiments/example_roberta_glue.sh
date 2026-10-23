#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
source "${REPO_ROOT}/experiments/runtime.sh"

MODEL="${MODEL:-roberta-base}"
TASK="${TASK:-sst2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./log_examples/roberta_glue}"
LR="${LR:-1e-5}"
LR_RATIO="${LR_RATIO:-50}"
INIT_PARA="${INIT_PARA:-A}"
LORA_A_STD="${LORA_A_STD:-1.0}"
LORA_B_STD="${LORA_B_STD:-1.0}"
LORA_A_SCALE="${LORA_A_SCALE:-0.5}"
LORA_B_SCALE="${LORA_B_SCALE:-0.5}"
LORA_NORMAL="${LORA_NORMAL:-False}"
TARGET_MODULES="${TARGET_MODULES:-query, value}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
SEED="${SEED:-42}"
USE_LOCAL="${USE_LOCAL:-0}"
DATA_DIR="${DATA_DIR:-data}"
FP16="${FP16:-1}"
BF16="${BF16:-0}"
DRY_RUN="${DRY_RUN:-0}"

export TOKENIZERS_PARALLELISM=false

model_tag="${MODEL##*/}"
output_dir="${OUTPUT_ROOT%/}/${TASK}/${model_tag}_init${INIT_PARA}_lr${LR}_ratio${LR_RATIO}_s${SEED}"

cmd=(
    "${PYTHON_BIN}" scripts/run_glue.py
    --model_name_or_path "${MODEL}"
    --task_name "${TASK}"
    --use_lora
    --target_modules "${TARGET_MODULES}"
    --do_train
    --do_eval
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --eval_steps "${EVAL_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --logging_steps "${LOGGING_STEPS}"
    --num_train_epochs "${EPOCHS}"
    --learning_rate "${LR}"
    --loraplus_lr_ratio "${LR_RATIO}"
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --lr_scheduler_type linear
    --adam_beta1 0.9
    --adam_beta2 0.99
    --adam_epsilon 1e-8
    --output_dir "${output_dir}"
    --logging_dir "${output_dir}/logs"
    --evaluation_strategy steps
    --save_strategy no
    --report_to none
    --overwrite_output_dir
    --save_total_limit 0
    --ignore_mismatched_sizes
    --seed "${SEED}"
    --lora_init_method "${INIT_PARA}"
    --lora_A_std "${LORA_A_STD}"
    --lora_B_std "${LORA_B_STD}"
    --lora_A_scale "${LORA_A_SCALE}"
    --lora_B_scale "${LORA_B_SCALE}"
    --lora_normal "${LORA_NORMAL}"
)

if [[ "${USE_LOCAL}" == "1" ]]; then
    cmd+=(--use_local --data_dir "${DATA_DIR}")
fi
if [[ "${FP16}" == "1" ]]; then
    cmd+=(--fp16)
fi
if [[ "${BF16}" == "1" ]]; then
    cmd+=(--bf16)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
else
    "${cmd[@]}"
fi
