#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
source "${REPO_ROOT}/experiments/runtime.sh"

MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
DATASET="${DATASET:-math_10k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./log_examples/single_run}"
INIT_METHOD="${INIT_METHOD:-Kaiming}"
INIT_PARA="${INIT_PARA:-A}"
INIT_STD="${INIT_STD:-1.0}"
LR="${LR:-3e-5}"
LR_RATIO="${LR_RATIO:-1}"
ADAPTER_NAME="${ADAPTER_NAME:-lora}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
VAL_SET_SIZE="${VAL_SET_SIZE:-0}"
LORA_R="${LORA_R:-8}"
ALPHA="${ALPHA:-8}"
LORA_NORMAL="${LORA_NORMAL:-False}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
CUTOFF_LEN="${CUTOFF_LEN:-256}"
LORA_A_SCALE="${LORA_A_SCALE:-}"
LORA_B_SCALE="${LORA_B_SCALE:-}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_SAVE_ADAPTER="${SKIP_SAVE_ADAPTER:-0}"
SKIP_SAVE_INIT_CKPT="${SKIP_SAVE_INIT_CKPT:-0}"
BNB_4BIT_COMPUTE_DTYPE="${BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
BNB_4BIT_QUANT_TYPE="${BNB_4BIT_QUANT_TYPE:-nf4}"
BNB_4BIT_USE_DOUBLE_QUANT="${BNB_4BIT_USE_DOUBLE_QUANT:-True}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
    bash "${SCRIPT_DIR}/run_llama_test.sh"
    --dataset "${DATASET}"
    --output_root "${OUTPUT_ROOT}"
    --model "${MODEL}"
    --adapter_name "${ADAPTER_NAME}"
    --init_method "${INIT_METHOD}"
    --init_para "${INIT_PARA}"
    --init_std "${INIT_STD}"
    --lrs "${LR}"
    --lr_ratios "${LR_RATIO}"
    --alpha "${ALPHA}"
    --lora_r "${LORA_R}"
    --gpu "${GPU}"
    --seed "${SEED}"
    --batch_size "${BATCH_SIZE}"
    --micro_batch_size "${MICRO_BATCH_SIZE}"
    --val_set_size "${VAL_SET_SIZE}"
    --lora_normal "${LORA_NORMAL}"
    --bnb_4bit_compute_dtype "${BNB_4BIT_COMPUTE_DTYPE}"
    --bnb_4bit_quant_type "${BNB_4BIT_QUANT_TYPE}"
    --bnb_4bit_use_double_quant "${BNB_4BIT_USE_DOUBLE_QUANT}"
)

if [[ -n "${NUM_EPOCHS}" ]]; then
    cmd+=(--num_epochs "${NUM_EPOCHS}")
fi
if [[ -n "${CUTOFF_LEN}" ]]; then
    cmd+=(--cutoff_len "${CUTOFF_LEN}")
fi
if [[ -n "${LORA_A_SCALE}" ]]; then
    cmd+=(--lora_A_scale "${LORA_A_SCALE}")
fi
if [[ -n "${LORA_B_SCALE}" ]]; then
    cmd+=(--lora_B_scale "${LORA_B_SCALE}")
fi
if [[ "${SKIP_EVAL}" == "1" ]]; then
    cmd+=(--skip_eval)
fi
if [[ "${SKIP_SAVE_ADAPTER}" == "1" ]]; then
    cmd+=(--skip_save_adapter)
fi
if [[ "${SKIP_SAVE_INIT_CKPT}" == "1" ]]; then
    cmd+=(--skip_save_init_ckpt)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
else
    "${cmd[@]}"
fi
