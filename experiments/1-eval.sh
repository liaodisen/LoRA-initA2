#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
source "${REPO_ROOT}/experiments/runtime.sh"

usage() {
    cat <<'EOF'
Usage: bash experiments/1-eval.sh MODEL DATASET ADAPTER GPU SEED OUTPUT_DIR INIT_PARA

DATASET may be either:
  - a training-set alias such as math_10k or commonsense_170k
  - a single eval dataset accepted by scripts/eval_llama.py
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

model="${1:-meta-llama/Meta-Llama-3-8B}"
dataset="${2:-math_10k}"
adapter="${3:-lora}"
gpu="${4:-0}"
seed="${5:-42}"
output_dir="${6:-./log/debug}"
init_para="${7:-A}"

eval_batch_size() {
    case "$1" in
        hellaswag) echo 10 ;;
        *) echo 16 ;;
    esac
}

case "${dataset}" in
    commonsense_170k|*common*)
        eval_datasets=("openbookqa" "ARC-Challenge" "winogrande" "piqa" "social_i_qa" "ARC-Easy" "boolq" "hellaswag")
        ;;
    math_10k|*math*)
        eval_datasets=("AddSub" "MultiArith" "SingleEq" "SVAMP" "gsm8k" "AQuA")
        ;;
    *)
        eval_datasets=("${dataset}")
        ;;
esac

mkdir -p "${output_dir}/evaluated_result"
for eval_dataset in "${eval_datasets[@]}"; do
    batch_size="$(eval_batch_size "${eval_dataset}")"
    echo "Evaluating ${eval_dataset}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/eval_llama.py \
        --base_model "${model}" \
        --adapter "${adapter}" \
        --seed "${seed}" \
        --init_para "${init_para}" \
        --dataset "${eval_dataset}" \
        --data_root "${LLM_ADAPTERS_DIR}" \
        --batch_size "${batch_size}" \
        --output_dir "${output_dir}/evaluated_result" \
        --lora_weights "${output_dir}/finetuned_result"
done
