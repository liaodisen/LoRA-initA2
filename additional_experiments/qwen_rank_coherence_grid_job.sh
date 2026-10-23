#!/bin/bash
set -euo pipefail

if [[ -n "${LORA_REPO_ROOT:-}" && -d "${LORA_REPO_ROOT}/scripts" ]]; then
    REPO_ROOT="${LORA_REPO_ROOT}"
    SCRIPT_DIR="${REPO_ROOT}/additional_experiments"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/finetune.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    SCRIPT_DIR="${REPO_ROOT}/additional_experiments"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}" || exit 1
source "${REPO_ROOT}/experiments/runtime.sh"

DATASET="math_10k"
MODEL="Qwen/Qwen3-0.6B"
ADAPTER_NAME="lora"
DEFAULT_OUTPUT_BASE="${ADDITIONAL_OUTPUT_BASE:-${SCRATCH:-${REPO_ROOT}}/additional_outputs}"
OUTPUT_ROOT="${DEFAULT_OUTPUT_BASE}/qwen3_0p6b_rank_coherence_math10k"
SEEDS="42"
RANKS="4 8 16 32 64 128"
LR_A="1e-5"
LR_RATIOS="200 1000 2000"
INIT_STD="1"
NUM_EPOCHS="1"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
CUTOFF_LEN="${CUTOFF_LEN:-512}"
VAL_SET_SIZE="${VAL_SET_SIZE:-0}"
LORA_NORMAL="${LORA_NORMAL:-False}"
TRAIN_ONLY="${TRAIN_ONLY:-1}"
GPU="${GPU:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
COHERENCE_PROBE_SIZE="${COHERENCE_PROBE_SIZE:-1}"
COHERENCE_PROBE_SEED="${COHERENCE_PROBE_SEED:-1234}"
COHERENCE_MAX_MODULES="${COHERENCE_MAX_MODULES:-12}"
COHERENCE_MAX_TOKENS="${COHERENCE_MAX_TOKENS:-64}"
COHERENCE_MAX_ROWS="${COHERENCE_MAX_ROWS:-64}"
COHERENCE_LOG_FRACTIONS="${COHERENCE_LOG_FRACTIONS:-0.01 0.05 0.25 0.5 1.0}"
COHERENCE_LOG_STEPS="${COHERENCE_LOG_STEPS:-1}"
COHERENCE_EVERY_N_STEPS="${COHERENCE_EVERY_N_STEPS:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) shift; DATASET="$1" ;;
        --model) shift; MODEL="$1" ;;
        --adapter_name) shift; ADAPTER_NAME="$1" ;;
        --output_root) shift; OUTPUT_ROOT="$1" ;;
        --seeds) shift; SEEDS="$1" ;;
        --ranks) shift; RANKS="$1" ;;
        --lr_a) shift; LR_A="$1" ;;
        --lr_ratios) shift; LR_RATIOS="$1" ;;
        --init_std) shift; INIT_STD="$1" ;;
        --num_epochs) shift; NUM_EPOCHS="$1" ;;
        --batch_size) shift; BATCH_SIZE="$1" ;;
        --micro_batch_size) shift; MICRO_BATCH_SIZE="$1" ;;
        --cutoff_len) shift; CUTOFF_LEN="$1" ;;
        --val_set_size) shift; VAL_SET_SIZE="$1" ;;
        --train_only) shift; TRAIN_ONLY="$1" ;;
        --gpu) shift; GPU="$1" ;;
        --skip_completed) SKIP_COMPLETED="1" ;;
        --no_skip_completed) SKIP_COMPLETED="0" ;;
        --coherence_probe_size) shift; COHERENCE_PROBE_SIZE="$1" ;;
        --coherence_probe_seed) shift; COHERENCE_PROBE_SEED="$1" ;;
        --coherence_max_modules) shift; COHERENCE_MAX_MODULES="$1" ;;
        --coherence_max_tokens) shift; COHERENCE_MAX_TOKENS="$1" ;;
        --coherence_max_rows) shift; COHERENCE_MAX_ROWS="$1" ;;
        --coherence_log_fractions) shift; COHERENCE_LOG_FRACTIONS="$1" ;;
        --coherence_log_steps) shift; COHERENCE_LOG_STEPS="$1" ;;
        --coherence_every_n_steps) shift; COHERENCE_EVERY_N_STEPS="$1" ;;
        --help|-h)
            echo "Usage: sbatch $0 --dataset math_10k --model Qwen/Qwen3-0.6B --ranks \"4 8 16\" --lr_ratios \"200 1000 2000\""
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a RANK_ARRAY <<< "${RANKS}"
read -r -a LR_RATIO_ARRAY <<< "${LR_RATIOS}"

INIT_CONFIGS=(
    "InitA:A:0.5:0.5"
    "InitA2:AA:1:1"
)

model_tag="${MODEL%/}"
model_tag="${model_tag##*/}"
if [[ -z "${model_tag}" ]]; then
    echo "Invalid model path: ${MODEL}" >&2
    exit 1
fi

extra_args=(--log_coherence_stats)
extra_args+=(
    --coherence_probe_size "${COHERENCE_PROBE_SIZE}"
    --coherence_probe_seed "${COHERENCE_PROBE_SEED}"
    --coherence_max_modules "${COHERENCE_MAX_MODULES}"
    --coherence_max_tokens "${COHERENCE_MAX_TOKENS}"
    --coherence_max_rows "${COHERENCE_MAX_ROWS}"
    --coherence_log_fractions "${COHERENCE_LOG_FRACTIONS}"
    --coherence_log_steps "${COHERENCE_LOG_STEPS}"
    --coherence_every_n_steps "${COHERENCE_EVERY_N_STEPS}"
)

if [[ "${TRAIN_ONLY}" == "1" ]]; then
    extra_args+=(--skip_eval --skip_save_adapter --skip_save_init_ckpt)
fi

echo "Job starting on $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "dataset=${DATASET}, model=${MODEL}, adapter=${ADAPTER_NAME}, output_root=${OUTPUT_ROOT}"
echo "seeds=${SEEDS}"
echo "ranks=${RANKS}"
echo "lr_A=${LR_A}, lr_ratios=${LR_RATIOS}, init_std=${INIT_STD}, epochs=${NUM_EPOCHS}, train_only=${TRAIN_ONLY}"

for seed in "${SEED_ARRAY[@]}"; do
    for config in "${INIT_CONFIGS[@]}"; do
        IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
        for rank in "${RANK_ARRAY[@]}"; do
            alpha="${rank}"

            if [[ "${SKIP_COMPLETED}" == "1" ]]; then
                completed=1
                for lr_ratio in "${LR_RATIO_ARRAY[@]}"; do
                    run_dir="${OUTPUT_ROOT}/${DATASET}/${model_tag}_${ADAPTER_NAME}_Kaiming_${init_para}_A${lora_A_scale}_B${lora_B_scale}_normal${LORA_NORMAL}_init_std${INIT_STD}/${model_tag}_${ADAPTER_NAME}_${init_para}_r${rank}_a${alpha}_lr${LR_A}_ratio${lr_ratio}_s${seed}"
                    if [[ ! -s "${run_dir}/finetuned_result/coherence_stats/coherence_stats.jsonl" ]]; then
                        completed=0
                        break
                    fi
                done
                if [[ "${completed}" == "1" ]]; then
                    echo "Skipping completed Qwen rank group: init=${init_label}, seed=${seed}, rank=${rank}"
                    continue
                fi
            fi

            echo "Running Qwen rank-coherence group: init=${init_label}, seed=${seed}, rank=${rank}, lr_A=${LR_A}, ratios=${LR_RATIOS}"
            bash experiments/run_llama_test.sh \
                --dataset "${DATASET}" \
                --output_root "${OUTPUT_ROOT}" \
                --model "${MODEL}" \
                --adapter_name "${ADAPTER_NAME}" \
                --init_method "Kaiming" \
                --init_std "${INIT_STD}" \
                --lrs "${LR_A}" \
                --lr_ratios "${LR_RATIOS}" \
                --init_para "${init_para}" \
                --lora_A_scale "${lora_A_scale}" \
                --lora_B_scale "${lora_B_scale}" \
                --lora_normal "${LORA_NORMAL}" \
                --alpha "${alpha}" \
                --lora_r "${rank}" \
                --cutoff_len "${CUTOFF_LEN}" \
                --val_set_size "${VAL_SET_SIZE}" \
                --num_epochs "${NUM_EPOCHS}" \
                --seed "${seed}" \
                --batch_size "${BATCH_SIZE}" \
                --micro_batch_size "${MICRO_BATCH_SIZE}" \
                --gpu "${GPU}" \
                "${extra_args[@]}"
        done
    done
done

echo "Completed Qwen rank-coherence sweep for dataset=${DATASET}."
