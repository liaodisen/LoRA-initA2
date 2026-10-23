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
SBATCH_MEM="${SBATCH_MEM:-64G}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-slurm_logs/rank_coherence_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-slurm_logs/rank_coherence_%j.err}"

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
MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
ADAPTER_NAME="${ADAPTER_NAME:-lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/rank_coherence_sweep}"
SEEDS=(${SEEDS:-42 43 44})
RANKS=(${RANKS:-4 8 16 32 64 128})
LR_A="${LR_A:-1e-5}"
LR_RATIOS=(${LR_RATIOS:-200 1000 2000})
INIT_STD="${INIT_STD:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
CUTOFF_LEN="${CUTOFF_LEN:-512}"
VAL_SET_SIZE="${VAL_SET_SIZE:-0}"
LORA_NORMAL="${LORA_NORMAL:-False}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
LOG_COHERENCE_STATS="${LOG_COHERENCE_STATS:-1}"
COHERENCE_PROBE_SIZE="${COHERENCE_PROBE_SIZE:-1}"
COHERENCE_PROBE_SEED="${COHERENCE_PROBE_SEED:-1234}"
COHERENCE_MAX_MODULES="${COHERENCE_MAX_MODULES:-12}"
COHERENCE_MAX_TOKENS="${COHERENCE_MAX_TOKENS:-64}"
COHERENCE_MAX_ROWS="${COHERENCE_MAX_ROWS:-64}"
COHERENCE_LOG_FRACTIONS="${COHERENCE_LOG_FRACTIONS:-0.01 0.05 0.25 0.5 1.0}"
COHERENCE_LOG_STEPS="${COHERENCE_LOG_STEPS:-1}"
COHERENCE_EVERY_N_STEPS="${COHERENCE_EVERY_N_STEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"

INIT_CONFIGS=(
    "Init[A]:A:0.5:0.5"
    "Init[A^2]:AA:1:1"
)

export WANDB_PROJECT="${WANDB_PROJECT:-lora-init-additional-plan-exp1-rank-coherence}"

submit_job() { submit_experiment "$@"; }

extra_args=()
if [[ "${TRAIN_ONLY}" == "1" ]]; then
    extra_args+=(--skip_save_adapter --skip_save_init_ckpt --skip_eval)
fi
if [[ "${LOG_COHERENCE_STATS}" == "1" ]]; then
    extra_args+=(
        --log_coherence_stats
        --coherence_probe_size "${COHERENCE_PROBE_SIZE}"
        --coherence_probe_seed "${COHERENCE_PROBE_SEED}"
        --coherence_max_modules "${COHERENCE_MAX_MODULES}"
        --coherence_max_tokens "${COHERENCE_MAX_TOKENS}"
        --coherence_max_rows "${COHERENCE_MAX_ROWS}"
        --coherence_log_fractions "${COHERENCE_LOG_FRACTIONS}"
        --coherence_log_steps "${COHERENCE_LOG_STEPS}"
        --coherence_every_n_steps "${COHERENCE_EVERY_N_STEPS}"
    )
fi

job_count=0
for seed in "${SEEDS[@]}"; do
    for rank in "${RANKS[@]}"; do
        alpha="${rank}"
        for lr_ratio in "${LR_RATIOS[@]}"; do
            for config in "${INIT_CONFIGS[@]}"; do
                IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
                echo "Submitting Plan Exp1 rank sweep: init=${init_label}, seed=${seed}, rank=${rank}, alpha=${alpha}, init_std=${INIT_STD}, lr_ratio=${lr_ratio}, coherence=${LOG_COHERENCE_STATS}"
                submit_job sbatch "${SBATCH_ARGS[@]}" "${REPO_ROOT}/experiments/run_llama_test.sh" \
                    --dataset "${DATASET}" \
                    --output_root "${OUTPUT_ROOT}" \
                    --model "${MODEL}" \
                    --adapter_name "${ADAPTER_NAME}" \
                    --init_method "Kaiming" \
                    --init_std "${INIT_STD}" \
                    --lrs "${LR_A}" \
                    --lr_ratios "${lr_ratio}" \
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
                    "${extra_args[@]}"
                job_count=$((job_count + 1))
            done
        done
    done
done

echo "Prepared ${job_count} Plan Exp1 rank-coherence sweep jobs under ${OUTPUT_ROOT}."
echo "Coherence JSONL is written under each run's finetuned_result/coherence_stats/ directory when LOG_COHERENCE_STATS=1."
