#!/bin/bash
set -euo pipefail

# Optional comprehensive Figure 5 follow-up described in docs/EXPERIMENTS.md.
#
# Purpose:
# - separate Init[A] instability from slow convergence under the original
#   fixed-LR Figure 5 setting;
# - check whether matched LR retuning removes the apparent Init[A] gap;
# - measure whether Init[A^2] has a larger near-optimal hyperparameter region.
#
# The default is intentionally broad and expensive:
# stage1=48 jobs, screen=72 jobs, long_fixed=162 jobs, total=282 jobs.
# Prefer PHASES="screen" or PHASES="long_fixed" with narrowed grids unless
# the study includes both tuning and long-horizon comparisons.

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
SBATCH_TIME="${SBATCH_TIME:-24:00:00}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-slurm_logs/fig5_tuning_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-slurm_logs/fig5_tuning_%j.err}"

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
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
ADAPTER_NAME="${ADAPTER_NAME:-lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/fig5_tuning_long_horizon_llama31_8b}"
PHASES=(${PHASES:-stage1 screen long_fixed})
SEEDS=(${SEEDS:-42 43 44})
SCREEN_SEEDS=(${SCREEN_SEEDS:-42})
LORA_R="${LORA_R:-8}"
INIT_STD="${INIT_STD:-0.5}"
FIXED_LR_A="${FIXED_LR_A:-1e-5}"
FIXED_LR_RATIO="${FIXED_LR_RATIO:-200}"
STAGE1_MULTIPLIERS=(${STAGE1_MULTIPLIERS:-0.5 1 2 3 4 8 10 12})
PRIMARY_MULTIPLIERS=(${PRIMARY_MULTIPLIERS:-3 10 12})
SCREEN_LR_AS=(${SCREEN_LR_AS:-5e-6 1e-5 2e-5})
SCREEN_LR_RATIOS=(${SCREEN_LR_RATIOS:-50 100 200 400})
LONG_LR_AS=(${LONG_LR_AS:-${FIXED_LR_A}})
LONG_LR_RATIOS=(${LONG_LR_RATIOS:-${FIXED_LR_RATIO}})
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
SCREEN_EPOCHS="${SCREEN_EPOCHS:-3}"
LONG_EPOCHS=(${LONG_EPOCHS:-1 2 3 4 5 6 7 8 9})
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
CUTOFF_LEN="${CUTOFF_LEN:-256}"
VAL_SET_SIZE="${VAL_SET_SIZE:-0}"
LORA_NORMAL="${LORA_NORMAL:-False}"
DRY_RUN="${DRY_RUN:-0}"

INIT_CONFIGS=(
    "Init[A]:A:0.5:0.5"
    "Init[A^2]:AA:1:1"
)

export WANDB_PROJECT="${WANDB_PROJECT:-lora-init-additional-plan-exp2-fig5-tuning}"

compute_alpha() {
    local rank=$1
    local multiplier=$2
    awk -v r="${rank}" -v m="${multiplier}" 'BEGIN {
        value = r * m
        if (value == int(value)) {
            printf "%d", value
        } else {
            printf "%.10g", value
        }
    }'
}

submit_job() { submit_experiment "$@"; }

submit_llama_job() {
    local phase_root=$1
    local multiplier=$2
    local lr_a=$3
    local lr_ratio=$4
    local epochs=$5
    local seed=$6
    local init_label=$7
    local init_para=$8
    local lora_A_scale=$9
    local lora_B_scale=${10}
    local alpha
    alpha="$(compute_alpha "${LORA_R}" "${multiplier}")"

    echo "Submitting Plan Exp2: root=${phase_root}, init=${init_label}, seed=${seed}, epochs=${epochs}, alpha/r=${multiplier}, alpha=${alpha}, lr_A=${lr_a}, lr_ratio=${lr_ratio}"
    submit_job sbatch "${SBATCH_ARGS[@]}" "${REPO_ROOT}/experiments/run_llama_test.sh" \
        --dataset "${DATASET}" \
        --output_root "${phase_root}" \
        --model "${MODEL}" \
        --adapter_name "${ADAPTER_NAME}" \
        --init_method "Kaiming" \
        --init_std "${INIT_STD}" \
        --lrs "${lr_a}" \
        --lr_ratios "${lr_ratio}" \
        --init_para "${init_para}" \
        --lora_A_scale "${lora_A_scale}" \
        --lora_B_scale "${lora_B_scale}" \
        --lora_normal "${LORA_NORMAL}" \
        --alpha "${alpha}" \
        --lora_r "${LORA_R}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --val_set_size "${VAL_SET_SIZE}" \
        --num_epochs "${epochs}" \
        --seed "${seed}" \
        --batch_size "${BATCH_SIZE}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}"
}

has_phase() {
    local wanted=$1
    local phase
    for phase in "${PHASES[@]}"; do
        [[ "${phase}" == "${wanted}" ]] && return 0
    done
    return 1
}

job_count=0

if has_phase "stage1"; then
    phase_root="${OUTPUT_ROOT}/stage1_fixed_lr_seed_sweep"
    for seed in "${SEEDS[@]}"; do
        for multiplier in "${STAGE1_MULTIPLIERS[@]}"; do
            for config in "${INIT_CONFIGS[@]}"; do
                IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
                submit_llama_job "${phase_root}" "${multiplier}" "${FIXED_LR_A}" "${FIXED_LR_RATIO}" "${STAGE1_EPOCHS}" "${seed}" "${init_label}" "${init_para}" "${lora_A_scale}" "${lora_B_scale}"
                job_count=$((job_count + 1))
            done
        done
    done
fi

if has_phase "screen"; then
    phase_root="${OUTPUT_ROOT}/stage2_matched_lr_screen"
    for seed in "${SCREEN_SEEDS[@]}"; do
        for multiplier in "${PRIMARY_MULTIPLIERS[@]}"; do
            for lr_a in "${SCREEN_LR_AS[@]}"; do
                for lr_ratio in "${SCREEN_LR_RATIOS[@]}"; do
                    for config in "${INIT_CONFIGS[@]}"; do
                        IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
                        submit_llama_job "${phase_root}" "${multiplier}" "${lr_a}" "${lr_ratio}" "${SCREEN_EPOCHS}" "${seed}" "${init_label}" "${init_para}" "${lora_A_scale}" "${lora_B_scale}"
                        job_count=$((job_count + 1))
                    done
                done
            done
        done
    done
fi

if has_phase "long_fixed"; then
    phase_root="${OUTPUT_ROOT}/stage3_long_fixed_lr_epochs"
    for seed in "${SEEDS[@]}"; do
        for multiplier in "${PRIMARY_MULTIPLIERS[@]}"; do
            for lr_a in "${LONG_LR_AS[@]}"; do
                for lr_ratio in "${LONG_LR_RATIOS[@]}"; do
                    for epoch in "${LONG_EPOCHS[@]}"; do
                        epoch_root="${phase_root}/epoch_${epoch}"
                        for config in "${INIT_CONFIGS[@]}"; do
                            IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
                            submit_llama_job "${epoch_root}" "${multiplier}" "${lr_a}" "${lr_ratio}" "${epoch}" "${seed}" "${init_label}" "${init_para}" "${lora_A_scale}" "${lora_B_scale}"
                            job_count=$((job_count + 1))
                        done
                    done
                done
            done
        done
    done
fi

echo "Prepared ${job_count} Plan Exp2 Figure-5 tuning/long-horizon jobs under ${OUTPUT_ROOT}."
echo "Use PHASES='stage1', PHASES='screen', or PHASES='long_fixed' to submit one part at a time."
