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
SBATCH_MEM="${SBATCH_MEM:-96G}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-slurm_logs/roberta_heatmap_missing_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-slurm_logs/roberta_heatmap_missing_%j.err}"
SBATCH_EXPORT="${SBATCH_EXPORT:-ALL,SKIP_COMPLETED=1,CLEAN_INCOMPLETE=1,ISOLATE_HF_CACHE=1}"

SBATCH_ARGS=(
    "--nodes=${SBATCH_NODES}"
    "--ntasks=${SBATCH_NTASKS}"
    "--cpus-per-task=${SBATCH_CPUS_PER_TASK}"
    "--mem=${SBATCH_MEM}"
    "--time=${SBATCH_TIME}"
    "--gres=${SBATCH_GRES}"
    "--output=${SBATCH_OUTPUT}"
    "--error=${SBATCH_ERROR}"
    "--export=${SBATCH_EXPORT}"
)
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then SBATCH_ARGS+=("--account=${SBATCH_ACCOUNT}"); fi

TASK="${TASK:-sst2}"
MODEL="${MODEL:-roberta-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/roberta_heatmap_5seeds}"
SEEDS=(${SEEDS:-42 43 44 45 46})
INIT_STDS=(${INIT_STDS:-8})
LR_VALUES=(${LR_VALUES:-1e-6 5e-6 1e-5 5e-5 1e-4})
LR_RATIOS=(${LR_RATIOS:-1 50 200 400 800})
LORA_NORMAL="${LORA_NORMAL:-False}"
FREEZE_LORA_A="${FREEZE_LORA_A:-False}"
FREEZE_LORA_B="${FREEZE_LORA_B:-False}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
DRY_RUN="${DRY_RUN:-0}"
USE_LOCAL="${USE_LOCAL:-0}"
DATA_DIR="${DATA_DIR:-data}"

INIT_CONFIGS=(
    "InitA:A:0.5:0.5"
    "InitA2:AA:1:1"
)

submit_job() { submit_experiment "$@"; }

count_missing_cells() {
    local init_label=$1
    local init_std=$2
    local seed=$3
    local missing=0
    local lr
    local lr_ratio
    local eval_path

    for lr in "${LR_VALUES[@]}"; do
        for lr_ratio in "${LR_RATIOS[@]}"; do
            eval_path="${OUTPUT_ROOT}/${TASK}/${init_label}_beta${init_std}_seed${seed}/lr-${lr}_ratio-${lr_ratio}/eval_results.json"
            if [[ ! -f "${eval_path}" ]]; then
                missing=$((missing + 1))
            fi
        done
    done
    printf '%s' "${missing}"
}

job_count=0
missing_cell_count=0
total_cells_per_group=$((${#LR_VALUES[@]} * ${#LR_RATIOS[@]}))
for seed in "${SEEDS[@]}"; do
    for init_std in "${INIT_STDS[@]}"; do
        for config in "${INIT_CONFIGS[@]}"; do
            IFS=':' read -r init_label init_para lora_A_scale lora_B_scale <<< "${config}"
            missing="$(count_missing_cells "${init_label}" "${init_std}" "${seed}")"
            if [[ "${missing}" == "0" ]]; then
                echo "Complete: task=${TASK}, init=${init_label}, beta=${init_std}, seed=${seed}"
                continue
            fi

            echo "Submitting missing RoBERTa cells: task=${TASK}, init=${init_label}, beta=${init_std}, seed=${seed}, missing=${missing}/${total_cells_per_group}"
            submit_job sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/roberta_heatmap_grid_job.sh" \
                --task "${TASK}" \
                --model "${MODEL}" \
                --output_root "${OUTPUT_ROOT}" \
                --seed "${seed}" \
                --init_std "${init_std}" \
                --init_label "${init_label}" \
                --init_para "${init_para}" \
                --lora_A_scale "${lora_A_scale}" \
                --lora_B_scale "${lora_B_scale}" \
                --lora_normal "${LORA_NORMAL}" \
                --freeze_lora_A "${FREEZE_LORA_A}" \
                --freeze_lora_B "${FREEZE_LORA_B}" \
                --num_epochs "${NUM_EPOCHS}" \
                --lr_values "${LR_VALUES[*]}" \
                --lr_ratios "${LR_RATIOS[*]}" \
                --use_local "${USE_LOCAL}" \
                --data_dir "${DATA_DIR}" \
                --skip_completed \
                --clean_incomplete \
                --isolate_hf_cache

            job_count=$((job_count + 1))
            missing_cell_count=$((missing_cell_count + missing))
        done
    done
done

echo "Prepared ${job_count} RoBERTa completion jobs for ${missing_cell_count} missing cells under ${OUTPUT_ROOT}."
echo "Recheck after completion with: find ${OUTPUT_ROOT}/${TASK} -mindepth 3 -maxdepth 3 -name eval_results.json | wc -l"
