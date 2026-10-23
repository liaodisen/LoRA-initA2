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
SBATCH_OUTPUT="${SBATCH_OUTPUT:-slurm_logs/llama31_init_variants_beta_lr_%j.out}"
SBATCH_ERROR="${SBATCH_ERROR:-slurm_logs/llama31_init_variants_beta_lr_%j.err}"

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_BASE}/llama31_init_variants_beta_lr}"

# The default is a 33-job screening grid:
#   4 beta-aware methods x 2 betas x 3 ratios
#   + 3 beta-independent controls x 3 ratios.
# Set SEEDS="43 44" after screening to add two confirmation seeds without
# rerunning the seed-42 screen.
VARIANTS=(${VARIANTS:-init_a2 orthogonal orthogonal_a2 eva eva_a2 pissa pissa_a2})
BETAS=(${BETAS:-0.5 1})
LR_RATIOS=(${LR_RATIOS:-1 200 4000})
SEEDS=(${SEEDS:-42})

LR_A="${LR_A:-1e-5}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
CUTOFF_LEN="${CUTOFF_LEN:-256}"
VAL_SET_SIZE="${VAL_SET_SIZE:-0}"
LORA_DROPOUT="${LORA_DROPOUT:-0}"
EVA_MAX_SAMPLES="${EVA_MAX_SAMPLES:-256}"
EVA_RHO="${EVA_RHO:-1.0}"
DRY_RUN="${DRY_RUN:-0}"

export WANDB_PROJECT="${WANDB_PROJECT:-lora-init-additional-llama31-init-variants}"

submit_job() { submit_experiment "$@"; }

submit_cell() {
    local variant=$1
    local beta=$2
    local ratio=$3
    local seed=$4
    local init_method
    local init_para="A"
    local lora_A_scale="0.5"
    local lora_B_scale="0.5"
    local init_std="${beta}"

    case "${variant}" in
        init_a2)
            init_method="Kaiming"
            init_para="AA"
            lora_A_scale="1"
            lora_B_scale="1"
            ;;
        orthogonal_a2)
            init_method="orthogonal_a2"
            ;;
        orthogonal)
            init_method="orthogonal"
            # PEFT's original orthogonal initializer has no beta parameter.
            init_std="1"
            ;;
        eva_a2)
            init_method="eva_a2"
            ;;
        eva)
            init_method="eva"
            # Original EVA is beta-independent; eva_rho=1 keeps matched rank.
            init_std="1"
            ;;
        pissa)
            init_method="pissa"
            # Vanilla PiSSA is beta-independent. The value is ignored and is
            # supplied only because the shared runner requires --init_std.
            init_std="1"
            ;;
        pissa_a2)
            init_method="pissa_a2"
            ;;
        *)
            echo "Unknown variant '${variant}'. Expected init_a2, orthogonal, orthogonal_a2, eva, eva_a2, pissa, or pissa_a2." >&2
            exit 1
            ;;
    esac

    echo "Submitting Llama-3.1-8B init/LR cell: variant=${variant}, beta=${beta}, seed=${seed}, lr_A=${LR_A}, ratio=${ratio}, dropout=${LORA_DROPOUT}"
    submit_job sbatch "${SBATCH_ARGS[@]}" "${REPO_ROOT}/experiments/run_llama_test.sh" \
        --dataset "${DATASET}" \
        --output_root "${OUTPUT_ROOT}" \
        --model "${MODEL}" \
        --adapter_name "${ADAPTER_NAME}" \
        --init_method "${init_method}" \
        --init_para "${init_para}" \
        --init_std "${init_std}" \
        --lora_A_scale "${lora_A_scale}" \
        --lora_B_scale "${lora_B_scale}" \
        --lora_normal "False" \
        --lora_dropout "${LORA_DROPOUT}" \
        --eva_max_samples "${EVA_MAX_SAMPLES}" \
        --eva_rho "${EVA_RHO}" \
        --lrs "${LR_A}" \
        --lr_ratios "${ratio}" \
        --alpha "${LORA_ALPHA}" \
        --lora_r "${LORA_R}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --val_set_size "${VAL_SET_SIZE}" \
        --num_epochs "${NUM_EPOCHS}" \
        --seed "${seed}" \
        --batch_size "${BATCH_SIZE}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}"
}

job_count=0
for variant in "${VARIANTS[@]}"; do
    if [[ "${variant}" == "orthogonal" || "${variant}" == "eva" || "${variant}" == "pissa" ]]; then
        variant_betas=("none")
    else
        variant_betas=("${BETAS[@]}")
    fi

    for beta in "${variant_betas[@]}"; do
        for ratio in "${LR_RATIOS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                submit_cell "${variant}" "${beta}" "${ratio}" "${seed}"
                job_count=$((job_count + 1))
            done
        done
    done
done

echo "Prepared ${job_count} Llama-3.1-8B initialization/asymmetric-LR jobs under ${OUTPUT_ROOT}."
echo "Screening defaults: variants=${VARIANTS[*]}, betas=${BETAS[*]}, ratios=${LR_RATIOS[*]}, seeds=${SEEDS[*]}."
echo "Summarize after completion with:"
echo "  python additional_experiments/summarize_llama31_init_variants.py --input_root ${OUTPUT_ROOT}"
