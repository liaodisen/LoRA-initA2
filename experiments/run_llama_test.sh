#!/bin/bash
set -euo pipefail

if [[ -n "${LORA_REPO_ROOT:-}" && -d "${LORA_REPO_ROOT}/scripts" ]]; then
    REPO_ROOT="${LORA_REPO_ROOT}"
    SCRIPT_DIR="${REPO_ROOT}/experiments"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/finetune.py" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    SCRIPT_DIR="${REPO_ROOT}/experiments"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}" || exit 1
source "${REPO_ROOT}/experiments/runtime.sh"


dataset="commonsense_170k"
output_root="./log"
lrs="1e-6"
init_std=1.0
init_method="Kaiming"
init_para="A"
lora_A_scale=""
lora_B_scale=""
lora_normal="False"
lora_dropout="0.05"
eva_max_samples="0"
eva_rho="1.0"
lr_ratios="1"
alpha=8
model="meta-llama/Meta-Llama-3-8B"
adapter_name="lora"
gpu=0
seed=42
batch_size=16
micro_batch_size=8
val_size=0
lora_r=8
cutoff_len="256"
num_epochs=""
skip_eval=0
skip_save_adapter=0
skip_save_init_ckpt=0
log_coherence_stats=0
coherence_probe_size=""
coherence_probe_seed=""
coherence_max_modules=""
coherence_max_tokens=""
coherence_max_rows=""
coherence_log_fractions=""
coherence_log_steps=""
coherence_every_n_steps=""
log_training_dynamics=0
dynamics_output_dir=""
dynamics_max_modules=""
dynamics_log_fractions=""
dynamics_log_steps=""
dynamics_every_n_steps=""
log_feature_decomposition=0
decomposition_output_dir=""
decomposition_probe_size=""
decomposition_probe_seed=""
decomposition_max_modules=""
decomposition_max_tokens=""
decomposition_log_fractions=""
decomposition_log_steps=""
decomposition_every_n_steps=""
log_width_alignment=0
alignment_output_dir=""
alignment_module_pattern=""
alignment_max_modules=""
alignment_max_tokens=""
alignment_log_fractions=""
alignment_log_steps=""
alignment_every_n_steps=""
disable_alignment_shuffled_control=0
logging_steps=10
bnb_4bit_compute_dtype="bfloat16"
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant="True"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) shift; dataset="$1" ;;
        --data_root) shift; LLM_ADAPTERS_DIR="$1" ;;
        --output_root) shift; output_root="$1" ;;
        --alpha) shift; alpha="$1" ;;
        --lrs) shift; lrs="$1" ;;
        --init_std) shift; init_std="$1" ;;
        --init_para|--init) shift; init_para="$1" ;;
        --lora_A_scale) shift; lora_A_scale="$1" ;;
        --lora_B_scale) shift; lora_B_scale="$1" ;;
        --lora_normal) shift; lora_normal="$1" ;;
        --lora_dropout) shift; lora_dropout="$1" ;;
        --eva_max_samples) shift; eva_max_samples="$1" ;;
        --eva_rho) shift; eva_rho="$1" ;;
        --lr_ratios) shift; lr_ratios="$1" ;;
        --init_method) shift; init_method="$1" ;;
        --model) shift; model="$1" ;;
        --adapter_name) shift; adapter_name="$1" ;;
        --gpu) shift; gpu="$1" ;;
        --seed) shift; seed="$1" ;;
        --batch_size) shift; batch_size="$1" ;;
        --micro_batch_size) shift; micro_batch_size="$1" ;;
        --val_set_size) shift; val_size="$1" ;;
        --lora_r) shift; lora_r="$1" ;;
        --cutoff_len) shift; cutoff_len="$1" ;;
        --num_epochs) shift; num_epochs="$1" ;;
        --skip_eval) skip_eval=1 ;;
        --skip_save_adapter) skip_save_adapter=1 ;;
        --skip_save_init_ckpt) skip_save_init_ckpt=1 ;;
        --log_coherence_stats) log_coherence_stats=1 ;;
        --coherence_probe_size) shift; coherence_probe_size="$1" ;;
        --coherence_probe_seed) shift; coherence_probe_seed="$1" ;;
        --coherence_max_modules) shift; coherence_max_modules="$1" ;;
        --coherence_max_tokens) shift; coherence_max_tokens="$1" ;;
        --coherence_max_rows) shift; coherence_max_rows="$1" ;;
        --coherence_log_fractions) shift; coherence_log_fractions="$1" ;;
        --coherence_log_steps) shift; coherence_log_steps="$1" ;;
        --coherence_every_n_steps) shift; coherence_every_n_steps="$1" ;;
        --log_training_dynamics) log_training_dynamics=1 ;;
        --dynamics_output_dir) shift; dynamics_output_dir="$1" ;;
        --dynamics_max_modules) shift; dynamics_max_modules="$1" ;;
        --dynamics_log_fractions) shift; dynamics_log_fractions="$1" ;;
        --dynamics_log_steps) shift; dynamics_log_steps="$1" ;;
        --dynamics_every_n_steps) shift; dynamics_every_n_steps="$1" ;;
        --log_feature_decomposition) log_feature_decomposition=1 ;;
        --decomposition_output_dir) shift; decomposition_output_dir="$1" ;;
        --decomposition_probe_size) shift; decomposition_probe_size="$1" ;;
        --decomposition_probe_seed) shift; decomposition_probe_seed="$1" ;;
        --decomposition_max_modules) shift; decomposition_max_modules="$1" ;;
        --decomposition_max_tokens) shift; decomposition_max_tokens="$1" ;;
        --decomposition_log_fractions) shift; decomposition_log_fractions="$1" ;;
        --decomposition_log_steps) shift; decomposition_log_steps="$1" ;;
        --decomposition_every_n_steps) shift; decomposition_every_n_steps="$1" ;;
        --log_width_alignment) log_width_alignment=1 ;;
        --alignment_output_dir) shift; alignment_output_dir="$1" ;;
        --alignment_module_pattern) shift; alignment_module_pattern="$1" ;;
        --alignment_max_modules) shift; alignment_max_modules="$1" ;;
        --alignment_max_tokens) shift; alignment_max_tokens="$1" ;;
        --alignment_log_fractions) shift; alignment_log_fractions="$1" ;;
        --alignment_log_steps) shift; alignment_log_steps="$1" ;;
        --alignment_every_n_steps) shift; alignment_every_n_steps="$1" ;;
        --disable_alignment_shuffled_control) disable_alignment_shuffled_control=1 ;;
        --logging_steps) shift; logging_steps="$1" ;;
        --bnb_4bit_compute_dtype) shift; bnb_4bit_compute_dtype="$1" ;;
        --bnb_4bit_quant_type) shift; bnb_4bit_quant_type="$1" ;;
        --bnb_4bit_use_double_quant) shift; bnb_4bit_use_double_quant="$1" ;;
        --help)
            echo "Usage: $0 [--dataset VALUE] [--output_root PATH] [--model VALUE] [--adapter_name lora|qlora] [--init_method Kaiming|pissa|orthogonal_a2|eva_a2|pissa_a2|...] [--init_para A|AA|AB|AB_reset|B|...] [--lora_A_scale VALUE] [--lora_B_scale VALUE] [--lora_dropout VALUE] [--skip_save_adapter] [--skip_save_init_ckpt]"
            echo "Defaults: Kaiming A/AB/AB_reset use lora_A_scale=0.5/lora_B_scale=0.5; AA uses 1/1."
            echo "Outputs are written under --output_root (default: ./log)."
            echo "Any init_para containing 'reset' enables base-weight subtraction. Use AB_reset for non-zero A/B init plus subtraction."
            echo "orthogonal_a2, eva_a2, and pissa_a2 use init_std as beta and target RMS(A)=beta/in_features."
            echo "Other non-Kaiming init_method values are passed through to PEFT; init_std and scale arguments are ignored."
            echo "eva and eva_a2 use --eva_max_samples examples for activation SVD; 0 means the full training set."
            echo "Use --eva_rho to control EVA rank redistribution; the matched-rank default is 1."
            echo "Use --log_coherence_stats plus --coherence_* options to log rank-coherence diagnostics under finetuned_result/coherence_stats."
            echo "Use --log_width_alignment plus --alignment_* options to test minibatch update-input scaling under finetuned_result/width_alignment."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
    shift
done

if [[ "${adapter_name}" != "lora" && "${adapter_name}" != "qlora" ]]; then
    echo "Invalid --adapter_name value: '${adapter_name}' (expected lora or qlora)"
    exit 1
fi

output_root="${output_root%/}"
if [[ -z "${output_root}" ]]; then
    echo "Invalid --output_root value"
    exit 1
fi

if [[ -z "${lora_A_scale}" ]]; then
    if [[ "${init_para}" == "AA" ]]; then
        lora_A_scale=1
    else
        lora_A_scale=0.5
    fi
fi

if [[ -z "${lora_B_scale}" ]]; then
    if [[ "${init_para}" == "AA" ]]; then
        lora_B_scale=1
    else
        lora_B_scale=0.5
    fi
fi

if [[ -z "${cutoff_len}" ]]; then cutoff_len=256; fi

model_tag="${model%/}"
model_tag="${model_tag##*/}"
if [[ -z "${model_tag}" ]]; then
    echo "Invalid --model value: '${model}'"
    exit 1
fi

if [[ "${dataset}" == "math_10k" ]]; then
    epoch=3
elif [[ "${dataset}" == "commonsense_170k" ]]; then
    epoch=1
else
    epoch=1
fi

if [[ -n "${num_epochs}" ]]; then
    epoch="${num_epochs}"
fi

lora_alpha=${alpha}
lora_A_std=${init_std}
lora_B_std=${init_std}
export WANDB_PROJECT="${WANDB_PROJECT:-lora-init-llm}"

init_method_key="${init_method,,}"
init_method_key="${init_method_key//-/_}"

if [[ "${init_para,,}" == *reset* || "${init_method_key}" == pissa* || "${init_method_key}" == olora ]]; then
    lora_dropout=0
fi

for lr in ${lrs}; do
    for lr_ratio in ${lr_ratios}; do
        if [[ "${init_method_key}" == "kaiming" ]]; then
            output_dir="${output_root}/${dataset}/${model_tag}_${adapter_name}_${init_method}_${init_para}_A${lora_A_scale}_B${lora_B_scale}_normal${lora_normal}_init_std${init_std}/${model_tag}_${adapter_name}_${init_para}_r${lora_r}_a${lora_alpha}_lr${lr}_ratio${lr_ratio}_s${seed}"
            export WANDB_RUN_NAME="${dataset}_${model_tag}_${adapter_name}_initmethod${init_method}_init${init_para}_A${lora_A_scale}_B${lora_B_scale}_r${lora_r}a${lora_alpha}_lr${lr}_ratio${lr_ratio}_s${seed}_init_std${init_std}_normal${lora_normal}"
            export WANDB_TAGS="${model_tag},${adapter_name},initmethod${init_method},${init_para},A${lora_A_scale},B${lora_B_scale},ratio${lr_ratio},lr${lr},seed${seed},init_std${init_std},normal${lora_normal}"
        elif [[ "${init_method_key}" == "orthogonal_a2" || "${init_method_key}" == "eva_a2" || "${init_method_key}" == "pissa_a2" ]]; then
            output_dir="${output_root}/${dataset}/${model_tag}_${adapter_name}_${init_method_key}_beta${init_std}/${model_tag}_${adapter_name}_${init_method_key}_beta${init_std}_r${lora_r}_a${lora_alpha}_lr${lr}_ratio${lr_ratio}_s${seed}"
            export WANDB_RUN_NAME="${dataset}_${model_tag}_${adapter_name}_initmethod${init_method_key}_beta${init_std}_r${lora_r}a${lora_alpha}_lr${lr}_ratio${lr_ratio}_s${seed}"
            export WANDB_TAGS="${model_tag},${adapter_name},initmethod${init_method_key},beta${init_std},ratio${lr_ratio},lr${lr},seed${seed}"
        else
            output_dir="${output_root}/${dataset}/${model_tag}_${adapter_name}_${init_method}/${model_tag}_${adapter_name}_${init_method}_r${lora_r}_a${lora_alpha}_lr${lr}_ratio${lr_ratio}_s${seed}"
            export WANDB_RUN_NAME="${dataset}_${model_tag}_${adapter_name}_initmethod${init_method}_r${lora_r}a${lora_alpha}_lr${lr}_ratio${lr_ratio}_s${seed}"
            export WANDB_TAGS="${model_tag},${adapter_name},initmethod${init_method},ratio${lr_ratio},lr${lr},seed${seed}"
        fi
        mkdir -p "${output_dir}/finetuned_result"

        finetune_extra_args=()
        if [[ "${skip_save_adapter}" == "1" ]]; then
            finetune_extra_args+=(--skip_save_adapter)
        fi
        if [[ "${skip_save_init_ckpt}" == "1" ]]; then
            finetune_extra_args+=(--skip_save_init_ckpt)
        fi
        if [[ "${log_coherence_stats}" == "1" ]]; then
            finetune_extra_args+=(--log_coherence_stats)
        fi
        if [[ -n "${coherence_probe_size}" ]]; then
            finetune_extra_args+=(--coherence_probe_size "${coherence_probe_size}")
        fi
        if [[ -n "${coherence_probe_seed}" ]]; then
            finetune_extra_args+=(--coherence_probe_seed "${coherence_probe_seed}")
        fi
        if [[ -n "${coherence_max_modules}" ]]; then
            finetune_extra_args+=(--coherence_max_modules "${coherence_max_modules}")
        fi
        if [[ -n "${coherence_max_tokens}" ]]; then
            finetune_extra_args+=(--coherence_max_tokens "${coherence_max_tokens}")
        fi
        if [[ -n "${coherence_max_rows}" ]]; then
            finetune_extra_args+=(--coherence_max_rows "${coherence_max_rows}")
        fi
        if [[ -n "${coherence_log_fractions}" ]]; then
            finetune_extra_args+=(--coherence_log_fractions "${coherence_log_fractions}")
        fi
        if [[ -n "${coherence_log_steps}" ]]; then
            finetune_extra_args+=(--coherence_log_steps "${coherence_log_steps}")
        fi
        if [[ -n "${coherence_every_n_steps}" ]]; then
            finetune_extra_args+=(--coherence_every_n_steps "${coherence_every_n_steps}")
        fi
        if [[ "${log_training_dynamics}" == "1" ]]; then
            finetune_extra_args+=(--log_training_dynamics)
        fi
        if [[ -n "${dynamics_output_dir}" ]]; then
            finetune_extra_args+=(--dynamics_output_dir "${dynamics_output_dir}")
        fi
        if [[ -n "${dynamics_max_modules}" ]]; then
            finetune_extra_args+=(--dynamics_max_modules "${dynamics_max_modules}")
        fi
        if [[ -n "${dynamics_log_fractions}" ]]; then
            finetune_extra_args+=(--dynamics_log_fractions "${dynamics_log_fractions}")
        fi
        if [[ -n "${dynamics_log_steps}" ]]; then
            finetune_extra_args+=(--dynamics_log_steps "${dynamics_log_steps}")
        fi
        if [[ -n "${dynamics_every_n_steps}" ]]; then
            finetune_extra_args+=(--dynamics_every_n_steps "${dynamics_every_n_steps}")
        fi
        if [[ "${log_feature_decomposition}" == "1" ]]; then
            finetune_extra_args+=(--log_feature_decomposition)
        fi
        if [[ -n "${decomposition_output_dir}" ]]; then
            finetune_extra_args+=(--decomposition_output_dir "${decomposition_output_dir}")
        fi
        if [[ -n "${decomposition_probe_size}" ]]; then
            finetune_extra_args+=(--decomposition_probe_size "${decomposition_probe_size}")
        fi
        if [[ -n "${decomposition_probe_seed}" ]]; then
            finetune_extra_args+=(--decomposition_probe_seed "${decomposition_probe_seed}")
        fi
        if [[ -n "${decomposition_max_modules}" ]]; then
            finetune_extra_args+=(--decomposition_max_modules "${decomposition_max_modules}")
        fi
        if [[ -n "${decomposition_max_tokens}" ]]; then
            finetune_extra_args+=(--decomposition_max_tokens "${decomposition_max_tokens}")
        fi
        if [[ -n "${decomposition_log_fractions}" ]]; then
            finetune_extra_args+=(--decomposition_log_fractions "${decomposition_log_fractions}")
        fi
        if [[ -n "${decomposition_log_steps}" ]]; then
            finetune_extra_args+=(--decomposition_log_steps "${decomposition_log_steps}")
        fi
        if [[ -n "${decomposition_every_n_steps}" ]]; then
            finetune_extra_args+=(--decomposition_every_n_steps "${decomposition_every_n_steps}")
        fi
        if [[ "${log_width_alignment}" == "1" ]]; then
            finetune_extra_args+=(--log_width_alignment)
        fi
        if [[ -n "${alignment_output_dir}" ]]; then
            finetune_extra_args+=(--alignment_output_dir "${alignment_output_dir}")
        fi
        if [[ -n "${alignment_module_pattern}" ]]; then
            finetune_extra_args+=(--alignment_module_pattern "${alignment_module_pattern}")
        fi
        if [[ -n "${alignment_max_modules}" ]]; then
            finetune_extra_args+=(--alignment_max_modules "${alignment_max_modules}")
        fi
        if [[ -n "${alignment_max_tokens}" ]]; then
            finetune_extra_args+=(--alignment_max_tokens "${alignment_max_tokens}")
        fi
        if [[ -n "${alignment_log_fractions}" ]]; then
            finetune_extra_args+=(--alignment_log_fractions "${alignment_log_fractions}")
        fi
        if [[ -n "${alignment_log_steps}" ]]; then
            finetune_extra_args+=(--alignment_log_steps "${alignment_log_steps}")
        fi
        if [[ -n "${alignment_every_n_steps}" ]]; then
            finetune_extra_args+=(--alignment_every_n_steps "${alignment_every_n_steps}")
        fi
        if [[ "${disable_alignment_shuffled_control}" == "1" ]]; then
            finetune_extra_args+=(--disable_alignment_shuffled_control)
        fi

        CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON_BIN}" scripts/finetune.py \
            --base_model "${model}" \
            --data_path "${LLM_ADAPTERS_DIR}/ft-training_set/${dataset}.json" \
            --output_dir "${output_dir}/finetuned_result" \
            --batch_size "${batch_size}" \
            --micro_batch_size "${micro_batch_size}" \
            --num_epochs "${epoch}" \
            --cutoff_len "${cutoff_len}" \
            --eval_step 500 \
            --save_step 500 \
            --logging_steps "${logging_steps}" \
            --target_modules "q_proj" "k_proj" "v_proj" "up_proj" "down_proj" \
            --seed "${seed}" \
            --val_set_size "${val_size}" \
            --adapter_name "${adapter_name}" \
            --lora_r "${lora_r}" \
            --lora_alpha "${lora_alpha}" \
            --learning_rate "${lr}" \
            --init_para "${init_para}" \
            --lora_A_std "${lora_A_std}" \
            --lora_B_std "${lora_B_std}" \
            --lora_A_scale "${lora_A_scale}" \
            --lora_B_scale "${lora_B_scale}" \
            --lora_normal "${lora_normal}" \
            --lora_dropout "${lora_dropout}" \
            --init_method "${init_method}" \
            --eva_max_samples "${eva_max_samples}" \
            --eva_rho "${eva_rho}" \
            --loraplus_lr_ratio "${lr_ratio}" \
            --bnb_4bit_compute_dtype "${bnb_4bit_compute_dtype}" \
            --bnb_4bit_quant_type "${bnb_4bit_quant_type}" \
            --bnb_4bit_use_double_quant "${bnb_4bit_use_double_quant}" \
            "${finetune_extra_args[@]}"

        if [[ "${skip_eval}" != "1" ]]; then
            bash "${SCRIPT_DIR}/1-eval.sh" "${model}" "${dataset}" "${adapter_name}" "${gpu}" "${seed}" "${output_dir}" "${init_para}"
        fi
    done
done
