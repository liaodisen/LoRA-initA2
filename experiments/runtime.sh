#!/usr/bin/env bash
# Shared environment and submission helper. Source after setting REPO_ROOT.
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="$(command -v python)"; fi
export PYTHON_BIN
export LORA_REPO_ROOT="${REPO_ROOT}"
export LLM_ADAPTERS_DIR="${LLM_ADAPTERS_DIR:-${REPO_ROOT}/LLM-Adapters-main}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

submit_experiment() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'DRY_RUN '
        printf '%q ' "$@"
        printf '\n'
    elif [[ "${RUN_LOCAL:-0}" == "1" ]]; then
        # Strip sbatch and its resource options; invoke the same job script.
        while [[ $# -gt 0 && "$1" != *.sh ]]; do shift; done
        if [[ $# -eq 0 ]]; then echo "No job script found" >&2; return 2; fi
        bash "$@"
    else
        "$@"
    fi
}
