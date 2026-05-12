#!/bin/bash
# vLLM eval — the 6-task Open-LLM-Leaderboard suite in ONE lm_eval call, adapted for a
# 2x B200 box (tensor_parallel_size=2, CUDA_VISIBLE_DEVICES=0,1). Mirrors
# eval_script/all_oneshot.sh (which is TP=8 / GPUs 0..7) and reuses the same
# eval_script/leaderboard_custom.yaml task group:
#   mmlu(5-shot) / truthfulqa_mc2(0) / winogrande(5) / hellaswag(10) / gsm8k(5) / arc_challenge(25)
#
# vLLM pins torch tightly, so it lives in its own venv (.venv-vllm) — kept separate from
# the main .venv (torch 2.8.0+cu128). Falls back to .venv if .venv-vllm is absent.
#
# Usage: bash eval_script/all_oneshot_b200.sh <MODEL_PATH_OR_HF_ID> [OUTPUT_DIR]
#   e.g. bash eval_script/all_oneshot_b200.sh /workspace/riskKD/output/TVKD
#        bash eval_script/all_oneshot_b200.sh vukien2301/llama-3.2-1b-tvkd-ultrafeedback
# Overridable via env: TP_SIZE, DTYPE, GPU_UTIL, MAX_LEN, BATCH_SIZE, CUDA_VISIBLE_DEVICES.
set -e
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${1:?Usage: bash all_oneshot_b200.sh <model_path_or_hf_id> [output_dir]}"
OUT_DIR="${2:-${REPO_DIR}/results/eval/$(basename "${MODEL_NAME}")_vllm}"

if [[ -x "${REPO_DIR}/.venv-vllm/bin/python" ]]; then
  source "${REPO_DIR}/.venv-vllm/bin/activate"
  echo "[eval-vllm] using ${REPO_DIR}/.venv-vllm"
else
  source "${REPO_DIR}/.venv/bin/activate"
  echo "[eval-vllm] WARNING: .venv-vllm not found, using main .venv (vllm may not be installed there)"
fi

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
[[ -f /workspace/.hf_home/token ]] && export HF_TOKEN="$(cat /workspace/.hf_home/token)"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

TP_SIZE="${TP_SIZE:-2}"
DTYPE="${DTYPE:-bfloat16}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

MODEL_ARGS="pretrained=${MODEL_NAME},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_UTIL},max_model_len=${MAX_LEN}"

echo "[eval-vllm] model=${MODEL_NAME}  TP=${TP_SIZE}  GPUs=${CUDA_VISIBLE_DEVICES}  out=${OUT_DIR}"
lm_eval --model vllm \
        --model_args "${MODEL_ARGS}" \
        --include_path "${SCRIPT_DIR}" \
        --tasks leaderboard_custom \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${OUT_DIR}" \
        --log_samples
echo "[eval-vllm] done -> ${OUT_DIR}"
