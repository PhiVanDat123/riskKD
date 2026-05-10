#!/bin/bash
# Same 6 benchmarks as all.sh, but in ONE lm_eval call so vLLM is initialized once
# (the per-task all.sh reloads weights + KV cache + NCCL 6 times). Per-task few-shot
# counts come from eval_script/leaderboard_custom.yaml: mmlu=5, truthfulqa_mc2=0,
# winogrande=5, hellaswag=10, gsm8k=5, arc_challenge=25 (Open-LLM-Leaderboard convention).
#
# Output: one results JSON under output/<MODEL_NAME>/all/ containing all tasks
# (vs all.sh's six output/<MODEL_NAME>/<task>/ dirs).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="${1:-$MODEL_NAME}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "Usage: bash all_oneshot.sh <MODEL_NAME>"
  exit 1
fi

export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export LM_EVAL_LOGLEVEL=DEBUG
export VLLM_LOGLEVEL=INFO

TP_SIZE=8
DTYPE="auto"
GPU_UTIL=0.8
BATCH_SIZE="auto:4"
MAX_LEN=4096

MODEL_ARGS="pretrained=${MODEL_NAME},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_UTIL},max_model_len=${MAX_LEN}"

lm_eval --model vllm \
        --model_args "${MODEL_ARGS}" \
        --include_path "${SCRIPT_DIR}" \
        --tasks leaderboard_custom \
        --batch_size "${BATCH_SIZE}" \
        --output_path "output/${MODEL_NAME}/all" \
        --log_samples \
        2>&1 | tee /tmp/lm_eval_oneshot.log

echo "[DONE] all 6 benchmarks (one vLLM load) finished for ${MODEL_NAME}."
