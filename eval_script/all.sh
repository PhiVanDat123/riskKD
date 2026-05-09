#!/bin/bash
set -e

# Self-locate so the script works regardless of cwd or repo location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="${1:-$MODEL_NAME}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "Usage: bash all.sh <MODEL_NAME>"
  exit 1
fi
export MODEL_NAME

# Run all 6 benchmarks sequentially. Each inner script uses all 8 GPUs via vLLM
# tensor parallelism (TP=8). Sequential is required because TP=8 already binds
# every GPU per task.
bash "${SCRIPT_DIR}/mmlu.sh"
bash "${SCRIPT_DIR}/qa.sh"
bash "${SCRIPT_DIR}/wino.sh"
bash "${SCRIPT_DIR}/hellaswag.sh"
bash "${SCRIPT_DIR}/gsm8k.sh"
bash "${SCRIPT_DIR}/arc.sh"

echo "[DONE] All 6 benchmarks finished for ${MODEL_NAME}."
