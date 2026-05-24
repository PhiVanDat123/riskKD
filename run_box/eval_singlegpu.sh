#!/bin/bash
# Run lm-eval (TP=1) on GPU 1 for already-finished models, sequentially.
# Skips models whose eval results already exist.
set -u
cd /workspace/riskKD
export HF_HOME=/workspace/.hf_home
export PYTHONPATH=/workspace/riskKD
export PATH=/workspace/riskKD/.venv-vllm/bin:$PATH
LMH=/workspace/riskKD/.venv-vllm/bin/lm_eval
RES_ROOT=/workspace/riskKD/results/eval/qwen3-0.6b
mkdir -p "${RES_ROOT}/lm_eval"
LOG=/workspace/riskKD/logs/eval_singlegpu.log
log(){ echo "[evals $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Priority order: our methods first, then baselines
MODELS=(
  "pfw_tail|/workspace/riskKD/output/qwen3-0.6b-pfw-tail"
  "riskkd|/workspace/riskKD/output/qwen3-0.6b-riskkd"
  "dpo|/workspace/riskKD/output/qwen3-0.6b-dpo"
  "dckd|/workspace/riskKD/output/qwen3-0.6b-dckd"
  "vanilakd|/workspace/riskKD/output/qwen3-0.6b-vanilakd"
  "adpa|/workspace/riskKD/output/qwen3-0.6b-adpa"
)

for pair in "${MODELS[@]}"; do
  NAME="${pair%%|*}"; MODEL="${pair##*|}"
  OUT="${RES_ROOT}/lm_eval/${NAME}"
  if ls "${OUT}/results"*.json >/dev/null 2>&1; then
    log "T1 ${NAME} done already, skip"; continue
  fi
  if ! ls "${MODEL}"/model*.safetensors >/dev/null 2>&1; then
    log "T1 ${NAME} missing model, skip"; continue
  fi
  mkdir -p "$OUT"
  log "T1 lm-eval (TP=1, GPU 1): ${NAME}"
  CUDA_VISIBLE_DEVICES=1 ${LMH} \
    --model vllm \
    --model_args "pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.85,max_model_len=4096" \
    --tasks hellaswag,arc_challenge,mmlu,truthfulqa_mc2,winogrande,gsm8k \
    --batch_size auto \
    --output_path "${OUT}" \
    > "/workspace/riskKD/logs/ph3_t1_${NAME}.log" 2>&1 && log "  ${NAME} OK" || log "  ${NAME} FAILED"
done
log "eval_singlegpu chain done"
