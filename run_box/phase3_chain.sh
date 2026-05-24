#!/bin/bash
# Waits for the Phase 2 dispatcher (PID arg) to exit, then runs Phase 3 lm-eval on all 7 models.
set -u
DISPATCHER_PID="$1"
cd /workspace/riskKD
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(cat /workspace/.hf_home/token 2>/dev/null)
export OPENAI_API_KEY=$(cat /workspace/.openai_key 2>/dev/null)
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH=/workspace/riskKD
LOG=/workspace/riskKD/logs/phase3_chain.log
log(){ echo "[ph3 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "waiting for dispatcher PID=${DISPATCHER_PID} to exit..."
while kill -0 "$DISPATCHER_PID" 2>/dev/null; do sleep 60; done
log "dispatcher exited. Phase 2 done. Starting Phase 3."

LMH=/workspace/riskKD/.venv-vllm/bin/lm_eval
RES_ROOT=/workspace/riskKD/results/eval/qwen3-0.6b
mkdir -p "${RES_ROOT}/lm_eval"

MODELS=(
  "dpo|/workspace/riskKD/output/qwen3-0.6b-dpo"
  "riskkd|/workspace/riskKD/output/qwen3-0.6b-riskkd"
  "pfw_tail|/workspace/riskKD/output/qwen3-0.6b-pfw-tail"
  "dckd|/workspace/riskKD/output/qwen3-0.6b-dckd"
  "vanilakd|/workspace/riskKD/output/qwen3-0.6b-vanilakd"
  "adpa|/workspace/riskKD/output/qwen3-0.6b-adpa"
  "tvkd|/workspace/riskKD/output/qwen3-0.6b-tvkd"
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
  log "T1 lm-eval: ${NAME}"
  CUDA_VISIBLE_DEVICES=0,1 ${LMH} \
    --model vllm \
    --model_args "pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.85,max_model_len=4096" \
    --tasks hellaswag,arc_challenge,mmlu,truthfulqa_mc2,winogrande,gsm8k \
    --batch_size auto \
    --output_path "${OUT}" \
    > "/workspace/riskKD/logs/ph3_t1_${NAME}.log" 2>&1 && log "  ${NAME} OK" || log "  ${NAME} FAILED"
done

log "Phase 3 T1 (lm-eval) complete."
