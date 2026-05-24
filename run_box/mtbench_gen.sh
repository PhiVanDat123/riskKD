#!/bin/bash
set -u
cd /workspace/riskKD
export HF_HOME=/workspace/.hf_home
export PYTHONPATH=/workspace/riskKD
export PATH=/workspace/riskKD/.venv-vllm/bin:$PATH
QF=/workspace/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
AD=/workspace/FastChat/fastchat/llm_judge/data/mt_bench/model_answer
mkdir -p "$AD"
LOG=/workspace/riskKD/logs/mtbench_gen.log
log(){ echo "[mtg $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

MODELS=( pfw_tail riskkd dpo dckd vanilakd adpa )
for m in "${MODELS[@]}"; do
  REPO=/workspace/riskKD/output/qwen3-0.6b-${m//_/-}
  OUT="$AD/qwen3-0.6b-${m}.jsonl"
  if [ -f "$OUT" ] && [ $(wc -l < "$OUT") -ge 80 ]; then
    log "skip $m (already $(wc -l < "$OUT") answers)"; continue
  fi
  if ! ls "$REPO"/model*.safetensors >/dev/null 2>&1; then
    log "skip $m (no model.safetensors)"; continue
  fi
  log "gen $m"
  CUDA_VISIBLE_DEVICES=1 /workspace/riskKD/.venv-vllm/bin/python scripts/mtbench_gen_vllm.py \
    --model-path "$REPO" \
    --model-id "qwen3-0.6b-${m}" \
    --question-file "$QF" \
    --answer-file "$OUT" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-new-token 1024 \
    --gpu-memory-utilization 0.85 \
    >> /workspace/riskKD/logs/mtbench_gen_${m}.log 2>&1 && log "  ${m} OK (lines: $(wc -l < "$OUT"))" || log "  ${m} FAILED"
done
log "mtbench_gen chain done"
