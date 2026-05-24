#!/bin/bash
# Pairwise MT-Bench: pfw_tail and riskkd anchors vs each baseline. Judge: gpt-5.4-mini.
set -u
cd /workspace/riskKD
export OPENAI_API_KEY=$(cat /workspace/.openai_key 2>/dev/null)
export PYTHONPATH=/workspace/riskKD
QF=/workspace/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
AD=/workspace/FastChat/fastchat/llm_judge/data/mt_bench/model_answer
OUT_ROOT=/workspace/riskKD/results/eval/qwen3-0.6b/mt_bench_pairwise
mkdir -p "$OUT_ROOT/gpt-5.4-mini"
LOG=/workspace/riskKD/logs/mtbench_judge.log
log(){ echo "[mtj $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

ANCHORS=( pfw_tail riskkd )
BASELINES=( dpo dckd vanilakd adpa tvkd )
JUDGE=gpt-5.4-mini

for anchor in "${ANCHORS[@]}"; do
  for baseline in "${BASELINES[@]}"; do
    OUT="$OUT_ROOT/$JUDGE/qwen3-0.6b-${anchor}__vs__qwen3-0.6b-${baseline}.json"
    if [ -f "$OUT" ]; then log "skip $anchor vs $baseline ($OUT exists)"; continue; fi
    log "judging $anchor vs $baseline"
    /workspace/riskKD/.venv-vllm/bin/python scripts/pairwise_judge.py \
      --anchor "qwen3-0.6b-${anchor}" \
      --baseline "qwen3-0.6b-${baseline}" \
      --judge "$JUDGE" \
      --answer-dir "$AD" \
      --question-file "$QF" \
      --output "$OUT" \
      --parallel 8 \
      --resume \
      >> "/workspace/riskKD/logs/pairwise_${JUDGE}_${anchor}_vs_${baseline}.log" 2>&1 \
      && log "  ${anchor} vs ${baseline} OK" || log "  ${anchor} vs ${baseline} FAILED"
  done
done
log "mtbench_judge chain done"
