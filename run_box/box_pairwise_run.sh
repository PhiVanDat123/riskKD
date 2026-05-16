#!/bin/bash
# Pairwise MT-Bench: anchor (Ours) vs each baseline, two judges.
# Reuses existing MT-Bench answer JSONLs - no model generation needed.
#
# Layout of outputs (under results/eval/mtbench-pair-v2/):
#   <judge>/<anchor>__vs__<baseline>.json
#   summary.json (aggregated win/tie/loss per pair per judge)
#
# Usage:
#   SMOKE=1 bash run_box/box_pairwise_run.sh                # 5 questions, 1 pair
#   bash run_box/box_pairwise_run.sh                        # full sweep
set -u -o pipefail
cd /workspace/riskKD
mkdir -p logs

source /workspace/riskKD/.venv-vllm/bin/activate
export OPENAI_API_KEY="$(cat /workspace/.openai_key)"
export ANTHROPIC_API_KEY="$(cat /workspace/.anthropic_key)"

SMOKE="${SMOKE:-0}"
NQ_FLAG=""
PARALLEL=10
if [ "$SMOKE" = "1" ]; then
  NQ_FLAG="--num-questions 5"
  PARALLEL=4
fi

ANSWER_DIR=/workspace/FastChat/fastchat/llm_judge/data/mt_bench/model_answer
QUESTION_FILE=/workspace/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
OUT_ROOT=results/eval/mtbench-pair-v2

JUDGES=(
  "claude-sonnet-4-6"
  "gpt-5.4-mini"
)

# anchor:baseline pairs (model_id of answer JSONL filename)
ANCHOR_QWEN="qwen3-1.7b-tailriskKD"
QWEN_BASELINES=(
  "qwen3-1.7b-ultrafeedback-dpo"
  "qwen3-1.7b-ultrafeedback-wpo"
  "qwen3-1.7b-ultrafeedback-radpo"
  "qwen3-1.7b-ultrafeedback-dckd"
  "qwen3-1.7b-ultrafeedback-adpa"
  "qwen3-1.7b-ultrafeedback-tvkd"
)
ANCHOR_LLAMA="llama3.2-1b-tailriskKD"
LLAMA_BASELINES=(
  "llama-3.2-1b-dckd-ultrafeedback"
  "llama-3.2-1b-adpa-ultrafeedback"
  "llama-3.2-1b-tvkd-ultrafeedback"
)

LOG=logs/pairwise_$(date +%Y%m%d_%H%M%S).log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_one() {
  local anchor="$1" baseline="$2" judge="$3"
  local out_dir="${OUT_ROOT}/${judge}"
  local out_file="${out_dir}/${anchor}__vs__${baseline}.json"
  mkdir -p "$out_dir"
  log "start: anchor=$anchor baseline=$baseline judge=$judge"
  python scripts/pairwise_judge.py \
    --anchor "$anchor" \
    --baseline "$baseline" \
    --judge "$judge" \
    --answer-dir "$ANSWER_DIR" \
    --question-file "$QUESTION_FILE" \
    --output "$out_file" \
    --parallel $PARALLEL \
    --resume \
    $NQ_FLAG \
    >> "logs/pairwise_${judge}_${anchor}_vs_${baseline}.log" 2>&1
  local rc=$?
  log "done: anchor=$anchor baseline=$baseline judge=$judge rc=$rc"
}

# Iterate: (anchor, baseline, judge) tuples
TOTAL=0
START=$(date +%s)
for judge in "${JUDGES[@]}"; do
  for baseline in "${QWEN_BASELINES[@]}"; do
    run_one "$ANCHOR_QWEN" "$baseline" "$judge"
    TOTAL=$((TOTAL+1))
    [ "$SMOKE" = "1" ] && break  # smoke: just one pair per judge
  done
  [ "$SMOKE" = "1" ] && continue
  for baseline in "${LLAMA_BASELINES[@]}"; do
    run_one "$ANCHOR_LLAMA" "$baseline" "$judge"
    TOTAL=$((TOTAL+1))
  done
done
DURATION=$(($(date +%s) - START))
log "ALL DONE: ran $TOTAL pair-judge sweeps in ${DURATION}s"
