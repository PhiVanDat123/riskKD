#!/bin/bash
# Run pairwise MT-Bench judging from LOCAL (no box needed).
# Pure API workload (Anthropic + OpenAI), uses already-pulled MT-Bench answer JSONLs.
#
# Setup:
#   pip install openai anthropic
#   export OPENAI_API_KEY="<your key>"
#   export ANTHROPIC_API_KEY="<your key>"
#
# Usage:
#   bash scripts/run_pairwise_local.sh                       # full sweep
#   SMOKE=1 bash scripts/run_pairwise_local.sh               # 5 questions, one pair per judge
#
# Expected wall time: ~60 min (Sonnet 4.6 + gpt-5.4-mini, 9 pairs each).
# Expected cost: ~$11 ($10 Sonnet + $1 gpt-5.4-mini).
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE="${SMOKE:-0}"
NQ_FLAG=""
PARALLEL=10
if [ "$SMOKE" = "1" ]; then
  NQ_FLAG="--num-questions 5"
  PARALLEL=4
fi

ANSWER_DIR="results/eval/mtbench/raw/model_answer"
QUESTION_FILE="results/eval/mtbench/raw/question.jsonl"
OUT_ROOT="results/eval/mtbench-pair-v2"
mkdir -p logs

test -d "$ANSWER_DIR" || { echo "ERROR: $ANSWER_DIR missing (run rsync from box first)"; exit 1; }
test -f "$QUESTION_FILE" || { echo "ERROR: $QUESTION_FILE missing"; exit 1; }
test -n "${OPENAI_API_KEY:-}" || { echo "ERROR: OPENAI_API_KEY not set"; exit 1; }
test -n "${ANTHROPIC_API_KEY:-}" || { echo "ERROR: ANTHROPIC_API_KEY not set"; exit 1; }

JUDGES=("claude-sonnet-4-6" "gpt-5.4-mini")

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

LOG="logs/pairwise_local_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_one() {
  local anchor="$1" baseline="$2" judge="$3"
  local out_dir="${OUT_ROOT}/${judge}"
  local out_file="${out_dir}/${anchor}__vs__${baseline}.json"
  mkdir -p "$out_dir"
  log "start: anchor=$anchor baseline=$baseline judge=$judge"
  python3 scripts/pairwise_judge.py \
    --anchor "$anchor" --baseline "$baseline" --judge "$judge" \
    --answer-dir "$ANSWER_DIR" --question-file "$QUESTION_FILE" \
    --output "$out_file" --parallel $PARALLEL --resume $NQ_FLAG \
    >> "logs/pairwise_local_${judge}_${anchor}_vs_${baseline}.log" 2>&1
  log "done: anchor=$anchor baseline=$baseline judge=$judge rc=$?"
}

TOTAL=0
START=$(date +%s)
for judge in "${JUDGES[@]}"; do
  for baseline in "${QWEN_BASELINES[@]}"; do
    run_one "$ANCHOR_QWEN" "$baseline" "$judge"
    TOTAL=$((TOTAL+1))
    [ "$SMOKE" = "1" ] && break
  done
  [ "$SMOKE" = "1" ] && continue
  for baseline in "${LLAMA_BASELINES[@]}"; do
    run_one "$ANCHOR_LLAMA" "$baseline" "$judge"
    TOTAL=$((TOTAL+1))
  done
done
DURATION=$(($(date +%s) - START))
log "ALL DONE: ran $TOTAL pair-judge sweeps in ${DURATION}s"
