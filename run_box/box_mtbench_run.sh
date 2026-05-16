#!/bin/bash
# MT-Bench orchestrator for the 25-model sweep on 2xB200.
# Pairs models from the model list, runs answer-gen on GPU 0 and GPU 1 in parallel,
# then judges each completed model with GPT-4 in the background and updates scoreboard JSON.
#
# Usage:
#   bash run_box/box_mtbench_run.sh                                  # full run
#   SMOKE=1 bash run_box/box_mtbench_run.sh                          # 2 questions, judge with parallel=2
#   bash run_box/box_mtbench_run.sh eval_script/mtbench_subset.txt   # custom list
#
# Reads tokens from /workspace/.hf_home/token and /workspace/.openai_key.

set -u -o pipefail
cd /workspace/riskKD
mkdir -p logs results/eval/mtbench
LOG=logs/mtbench_$(date +%Y%m%d_%H%M%S).log

source /workspace/riskKD/.venv-vllm/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN="$(cat /workspace/.hf_home/token)"
export OPENAI_API_KEY="$(cat /workspace/.openai_key)"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# We import fastchat from the cloned source, not a pip install
export PYTHONPATH="/workspace/FastChat${PYTHONPATH:+:$PYTHONPATH}"

MODEL_LIST="${1:-eval_script/mtbench_model_list.txt}"
SMOKE="${SMOKE:-0}"
MODE="${MODE:-single}"                 # single | pairwise-baseline
BASELINE_MODEL="${BASELINE_MODEL:-gpt-3.5-turbo}"
FC_DIR=/workspace/FastChat
QUESTION_FILE="${FC_DIR}/fastchat/llm_judge/data/mt_bench/question.jsonl"
ANSWER_DIR="${FC_DIR}/fastchat/llm_judge/data/mt_bench/model_answer"
JUDGMENT_DIR="${FC_DIR}/fastchat/llm_judge/data/mt_bench/model_judgment"
if [ "$MODE" = "pairwise-baseline" ]; then
  RESULTS_DIR=results/eval/mtbench-pair
  JUDGMENT_FILE="${JUDGMENT_DIR}/gpt-4_pair.jsonl"
  SCORE_SCRIPT=scripts/mtbench_pair_score_to_json.py
else
  RESULTS_DIR=results/eval/mtbench
  JUDGMENT_FILE="${JUDGMENT_DIR}/gpt-4_single.jsonl"
  SCORE_SCRIPT=scripts/mtbench_score_to_json.py
fi
SCOREBOARD="${RESULTS_DIR}/scores.json"
mkdir -p "${ANSWER_DIR}" "${JUDGMENT_DIR}" "${RESULTS_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

NQ_FLAG=""
PARALLEL_FLAG="--parallel 8"
JUDGE_FIRST_N=""
if [ "$SMOKE" = "1" ]; then
  # Gen still produces all 80 answers (check_data requires it), but judge only first 2.
  PARALLEL_FLAG="--parallel 2"
  JUDGE_FIRST_N="--first-n 2"
fi

mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "$MODEL_LIST")
log "loaded ${#MODELS[@]} models from $MODEL_LIST (SMOKE=$SMOKE, MODE=$MODE)"

score_json_path() {
  local repo="$1" fam safe
  case "$repo" in
    *qwen3-1.7b*) fam=qwen3-1.7b;;
    *qwen3-8b*) fam=qwen3-8b;;
    *llama-3.2-1b*|*llama3.2-1b*) fam=llama-3.2-1b;;
    *llama-3.1-8b*) fam=llama-3.1-8b;;
    *) fam=other;;
  esac
  safe="${repo//\//__}"
  echo "${RESULTS_DIR}/${fam}/per_model/${safe}.json"
}

gen_one() {
  local repo="$1" gpu="$2"
  local model_id="${repo##*/}"
  local answer_file="${ANSWER_DIR}/${model_id}.jsonl"
  log "gen[GPU $gpu] start ${repo} -> ${model_id}.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/mtbench_gen_vllm.py \
    --model-path "$repo" \
    --model-id "$model_id" \
    --question-file "$QUESTION_FILE" \
    --answer-file "$answer_file" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-new-token 1024 \
    --gpu-memory-utilization 0.85 \
    $NQ_FLAG \
    >> "logs/mtbench_gen_${model_id}.log" 2>&1
  local rc=$?
  log "gen[GPU $gpu] done ${model_id} rc=$rc"
  return $rc
}

judge_one() {
  local repo="$1"
  local model_id="${repo##*/}"
  log "judge start ${model_id} (mode=${MODE})"
  # gen_judgment.py has an `input("Press Enter to confirm...")` that needs a real
  # newline (not EOF, which raises EOFError in Python's input()). Use printf '\n'.
  if [ "$MODE" = "pairwise-baseline" ]; then
    ( cd "$FC_DIR/fastchat/llm_judge" && \
      printf '\n' | python gen_judgment.py \
        --bench-name mt_bench \
        --model-list "$model_id" \
        --judge-model gpt-4 \
        --mode pairwise-baseline \
        --baseline-model "$BASELINE_MODEL" \
        $PARALLEL_FLAG $JUDGE_FIRST_N ) \
      >> "/workspace/riskKD/logs/mtbench_judge_pair_${model_id}.log" 2>&1
    local rc=$?
    log "judge done ${model_id} rc=$rc"
    if [ $rc -ne 0 ]; then return $rc; fi
    python "$SCORE_SCRIPT" \
      --model-id "$model_id" \
      --repo-id "$repo" \
      --baseline-model "$BASELINE_MODEL" \
      --judgment-file "${JUDGMENT_FILE}" \
      --question-file "$QUESTION_FILE" \
      --judge-model gpt-4 \
      --out-dir "${RESULTS_DIR}" \
      >> "logs/mtbench_score_pair_${model_id}.log" 2>&1
  else
    ( cd "$FC_DIR/fastchat/llm_judge" && \
      printf '\n' | python gen_judgment.py \
        --bench-name mt_bench \
        --model-list "$model_id" \
        --judge-model gpt-4 \
        --mode single \
        $PARALLEL_FLAG $JUDGE_FIRST_N ) \
      >> "/workspace/riskKD/logs/mtbench_judge_${model_id}.log" 2>&1
    local rc=$?
    log "judge done ${model_id} rc=$rc"
    if [ $rc -ne 0 ]; then return $rc; fi
    python "$SCORE_SCRIPT" \
      --model-id "$model_id" \
      --repo-id "$repo" \
      --judgment-file "${JUDGMENT_FILE}" \
      --question-file "$QUESTION_FILE" \
      --judge-model gpt-4 \
      --out-dir "${RESULTS_DIR}" \
      >> "logs/mtbench_score_${model_id}.log" 2>&1
  fi
  log "score done ${model_id}"
}

i=0
while [ $i -lt ${#MODELS[@]} ]; do
  m0="${MODELS[$i]}"
  m1="${MODELS[$((i + 1))]:-}"

  if [ "$SMOKE" != "1" ] && [ -f "$(score_json_path "$m0")" ]; then
    log "SKIP $m0 (already scored)"; m0=""
  fi
  if [ -n "$m1" ] && [ "$SMOKE" != "1" ] && [ -f "$(score_json_path "$m1")" ]; then
    log "SKIP $m1 (already scored)"; m1=""
  fi

  pids=()
  if [ -n "$m0" ]; then
    gen_one "$m0" 0 & pids+=("$!|0|$m0")
  fi
  if [ -n "$m1" ]; then
    gen_one "$m1" 1 & pids+=("$!|1|$m1")
  fi

  for info in "${pids[@]}"; do
    pid="${info%%|*}"; rest="${info#*|}"; gpu="${rest%%|*}"; repo="${rest#*|}"
    wait "$pid"; rc=$?
    if [ $rc -eq 0 ]; then
      log "queueing judge for $repo"
      judge_one "$repo" &
    else
      log "FAILED gen for $repo (rc=$rc) — not queueing judge"
    fi
  done

  i=$((i + 2))
done

log "all gen rounds done; waiting for outstanding judges"
wait
log "all done."
if [ -f "$SCOREBOARD" ]; then
  python -c "
import json
d = json.load(open('$SCOREBOARD'))
ms = d['models']
print(f\"\\nScoreboard: {len(ms)} models scored\")
for k, v in sorted(ms.items(), key=lambda x: -(x[1]['overall'] or 0)):
    print(f\"  {v['overall']:.3f}  {k}\")"
fi
