#!/bin/bash
# AlpacaEval 2 (LC) orchestrator for the 25-model sweep on 2xB200.
# Same pattern as box_mtbench_run.sh: pairs of models on GPU 0/1, ours-first.
# Per-model: vLLM gen (805 prompts) → alpaca_eval CLI (GPT-4-turbo judge) → score JSON.
#
# Usage:
#   bash run_box/box_alpaca_run.sh                                # full run
#   SMOKE=1 bash run_box/box_alpaca_run.sh                        # 5 prompts/model, single-model
#   bash run_box/box_alpaca_run.sh eval_script/alpaca_subset.txt  # custom list
set -u -o pipefail
cd /workspace/riskKD
mkdir -p logs results/eval/alpaca
LOG=logs/alpaca_$(date +%Y%m%d_%H%M%S).log

source /workspace/riskKD/.venv-vllm/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN="$(cat /workspace/.hf_home/token)"
export OPENAI_API_KEY="$(cat /workspace/.openai_key)"
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_LIST="${1:-eval_script/alpaca_model_list.txt}"
SMOKE="${SMOKE:-0}"
ANNOTATOR="${ANNOTATOR:-weighted_alpaca_eval_gpt-4o-mini-2024-07-18}"  # LC v2 with gpt-4o-mini judge (cheap)
ALPACA_OUT_DIR=/workspace/alpaca_outputs           # per-model JSON + alpaca-eval workspace
MODEL_OUTPUTS_DIR="${ALPACA_OUT_DIR}/model_outputs"
EVAL_RESULTS_DIR="${ALPACA_OUT_DIR}/eval_results"
SCOREBOARD=results/eval/alpaca/scores.json
mkdir -p "${MODEL_OUTPUTS_DIR}" "${EVAL_RESULTS_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

NQ_FLAG=""
if [ "$SMOKE" = "1" ]; then
  NQ_FLAG="--num-prompts 5"
fi

mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "$MODEL_LIST")
log "loaded ${#MODELS[@]} models from $MODEL_LIST (SMOKE=$SMOKE, annotator=$ANNOTATOR)"

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
  echo "results/eval/alpaca/${fam}/per_model/${safe}.json"
}

gen_one() {
  local repo="$1" gpu="$2"
  local model_id="${repo##*/}"
  local output_file="${MODEL_OUTPUTS_DIR}/${model_id}.json"
  log "gen[GPU $gpu] start ${repo} -> ${model_id}.json"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/alpaca_gen_vllm.py \
    --model-path "$repo" \
    --model-id "$model_id" \
    --output-file "$output_file" \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-new-token 2048 \
    --gpu-memory-utilization 0.85 \
    $NQ_FLAG \
    >> "logs/alpaca_gen_${model_id}.log" 2>&1
  local rc=$?
  log "gen[GPU $gpu] done ${model_id} rc=$rc"
  return $rc
}

eval_one() {
  local repo="$1"
  local model_id="${repo##*/}"
  local output_file="${MODEL_OUTPUTS_DIR}/${model_id}.json"
  local results_dir="${EVAL_RESULTS_DIR}/${model_id}"
  log "eval start ${model_id} (annotator=$ANNOTATOR)"
  alpaca_eval --model_outputs "$output_file" \
    --annotators_config "$ANNOTATOR" \
    --output_path "$results_dir" \
    >> "logs/alpaca_eval_${model_id}.log" 2>&1
  local rc=$?
  log "eval done ${model_id} rc=$rc"
  if [ $rc -ne 0 ]; then return $rc; fi
  python scripts/alpaca_score_to_json.py \
    --model-id "$model_id" \
    --repo-id "$repo" \
    --leaderboard-csv "${results_dir}/${ANNOTATOR}/leaderboard.csv" \
    --annotator "$ANNOTATOR" \
    --out-dir results/eval/alpaca \
    >> "logs/alpaca_score_${model_id}.log" 2>&1
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
  if [ -n "$m0" ]; then gen_one "$m0" 0 & pids+=("$!|0|$m0"); fi
  if [ -n "$m1" ]; then gen_one "$m1" 1 & pids+=("$!|1|$m1"); fi

  for info in "${pids[@]}"; do
    pid="${info%%|*}"; rest="${info#*|}"; gpu="${rest%%|*}"; repo="${rest#*|}"
    wait "$pid"; rc=$?
    if [ $rc -eq 0 ]; then
      log "queueing eval for $repo"
      eval_one "$repo" &
    else
      log "FAILED gen for $repo (rc=$rc) — not queueing eval"
    fi
  done

  i=$((i + 2))
done

log "all gen rounds done; waiting for outstanding evals"
wait
log "all done."
if [ -f "$SCOREBOARD" ]; then
  python -c "
import json
d = json.load(open('$SCOREBOARD'))
ms = d['models']
print(f'\\nScoreboard: {len(ms)} models')
for k, v in sorted(ms.items(), key=lambda x: -(x[1].get('lc_win_rate') or 0)):
    print(f\"  lc={v['lc_win_rate']:.2f}  wr={v['win_rate']:.2f}  {k}\")"
fi
