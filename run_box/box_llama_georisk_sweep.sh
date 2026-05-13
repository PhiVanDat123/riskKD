#!/bin/bash
# Llama 3.2-1B GeoRiskKD sweep on 2×B200: 5 configurations targeting beat-42.45
# (the Llama riskKD baseline). Each config: train → vllm lm-eval → hf upload.
# Sequential, idempotent (per-stage skip on existing outputs).
# Log lives at logs/llama-georisk-sweep.log.
set -e
set -o pipefail
cd /workspace/riskKD

source .venv/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(cat /workspace/.hf_home/token 2>/dev/null)
export WANDB_API_KEY=$(cat /workspace/.wandb_key 2>/dev/null)
export WANDB_PROJECT=riskKD-llama
export WANDB_DIR=/workspace/riskKD
export HF_HUB_ENABLE_HF_TRANSFER=1
export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=0,1

ACC=recipes/accelerate_config/deepspeed_zero3_2_nooffload.yaml
VLLM_LM_EVAL=/workspace/riskKD/.venv-vllm/bin/lm_eval

mkdir -p logs results/eval

log(){ echo "[llama-sweep $(date +%H:%M:%S)] $*"; }
has_model(){ ls "$1"/model*.safetensors >/dev/null 2>&1; }

# Config table: tag | recipe | output_dir | hf_repo
# Run order: default first (v1 anchor) -> align-heavy -> target=all -> kl-conservative -> risk-only.
RUNS=(
  "risk|recipes/llama3.2-1b-deita-dpomix/georiskKD.yaml|/workspace/riskKD/output/llama-3.2-1b-georiskKD-risk|vukien2301/llama-3.2-1b-georiskKD-risk"
  "align-heavy|recipes/llama3.2-1b-deita-dpomix/georiskKD_align_heavy.yaml|/workspace/riskKD/output/llama-3.2-1b-georiskKD-align-heavy|vukien2301/llama-3.2-1b-georiskKD-align-heavy"
  "all|recipes/llama3.2-1b-deita-dpomix/georiskKD_targetall.yaml|/workspace/riskKD/output/llama-3.2-1b-georiskKD-all|vukien2301/llama-3.2-1b-georiskKD-all"
  "kl-conservative|recipes/llama3.2-1b-deita-dpomix/georiskKD_kl_conservative.yaml|/workspace/riskKD/output/llama-3.2-1b-georiskKD-kl-conservative|vukien2301/llama-3.2-1b-georiskKD-kl-conservative"
  "risk-only|recipes/llama3.2-1b-deita-dpomix/georiskKD_risk_only.yaml|/workspace/riskKD/output/llama-3.2-1b-georiskKD-risk-only|vukien2301/llama-3.2-1b-georiskKD-risk-only"
)

bench_model(){  # $1=model_path  $2=out_subdir
  local OUTD="results/eval/$2"
  mkdir -p "$OUTD"
  if ls "$OUTD/mmlu_full"/*/results*.json >/dev/null 2>&1 && ls "$OUTD/rest"/*/results*.json >/dev/null 2>&1; then
    log "$OUTD already complete (mmlu_full + rest) — skipping re-eval"; return 0
  fi
  if ! ls "$OUTD/mmlu_full"/*/results*.json >/dev/null 2>&1; then
    log "lm-eval (vllm) MMLU 5-shot: $1 -> $OUTD/mmlu_full"
    rm -rf "$OUTD/mmlu_full" 2>/dev/null || true
    CUDA_VISIBLE_DEVICES=0,1 "$VLLM_LM_EVAL" --model vllm \
      --model_args pretrained="$1",tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192 \
      --tasks mmlu --num_fewshot 5 --batch_size auto --output_path "$OUTD/mmlu_full" \
      || log "lm_eval mmlu exited non-zero (cosmetic make_table RecursionError) — verifying results.json"
    ls "$OUTD/mmlu_full"/*/results*.json >/dev/null 2>&1 || { log "ERROR: no mmlu results.json for $2"; return 1; }
  fi
  if ! ls "$OUTD/rest"/*/results*.json >/dev/null 2>&1; then
    log "lm-eval (vllm) REST (arc_25/hellaswag_10/truthfulqa_0/winogrande_5/gsm8k_5): $1 -> $OUTD/rest"
    rm -rf "$OUTD/rest" 2>/dev/null || true
    CUDA_VISIBLE_DEVICES=0,1 "$VLLM_LM_EVAL" --model vllm \
      --model_args pretrained="$1",tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192 \
      --gen_kwargs max_gen_toks=256 \
      --include_path eval_script --tasks lc_rest --batch_size auto --output_path "$OUTD/rest" \
      || log "lm_eval rest exited non-zero (cosmetic make_table RecursionError) — verifying results.json"
    ls "$OUTD/rest"/*/results*.json >/dev/null 2>&1 || { log "ERROR: no rest results.json for $2"; return 1; }
  fi
  log "lm-eval done: $2"
}

for entry in "${RUNS[@]}"; do
  IFS="|" read -r TAG RECIPE OUTDIR HFREPO <<< "$entry"
  EVAL_TAG="llama-3.2-1b-georiskKD-${TAG}"
  log "================ CONFIG: $TAG ================"
  log "recipe=$RECIPE  output=$OUTDIR  hf=$HFREPO"

  # Stage 1: train
  if has_model "$OUTDIR"; then
    log "[1/3] training output already exists at $OUTDIR — skipping"
  else
    log "[1/3] training..."
    python -m accelerate.commands.launch --config_file $ACC scripts/run_distill_dpo.py "$RECIPE" || {
      log "ERROR: training failed for $TAG — leaving log + skipping remaining stages"; continue
    }
    has_model "$OUTDIR" || { log "ERROR: no model shards in $OUTDIR after training"; continue; }
    log "[1/3] training done -> $OUTDIR"
  fi

  # Stage 2: lm-eval (vllm)
  log "[2/3] lm-eval (vllm) on $OUTDIR"
  bench_model "$OUTDIR" "$EVAL_TAG" || { log "ERROR: eval failed for $TAG"; continue; }

  # Stage 3: push to HF (model + eval artifacts)
  log "[3/3] push to HF ($HFREPO)"
  hf upload "$HFREPO" "$OUTDIR" --commit-message "GeoRiskKD ${TAG} — Llama 3.2-1B" 2>&1 | tail -3
  if [ -d "results/eval/$EVAL_TAG" ]; then
    hf upload "$HFREPO" "results/eval/$EVAL_TAG" "eval/" --commit-message "lm-eval (mmlu + rest) — ${TAG}" 2>&1 | tail -3
  fi
  log "================ DONE: $TAG -> https://huggingface.co/$HFREPO ================"
done

log "=========== SWEEP COMPLETE ==========="
log "Compare results against riskKD baseline avg = 42.45"
log "Eval JSONs: results/eval/llama-3.2-1b-georiskKD-*"
