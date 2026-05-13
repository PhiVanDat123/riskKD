#!/bin/bash
# Qwen3 GeoRiskKD (target=risk) end-to-end chain on the 2xB200 box:
#   1. accelerate.launch the riskKD trainer with georiskKD.yaml
#   2. on success: vllm lm-eval (MMLU 5-shot + the rest pack from eval_script/lc_rest.yaml)
#   3. on success: hf upload to vukien2301/qwen3-1.7b-georiskKD-risk
# Idempotent: each stage skips if its outputs already exist, so re-running picks up where
# it left off. Logs to this script's stdout (logs/georiskKD-risk-chain.log).
set -e
set -o pipefail
cd /workspace/riskKD

source .venv/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(cat /workspace/.hf_home/token 2>/dev/null)
export WANDB_API_KEY=$(cat /workspace/.wandb_key 2>/dev/null)
export WANDB_PROJECT=riskKD-qwen3
export WANDB_DIR=/workspace/riskKD
export HF_HUB_ENABLE_HF_TRANSFER=1
export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=0,1

ACC=recipes/accelerate_config/deepspeed_zero3_2_nooffload.yaml
RECIPE=recipes/qwen3-1.7b-ultrafeedback/georiskKD.yaml
OUTDIR=/workspace/riskKD/output/qwen3-1.7b-georiskKD-risk
HF_REPO=vukien2301/qwen3-1.7b-georiskKD-risk
EVAL_TAG=qwen3-1.7b-georiskKD-risk
VLLM_LM_EVAL=/workspace/riskKD/.venv-vllm/bin/lm_eval

mkdir -p logs results/eval

log(){ echo "[georisk-risk $(date +%H:%M:%S)] $*"; }
has_model(){ ls "$1"/model*.safetensors >/dev/null 2>&1; }

# --- Stage 1/3: train ---------------------------------------------------------
log "=== STAGE 1/3: train (recipe=$RECIPE) ==="
if has_model "$OUTDIR"; then
  log "training output already exists at $OUTDIR — skipping training"
else
  python -m accelerate.commands.launch --config_file $ACC scripts/run_distill_dpo.py "$RECIPE"
  has_model "$OUTDIR" || { log "ERROR: training finished without model shards in $OUTDIR"; exit 1; }
  log "training done -> $OUTDIR"
fi

# --- Stage 2/3: lm-eval via vllm ---------------------------------------------
log "=== STAGE 2/3: lm-eval (vllm) ==="
EVALD="results/eval/$EVAL_TAG"
mkdir -p "$EVALD"

# Two-call split to dodge lm-eval 0.4.12's nested-group bug on mmlu (same pattern as
# box_qwen_chain_a.sh). Each call tolerates the cosmetic make_table RecursionError but
# verifies the results.json was written before continuing.
if ls "$EVALD/mmlu_full"/*/results*.json >/dev/null 2>&1 && ls "$EVALD/rest"/*/results*.json >/dev/null 2>&1; then
  log "eval already complete (mmlu_full + rest) — skipping"
else
  if ! ls "$EVALD/mmlu_full"/*/results*.json >/dev/null 2>&1; then
    log "lm-eval (vllm) MMLU 5-shot -> $EVALD/mmlu_full"
    rm -rf "$EVALD/mmlu_full" 2>/dev/null || true
    CUDA_VISIBLE_DEVICES=0,1 "$VLLM_LM_EVAL" --model vllm \
      --model_args pretrained="$OUTDIR",tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192 \
      --tasks mmlu --num_fewshot 5 --batch_size auto --output_path "$EVALD/mmlu_full" \
      || log "lm_eval mmlu exited non-zero (cosmetic make_table RecursionError) — verifying results.json"
    ls "$EVALD/mmlu_full"/*/results*.json >/dev/null 2>&1 || { log "ERROR: no mmlu results.json"; exit 1; }
  fi
  if ! ls "$EVALD/rest"/*/results*.json >/dev/null 2>&1; then
    log "lm-eval (vllm) REST (arc_25 / hellaswag_10 / truthfulqa_0 / winogrande_5 / gsm8k_5) -> $EVALD/rest"
    rm -rf "$EVALD/rest" 2>/dev/null || true
    CUDA_VISIBLE_DEVICES=0,1 "$VLLM_LM_EVAL" --model vllm \
      --model_args pretrained="$OUTDIR",tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192 \
      --gen_kwargs max_gen_toks=256 \
      --include_path eval_script --tasks lc_rest --batch_size auto --output_path "$EVALD/rest" \
      || log "lm_eval rest exited non-zero (cosmetic make_table RecursionError) — verifying results.json"
    ls "$EVALD/rest"/*/results*.json >/dev/null 2>&1 || { log "ERROR: no rest results.json"; exit 1; }
  fi
  log "lm-eval done -> $EVALD"
fi

# --- Stage 3/3: push to HF ----------------------------------------------------
log "=== STAGE 3/3: push to HF ($HF_REPO) ==="
# hf upload is idempotent (skips unchanged blobs).
hf upload "$HF_REPO" "$OUTDIR" --commit-message "GeoRiskKD (target=risk) Qwen3 1.7B" 2>&1 | tail -5
# Also upload the eval results next to the model under eval/ so they live with the run.
if [ -d "$EVALD" ]; then
  hf upload "$HF_REPO" "$EVALD" "eval/" --commit-message "lm-eval (mmlu + rest)" 2>&1 | tail -3
fi
log "=== CHAIN COMPLETE === model + eval at https://huggingface.co/$HF_REPO"
