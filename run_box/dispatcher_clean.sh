#!/bin/bash
set -u
cd /workspace/riskKD
LOG=/workspace/riskKD/logs/dispatcher.log
log(){ echo "[disp $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

METHODS=( riskKD pfw_tail dckd vanilakd adpa tvkd )
declare -A OUTDIR=(
  [riskKD]=/workspace/riskKD/output/qwen3-0.6b-riskkd
  [pfw_tail]=/workspace/riskKD/output/qwen3-0.6b-pfw-tail
  [dckd]=/workspace/riskKD/output/qwen3-0.6b-dckd
  [vanilakd]=/workspace/riskKD/output/qwen3-0.6b-vanilakd
  [adpa]=/workspace/riskKD/output/qwen3-0.6b-adpa
  [tvkd]=/workspace/riskKD/output/qwen3-0.6b-tvkd
)

declare -A GPU_PID=( [0]="" [1]="" )
declare -A GPU_METHOD=( [0]="" [1]="" )

next_method(){
  for M in "${METHODS[@]}"; do
    local OUT="${OUTDIR[$M]}"
    if ls "$OUT"/model*.safetensors >/dev/null 2>&1; then continue; fi
    if [ "${GPU_METHOD[0]}" = "$M" ] || [ "${GPU_METHOD[1]}" = "$M" ]; then continue; fi
    echo "$M"; return
  done
  echo ""
}

launch(){
  local M="$1" G="$2"
  local RECIPE="recipes/qwen3-0.6b-ultrafeedback/${M}.yaml"
  local OUT="${OUTDIR[$M]}"
  local TLOG="logs/ph2_${M}.log"
  log "GPU $G launching $M"
  rm -rf "$OUT"
  # CRITICAL: explicit env vars before each python launch (venv activate clobbers PYTHONPATH)
  CUDA_VISIBLE_DEVICES="$G" \
    HF_HOME=/workspace/.hf_home \
    HF_TOKEN=$(cat /workspace/.hf_home/token 2>/dev/null) \
    WANDB_API_KEY=$(cat /workspace/.wandb_key 2>/dev/null) \
    WANDB_PROJECT=riskKD-qwen3-0.6b \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=/workspace/riskKD \
    /workspace/riskKD/.venv/bin/python -m accelerate.commands.launch \
      --config_file recipes/accelerate_config/single_gpu_nods.yaml \
      scripts/run_distill_dpo.py "$RECIPE" > "$TLOG" 2>&1 &
  GPU_PID[$G]=$!
  GPU_METHOD[$G]=$M
  log "GPU $G PID=${GPU_PID[$G]}"
}

# SAFETY: if a launch fails (process exits with rc!=0), DO NOT auto-retry. Mark as failed and move on.
declare -A FAILED=()

log "dispatcher start; queue: ${METHODS[*]}"
for G in 0 1; do
  M=$(next_method)
  [ -n "$M" ] && launch "$M" "$G"
done

while true; do
  ANY_RUNNING=false
  for G in 0 1; do
    PID="${GPU_PID[$G]}"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
      ANY_RUNNING=true; continue
    fi
    if [ -n "$PID" ]; then
      wait "$PID" 2>/dev/null; RC=$?
      M="${GPU_METHOD[$G]}"
      log "GPU $G freed -- $M rc=$RC"
      if [ $RC -ne 0 ] && ! ls "${OUTDIR[$M]}"/model*.safetensors >/dev/null 2>&1; then
        FAILED[$M]=1
        log "  MARKED $M as failed; will not retry"
      fi
      GPU_PID[$G]=""; GPU_METHOD[$G]=""
    fi
    # find next that is not in FAILED
    M=""
    for cand in "${METHODS[@]}"; do
      [ -n "${FAILED[$cand]:-}" ] && continue
      if ls "${OUTDIR[$cand]}"/model*.safetensors >/dev/null 2>&1; then continue; fi
      if [ "${GPU_METHOD[0]}" = "$cand" ] || [ "${GPU_METHOD[1]}" = "$cand" ]; then continue; fi
      M=$cand; break
    done
    [ -n "$M" ] && { launch "$M" "$G"; ANY_RUNNING=true; }
  done
  [ "$ANY_RUNNING" = false ] && { log "dispatcher done"; break; }
  sleep 30
done

log "FAILED methods: ${!FAILED[*]:-(none)}"
