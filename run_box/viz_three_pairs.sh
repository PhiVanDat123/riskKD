#!/bin/bash
# Generate PFW heatmap viz for 3 teacher-student pairs:
#   1) qwen3_big   : Qwen3-8B (DPO teacher)      -> Qwen3-1.7B  pfw_tail
#   2) qwen3_small : Qwen3-1.7B (DPO teacher)    -> Qwen3-0.6B  pfw_tail
#   3) llama       : Llama-3.1-8B (DPO teacher)  -> Llama-3.2-1B pfw_tail
# Pair 1 on GPU 0, pair 3 on GPU 1 in parallel, pair 2 second on the first
# free GPU.
set -u
cd /workspace/riskKD
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(cat /workspace/.hf_home/token)
export PYTHONPATH=/workspace/riskKD
export PATH=/workspace/riskKD/.venv/bin:$PATH
PY=/workspace/riskKD/.venv/bin/python
LOG=/workspace/riskKD/logs/viz_three_pairs.log
mkdir -p logs results/viz/pfw_pairs
log(){ echo "[viz $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_pair(){
  local NAME="$1" TEACHER="$2" STUDENT="$3" GPU="$4"
  local OUT="results/viz/pfw_pairs/${NAME}"
  log "PAIR ${NAME} on GPU ${GPU}: ${STUDENT}  <-  ${TEACHER}"
  CUDA_VISIBLE_DEVICES=${GPU} ${PY} scripts/visualize_pfw_weights.py \
    --teacher "${TEACHER}" \
    --student "${STUDENT}" \
    --num_examples 24 \
    --max_length 512 --max_prompt_length 256 \
    --num_text_examples 4 \
    --out_dir "${OUT}" \
    > "logs/viz_${NAME}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "  ${NAME} FAILED rc=${rc} - see logs/viz_${NAME}.log"; return $rc; fi
  log "  ${NAME} OK"
}

(
  run_pair qwen3_big \
    vukien2301/qwen3-8b-ultrafeedback-dpo-teacher \
    vukien2301/qwen3-1.7b-pfw-tail \
    0
  # GPU 0 freed -> run qwen3_small here second
  run_pair qwen3_small \
    /workspace/riskKD/output/qwen3-1.7b-ultrafeedback-dpo-teacher \
    /workspace/riskKD/output/qwen3-0.6b-pfw-tail \
    0
) &
PID0=$!

run_pair llama \
  vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1 \
  vukien2301/llama3.2-1b-pfw-tail \
  1 &
PID1=$!

wait $PID0; wait $PID1
log "all 3 pairs done"
