#!/bin/bash
# lm-eval for the three baseline-of-baselines models on the Qwen3-0.6B campaign:
#   teacher_sft : vukien2301/qwen3-1.7b-deita-sft-student (HF Hub, cached)
#   teacher_dpo : /workspace/riskKD/output/qwen3-1.7b-ultrafeedback-dpo-teacher
#   student_sft : /workspace/riskKD/output/qwen3-0.6b-deita-sft-student
# Runs:
#   GPU 0 -> teacher_dpo then teacher_sft  (both ~1.7B; sequential)
#   GPU 1 -> student_sft                   (0.6B; fast)
set -u
cd /workspace/riskKD
export HF_HOME=/workspace/.hf_home
export PYTHONPATH=/workspace/riskKD
export PATH=/workspace/riskKD/.venv-vllm/bin:$PATH
LMH=/workspace/riskKD/.venv-vllm/bin/lm_eval
RES_ROOT=/workspace/riskKD/results/eval/qwen3-0.6b/lm_eval
mkdir -p "${RES_ROOT}"
LOG=/workspace/riskKD/logs/eval_baseline_models.log
log(){ echo "[evbl $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

TASKS=hellaswag,arc_challenge,mmlu,truthfulqa_mc2,winogrande,gsm8k

run_one(){
  local NAME="$1" MODEL="$2" GPU="$3"
  local OUT="${RES_ROOT}/${NAME}"
  if ls "${OUT}/results"*.json >/dev/null 2>&1; then
    log "${NAME} done already, skip"; return 0
  fi
  mkdir -p "$OUT"
  log "lm-eval GPU ${GPU}: ${NAME}  (model: ${MODEL})"
  CUDA_VISIBLE_DEVICES=${GPU} ${LMH} \
    --model vllm \
    --model_args "pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.85,max_model_len=4096" \
    --tasks ${TASKS} \
    --batch_size auto \
    --output_path "${OUT}" \
    > "/workspace/riskKD/logs/evbl_${NAME}.log" 2>&1 && log "  ${NAME} OK" || log "  ${NAME} FAILED rc=$?"
}

(
  run_one teacher_dpo /workspace/riskKD/output/qwen3-1.7b-ultrafeedback-dpo-teacher 0
  run_one teacher_sft vukien2301/qwen3-1.7b-deita-sft-student 0
) &
PID0=$!

run_one student_sft /workspace/riskKD/output/qwen3-0.6b-deita-sft-student 1 &
PID1=$!

wait $PID0
wait $PID1
log "all baseline-model evals done"
