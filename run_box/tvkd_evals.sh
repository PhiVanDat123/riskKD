#!/bin/bash
set -u
cd /workspace/riskKD
export HF_HOME=/workspace/.hf_home
export PYTHONPATH=/workspace/riskKD
export PATH=/workspace/riskKD/.venv-vllm/bin:$PATH
LOG=/workspace/riskKD/logs/tvkd_evals.log
log(){ echo "[tvkd_eval $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
RES_ROOT=/workspace/riskKD/results/eval/qwen3-0.6b
LMH=/workspace/riskKD/.venv-vllm/bin/lm_eval
QF=/workspace/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
AD=/workspace/FastChat/fastchat/llm_judge/data/mt_bench/model_answer

# 1) lm-eval on GPU 0
log "tvkd lm-eval (GPU 0)"
mkdir -p "${RES_ROOT}/lm_eval/tvkd"
CUDA_VISIBLE_DEVICES=0 $LMH \
  --model vllm \
  --model_args "pretrained=/workspace/riskKD/output/qwen3-0.6b-tvkd,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.85,max_model_len=4096" \
  --tasks hellaswag,arc_challenge,mmlu,truthfulqa_mc2,winogrande,gsm8k \
  --batch_size auto \
  --output_path "${RES_ROOT}/lm_eval/tvkd" \
  > /workspace/riskKD/logs/ph3_t1_tvkd.log 2>&1 && log "  lm-eval OK" || log "  lm-eval FAILED"

# 2) MT-Bench gen on GPU 0
log "tvkd MT-Bench gen"
CUDA_VISIBLE_DEVICES=0 /workspace/riskKD/.venv-vllm/bin/python scripts/mtbench_gen_vllm.py \
  --model-path /workspace/riskKD/output/qwen3-0.6b-tvkd \
  --model-id qwen3-0.6b-tvkd \
  --question-file "$QF" \
  --answer-file "$AD/qwen3-0.6b-tvkd.jsonl" \
  --tensor-parallel-size 1 --max-model-len 4096 --max-new-token 1024 \
  --gpu-memory-utilization 0.85 \
  >> /workspace/riskKD/logs/mtbench_gen_tvkd.log 2>&1 && log "  mtbench gen OK" || log "  mtbench gen FAILED"

log "tvkd evals done"
