#!/bin/bash
# Run the lm-eval leaderboard suite on the TVKD final model (HF backend, single GPU).
cd /workspace/riskKD
source .venv/bin/activate
export HF_HOME=/workspace/.hf_home
export CUDA_VISIBLE_DEVICES=0
rm -rf results/eval/tvkd_b200 2>/dev/null
echo "[eval] $(date) starting lm_eval on output/TVKD ..."
lm_eval --model hf \
  --model_args pretrained=/workspace/riskKD/output/TVKD,dtype=bfloat16 \
  --include_path eval_script \
  --tasks leaderboard_custom \
  --batch_size auto \
  --output_path results/eval/tvkd_b200
echo "[eval] $(date) lm_eval exit=$?"
