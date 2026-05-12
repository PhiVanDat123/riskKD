#!/bin/bash
cd /workspace/riskKD
source .venv/bin/activate
export HF_HOME=/workspace/.hf_home
export CUDA_VISIBLE_DEVICES=0
rm -rf results/eval/adpa_b200 2>/dev/null
echo "[eval-adpa] $(date) starting lm_eval on output/ADPA ..."
lm_eval --model hf --model_args pretrained=/workspace/riskKD/output/ADPA,dtype=bfloat16 --include_path eval_script --tasks leaderboard_custom --batch_size auto --output_path results/eval/adpa_b200
echo "[eval-adpa] $(date) exit=$?"
