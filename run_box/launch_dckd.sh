#!/bin/bash
cd /workspace/riskKD
source .venv/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(cat /workspace/.hf_home/token 2>/dev/null)
export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=0,1
rm -rf output/DCKD 2>/dev/null
echo "[dckd] $(date) launching ..."
exec python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3_2_nooffload.yaml \
  scripts/run_distill_dpo.py DCKD.yaml
