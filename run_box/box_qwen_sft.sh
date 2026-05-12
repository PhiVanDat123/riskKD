#!/bin/bash
# arg1 = recipe yaml path (relative to /workspace/riskKD), arg2 = log file name
cd /workspace/riskKD
source .venv/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(cat /workspace/.hf_home/token 2>/dev/null)
export WANDB_API_KEY=$(cat /workspace/.wandb_key 2>/dev/null)
export WANDB_PROJECT=riskKD-qwen3
export WANDB_DIR=/workspace/riskKD
export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=0,1
echo "[qwen-sft] $(date) launching SFT: $1 -> logs/$2 (wandb project=$WANDB_PROJECT)"
exec python -m accelerate.commands.launch --config_file recipes/accelerate_config/deepspeed_zero3_2_nooffload.yaml scripts/run_sft.py "$1"
