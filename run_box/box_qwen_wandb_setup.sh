#!/bin/bash
# Run ON the box. Wires up W&B for the Qwen campaign: writes the API key, flips the
# SFT recipes to report_to=wandb with run names, and rewrites box_qwen_sft.sh to export
# the W&B env (project=riskKD-qwen3, dir on /workspace).
set -e
cd /workspace/riskKD
echo "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" > /workspace/.wandb_key
chmod 600 /workspace/.wandb_key

# SFT recipes: report_to wandb + run_name
sed -i 's/^report_to: none/report_to: wandb/' recipes/qwen3-1.7b-ultrafeedback/teacher_sft.yaml recipes/qwen3-1.7b-ultrafeedback/student_sft_init.yaml
grep -q '^run_name:' recipes/qwen3-1.7b-ultrafeedback/teacher_sft.yaml || sed -i '/^report_to: wandb/a run_name: qwen3-8b-sft-teacher' recipes/qwen3-1.7b-ultrafeedback/teacher_sft.yaml
grep -q '^run_name:' recipes/qwen3-1.7b-ultrafeedback/student_sft_init.yaml || sed -i '/^report_to: wandb/a run_name: qwen3-1.7b-sft-student' recipes/qwen3-1.7b-ultrafeedback/student_sft_init.yaml

# launch wrapper with W&B env
cat > box_qwen_sft.sh <<'WRAP'
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
WRAP
chmod +x box_qwen_sft.sh

echo "RESULT:"
grep -nE "report_to|run_name|num_train_epochs" recipes/qwen3-1.7b-ultrafeedback/teacher_sft.yaml recipes/qwen3-1.7b-ultrafeedback/student_sft_init.yaml
echo "wandb key: $(wc -c < /workspace/.wandb_key) bytes | wandb installed: $(.venv/bin/python -c 'import wandb;print(wandb.__version__)' 2>/dev/null||echo NO)"
