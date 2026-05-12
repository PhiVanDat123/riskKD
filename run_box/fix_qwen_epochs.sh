#!/bin/bash
cd /workspace/riskKD
pkill -9 -f "run_sft.py" 2>/dev/null || true
pkill -9 -f "accelerate.commands.launch" 2>/dev/null || true
sleep 2
rm -rf output/qwen3-8b-deita-sft-teacher 2>/dev/null || true
sed -i 's/^num_train_epochs: 3/num_train_epochs: 1/' recipes/qwen3-1.7b-ultrafeedback/teacher_sft.yaml recipes/qwen3-1.7b-ultrafeedback/student_sft_init.yaml
echo "RESULT:"
grep num_train_epochs recipes/qwen3-1.7b-ultrafeedback/teacher_sft.yaml recipes/qwen3-1.7b-ultrafeedback/student_sft_init.yaml
echo "procs (run_sft/accelerate): $(ps -eo cmd|grep -cE '[r]un_sft|[a]ccelerate.commands')"
echo "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader|tr '\n' ' ')"
