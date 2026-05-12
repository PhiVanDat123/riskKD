#!/bin/bash
cd /workspace/riskKD
pkill -f "lm_eval.*adpa_b200" 2>/dev/null || true
rm -rf results/eval/adpa_b200 2>/dev/null || true
sleep 1
nohup bash run_eval_adpa.sh > logs/eval_adpa.log 2>&1 & disown
echo "[box] ADPA eval launched PID=$! -> logs/eval_adpa.log"
