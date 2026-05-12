#!/bin/bash
cd /workspace/riskKD
pkill -f "lm_eval.*dckd_b200" 2>/dev/null || true
rm -rf results/eval/dckd_b200 2>/dev/null || true
sleep 1
nohup bash run_eval_dckd.sh > logs/eval_dckd.log 2>&1 & disown
echo "[box] DCKD eval launched PID=$! -> logs/eval_dckd.log"
