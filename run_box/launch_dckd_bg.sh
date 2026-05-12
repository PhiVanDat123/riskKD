#!/bin/bash
cd /workspace/riskKD
pkill -f "run_distill_dpo.py.*DCKD" 2>/dev/null || true
rm -rf output/DCKD 2>/dev/null || true
sleep 1
nohup bash launch_dckd.sh > logs/dckd_train.log 2>&1 & disown
echo "[box] DCKD launched PID=$! -> logs/dckd_train.log"
