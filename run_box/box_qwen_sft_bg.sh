#!/bin/bash
cd /workspace/riskKD
nohup bash box_qwen_sft.sh "$1" "$2" > "logs/$2" 2>&1 & disown
echo "[box] SFT launched PID=$! recipe=$1 -> logs/$2"
