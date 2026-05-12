#!/bin/bash
# Qwen campaign — "chunk A" (run ON the box, fire-and-forget after the teacher SFT finishes):
#   1. lm-eval the SFT teacher
#   2. teacher DPO (1 epoch from the SFT teacher)
#   3. lm-eval the DPO teacher
#   4. student SFT (Qwen3-1.7B-Base, 1 epoch)
#   5. lm-eval the SFT student
# Each stage logs to this script's stdout (logs/qwen_chainA.log).
set -e
set -o pipefail
cd /workspace/riskKD
source .venv/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(cat /workspace/.hf_home/token 2>/dev/null)
export WANDB_API_KEY=$(cat /workspace/.wandb_key 2>/dev/null)
export WANDB_PROJECT=riskKD-qwen3
export WANDB_DIR=/workspace/riskKD
export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_LOG_LEVEL=info
ACC=recipes/accelerate_config/deepspeed_zero3_2_nooffload.yaml
log(){ echo "[qwen-chainA $(date +%H:%M:%S)] $*"; }

bench_model(){  # $1=model_path  $2=out_subdir
  log "lm-eval: $1 -> results/eval/$2"
  rm -rf "results/eval/$2" 2>/dev/null || true
  CUDA_VISIBLE_DEVICES=0 lm_eval --model hf --model_args pretrained="$1",dtype=bfloat16 \
    --include_path eval_script --tasks leaderboard_custom --batch_size auto --output_path "results/eval/$2"
  log "lm-eval done: $2"
}

TEACHER_SFT=output/qwen3-8b-deita-sft-teacher
TEACHER_DPO=output/qwen3-8b-ultrafeedback-dpo-teacher
STUDENT_SFT=output/qwen3-1.7b-deita-sft-student

[ -f "$TEACHER_SFT/model.safetensors" ] || { log "ERROR: $TEACHER_SFT/model.safetensors not found — aborting chunk A"; exit 1; }

log "=== STAGE 1/5: lm-eval SFT teacher ==="
bench_model "$TEACHER_SFT" qwen_sft_teacher

log "=== STAGE 2/5: teacher DPO (1 epoch on UltraFeedback) ==="
export CUDA_VISIBLE_DEVICES=0,1
python -m accelerate.commands.launch --config_file $ACC scripts/run_distill_dpo.py recipes/qwen3-1.7b-ultrafeedback/teacher_dpo.yaml
[ -f "$TEACHER_DPO/model.safetensors" ] || { log "ERROR: teacher DPO produced no model — aborting"; exit 1; }
log "teacher DPO done -> $TEACHER_DPO"

log "=== STAGE 3/5: lm-eval DPO teacher ==="
bench_model "$TEACHER_DPO" qwen_dpo_teacher

log "=== STAGE 4/5: student SFT (Qwen3-1.7B-Base, 1 epoch) ==="
export CUDA_VISIBLE_DEVICES=0,1
python -m accelerate.commands.launch --config_file $ACC scripts/run_sft.py recipes/qwen3-1.7b-ultrafeedback/student_sft_init.yaml
[ -f "$STUDENT_SFT/model.safetensors" ] || { log "ERROR: student SFT produced no model — aborting"; exit 1; }
log "student SFT done -> $STUDENT_SFT"

log "=== STAGE 5/5: lm-eval SFT student ==="
bench_model "$STUDENT_SFT" qwen_sft_student

log "=== CHUNK A DONE === Qwen3-8B SFT teacher + DPO teacher + Qwen3-1.7B SFT student all built & evaled. Next (chunk B): rebuild ultrafeedback-dckd + ultrafeedback-adpa with the Qwen DPO teacher logits, then train DPO/WPO/Ra-DPO/DCKD/ADPA/TVKD/riskKD + eval each + push to HF."
