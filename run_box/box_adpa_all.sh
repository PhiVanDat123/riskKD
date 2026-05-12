#!/bin/bash
# Full ADPA pipeline (run ON the B200 box, fire-and-forget):
#   1. wait for the SFT teacher download to finish
#   2. precompute the SFT-teacher's logits on the rejected responses (train+test)
#   3. merge with the already-computed DPO-teacher rejected logits -> ultrafeedback-adpa
#      (rejected_margin_logp_every = per-token log(p_dpo / p_sft) on the rejected response)
#   4. patch ADPA.yaml for the box (TVKD-matched: 1 epoch, eff-batch 64, num_proc 16, max_length 512)
#   5. launch ADPA training
set -e
set -o pipefail
cd /workspace/riskKD
source .venv/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_TOKEN="$(cat /workspace/.hf_home/token 2>/dev/null)"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_LOG_LEVEL=info
export DS_SKIP_CUDA_CHECK=1
log(){ echo "[adpa-all $(date +%H:%M:%S)] $*"; }

DATA="pvdhihihi/ultra-feedback"
SFT_TEACHER="vukien2301/llama-3.1-8b-deita-sft-teacher-epoch1"
STUDENT="vukien2301/llama-3.2-1b-deita-sft-student"
DPO_TEACHER="vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1"
D="data/llama3.2-1b-deita-dpomix"
DPO_REJ_TRAIN="$D/ultrafeedback-dpoteacher-rejected-logp-train"
DPO_REJ_TEST="$D/ultrafeedback-dpoteacher-rejected-logp-test"
SFT_REJ_TRAIN="$D/ultrafeedback-sftteacher-rejected-logp-train"
SFT_REJ_TEST="$D/ultrafeedback-sftteacher-rejected-logp-test"
ADPA_DS="$D/ultrafeedback-adpa"
PAD=128001; GPUS="0,1"; NPROC=2; MAX_TOK=16384; MAX_BS=16
U_BEGIN='<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'; U_END='<|eot_id|>'
A_BEGIN='<|start_header_id|>assistant<|end_header_id|>\n\n'; A_END='<|eot_id|>'
SNAP_GLOB="/workspace/.hf_home/hub/models--vukien2301--llama-3.1-8b-deita-sft-teacher-epoch1/snapshots/*/model-00004-of-00004.safetensors"

# ---- 1) wait for SFT teacher download ----
log "waiting for SFT teacher download ($SFT_TEACHER) ..."
for i in $(seq 1 160); do
  ls $SNAP_GLOB >/dev/null 2>&1 && break
  sleep 15
done
ls $SNAP_GLOB >/dev/null 2>&1 || { log "SFT teacher NOT downloaded after waiting — abort"; exit 1; }
log "SFT teacher ready."

# ---- 2) precompute SFT-teacher logits on rejected responses ----
pkill -f "accelerate.commands.launch" 2>/dev/null || true
for SPLIT in train test; do
  SAVE="$D/ultrafeedback-sftteacher-rejected-logp-$SPLIT"
  if [ -f "$SAVE/state.json" ]; then log "precompute $SPLIT already done, skip"; continue; fi
  rm -rf "$SAVE" 2>/dev/null || true
  log "precompute SFT-teacher rejected logits, split=$SPLIT ..."
  CUDA_VISIBLE_DEVICES=$GPUS python -m accelerate.commands.launch --num_processes=$NPROC --main_process_port 29503 \
    utils/precompute_logits.py --data "$DATA" --split "$SPLIT" --model "$SFT_TEACHER" \
    --conversation-key rejected --prompt-key prompt \
    --user-begin "$U_BEGIN" --user-end "$U_END" --assistant-begin "$A_BEGIN" --assistant-end "$A_END" \
    --save-to "$SAVE" --pad-token-id $PAD --max-tokens-per-batch $MAX_TOK --max-batch-size $MAX_BS
  rm -f "$SAVE"/results_rank_*.jsonl
  log "precompute $SPLIT done."
done

# ---- 3) merge -> ultrafeedback-adpa ----
log "merging DPO-teacher + SFT-teacher rejected logits -> $ADPA_DS ..."
rm -rf "$ADPA_DS" 2>/dev/null || true
python utils/merge_logits_adpa_dataset.py \
  --input-dataset-dict "$DATA" \
  --dpo-teacher-logp-train "$DPO_REJ_TRAIN" --ref-teacher-logp-train "$SFT_REJ_TRAIN" \
  --dpo-teacher-logp-test "$DPO_REJ_TEST"  --ref-teacher-logp-test "$SFT_REJ_TEST" \
  --label-key rejected_labels --logits-key rejected_compressed_probs --output-key rejected_margin_logp_every \
  --save-to "$ADPA_DS"
log "ADPA dataset ready: $ADPA_DS"

# ---- 4) patch ADPA.yaml ----
python3 - "$STUDENT" "$DPO_TEACHER" "$ADPA_DS" <<'PY'
import sys, pathlib, re
student, teacher, ds = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path("ADPA.yaml"); s = p.read_text()
s = s.replace("model_name_or_path: model/student_sft_init", f"model_name_or_path: {student}")
s = s.replace("ref_model_name_or_path: model/teacher_dpo", f"ref_model_name_or_path: {teacher}")
s = s.replace("  data/llama3.2-1b-deita-dpomix/dpomix7k-adpa: 1.0", f"  /workspace/riskKD/{ds}: 1.0")
s = s.replace("output_dir: model/adpa", "output_dir: /workspace/riskKD/output/ADPA")
s = re.sub(r"num_train_epochs: \d+", "num_train_epochs: 1", s)
s = re.sub(r"per_device_train_batch_size: \d+.*", "per_device_train_batch_size: 32  # 32 x 2 GPUs x ga1 = 64 (matches TVKD/DCKD)", s)
s = re.sub(r"gradient_accumulation_steps: \d+", "gradient_accumulation_steps: 1", s)
s = re.sub(r"dataset_num_proc: \d+", "dataset_num_proc: 16", s)
s = re.sub(r"max_length: \d+", "max_length: 512", s)
s = re.sub(r"max_prompt_length: \d+", "max_prompt_length: 256", s)
s = s.replace("report_to: wandb", "report_to: none")
p.write_text(s)
print("[adpa-all] patched ADPA.yaml:")
for ln in s.splitlines():
    if any(k in ln for k in ("model_name_or_path","ref_model","ultrafeedback-adpa","output_dir","num_train_epochs","per_device","gradient_accumulation","dataset_num_proc","max_length","max_prompt_length","report_to","adpa_weight","adpa_loss_type")):
        print("   ", ln.strip())
PY

# ---- 5) launch ADPA training ----
log "launching ADPA training -> logs/adpa_train.log"
export CUDA_VISIBLE_DEVICES=0,1
nohup python -m accelerate.commands.launch --config_file recipes/accelerate_config/deepspeed_zero3_2_nooffload.yaml scripts/run_distill_dpo.py ADPA.yaml > logs/adpa_train.log 2>&1 & disown
log "ADPA training launched PID=$! . pipeline script done."
