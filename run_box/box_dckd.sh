#!/bin/bash
# Run ON the B200 box. Patches /workspace/riskKD/DCKD.yaml for this box + the TVKD-matched
# settings, then launches DCKD training (2x B200, ZeRO-3 no-offload, eff-batch 64, 1 epoch).
set -e
cd /workspace/riskKD
STUDENT="vukien2301/llama-3.2-1b-deita-sft-student"
TEACHER="vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1"

python3 - "$STUDENT" "$TEACHER" <<'PY'
import sys, pathlib
student, teacher = sys.argv[1], sys.argv[2]
p = pathlib.Path("DCKD.yaml"); s = p.read_text()
repl = {
  "model_name_or_path: model/student_sft_init": f"model_name_or_path: {student}",
  "ref_model_name_or_path: model/teacher_dpo": f"ref_model_name_or_path: {teacher}",
  "  data/llama3.2-1b-deita-dpomix/dpomix7k-dckd: 1.0": "  /workspace/riskKD/data/llama3.2-1b-deita-dpomix/ultrafeedback-dckd: 1.0",
  "output_dir: model/dckd": "output_dir: /workspace/riskKD/output/DCKD",
  "num_train_epochs: 5": "num_train_epochs: 1",
  "per_device_train_batch_size: 1": "per_device_train_batch_size: 8  # 8 x 2 GPUs x ga4 = 64 (matches TVKD)",
  "gradient_accumulation_steps: 16": "gradient_accumulation_steps: 4",
  "dataset_num_proc: 64": "dataset_num_proc: 16",
  "max_length: 2048": "max_length: 512",
  "max_prompt_length: 1800": "max_prompt_length: 256",
  "report_to: wandb": "report_to: none",
}
for a, b in repl.items():
    assert a in s, f"pattern not found: {a!r}"
    s = s.replace(a, b)
p.write_text(s)
print("[box] patched DCKD.yaml:")
for ln in s.splitlines():
    if any(k in ln for k in ("model_name_or_path","ref_model","ultrafeedback-dckd","output_dir","num_train_epochs","per_device","gradient_accumulation","dataset_num_proc","max_length","max_prompt_length","report_to","sft_on_chosen","distillation_weight","chosen_distil_weight","rejected_distil_weight","beta:")):
        print("   ", ln.strip())
PY

# launch DCKD training
source .venv/bin/activate
export HF_HOME=/workspace/.hf_home HF_TOKEN=$(cat /workspace/.hf_home/token) DS_SKIP_CUDA_CHECK=1 ACCELERATE_LOG_LEVEL=info CUDA_VISIBLE_DEVICES=0,1
mkdir -p logs output
pkill -f "run_distill_dpo.py.*DCKD" 2>/dev/null || true
nohup python -m accelerate.commands.launch --config_file recipes/accelerate_config/deepspeed_zero3_2_nooffload.yaml scripts/run_distill_dpo.py /workspace/riskKD/DCKD.yaml > logs/dckd_train.log 2>&1 & disown
echo "[box] DCKD training launched PID=$! -> logs/dckd_train.log"
