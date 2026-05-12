#!/bin/bash
# Run ON the B200 box (/workspace/riskKD). Patches dckd.sh + TVKD.yaml for this box,
# writes the fast merger, starts model downloads.
set -e
cd /workspace/riskKD
export HF_HOME=/workspace/.hf_home
export HF_TOKEN=$(cat /workspace/.hf_home/token 2>/dev/null)
export HF_HUB_ENABLE_HF_TRANSFER=1
TEACHER_ID="vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1"
STUDENT_ID="vukien2301/llama-3.2-1b-deita-sft-student"

# ---- fast zero-copy merger (replaces the slow merge_logits_dckd_dataset.py) ----
cat > utils/fast_merge_dckd.py <<'PY'
"""Zero-copy column-wise merge of per-split teacher-logp datasets into the *-dckd dataset.
concatenate_datasets(..., axis=1) instead of a Python-list->pyarrow round-trip. Atomic via .tmp rename."""
import os, shutil, argparse
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
try:
    base = load_dataset(a.base)
except Exception:
    base = load_from_disk(a.base)
print("[fast_merge] base splits:", list(base.keys()))
result = {}
for split in [s for s in ("train", "test") if s in base]:
    cp = os.path.join(a.root, f"{a.tag}-chosen-logp-{split}")
    rp = os.path.join(a.root, f"{a.tag}-rejected-logp-{split}")
    if not (os.path.isdir(cp) and os.path.isdir(rp)):
        print(f"[fast_merge] WARN missing logp dir for split={split}, skip"); continue
    c = load_from_disk(cp); r = load_from_disk(rp); b = base[split]
    assert len(b) == len(c) == len(r), f"row mismatch {split}: base={len(b)} chosen={len(c)} rejected={len(r)}"
    c2 = c.select_columns(["chosen_compressed_probs", "chosen_labels"])
    r2 = r.select_columns(["rejected_compressed_probs", "rejected_labels"])
    merged = concatenate_datasets([b.flatten_indices(), c2, r2], axis=1)
    print(f"[fast_merge] {split}: cols={merged.column_names} n={len(merged)}")
    result[split] = merged
dd = DatasetDict(result)
tmp = a.out + ".tmp"
if os.path.exists(tmp): shutil.rmtree(tmp)
print(f"[fast_merge] writing -> {tmp}"); dd.save_to_disk(tmp)
if os.path.exists(a.out): shutil.rmtree(a.out)
os.rename(tmp, a.out); print(f"[fast_merge] DONE -> {a.out}")
PY
echo "[box] wrote utils/fast_merge_dckd.py ($(wc -l < utils/fast_merge_dckd.py) lines)"

# ---- patch run/dckd.sh ----
python3 - "$TEACHER_ID" <<'PY'
import re, sys, pathlib
teacher = sys.argv[1]
p = pathlib.Path("run/dckd.sh"); s = p.read_text()
s = s.replace('TEACHER="/home/minchan.kwon/ADPA/model/llama3.2-1b-deita-dpomix/dpo_teacher_epoch1"', f'TEACHER="{teacher}"')
s = s.replace('GPUS="0,1,2,3,4,5,6,7"', 'GPUS="0,1"')
s = s.replace('NPROC=8', 'NPROC=2')
if 'HF_HOME=/workspace/.hf_home' not in s:
    s = s.replace('set -o pipefail\n', 'set -o pipefail\nexport HF_HOME=/workspace/.hf_home\nexport HF_TOKEN=$(cat /workspace/.hf_home/token 2>/dev/null)\nexport HF_HUB_ENABLE_HF_TRANSFER=1\n')
s = re.sub(r'log_and_run "merge" "python utils/merge_logits_dckd_dataset\.py \$\{MERGE_ARGS\}"',
           'log_and_run "merge" "python utils/fast_merge_dckd.py --base ${DATA} --root ${OUTDIR} --tag ${TAG} --out ${MERGED_OUT}"', s)
p.write_text(s)
print("[box] patched run/dckd.sh:")
for ln in s.splitlines():
    if any(k in ln for k in ('TEACHER=','GPUS=','NPROC=','HF_HOME','HF_TOKEN','HF_HUB_ENABLE','fast_merge_dckd')):
        print("   ", ln.strip())
PY

# ---- patch recipes/.../TVKD.yaml paths ----
python3 - "$TEACHER_ID" <<'PY'
import sys, pathlib
teacher = sys.argv[1]
p = pathlib.Path("recipes/llama3.2-1b-deita-dpomix/TVKD.yaml"); s = p.read_text()
s = s.replace("ref_model_name_or_path: /home/minchan.kwon/ADPA/model/llama3.2-1b-deita-dpomix/dpo_teacher_epoch1",
              f"ref_model_name_or_path: {teacher}")
s = s.replace("  /root/riskKD/data/llama3.2-1b-deita-dpomix/ultrafeedback-dckd: 1.0",
              "  /workspace/riskKD/data/llama3.2-1b-deita-dpomix/ultrafeedback-dckd: 1.0")
s = s.replace("output_dir: /home/minchan.kwon/ADPA/model/llama3.2-1b-deita-dpomix/TVKD_output",
              "output_dir: /workspace/riskKD/output/TVKD")
p.write_text(s)
print("[box] patched TVKD.yaml:")
for ln in s.splitlines():
    if any(k in ln for k in ('model_name_or_path','ref_model','ultrafeedback-dckd','output_dir','num_train_epochs','per_device_train_batch_size','dataset_num_proc','dpo_weight','distillation_weight','qadapter_distil_weight')):
        print("   ", ln.strip())
PY

# ---- start model downloads in background (verify repos exist + warm cache) ----
source /workspace/riskKD/.venv/bin/activate
nohup huggingface-cli download "$TEACHER_ID" > /workspace/riskKD/logs/dl_teacher.log 2>&1 &
echo "[box] teacher download PID=$! ($TEACHER_ID) -> logs/dl_teacher.log"
nohup huggingface-cli download "$STUDENT_ID" > /workspace/riskKD/logs/dl_student.log 2>&1 &
echo "[box] student download PID=$! ($STUDENT_ID) -> logs/dl_student.log"
echo "[box] done."
