#!/bin/bash
# Precompute DPO-teacher token-level distributions for the DCKD / vanila-KD / ADPA
# baselines, then merge them into the training dataset.
#
# NOTE: riskKD itself does NOT need this — it reads the teacher at train time as the
# reference model. This pipeline is only for the *baselines* whose loss consumes
# `chosen_compressed_probs` / `rejected_compressed_probs` (distillation_weight > 0).
#
# After it finishes, point `dataset_mixer` in the DCKD recipe YAML at $MERGED_OUT.
# `mix_datasets` in scripts/run_distill_dpo.py auto-renames *_compressed_probs ->
# teacher_*_probs.

set -e
set -o pipefail
trap 'echo -e "\n❌ [ERROR] Command failed: $BASH_COMMAND\n"' ERR

# ----------------------------------------------------------------------------
# Config — edit these
# ----------------------------------------------------------------------------
DATA="pvdhihihi/ultra-feedback"                                                       # dataset to precompute logits over (must match riskKD's dataset_mixer)
TEACHER="/home/minchan.kwon/ADPA/model/llama3.2-1b-deita-dpomix/dpo_teacher_epoch1"    # the DPO-trained teacher (same model used as ref_model_name_or_path in riskKD.yaml)
OUTDIR="data/llama3.2-1b-deita-dpomix"                                                 # where to write the per-split teacher-logp datasets + the merged dataset
TAG="ultrafeedback-dpoteacher"                                                        # filename tag for the per-split logp dirs
MERGED_OUT="${OUTDIR}/ultrafeedback-dckd"                                              # final merged dataset (-> dataset_mixer in the DCKD recipe)
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
PORT=29501
MAX_TOK_PER_BATCH=2048
PAD_TOKEN_ID=128001                                                                   # Llama-3 <|end_of_text|>

# Llama-3 chat-format delimiters (the precompute wraps each turn with these)
U_BEGIN='<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
U_END='<|eot_id|>'
A_BEGIN='<|start_header_id|>assistant<|end_header_id|>\n\n'
A_END='<|eot_id|>'

# ----------------------------------------------------------------------------
LOG_DIR="logs"
mkdir -p "$LOG_DIR" "$OUTDIR"

# kill any stray accelerate launcher so PORT is free (we've hit "address in use" before)
pkill -f "accelerate.commands.launch" 2>/dev/null || true

# detect which splits the dataset actually has (ultra-feedback may be train-only)
SPLITS=$(python -c "
from datasets import get_dataset_split_names
try:
    s = get_dataset_split_names('$DATA')
except Exception:
    s = ['train']
# keep only the ones we care about, train first
out = [x for x in ['train', 'test'] if x in s]
print(' '.join(out) if out else 'train')
")
echo "==> dataset '$DATA' splits to precompute: $SPLITS"

log_and_run () {
  local NAME="$1"; shift
  local LOG_FILE="$LOG_DIR/${NAME}.log"
  echo -e "\n🔹 [START] $NAME"
  echo "🔸 Logging to: $LOG_FILE"
  echo "------------------------------------------------"
  {
    echo ">>> [Start: $(date)]"
    echo ">>> [Command] $@"
    echo "------------------------------------------------"
    eval "$@"
    echo ">>> [Success: $(date)]"
  } 2>&1 | tee "$LOG_FILE"
  echo -e "✅ [DONE] $NAME\n"
}

precompute () {
  local SPLIT="$1" KEY="$2"
  local SAVE_TO="${OUTDIR}/${TAG}-${KEY}-logp-${SPLIT}"
  log_and_run "precompute_${SPLIT}_${KEY}" \
"CUDA_VISIBLE_DEVICES=${GPUS} python -m accelerate.commands.launch \
  --num_processes=${NPROC} \
  --main_process_port ${PORT} \
  utils/precompute_logits.py \
  --data ${DATA} \
  --split ${SPLIT} \
  --model ${TEACHER} \
  --conversation-key ${KEY} \
  --user-begin '${U_BEGIN}' \
  --user-end '${U_END}' \
  --assistant-begin '${A_BEGIN}' \
  --assistant-end '${A_END}' \
  --save-to ${SAVE_TO} \
  --pad-token-id ${PAD_TOKEN_ID} \
  --max-tokens-per-batch ${MAX_TOK_PER_BATCH}"
  log_and_run "rm_${SPLIT}_${KEY}_temp" "rm -f ${SAVE_TO}/results_rank_*.jsonl"
}

for SPLIT in $SPLITS; do
  for KEY in chosen rejected; do
    precompute "$SPLIT" "$KEY"
  done
done

# build merge args; include the test args only if we actually computed the test split
MERGE_ARGS="--input-dataset-dict ${DATA} \
  --teacher-chosen-logp-train   ${OUTDIR}/${TAG}-chosen-logp-train \
  --teacher-rejected-logp-train ${OUTDIR}/${TAG}-rejected-logp-train"
if echo " $SPLITS " | grep -q " test "; then
  MERGE_ARGS="${MERGE_ARGS} \
  --teacher-chosen-logp-test    ${OUTDIR}/${TAG}-chosen-logp-test \
  --teacher-rejected-logp-test  ${OUTDIR}/${TAG}-rejected-logp-test"
fi
MERGE_ARGS="${MERGE_ARGS} --save-to ${MERGED_OUT}"

log_and_run "merge" "python utils/merge_logits_dckd_dataset.py ${MERGE_ARGS}"

echo "==> Done. Merged dataset at: ${MERGED_OUT}"
echo "    Set 'dataset_mixer: {${MERGED_OUT}: 1.0}' in the DCKD recipe YAML."
