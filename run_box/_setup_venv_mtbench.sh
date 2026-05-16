#!/bin/bash
# Add MT-Bench / FastChat-judge runtime deps to the existing .venv-vllm.
# We run FastChat from the cloned /workspace/FastChat source (via PYTHONPATH),
# so we don't pip-install fschat itself - we only install its llm_judge runtime deps.
# Idempotent: re-runnable.
set -e -o pipefail
cd /workspace/riskKD
PY=/workspace/riskKD/.venv-vllm/bin/python
FC_DIR=/workspace/FastChat
test -x "$PY" || { echo "ERROR: .venv-vllm not found at $PY; run _setup_venv_vllm.sh first"; exit 1; }
test -d "$FC_DIR/fastchat/llm_judge" || { echo "ERROR: FastChat not cloned at $FC_DIR"; exit 1; }
# fschat[llm_judge] pyproject pins: openai<1, anthropic>=0.3, ray. Plus main deps: shortuuid, tiktoken.
# fastchat.llm_judge.common transitively imports fastchat.model.model_adapter which pulls
# accelerate, peft, sentencepiece, protobuf even though we don't use the model-loading path.
uv pip install --python "$PY" \
  "openai==0.28.1" "anthropic>=0.3,<1" "ray" "shortuuid" "tiktoken" \
  "accelerate>=0.21" "peft" "sentencepiece" "protobuf"
PYTHONPATH="$FC_DIR" "$PY" -c "from fastchat.llm_judge.common import temperature_config; print('temperature_config keys:', list(temperature_config))"
"$PY" -c "import openai, shortuuid, ray; print('openai', openai.__version__, '| shortuuid OK | ray', ray.__version__)"
echo "[venv-mtbench] done"
