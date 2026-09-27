#!/usr/bin/env bash
# Separate venv for the Q-Align OneAlign anchor: its remote code (mPLUG-Owl2) needs transformers 4.36.1
# (model card). Torch comes from the cu128 index (sm_120 / Blackwell support).
#
#   bash scripts/onealign_env.sh                     # create /root/venvs/onealign
#   /root/venvs/onealign/bin/python scripts/onealign_eval.py --out runs/eval-onealign t1 ...
set -euo pipefail
venv="${ONEALIGN_VENV:-/root/venvs/onealign}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
uv venv --python 3.10 "$venv"
uv pip install --python "$venv/bin/python" --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.*"
uv pip install --python "$venv/bin/python" \
    "transformers==4.36.1" "tokenizers<0.16" sentencepiece "accelerate<0.30" icecream \
    "numpy<2" scipy pillow pyyaml protobuf
"$venv/bin/python" -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
