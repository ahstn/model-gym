#!/usr/bin/env bash
# Stage 0 accuracy anchors on the locked manifests: Q-ReAlign Mini/Lite/Pro, then Q-Align OneAlign.
# Usage: scripts/anchors.sh <set>...   (for example: t1 t2, or the image sets only)
#   MODELS="Lite-4B Pro-9B"   Q-ReAlign sizes to run (default: all three; "" skips them)
#   ONEALIGN=0                skip OneAlign
#   QREALIGN_HUB_CACHE=dir    HF hub cache for the Q-ReAlign weights (Lite + Pro are ~29 GB;
#                             on the pod the container disk is too small, so use /workspace)
# Runs one model at a time. Speed numbers in these runs are not the speed benchmark (see `vq bench`).
set -euo pipefail
cd "$(dirname "$0")/.."
sets=("$@")
for m in ${MODELS-Mini-0.8B Lite-4B Pro-9B}; do
    HF_HUB_CACHE="${QREALIGN_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}" \
        uv run vq eval --model "q-future/Q-ReAlign-${m}" --out "runs/anchors/qrealign-${m}" \
        --batch-size 16 --workers 8 "${sets[@]}"
done
if [[ "${ONEALIGN:-1}" == 1 ]]; then
    /root/venvs/onealign/bin/python scripts/onealign_eval.py --workers 8 eval --out runs/anchors/onealign \
        --batch-size 32 "${sets[@]}"
fi
