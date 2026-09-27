#!/usr/bin/env bash
# Stage 0 accuracy anchors on the locked manifests: Q-ReAlign Mini/Lite/Pro, then Q-Align OneAlign.
# Usage: scripts/anchors.sh <set>...   (for example: t1 t2, or the image sets only)
# Runs one model at a time. Speed numbers in these runs are not the speed benchmark (see `vq bench`).
set -euo pipefail
cd "$(dirname "$0")/.."
sets=("$@")
for m in Mini-0.8B Lite-4B Pro-9B; do
    uv run vq eval --model "q-future/Q-ReAlign-${m}" --out "runs/anchors/qrealign-${m}" \
        --batch-size 16 --workers 8 "${sets[@]}"
done
/root/venvs/onealign/bin/python scripts/onealign_eval.py --workers 8 eval --out runs/anchors/onealign \
    --batch-size 32 "${sets[@]}"
