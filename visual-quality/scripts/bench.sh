#!/usr/bin/env bash
# Stage 0 speed harness (docs/evaluation/protocol.md section 9). Needs the GPU to itself.
# Usage: scripts/bench.sh [r0-model-dir]
#   QREALIGN_HUB_CACHE=dir   HF hub cache for Lite/Pro weights (see scripts/anchors.sh)
# Writes runs/bench/<name>/bench.json; the inference-config sweep also writes accuracy runs to runs/sweep/.
set -euo pipefail
cd "$(dirname "$0")/.."
r0="${1:-}"
big_cache="${QREALIGN_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

# 1. Every model at its default inference config.
/root/venvs/onealign/bin/python scripts/onealign_eval.py bench --out runs/bench/onealign
uv run vq bench --model q-future/Q-ReAlign-Mini-0.8B --out runs/bench/qrealign-Mini-0.8B
for m in Lite-4B Pro-9B; do
    HF_HUB_CACHE="$big_cache" uv run vq bench --model "q-future/Q-ReAlign-${m}" --out "runs/bench/qrealign-${m}"
done
[[ -n "$r0" ]] && uv run vq bench --model "$r0" --out runs/bench/r0

# 2. Inference-config sweep on the 0.8B model: one factor at a time against the default above.
mini=q-future/Q-ReAlign-Mini-0.8B
uv run vq bench --model "$mini" --out runs/bench/sweep-full-logits --full-logits
uv run vq bench --model "$mini" --out runs/bench/sweep-hub-kernels --use-kernels
uv run vq bench --model "$mini" --out runs/bench/sweep-flash-attn2 --attn kernels-community/flash-attn2
# Pixel caps: 256 / 576 / 1024 visual tokens per image (32 x 32 pixels per merged token).
for tok in 256 576 1024; do
    px=$((tok * 1024))
    uv run vq bench --model "$mini" --out "runs/bench/sweep-cap${tok}" --max-pixels "$px"
    uv run vq eval --model "$mini" --out "runs/sweep/mini-cap${tok}" --max-pixels "$px" \
        test_koniq test_spaq test_kadid agiqa3k live test_ava livec csiq
done
