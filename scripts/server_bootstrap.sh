#!/usr/bin/env bash
# Provision a Linux GPU host for training and report what it can run.
#
#   scripts/server_bootstrap.sh
#
# Safe to re-run: uv sync is idempotent.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== host =="
uname -srm
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
else
    echo "nvidia-smi not found: CUDA training is not possible on this host" >&2
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv =="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "== python =="
uv python install 3.13

echo "== dependencies (train + export extras) =="
if [[ -n "${UV_TORCH_INDEX:-}" ]]; then
    echo "using torch index ${UV_TORCH_INDEX}"
    uv sync --extra train --extra export --index "${UV_TORCH_INDEX}"
else
    # Linux wheels on PyPI already bundle the CUDA runtime.
    uv sync --extra train --extra export
fi

echo "== environment =="
uv run --extra train --extra export model-gym info

cat <<'EOF'

If `cuda available` above is false, the torch wheel does not match this driver.
Reinstall against the matching PyTorch index and re-check:

    UV_TORCH_INDEX=https://download.pytorch.org/whl/cu126 scripts/server_bootstrap.sh

Pick the CUDA flavour from the driver version printed by nvidia-smi, for example:

    nvidia-smi                                                # driver reports the max CUDA runtime
    nvidia-smi --query-gpu=compute_cap --format=csv           # 8.6 means sm_86 (RTX 3090)

Next:
    uv run model-gym build-data
    uv run --extra train model-gym train --config configs/base.yaml
EOF
