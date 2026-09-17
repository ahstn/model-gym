#!/usr/bin/env bash
# Bring a finished run back from the training host (checkpoints, reports, metadata).
#
#   scripts/fetch_artifacts.sh user@gpu-host runs/modernbert-base [local-dir]
set -euo pipefail

target="${1:-}"
remote="${2:-}"
local="${3:-$(basename "${remote:-run}")}"
if [[ -z "${target}" || -z "${remote}" ]]; then
    echo "usage: $(basename "$0") user@host remote-run-dir [local-run-dir]" >&2
    exit 2
fi

mkdir -p "${local}"
rsync -az --human-readable --info=progress2 \
    --include 'best/' --include 'best/***' \
    --include 'reports/' --include 'reports/***' \
    --include 'run_metadata.json' \
    --exclude '*' \
    "${target}:${remote}/" "${local}/"

echo "fetched ${target}:${remote} -> ${local}"
echo
echo "next:"
echo "  uv run --extra export model-gym export --model ${local}/best --precision int8"
