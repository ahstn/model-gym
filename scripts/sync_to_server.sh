#!/usr/bin/env bash
# Copy code, configs, and the seed corpus to a training host.
#
#   scripts/sync_to_server.sh user@gpu-host [remote-dir]
#
# data/processed and data/synthetic are excluded on purpose: the corpus is
# deterministic, so the host rebuilds byte-identical splits with `build-data`.
set -euo pipefail

target="${1:-}"
remote_dir="${2:-model-gym}"
if [[ -z "${target}" ]]; then
    echo "usage: $(basename "$0") user@host [remote-dir]" >&2
    exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -az --delete --human-readable --info=progress2 \
    --exclude '.venv/' \
    --exclude '.git/' \
    --exclude 'runs/' \
    --exclude 'artifacts/' \
    --exclude 'data/processed/' \
    --exclude 'data/synthetic/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    "${root}/" "${target}:${remote_dir}/"

echo "synced ${root}/ -> ${target}:${remote_dir}/"
echo
echo "next:"
echo "  ssh ${target} 'cd ${remote_dir} && scripts/server_bootstrap.sh'"
