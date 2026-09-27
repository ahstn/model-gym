#!/usr/bin/env bash
# Drive the RunPod training host from the local checkout.
#
#   scripts/pod.sh sync                        # rsync visual-quality/ -> $POD_DIR (no .venv, data, runs, caches)
#   scripts/pod.sh bootstrap                   # install uv, python 3.12, uv sync --extra gpu, data -> /dev/shm/vq, `vq info`
#   scripts/pod.sh run <name> [--] <cmd...>    # nohup `uv run <cmd...>`, log/pid/exit in runs/logs/<name>.*
#   scripts/pod.sh status                      # GPU usage and state of every run
#   scripts/pod.sh logs <name> [n]             # last n (default 50) log lines
#   scripts/pod.sh fetch <run-name>            # rsync runs/<run-name>/ and its logs back to visual-quality/runs/
#   scripts/pod.sh fetch-data-manifest         # rsync data/manifests/report.json back to visual-quality/data/
#   scripts/pod.sh backup <run-name>           # copy the run and its logs to /workspace/backup/ on the pod
#   scripts/pod.sh ssh                         # interactive shell in $POD_DIR
#
# Env (defaults in mise.toml): POD_HOST POD_PORT POD_USER POD_KEY POD_DIR.
# Code, caches and the venv live on the container disk (/root); /workspace is a slow FUSE mount used only for backups.
# `sync` deletes everything in $POD_DIR that is not in the local checkout, except .venv/, data/ and runs/:
# keep generated files (suite dirs, samples) under data/.
set -euo pipefail

POD_HOST="${POD_HOST:-216.243.220.174}"
POD_PORT="${POD_PORT:-16541}"
POD_USER="${POD_USER:-root}"
POD_KEY="${POD_KEY:-/home/ahstn/git/runpod.pem}"
POD_DIR="${POD_DIR:-/root/model-gym/visual-quality}"
REMOTE_HF_HOME="/root/hf"
# Media lives in RAM (/dev/shm): the container disk is small and /workspace reads at ~16 MB/s.
REMOTE_DATA="/dev/shm/vq"
BACKUP_DIR="/workspace/backup"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${POD_USER}@${POD_HOST}"
ssh_opts=(-p "${POD_PORT}" -i "${POD_KEY}" -o StrictHostKeyChecking=accept-new)
rsync_ssh="ssh -p ${POD_PORT} -i ${POD_KEY} -o StrictHostKeyChecking=accept-new"

usage() {
    sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

# Run a bash script on the pod; extra args become its positional parameters.
remote() {
    local script="$1"
    shift
    local args=""
    if (($# > 0)); then
        args="$(printf '%q ' "$@")"
    fi
    # shellcheck disable=SC2029  # expansion on the client side is intended
    ssh "${ssh_opts[@]}" "${target}" "bash -s -- ${args}" <<<"${script}"
}

require_name() {
    if [[ -z "${1:-}" || "$1" == */* || "$1" == .* ]]; then
        echo "error: a plain run name is required" >&2
        exit 2
    fi
}

cmd="${1:-}"
[[ -n "${cmd}" ]] || usage
shift

case "${cmd}" in
sync)
    ssh "${ssh_opts[@]}" "${target}" "mkdir -p $(printf '%q' "${POD_DIR}")"
    rsync -az --delete --human-readable \
        -e "${rsync_ssh}" \
        --exclude '.venv/' \
        --exclude 'data/' \
        --exclude 'runs/' \
        --exclude '__pycache__/' \
        --exclude '.ruff_cache/' \
        --exclude '.pytest_cache/' \
        "${root}/" "${target}:${POD_DIR}/"
    echo "synced ${root}/ -> ${target}:${POD_DIR}/"
    ;;
bootstrap)
    remote '
set -euo pipefail
pod_dir="$1"; hf_home="$2"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export HF_HOME="$hf_home"
mkdir -p "$HF_HOME"
cd "$pod_dir"
mkdir -p "$3"
ln -sfn "$3" data
uv python install 3.12
if [[ -f uv.lock ]]; then uv sync --frozen --extra gpu; else uv sync --extra gpu; fi
uv run vq info
' "${POD_DIR}" "${REMOTE_HF_HOME}" "${REMOTE_DATA}"
    ;;
run)
    name="${1:-}"
    require_name "${name}"
    shift
    # `mise run` swallows the first `--`, so the separator is optional.
    [[ "${1:-}" == "--" ]] && shift
    (($# > 0)) || usage
    inner="$(printf '%q ' "$@")"
    remote '
set -euo pipefail
pod_dir="$1"; hf_home="$2"; name="$3"; inner="$4"
export PATH="$HOME/.local/bin:$PATH"
cd "$pod_dir"
mkdir -p runs/logs
pidf="runs/logs/$name.pid"
if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "error: run $name is still alive (pid $(cat "$pidf"))" >&2
    exit 1
fi
rm -f "runs/logs/$name.exit"
job="env HF_HOME=$(printf %q "$hf_home") PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run $inner; echo \$? > $(printf %q "runs/logs/$name.exit")"
nohup bash -c "$job" > "runs/logs/$name.log" 2>&1 < /dev/null &
echo $! > "$pidf"
echo "started $name (pid $!): uv run $inner"
' "${POD_DIR}" "${REMOTE_HF_HOME}" "${name}" "${inner}"
    ;;
status)
    remote '
set -euo pipefail
cd "$1"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
shopt -s nullglob
for pidf in runs/logs/*.pid; do
    name="$(basename "$pidf" .pid)"
    if kill -0 "$(cat "$pidf")" 2>/dev/null; then
        state="alive"
    elif [[ -f "runs/logs/$name.exit" ]]; then
        state="exited($(cat "runs/logs/$name.exit"))"
    else
        state="dead(no exit code)"
    fi
    echo "== $name: $state"
    tail -n 3 "runs/logs/$name.log" 2>/dev/null || true
done
' "${POD_DIR}"
    ;;
logs)
    name="${1:-}"
    require_name "${name}"
    lines="${2:-50}"
    [[ "${lines}" =~ ^[0-9]+$ ]] || usage
    remote 'tail -n "$3" "$1/runs/logs/$2.log"' "${POD_DIR}" "${name}" "${lines}"
    ;;
fetch)
    name="${1:-}"
    require_name "${name}"
    mkdir -p "${root}/runs/${name}" "${root}/runs/logs"
    rsync -az --human-readable -e "${rsync_ssh}" "${target}:${POD_DIR}/runs/${name}/" "${root}/runs/${name}/"
    rsync -az --human-readable -e "${rsync_ssh}" --include "${name}.*" --exclude '*' \
        "${target}:${POD_DIR}/runs/logs/" "${root}/runs/logs/"
    echo "fetched runs/${name}"
    ;;
fetch-data-manifest)
    mkdir -p "${root}/data/manifests"
    rsync -az -e "${rsync_ssh}" "${target}:${POD_DIR}/data/manifests/report.json" "${root}/data/manifests/report.json"
    echo "fetched data/manifests/report.json"
    ;;
backup)
    name="${1:-}"
    require_name "${name}"
    remote '
set -euo pipefail
cd "$1"; name="$2"; dest="$3"
mkdir -p "$dest/runs/logs"
cp -a "runs/$name" "$dest/runs/"
shopt -s nullglob
logs=(runs/logs/"$name".*)
if ((${#logs[@]} > 0)); then cp -a "${logs[@]}" "$dest/runs/logs/"; fi
echo "backed up runs/$name -> $dest/runs/$name"
' "${POD_DIR}" "${name}" "${BACKUP_DIR}"
    ;;
ssh)
    exec ssh -t "${ssh_opts[@]}" "${target}" "cd $(printf '%q' "${POD_DIR}") && exec bash -l"
    ;;
*)
    usage
    ;;
esac
