#!/usr/bin/env bash
# Fetch media into data/ on the pod (run from $POD_DIR, after `scripts/pod.sh bootstrap`):
#
#   bash scripts/fetch_media.sh images    # Q-Align image tars (IQA + AVA + cross-sets) -> data/img/
#   bash scripts/fetch_media.sh lsvq      # LSVQ tar.gz parts, one at a time -> 8 frames each -> data/frames/
#   bash scripts/fetch_media.sh konvid    # KoNViD-1k -> frames
#   bash scripts/fetch_media.sh maxwell   # DIVIDE-MaxWell (eval only) -> frames
#
# Videos are deleted after frame extraction: 73 GB of LSVQ video does not fit next to the images in RAM.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
data="$(readlink -f data)"
auth=()
if [[ -f "${HF_HOME:-/root/hf}/token" ]]; then auth=(-H "Authorization: Bearer $(cat "${HF_HOME:-/root/hf}/token")"); fi
hf_url() { echo "https://huggingface.co/datasets/$1/resolve/main/$2"; }

case "${1:-}" in
images)
    mkdir -p "$data/img"
    for f in koniq.tar spaq.tar kadid10k.tar live.tar livec.tar csiq.tar agi-cgi.tar ava_images.tar; do
        curl -sfL "${auth[@]}" "$(hf_url q-future/q-align-datasets "$f")" | tar x -C "$data/img" &
    done
    wait
    find "$data/img" -name '._*' -delete
    ;;
lsvq)
    mkdir -p "$data/video/lsvq"
    parts=(backup backup2)
    for i in $(seq 1 25); do parts+=("yfcc-batch$i"); done
    for i in $(seq 1 24); do parts+=("ia-batch$i"); done
    for p in "${parts[@]}"; do
        if [[ -f "$data/video/.done-$p" ]]; then continue; fi
        curl -sfL "${auth[@]}" "$(hf_url teowu/LSVQ-videos "$p.tar.gz")" | tar xz -C "$data/video/lsvq"
        uv run vq frames lsvq test_lsvq test_lsvq_1080p --workers 12 --delete
        touch "$data/video/.done-$p"
        echo "done $p"
    done
    ;;
konvid)
    mkdir -p "$data/video/konvid"
    uv run hf download --repo-type dataset shuoxing/KoNViD_1k_videos --local-dir "$data/video/konvid"
    uv run vq frames konvid --workers 12 --delete
    ;;
maxwell)
    mkdir -p "$data/video/maxwell"
    curl -sfL "${auth[@]}" -o "$data/video/maxwell.zip" "$(hf_url teowu/DIVIDE-MaxWell videos.zip)"
    python3 -m zipfile -e "$data/video/maxwell.zip" "$data/video/maxwell"
    rm "$data/video/maxwell.zip"
    # The zip holds videos/NNNN.mp4 (plus a stray .ipynb_checkpoints); the labels name NNNN.mp4.
    find "$data/video/maxwell/videos" -name '*.mp4' -exec mv -t "$data/video/maxwell" {} +
    rm -r "$data/video/maxwell/videos"
    uv run vq frames maxwell_test --workers 12 --delete
    ;;
*)
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
    ;;
esac
