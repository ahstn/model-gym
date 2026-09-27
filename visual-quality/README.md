# visual-quality

Code for the Q-Align challenger work. Research notes are in [docs/](docs/README.md). The plan is in [docs/training/experiment-plan.md](docs/training/experiment-plan.md).

- `vq labels`, `vq build`: download the Q-Align label JSONs and write locked manifests (`data/manifests/*.jsonl`).
- `vq frames`: 8 uniform frames per video, long side 448, JPEG q90 (the Q-ReAlign recipe).
- `vq eval`: batched level-token scorer. Writes predictions and metrics (raw PLCC / SRCC / KRCC, bootstrap intervals, items/s, peak VRAM).
- `vq train`: full-parameter SFT with the Q-ReAlign recipe on one GPU.

## Pod workflow

```bash
bash scripts/pod.sh sync && bash scripts/pod.sh bootstrap
bash scripts/pod.sh run labels -- vq labels
bash scripts/pod.sh run media -- bash scripts/fetch_media.sh images   # then lsvq, konvid, maxwell
bash scripts/pod.sh run build -- vq build
bash scripts/pod.sh run eval-mini -- vq eval --model q-future/Q-ReAlign-Mini-0.8B --out runs/eval-mini t1 t2
```

Media lives in `/dev/shm/vq` on the pod (`data/` is a symlink). The container disk is small, and `/workspace` reads at about 16 MB/s.

Results are in [results/](results/).
