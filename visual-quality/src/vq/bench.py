"""Speed harness (docs/evaluation/protocol.md section 9).

Fixed inputs: 500 KonIQ + 500 SPAQ test images, 100 LSVQ + 100 LSVQ-1080p test videos (8 frames each),
evenly spaced over each manifest. Times include image decode and preprocessing.

- latency: batch 1, one item at a time in the main process (decode -> processor -> forward), p50 / p95 ms.
- throughput: DataLoader workers + padded batches for each batch size until OOM; items/s and peak VRAM.

The run also records visual tokens per item, the software stack, and the CPU count.
"""

from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Any

import numpy as np

from vq.data import read_manifest
from vq.evaluate import environment, subset

IMAGE_MIX = (("test_koniq", 500), ("test_spaq", 500))
VIDEO_MIX = (("test_lsvq", 100), ("test_lsvq_1080p", 100))
BATCH_SIZES = (1, 4, 8, 16, 32, 64)
LATENCY_ITEMS = 100


def bench_rows(root: Path, mix: tuple[tuple[str, int], ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, n in mix:
        rows += subset(read_manifest(root / "manifests" / f"{name}.jsonl"), n)
    return rows


def _visual_tokens(batch: dict[str, Any], merge: int) -> float:
    grid = batch.get("image_grid_thw")
    if grid is None:
        return float("nan")
    return float(grid.prod(-1).sum()) / (merge * merge)


def run(cfg, root: Path, out: Path, *, batch_sizes: tuple[int, ...] = BATCH_SIZES) -> dict[str, Any]:
    import torch

    from vq.scorer import Scorer, encode, load_images, score_rows

    scorer = Scorer(cfg)
    merge = getattr(scorer.processor.image_processor, "merge_size", 2)
    result: dict[str, Any] = {
        "config": cfg.asdict(),
        "env": environment(),
        "modes": {},
    }
    for mode, mix in (("image", IMAGE_MIX), ("video", VIDEO_MIX)):
        rows = bench_rows(root, mix)
        # visual tokens per item and batch-1 latency, in-process
        toks, lat = [], []
        for r in rows[:5]:  # warm-up
            scorer.forward(encode(scorer.processor, [r], [load_images(root, r)]))
        for r in subset(rows, LATENCY_ITEMS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            batch = encode(scorer.processor, [r], [load_images(root, r)])
            scorer.forward(batch)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
            toks.append(_visual_tokens(batch, merge))
        entry: dict[str, Any] = {
            "items": len(rows),
            "visual_tokens_mean": float(np.mean(toks)),
            "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p95": float(np.percentile(lat, 95)),
            "throughput": {},
        }
        for bs in batch_sizes:
            try:
                res = score_rows(_with_batch(scorer, bs), root, rows)
            except torch.OutOfMemoryError:
                entry["throughput"][str(bs)] = "oom"
                torch.cuda.empty_cache()
                break
            entry["throughput"][str(bs)] = {
                "items_per_s": res["items_per_s"],
                "peak_vram_gb": res["peak_vram_gb"],
            }
            print(json.dumps({mode: {"bs": bs, "items_per_s": round(res["items_per_s"], 2)}}), flush=True)
        ok = {k: v for k, v in entry["throughput"].items() if isinstance(v, dict)}
        best = max(ok, key=lambda k: ok[k]["items_per_s"])
        entry["peak"] = {"batch_size": int(best), **ok[best]}
        result["modes"][mode] = entry
        print(json.dumps({mode: {k: v for k, v in entry.items() if k != "throughput"}}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(result, indent=2))
    return result


def _with_batch(scorer, bs: int):
    from dataclasses import replace

    scorer.cfg = replace(scorer.cfg, batch_size=bs)
    return scorer
