"""Score locked manifests and write predictions, metrics, and the run environment."""

from __future__ import annotations

import json
import platform

from pathlib import Path
from typing import Any

import numpy as np

from vq.data import read_manifest
from vq.metrics import bootstrap, correlations
from vq.scorer import Scorer, ScorerConfig, score_rows


def subset(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Evenly spaced deterministic subset (manifests keep the label-file order)."""
    if not limit or limit >= len(rows):
        return rows
    idx = np.linspace(0, len(rows) - 1, limit).round().astype(int)
    return [rows[i] for i in idx]


def environment() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run(cfg: ScorerConfig, root: Path, sets: list[str], out: Path, *, limit: int = 0, boot: int = 1000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "preds").mkdir(exist_ok=True)
    scorer = Scorer(cfg)
    metrics: dict[str, Any] = {"config": cfg.asdict(), "limit": limit, "env": environment(), "sets": {}}
    # Add sets to an earlier run of the same config (for example video sets after their frames exist).
    prev = out / "metrics.json"
    if prev.exists():
        old = json.loads(prev.read_text())
        if old["config"] == metrics["config"] and old["limit"] == limit:
            metrics["sets"] = old["sets"]
    # Warm up kernels and allocator on a few items so the first set's timing is not penalised.
    first = subset(read_manifest(root / "manifests" / f"{sets[0]}.jsonl"), min(cfg.batch_size, 8) * 2)
    score_rows(scorer, root, first)
    for name in sets:
        rows = subset(read_manifest(root / "manifests" / f"{name}.jsonl"), limit)
        res = score_rows(scorer, root, rows)
        mos = np.array([r["mos"] for r in rows], dtype=np.float64)
        hb = rows[0]["higher_better"]
        m = correlations(res["preds"], mos, higher_better=hb)
        m["ci95"] = bootstrap(res["preds"], mos, higher_better=hb, n=boot) if boot else None
        m.update(n=len(rows), items_per_s=res["items_per_s"], wall_s=res["wall_s"], peak_vram_gb=res["peak_vram_gb"])
        metrics["sets"][name] = m
        with (out / "preds" / f"{name}.jsonl").open("w") as f:
            for r, p, pr in zip(rows, res["preds"], res["probs"], strict=True):
                f.write(
                    json.dumps({"id": r["id"], "pred": float(p), "probs": pr.round(5).tolist(), "mos": r["mos"]}) + "\n"
                )
        print(
            json.dumps({name: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "ci95"}}),
            flush=True,
        )
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
