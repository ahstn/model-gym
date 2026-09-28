"""Markdown tables: PLCC / SRCC per set with paired-bootstrap SRCC gaps to a reference, and speed-harness results."""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np

from vq.data import SOURCES
from vq.metrics import bootstrap


def _load(run: Path) -> dict:
    return json.loads((run / "metrics.json").read_text())


def _preds(run: Path, name: str) -> dict[str, tuple[float, float]]:
    with (run / "preds" / f"{name}.jsonl").open() as f:
        return {r["id"]: (r["pred"], r["mos"]) for r in map(json.loads, f)}


def table(runs: list[Path], sets: list[str], *, ref: Path | None = None, boot: int = 1000) -> str:
    """One row per run, one column per set: `PLCC / SRCC`; with `ref`, a second table of SRCC gaps (95% CI)."""
    metrics = {r: _load(r) for r in [*runs, *([ref] if ref else [])]}
    head = "| Model | " + " | ".join(sets) + " |\n|---|" + "---|" * len(sets) + "\n"
    rows = []
    for r in runs:
        cells = []
        for s in sets:
            m = metrics[r]["sets"].get(s)
            cells.append(f"{m['plcc']:.3f} / {m['srcc']:.3f}" if m else "n/a")
        rows.append(f"| {r.name} | " + " | ".join(cells) + " |")
    out = head + "\n".join(rows) + "\n"
    if ref is None:
        return out
    out += "\nSRCC minus reference `" + ref.name + "` (paired bootstrap over items, 95% CI):\n\n" + head
    for r in runs:
        if r == ref:
            continue
        cells = []
        for s in sets:
            if s not in metrics[r]["sets"] or s not in metrics[ref]["sets"]:
                cells.append("n/a")
                continue
            a, b = _preds(r, s), _preds(ref, s)
            ids = sorted(set(a) & set(b))
            pa = np.array([a[i][0] for i in ids])
            pb = np.array([b[i][0] for i in ids])
            mos = np.array([a[i][1] for i in ids])
            hb = SOURCES[s].higher_better
            ci = bootstrap(pa, mos, higher_better=hb, n=boot, ref=pb)["srcc"]
            d = metrics[r]["sets"][s]["srcc"] - metrics[ref]["sets"][s]["srcc"]
            mark = "**" if ci[0] > 0 or ci[1] < 0 else ""
            cells.append(f"{mark}{d:+.3f}{mark} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        out += f"| {r.name} | " + " | ".join(cells) + " |\n"
    return out


def bench_table(runs: list[Path], *, usd_per_hour: float) -> str:
    """One row per `bench.json` mode: visual tokens, batch-1 latency, peak throughput, VRAM, cost per 1M items."""
    out = (
        "| Model | Mode | Visual tokens / item | Batch-1 latency p50 / p95 (ms) | Peak items/s (batch) "
        f"| Peak VRAM (GB) | USD per 1M items at ${usd_per_hour:.2f}/h |\n|---|---|---:|---|---|---:|---:|\n"
    )
    for r in runs:
        b = json.loads((r / "bench.json").read_text())
        for mode, e in b["modes"].items():
            p = e["peak"]
            usd = usd_per_hour / 3600 / p["items_per_s"] * 1e6
            out += (
                f"| {r.name} | {mode} | {e['visual_tokens_mean']:.0f} "
                f"| {e['latency_ms_p50']:.0f} / {e['latency_ms_p95']:.0f} "
                f"| {p['items_per_s']:.1f} ({p['batch_size']}) | {p['peak_vram_gb']:.1f} | {usd:.2f} |\n"
            )
    return out
