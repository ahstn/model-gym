"""Panel evaluation: code-logit readout, per-kind temperature scaling, metrics, permutation probe."""

from __future__ import annotations

import json
import logging
import random
import time

from pathlib import Path
from typing import Any

import numpy as np
import torch

from decision.calibration import apply_temperature, fit_temperatures
from decision.metrics import summarize
from decision.modeling import code_logits, collate, load_model, token_budget_batches
from decision.prompt import Readout
from decision.schema import KINDS, Decision, read_jsonl

logger = logging.getLogger(__name__)

MAX_ROWS_PER_BATCH = 256
PERMUTATION_ROWS = 500
PERMUTATION_SEED = 0


def _logits(
    model: Any, readout: Readout, rows: list[Decision], encoded: list[list[int]], *, pad_id: int, batch_tokens: int
) -> tuple[torch.Tensor, int]:
    """Return code logits [N, MAX_OPTIONS] float32 cpu (row order preserved) and number of prompt tokens."""
    device = next(model.parameters()).device
    code_ids = torch.tensor(readout.code_token_ids, dtype=torch.long, device=device)
    out = torch.empty((len(rows), len(readout.code_token_ids)), dtype=torch.float32)
    batches = token_budget_batches(
        [len(e) for e in encoded], max_tokens=batch_tokens, max_rows=MAX_ROWS_PER_BATCH, shuffle=False, seed=0
    )
    with torch.inference_mode():
        for idx in batches:
            batch = collate([encoded[i] for i in idx], [rows[i] for i in idx], pad_id=pad_id).to(device)
            out[idx] = code_logits(model, batch, code_ids).float().cpu()
    return out, sum(len(e) for e in encoded)


def _encode(
    readout: Readout, rows: list[Decision], max_tokens: int
) -> tuple[list[Decision], list[list[int]], list[bool]]:
    kept, enc, answered = [], [], []
    for d in rows:
        ids = readout.encode(d, max_tokens=max_tokens)
        answered.append(ids is not None)
        if ids is not None:
            kept.append(d)
            enc.append(ids)
    return kept, enc, answered


def _targets(rows: list[Decision], width: int) -> torch.Tensor:
    t = torch.zeros((len(rows), width), dtype=torch.float32)
    for i, d in enumerate(rows):
        t[i, : len(d.target)] = torch.tensor(d.target)
    return t


def _group_masks(rows: list[Decision]) -> dict[str, np.ndarray]:
    from decision.data import split_of

    keys = [
        (f"dataset={d.dataset}", f"kind={d.kind}", f"split={split_of(d.id)}", f"dataset={d.dataset}/kind={d.kind}")
        for d in rows
    ]
    names = sorted({k for ks in keys for k in ks})
    return {n: np.array([n in ks for ks in keys], dtype=bool) for n in names}


def _probs_with_temps(logits: torch.Tensor, rows: list[Decision], temps: dict[str, float]) -> np.ndarray:
    probs = torch.empty_like(logits, dtype=torch.float64)
    for kind in KINDS:
        m = torch.tensor([d.kind == kind for d in rows], dtype=torch.bool)
        if m.any():
            probs[m] = apply_temperature(logits[m], temps.get(kind, 1.0))
    return probs.numpy()


def _permutation_probe(
    model: Any,
    readout: Readout,
    rows: list[Decision],
    *,
    max_tokens: int,
    batch_tokens: int,
    pad_id: int,
    base_probs: np.ndarray,
) -> dict[str, Any]:
    cand = [i for i, d in enumerate(rows) if d.kind == "choice" and len(d.options) >= 3]
    rng = random.Random(PERMUTATION_SEED)
    picked = sorted(rng.sample(cand, min(PERMUTATION_ROWS, len(cand))))
    orders, perm_rows, base_idx = [], [], []
    for i in picked:
        order = list(range(len(rows[i].options)))
        rng.shuffle(order)
        orders.append(order)
        perm_rows.append(rows[i].permuted(order))
        base_idx.append(i)
    kept, enc, answered = _encode(readout, perm_rows, max_tokens)
    if not kept:
        return {"n": 0}
    logits, _ = _logits(model, readout, kept, enc, pad_id=pad_id, batch_tokens=batch_tokens)
    perm_probs = torch.softmax(logits.double(), dim=-1).numpy()
    tvs, agree = [], []
    j = 0
    for order, i, ok in zip(orders, base_idx, answered, strict=True):
        if not ok:
            continue
        remapped = np.zeros(len(order))
        remapped[order] = perm_probs[j, : len(order)]  # new[p] = old[order[p]]
        base = base_probs[i, : len(order)]
        tvs.append(0.5 * np.abs(remapped - base).sum())
        agree.append(int(remapped.argmax() == base.argmax()))
        j += 1
    return {"n": len(tvs), "mean_tv": float(np.mean(tvs)), "argmax_agreement": float(np.mean(agree))}


def run(
    model_id: str,
    *,
    adapter: Path | None,
    panel: Path,
    calib: Path | None,
    out_dir: Path,
    max_tokens: int,
    batch_tokens: int,
    limit: int | None,
    trust_remote_code: bool = False,
    merge: bool = False,
) -> dict:
    model, tokenizer = load_model(model_id, adapter=adapter, trust_remote_code=trust_remote_code, merge=merge)
    readout = Readout(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    width = len(readout.code_token_ids)

    all_rows = list(read_jsonl(panel))
    if limit is not None:
        all_rows = all_rows[:limit]
    rows, enc, answered = _encode(readout, all_rows, max_tokens)
    all_masks = _group_masks(all_rows)
    answered_arr = np.array(answered, dtype=bool)
    unanswered = {"all": float(1 - answered_arr.mean()) if len(all_rows) else 0.0}
    unanswered |= {n: float(1 - answered_arr[m].mean()) for n, m in all_masks.items()}

    start = time.perf_counter()
    logits, n_tokens = _logits(model, readout, rows, enc, pad_id=pad_id, batch_tokens=batch_tokens)
    seconds = time.perf_counter() - start
    targets = _targets(rows, width)

    temps: dict[str, float] = dict.fromkeys(KINDS, 1.0)
    if calib is not None:
        c_rows, c_enc, _ = _encode(readout, list(read_jsonl(calib)), max_tokens)
        c_logits, _ = _logits(model, readout, c_rows, c_enc, pad_id=pad_id, batch_tokens=batch_tokens)
        c_targets = _targets(c_rows, width)
        by_kind = {}
        for kind in KINDS:
            m = torch.tensor([d.kind == kind for d in c_rows], dtype=torch.bool)
            if m.any():
                by_kind[kind] = (c_logits[m], c_targets[m])
        temps |= fit_temperatures(by_kind)

    raw_probs = _probs_with_temps(logits, rows, {})
    cal_probs = _probs_with_temps(logits, rows, temps)
    tgt = targets.double().numpy()
    k = np.array([len(d.options) for d in rows])
    groups = {n: m[answered_arr] for n, m in all_masks.items()}

    metrics: dict[str, Any] = {
        "raw": summarize(raw_probs, tgt, groups, num_options=k),
        "calibrated": summarize(cal_probs, tgt, groups, num_options=k),
        "unanswered": unanswered,
        "permutation": _permutation_probe(
            model,
            readout,
            rows,
            max_tokens=max_tokens,
            batch_tokens=batch_tokens,
            pad_id=pad_id,
            base_probs=raw_probs,
        ),
        "throughput": {
            "rows": len(rows),
            "tokens": n_tokens,
            "seconds": seconds,
            "rows_per_s": len(rows) / seconds if seconds else None,
            "tokens_per_s": n_tokens / seconds if seconds else None,
        },
        "model": model_id,
        "adapter": str(adapter) if adapter is not None else None,
        "adapter_merged": adapter is not None and merge,
        "temperatures": temps,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w") as f:
        for i, d in enumerate(rows):
            kk = len(d.options)
            rec = {
                "id": d.id,
                "dataset": d.dataset,
                "kind": d.kind,
                "K": kk,
                "probs": [round(float(p), 6) for p in cal_probs[i, :kk]],
                "target": list(d.target),
                "raw_argmax": int(raw_probs[i, :kk].argmax()),
            }
            f.write(json.dumps(rec) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "temperatures.json").write_text(json.dumps(temps, indent=2))
    logger.info("eval done: %d rows in %.1fs -> %s", len(rows), seconds, out_dir)
    return metrics
