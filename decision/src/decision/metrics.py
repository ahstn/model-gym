"""Deterministic numpy metrics for option distributions (padding slots are 0 in probs and targets)."""

from __future__ import annotations

from typing import Any

import numpy as np

ECE_BINS = 15
LOG_FLOOR = 1e-12


def _xlogy(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.where(t > 0, t * np.log(np.clip(p, LOG_FLOOR, None)), 0.0)


def _group(probs: np.ndarray, targets: np.ndarray, k: np.ndarray, name: str) -> dict[str, Any]:
    n = len(probs)
    out: dict[str, Any] = {"n": n}
    if n == 0:
        return out
    t_max = targets.max(axis=1)
    decisive = np.isclose(targets, t_max[:, None], rtol=0, atol=1e-9).sum(axis=1) == 1
    pred = probs.argmax(axis=1)
    correct = pred == targets.argmax(axis=1)
    conf = probs.max(axis=1)
    nd = int(decisive.sum())
    out["n_decisive"] = nd
    soft_nll = -_xlogy(targets, probs).sum(axis=1)
    entropy = -_xlogy(targets, targets).sum(axis=1)
    out["soft_nll"] = float(soft_nll.mean())
    out["kl"] = float((soft_nll - entropy).mean())
    out["brier"] = float(((probs - targets) ** 2).sum(axis=1).mean())
    if nd:
        c, w, p = conf[decisive], correct[decisive], conf[decisive]
        out["agreement"] = float(w.mean())
        bins = np.minimum((c * ECE_BINS).astype(np.int64), ECE_BINS - 1)
        ece = 0.0
        for b in range(ECE_BINS):
            m = bins == b
            if m.any():
                ece += m.mean() * abs(p[m].mean() - w[m].mean())
        out["ece_top"] = float(ece)
        out["confident_wrong_95"] = float(((c >= 0.95) & ~w).mean())
    else:
        out.update(agreement=None, ece_top=None, confident_wrong_95=None)
    if "kind=score" in name:
        denom = np.maximum(k - 1, 1).astype(np.float64)
        levels = np.arange(probs.shape[1], dtype=np.float64)
        ep = (probs * levels).sum(axis=1) / denom
        et = (targets * levels).sum(axis=1) / denom
        out["expected_level_mae"] = float(np.abs(ep - et).mean())
    if "kind=noul" in name:
        out["noul_brier_yes"] = float(((probs[:, 1] - targets[:, 1]) ** 2).mean())
    return out


def summarize(
    probs: np.ndarray,
    targets: np.ndarray,
    groups: dict[str, np.ndarray],
    *,
    num_options: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Metrics for "all" plus each named boolean mask. Names containing "kind=score"/"kind=noul" get extra metrics.

    `num_options` (K per row) is used for score level normalisation; inferred from non-zero slots when omitted.
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if num_options is None:
        nz = (probs > 0) | (targets > 0)
        num_options = probs.shape[1] - np.argmax(nz[:, ::-1], axis=1)
    k = np.asarray(num_options)
    masks = {"all": np.ones(len(probs), dtype=bool), **groups}
    return {name: _group(probs[m], targets[m], k[m], name) for name, m in masks.items()}
