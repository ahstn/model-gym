"""Raw (unmapped) PLCC, SRCC, KRCC and percentile bootstrap intervals."""

from __future__ import annotations

import numpy as np

from scipy.stats import kendalltau, pearsonr, spearmanr


def correlations(pred: np.ndarray, mos: np.ndarray, *, higher_better: bool = True) -> dict[str, float]:
    """Raw PLCC / SRCC / KRCC. DMOS sets (higher = worse) are negated so that all values are positive-is-good."""
    gt = mos if higher_better else -mos
    return {
        "plcc": float(pearsonr(pred, gt)[0]),
        "srcc": float(spearmanr(pred, gt)[0]),
        "krcc": float(kendalltau(pred, gt)[0]),
    }


def bootstrap(
    pred: np.ndarray,
    mos: np.ndarray,
    *,
    higher_better: bool = True,
    n: int = 1000,
    seed: int = 0,
    ref: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """95% percentile intervals for PLCC and SRCC over resampled items.

    With `ref` (another model's predictions on the same items), the interval is for the paired
    difference pred - ref, resampling the same items for both.
    """
    rng = np.random.default_rng(seed)
    gt = mos if higher_better else -mos
    m = len(gt)
    out: dict[str, list[float]] = {"plcc": [], "srcc": []}
    for _ in range(n):
        idx = rng.integers(0, m, m)
        for key, fn in (("plcc", pearsonr), ("srcc", spearmanr)):
            v = fn(pred[idx], gt[idx])[0]
            if ref is not None:
                v -= fn(ref[idx], gt[idx])[0]
            out[key].append(v)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in out.items()}
