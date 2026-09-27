import numpy as np
import pytest
import torch

from decision.calibration import apply_temperature, fit_temperatures
from decision.metrics import summarize

W = 255


def _pad(rows: list[list[float]]) -> np.ndarray:
    out = np.zeros((len(rows), W))
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def test_perfect_prediction_agreement_and_nll_equals_entropy() -> None:
    t = _pad([[0.7, 0.3], [0.1, 0.2, 0.7]])
    m = summarize(t.copy(), t, {})["all"]
    assert m["agreement"] == 1.0
    assert m["kl"] == pytest.approx(0.0, abs=1e-12)
    ent = -(0.7 * np.log(0.7) + 0.3 * np.log(0.3) + 0.1 * np.log(0.1) + 0.2 * np.log(0.2) + 0.7 * np.log(0.7)) / 2
    assert m["soft_nll"] == pytest.approx(ent)
    assert m["brier"] == pytest.approx(0.0)


def test_uniform_noul_brier_and_ties_not_decisive() -> None:
    p = _pad([[0.5, 0.5]] * 3)
    t = _pad([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    m = summarize(p, t, {"kind=noul": np.ones(3, dtype=bool)})["kind=noul"]
    assert m["noul_brier_yes"] == pytest.approx((0.25 + 0.25 + 0.0) / 3)
    assert m["n_decisive"] == 2


def test_score_expected_level_mae() -> None:
    p = _pad([[1.0, 0.0, 0.0]])
    t = _pad([[0.0, 0.0, 1.0]])
    m = summarize(p, t, {"kind=score": np.ones(1, dtype=bool)}, num_options=np.array([3]))["kind=score"]
    assert m["expected_level_mae"] == pytest.approx(1.0)
    assert m["confident_wrong_95"] == 1.0


def test_temperature_fit_recovers_known_t_with_padding() -> None:
    g = torch.Generator().manual_seed(0)
    n, k = 4000, 4
    logits = torch.full((n, W), float("-inf"), dtype=torch.float64)
    logits[:, :k] = torch.randn(n, k, generator=g, dtype=torch.float64) * 3
    targets = torch.softmax(logits / 2.0, dim=-1)
    temps = fit_temperatures({"choice": (logits, targets), "noul": (logits[:10], targets[:10])})
    assert temps["choice"] == pytest.approx(2.0, rel=1e-3)
    assert temps["noul"] == 1.0
    probs = apply_temperature(logits, temps["choice"])
    assert torch.isfinite(probs).all()
    assert (probs[:, k:] == 0).all()
