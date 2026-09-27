"""Per-kind scalar temperature scaling fitted on soft NLL."""

from __future__ import annotations

import logging
import math

import torch

from decision.schema import Kind

logger = logging.getLogger(__name__)

MIN_ROWS = 50
T_MIN, T_MAX = 0.05, 20.0


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """softmax(logits / T); -inf padding slots get probability 0."""
    return torch.softmax(logits.double() / temperature, dim=-1)


def _soft_nll(logits: torch.Tensor, targets: torch.Tensor, log_t: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(logits)
    # Divide only finite logits: -inf / T would put inf into the log-T gradient (0 * inf = nan).
    scaled = (torch.where(finite, logits, torch.zeros_like(logits)) / log_t.exp()).masked_fill(~finite, float("-inf"))
    logp = torch.log_softmax(scaled, dim=-1)
    return -torch.where(targets > 0, targets * logp, torch.zeros_like(logp)).sum(-1).mean()


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    logits, targets = logits.double(), targets.double()
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = _soft_nll(logits, targets, log_t)
        loss.backward()
        return loss

    opt.step(closure)
    t = math.exp(float(log_t.detach()))
    return min(max(t, T_MIN), T_MAX) if math.isfinite(t) else 1.0


def fit_temperatures(logits_by_kind: dict[Kind, tuple[torch.Tensor, torch.Tensor]]) -> dict[Kind, float]:
    out: dict[Kind, float] = {}
    for kind, (logits, targets) in logits_by_kind.items():
        if len(logits) < MIN_ROWS:
            logger.warning("kind=%s has %d calib rows (< %d); using T=1.0", kind, len(logits), MIN_ROWS)
            out[kind] = 1.0
            continue
        out[kind] = fit_temperature(logits, targets)
        logger.info("kind=%s T=%.4f (n=%d)", kind, out[kind], len(logits))
    return out
