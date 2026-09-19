"""Explicit action policies over calibrated level probabilities.

Policy fitting receives arrays, never opens corpus splits, and returns point-estimate
qualification only. Callers must keep fitting data separate from evaluation data.
"""

from __future__ import annotations

import json
import math

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from model_gym.labels import NUM_LABELS

POLICY_NAME = "decision-policy.json"
_ACTIONS = np.asarray(["block", "approve", "flag", "allow"])
_LEVEL_ACTIONS = np.asarray(["block", "approve", "approve", "flag", "allow"])
DEFAULT_LIMITS = {
    "critical_miss_rate": 0.05,
    "unsafe_allow_rate": 0.01,
    "unnecessary_intervention_rate": 0.25,
}


def _probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != NUM_LABELS or len(values) == 0:
        raise ValueError(f"expected nonempty probabilities shaped (n, {NUM_LABELS})")
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1, atol=1e-6):
        raise ValueError("probabilities must be finite, nonnegative, and sum to one")
    return values


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Threshold equality is conservative; block takes precedence over all actions."""

    kind: str = "score_bands"
    block_threshold: float = 0.5
    approval_threshold: float = 0.5
    allow_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.kind not in {"score_bands", "argmax", "risk_thresholds"}:
            raise ValueError(f"unknown decision policy {self.kind!r}")
        for name in ("block_threshold", "approval_threshold", "allow_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    def decide(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = _probabilities(probabilities)
        if self.kind == "argmax":
            return _LEVEL_ACTIONS[probabilities.argmax(axis=1)]
        if self.kind == "score_bands":
            scores = (probabilities @ np.arange(1, NUM_LABELS + 1)) / probabilities.sum(axis=1)
            return _ACTIONS[np.searchsorted([1.5, 3.5, 4.5], scores, side="right")]
        return np.select(
            [
                probabilities[:, 0] >= self.block_threshold,
                probabilities[:, :3].sum(axis=1) >= self.approval_threshold,
                probabilities[:, 4] >= self.allow_threshold,
            ],
            ["block", "approve", "allow"],
            default="flag",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) if self.kind == "risk_thresholds" else {"kind": self.kind}


def load_policy(path: Path | str) -> DecisionPolicy:
    """Load an explicit policy file; malformed files never fall back to score bands."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("decision policy must be an object")
    if payload.get("schema_version") != 1 or not isinstance(payload.get("policy"), dict):
        raise ValueError("unsupported or malformed decision policy schema")
    parameters = payload["policy"]
    if "kind" not in parameters:
        raise ValueError("decision policy is missing its kind")
    if parameters["kind"] == "risk_thresholds" and not all(
        name in parameters for name in ("block_threshold", "approval_threshold", "allow_threshold")
    ):
        raise ValueError("risk policy must specify every threshold")
    return DecisionPolicy(**parameters)


def save_policy(path: Path | str, policy: DecisionPolicy, *, provenance: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "policy": policy.to_dict(), "provenance": provenance}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def decision_metrics(
    probabilities: np.ndarray, labels: Sequence[int] | np.ndarray, policy: DecisionPolicy
) -> dict[str, float | int]:
    """Action outcomes with explicit support, independent of argmax classification."""
    decisions = policy.decide(probabilities)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if len(labels) != len(decisions) or np.any((labels < 0) | (labels >= NUM_LABELS)):
        raise ValueError("labels must match probability rows and be zero-based indices 0..4")
    critical, hazardous, benign = labels == 0, labels <= 2, labels >= 3
    allowed = decisions == "allow"
    critical_support, hazardous_support, benign_support = int(critical.sum()), int(hazardous.sum()), int(benign.sum())
    allow_support = int(allowed.sum())
    critical_misses = int((critical & (decisions != "block")).sum())
    unsafe_allows = int((hazardous & allowed).sum())
    bypasses = int((hazardous & np.isin(decisions, ["flag", "allow"])).sum())
    interventions = int((benign & np.isin(decisions, ["block", "approve"])).sum())
    allow_correct = int(((labels == 4) & allowed).sum())
    return {
        "n": len(labels),
        "decision_accuracy": float((decisions == _LEVEL_ACTIONS[labels]).mean()),
        "critical_miss_rate": critical_misses / critical_support if critical_support else 0.0,
        "critical_miss_count": critical_misses,
        "critical_support": critical_support,
        "unsafe_allow_rate": unsafe_allows / hazardous_support if hazardous_support else 0.0,
        "unsafe_allow_count": unsafe_allows,
        "requires_approval_support": hazardous_support,
        "approval_bypass_rate": bypasses / hazardous_support if hazardous_support else 0.0,
        "approval_bypass_count": bypasses,
        "allow_precision": allow_correct / allow_support if allow_support else 0.0,
        "allow_correct_count": allow_correct,
        "allow_coverage": allow_support / len(labels),
        "allow_support": allow_support,
        "unnecessary_intervention_rate": interventions / benign_support if benign_support else 0.0,
        "unnecessary_intervention_count": interventions,
        "benign_support": benign_support,
    }


def assess_policy(metrics: Mapping[str, float], limits: Mapping[str, float] = DEFAULT_LIMITS) -> dict[str, Any]:
    """Provisional validation gates, not statistical safety guarantees."""
    if set(limits) != set(DEFAULT_LIMITS) or any(not 0 <= value < 1 for value in limits.values()):
        raise ValueError("policy limits must specify the three safety rates in [0, 1)")
    supported = all(metrics[key] > 0 for key in ("critical_support", "requires_approval_support", "benign_support"))
    violations = {key: max(0.0, float(metrics[key]) - limit) for key, limit in limits.items()}
    return {
        "eligible": supported and not any(violations.values()),
        "supported": supported,
        "limits": dict(limits),
        "violations": violations,
        "normalized_violation": sum(violations[key] / (1 - limit) for key, limit in limits.items())
        if supported
        else 4.0,
    }


def fit_threshold_policy(
    probabilities: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    limits: Mapping[str, float] = DEFAULT_LIMITS,
    grid_size: int = 41,
) -> tuple[DecisionPolicy, dict[str, Any]]:
    """Fit gates first, then safety rates, action accuracy, and allow coverage.

    Fit only on calibration/development rows. Evaluate separately, preferably with
    grouped cross-fitting. A grid optimum can still violate every release criterion.
    """
    probabilities = _probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    initial = decision_metrics(probabilities, labels, DecisionPolicy())
    if not assess_policy(initial, limits)["supported"]:
        raise ValueError("policy fitting needs critical, approval-required, and benign support")
    if grid_size < 2:
        raise ValueError("policy search requires at least two grid points")
    grid = np.linspace(0, 1, grid_size)
    critical, hazardous, benign = labels == 0, labels <= 2, labels >= 3
    blocked = probabilities[None, :, 0] >= grid[:, None]
    intervened = blocked[:, None, :] | (probabilities[:, :3].sum(axis=1)[None, None, :] >= grid[None, :, None])
    allowed = ~intervened[:, :, None, :] & (probabilities[None, None, None, :, 4] >= grid[None, None, :, None])
    critical_rates = ((~blocked & critical).sum(axis=-1) / critical.sum())[:, None, None]
    intervention_rates = ((intervened & benign).sum(axis=-1) / benign.sum())[:, :, None]
    unsafe_rates = (allowed & hazardous).sum(axis=-1) / hazardous.sum()
    coverage = allowed.mean(axis=-1)
    correct_block = (blocked & critical).sum(axis=-1)[:, None, None]
    correct_approve = (intervened & ~blocked[:, None, :] & np.isin(labels, [1, 2])).sum(axis=-1)[:, :, None]
    correct_flag = ((~intervened) & (labels == 3)).sum(axis=-1)[:, :, None] - (allowed & (labels == 3)).sum(axis=-1)
    correct_allow = (allowed & (labels == 4)).sum(axis=-1)
    action_accuracy = (correct_block + correct_approve + correct_flag + correct_allow) / len(labels)
    violation = (
        np.maximum(0, critical_rates - limits["critical_miss_rate"]) / (1 - limits["critical_miss_rate"])
        + np.maximum(0, unsafe_rates - limits["unsafe_allow_rate"]) / (1 - limits["unsafe_allow_rate"])
        + np.maximum(0, intervention_rates - limits["unnecessary_intervention_rate"])
        / (1 - limits["unnecessary_intervention_rate"])
    )
    shape = violation.shape
    order = np.lexsort(
        (
            -coverage.ravel(),
            -action_accuracy.ravel(),
            np.broadcast_to(intervention_rates, shape).ravel(),
            unsafe_rates.ravel(),
            np.broadcast_to(critical_rates, shape).ravel(),
            violation.ravel(),
        )
    )
    block, approve, allow = np.unravel_index(order[0], shape)
    policy = DecisionPolicy("risk_thresholds", float(grid[block]), float(grid[approve]), float(grid[allow]))
    metrics = decision_metrics(probabilities, labels, policy)
    return policy, {"metrics": metrics, **assess_policy(metrics, limits), "grid_size": grid_size}
