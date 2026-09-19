"""Validation-only scalar temperature fitting and the checkpoint sidecar contract."""

from __future__ import annotations

import json
import math

from pathlib import Path
from typing import Any

import numpy as np

from model_gym.labels import NUM_LABELS
from model_gym.metrics import expected_calibration_error

CALIBRATION_NAME = "calibration.json"
TEMPERATURE_BOUNDS = (0.05, 100.0)


def validate_temperature(value: Any) -> float:
    """Reject invalid temperatures instead of silently disabling calibration."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("calibration temperature must be a number")
    temperature = float(value)
    low, high = TEMPERATURE_BOUNDS
    if not math.isfinite(temperature) or not low <= temperature <= high:
        raise ValueError(f"calibration temperature must be finite and in [{low}, {high}]")
    return temperature


def load_calibration(model_dir: Path | str) -> dict[str, Any] | None:
    """Read validated calibration metadata; only an absent sidecar means uncalibrated."""
    path = Path(model_dir) / CALIBRATION_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid calibration file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"invalid calibration file {path}: expected an object")
    if payload.get("schema_version") != 1 or payload.get("method") != "temperature_scaling":
        raise ValueError(f"invalid calibration file {path}: unsupported schema or method")
    if payload.get("split") != "validation":
        raise ValueError(f"invalid calibration file {path}: fit split must be validation")
    validate_temperature(payload.get("temperature"))
    count = payload.get("sample_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"invalid calibration file {path}: sample_count must be positive")
    digest = payload.get("validation_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"invalid calibration file {path}: invalid validation_sha256")
    for name in ("nll_before", "nll_after", "ece_before", "ece_after"):
        value = payload.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
            or (name.startswith("ece_") and value > 1)
        ):
            raise ValueError(f"invalid calibration file {path}: invalid {name}")
    return payload


def _log_probabilities(centered: np.ndarray, inverse_temperature: float) -> np.ndarray:
    scaled = centered * inverse_temperature
    return scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *, validation_sha256: str) -> dict[str, Any]:
    """Minimize validation NLL over a bounded positive scalar temperature.

    NLL is convex in inverse temperature. Bisecting its derivative gives a
    deterministic fit without optimizer state or an additional dependency.
    Callers must supply raw, never previously calibrated, logits.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 2 or logits.shape[1] != NUM_LABELS or not len(logits) or not np.isfinite(logits).all():
        raise ValueError(f"expected nonempty finite logits shaped (n, {NUM_LABELS})")
    if (
        labels.shape != (len(logits),)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any((labels < 0) | (labels >= NUM_LABELS))
    ):
        raise ValueError("labels must be one valid integer class index per logit row")
    centered = logits - logits.max(axis=1, keepdims=True)
    row_indices = np.arange(len(labels))
    true_logits = centered[row_indices, labels]
    lower, upper = 1 / TEMPERATURE_BOUNDS[1], 1 / TEMPERATURE_BOUNDS[0]
    for _ in range(64):
        midpoint = (lower + upper) / 2
        probabilities = np.exp(_log_probabilities(centered, midpoint))
        gradient = float(((probabilities * centered).sum(axis=1) - true_logits).mean())
        if gradient > 0:
            upper = midpoint
        else:
            lower = midpoint
    temperature = float(np.clip(1 / ((lower + upper) / 2), *TEMPERATURE_BOUNDS))
    before = _log_probabilities(centered, 1.0)
    after = _log_probabilities(centered, 1 / temperature)
    nll_before = float(-before[row_indices, labels].mean())
    nll_after = float(-after[row_indices, labels].mean())
    if nll_after >= nll_before:
        temperature, after, nll_after = 1.0, before, nll_before
    return {
        "schema_version": 1,
        "method": "temperature_scaling",
        "split": "validation",
        "sample_count": len(labels),
        "validation_sha256": validation_sha256,
        "temperature": temperature,
        "temperature_bounds": list(TEMPERATURE_BOUNDS),
        "nll_before": nll_before,
        "nll_after": nll_after,
        "ece_before": expected_calibration_error(np.exp(before), labels),
        "ece_after": expected_calibration_error(np.exp(after), labels),
    }
