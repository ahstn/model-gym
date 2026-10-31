"""Corpus quantiles that turn raw values into percentiles a teacher model can read."""

from __future__ import annotations

import json

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from image_stats.metrics import ALGORITHM_VERSION, METRIC_NAMES

SCHEMA = "image-stats-reference/v1"
LEVELS = np.linspace(0.0, 100.0, 101)


@dataclass(frozen=True)
class Reference:
    name: str
    version: str
    max_edge: int
    count: int
    quantiles: dict[str, list[float]]

    @classmethod
    def build(cls, metrics: Iterable[Mapping[str, float]], *, name: str, max_edge: int) -> Reference:
        rows = list(metrics)
        if len(rows) < 20:
            raise ValueError("A reference needs at least 20 analysed images")
        quantiles = {
            metric: [float(f"{value:.6g}") for value in np.percentile([row[metric] for row in rows], LEVELS)]
            for metric in METRIC_NAMES
        }
        return cls(name=name, version=ALGORITHM_VERSION, max_edge=max_edge, count=len(rows), quantiles=quantiles)

    def percentile(self, metric: str, value: float) -> float:
        """Percentile rank in [0, 100]; ties take the middle of the tied range."""
        xs = np.asarray(self.quantiles[metric])
        low, high = int(np.searchsorted(xs, value, "left")), int(np.searchsorted(xs, value, "right"))
        if high > low:
            return float((LEVELS[low] + LEVELS[high - 1]) / 2)
        if low == 0:
            return 0.0
        if low == len(xs):
            return 100.0
        x0, x1 = xs[low - 1], xs[low]
        return float(LEVELS[low - 1] + (LEVELS[low] - LEVELS[low - 1]) * (value - x0) / (x1 - x0))

    def check(self, version: str, max_edge: int) -> None:
        if version != self.version or max_edge != self.max_edge:
            raise ValueError(
                f"Reference built with {self.version} at max_edge {self.max_edge}; "
                f"analysis uses {version} at max_edge {max_edge}"
            )

    def save(self, path: str | Path) -> None:
        payload = {"schema": SCHEMA, **self.__dict__}
        Path(path).write_text(json.dumps(payload, indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Reference:
        payload = json.loads(Path(path).read_text())
        if payload.pop("schema", None) != SCHEMA:
            raise ValueError(f"Expected a {SCHEMA} file")
        return cls(**payload)
