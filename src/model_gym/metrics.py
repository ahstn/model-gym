"""Ordinal-aware metrics for the five-level risk classifier.

Plain accuracy hides the failure that matters most here: a critical command
classified as low risk. These metrics therefore report, on top of accuracy and
macro F1, quadratic weighted kappa (ordinal agreement), mean absolute level error,
and the two asymmetric error rates a guardrail cares about:

* ``severe_miss_rate`` -- true level 1-2 predicted as level 4-5 (unsafe allowed).
* ``over_block_rate``  -- true level 4-5 predicted as level 1-2 (safe blocked).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from model_gym.labels import LABELS, LEVELS, NUM_LABELS, decision_for_index, risk_score

SEVERE_LEVELS: tuple[int, ...] = (1, 2)
SAFE_LEVELS: tuple[int, ...] = (4, 5)
ECE_BINS: int = 15


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted for numerical stability."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != NUM_LABELS:
        raise ValueError(f"expected logits shaped (n, {NUM_LABELS}), got {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = ECE_BINS) -> float:
    """Gap between confidence and accuracy, averaged over confidence bins."""
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        error += mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


@dataclass(frozen=True, slots=True)
class Metrics:
    """Evaluation output for one split."""

    n: int
    accuracy: float
    macro_f1: float
    qwk: float
    mae: float
    ece: float
    mean_expected_level: float
    decision_accuracy: float
    severe_miss_rate: float
    severe_miss_support: int
    over_block_rate: float
    over_block_support: int
    per_level_support: tuple[int, ...]
    per_level_recall: tuple[float, ...]
    per_level_precision: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]

    def headline(self) -> dict[str, float]:
        """Flat scalar metrics, safe for ``Trainer`` logging and model selection."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "qwk": self.qwk,
            "mae": self.mae,
            "severe_miss_rate": self.severe_miss_rate,
            "over_block_rate": self.over_block_rate,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "n": self.n,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "qwk": self.qwk,
            "mae": self.mae,
            "ece": self.ece,
            "mean_expected_level": self.mean_expected_level,
            "decision_accuracy": self.decision_accuracy,
            "severe_miss_rate": self.severe_miss_rate,
            "severe_miss_support": self.severe_miss_support,
            "over_block_rate": self.over_block_rate,
            "over_block_support": self.over_block_support,
            "levels": [
                {
                    "level": int(level),
                    "slug": level.slug,
                    "support": support,
                    "recall": recall,
                    "precision": precision,
                }
                for level, support, recall, precision in zip(
                    LEVELS, self.per_level_support, self.per_level_recall, self.per_level_precision, strict=True
                )
            ],
            "confusion": [list(row) for row in self.confusion],
        }
        return payload

    def format(self, title: str = "evaluation") -> str:
        """Markdown report with the summary, per-level rows, and the confusion matrix."""
        summary = (
            (
                "accuracy",
                self.accuracy,
                "exact level match",
            ),
            (
                "decision accuracy",
                self.decision_accuracy,
                "match after mapping levels to block/approve/flag/allow",
            ),
            ("macro F1", self.macro_f1, "unweighted mean over the 5 levels"),
            ("quadratic weighted kappa", self.qwk, "ordinal agreement, 1.0 is perfect"),
            ("mean absolute level error", self.mae, "levels between prediction and truth"),
            ("expected calibration error", self.ece, f"confidence versus accuracy, {ECE_BINS} bins"),
            ("mean expected level", self.mean_expected_level, "mean E[level]; 1.0 critical, 5.0 low risk"),
            (
                "severe miss rate",
                self.severe_miss_rate,
                f"true level 1-2 predicted 4-5, over {self.severe_miss_support} rows",
            ),
            (
                "over block rate",
                self.over_block_rate,
                f"true level 4-5 predicted 1-2, over {self.over_block_support} rows",
            ),
        )
        lines = [
            f"# {title}",
            "",
            f"rows: {self.n}",
            "",
            "| metric | value | meaning |",
            "| --- | ---: | --- |",
        ]
        lines.extend(f"| {name} | {value:.4f} | {meaning} |" for name, value, meaning in summary)
        lines.extend(
            [
                "",
                "| level | support | recall | precision |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for level, support, recall, precision in zip(
            LEVELS, self.per_level_support, self.per_level_recall, self.per_level_precision, strict=True
        ):
            lines.append(f"| {int(level)} {level.slug} | {support} | {recall:.4f} | {precision:.4f} |")
        lines.extend(["", "confusion matrix (rows true, columns predicted):", ""])
        header = "| true \\ pred | " + " | ".join(str(int(level)) for level in LEVELS) + " |"
        lines.append(header)
        lines.append("| --- | " + " | ".join("---:" for _ in LEVELS) + " |")
        for level, row in zip(LEVELS, self.confusion, strict=True):
            lines.append(f"| **{int(level)} {level.slug}** | " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines) + "\n"


def compute_metrics(probabilities: np.ndarray, labels: Sequence[int] | np.ndarray) -> Metrics:
    """Compute every reported metric from probabilities and true label indices."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_LABELS:
        raise ValueError(f"expected probabilities shaped (n, {NUM_LABELS}), got {probabilities.shape}")
    if len(probabilities) != len(labels):
        raise ValueError(f"got {len(probabilities)} rows but {len(labels)} labels")
    if len(labels) == 0:
        raise ValueError("no rows to evaluate")

    predicted = probabilities.argmax(axis=1)
    predicted_levels = predicted + 1
    true_levels = labels + 1

    matrix = confusion_matrix(labels, predicted, labels=list(range(NUM_LABELS)))
    supports = matrix.sum(axis=1)
    true_positives = np.diag(matrix)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(supports > 0, true_positives / supports, 0.0)
        predicted_totals = matrix.sum(axis=0)
        precision = np.where(predicted_totals > 0, true_positives / predicted_totals, 0.0)

    severe_mask = np.isin(true_levels, SEVERE_LEVELS)
    over_block_mask = np.isin(true_levels, SAFE_LEVELS)
    severe_misses = int(np.sum(np.isin(predicted_levels[severe_mask], SAFE_LEVELS)))
    over_blocks = int(np.sum(np.isin(predicted_levels[over_block_mask], SEVERE_LEVELS)))

    true_decisions = [decision_for_index(int(index)) for index in labels]
    predicted_decisions = [decision_for_index(int(index)) for index in predicted]
    decision_hits = sum(1 for left, right in zip(true_decisions, predicted_decisions, strict=True) if left is right)

    return Metrics(
        n=len(labels),
        accuracy=float((predicted == labels).mean()),
        macro_f1=float(f1_score(labels, predicted, labels=list(range(NUM_LABELS)), average="macro", zero_division=0)),
        qwk=float(cohen_kappa_score(labels, predicted, weights="quadratic", labels=list(range(NUM_LABELS)))),
        mae=float(np.abs(predicted - labels).mean()),
        ece=expected_calibration_error(probabilities, labels),
        mean_expected_level=float(np.mean([risk_score(row) for row in probabilities])),
        decision_accuracy=decision_hits / len(labels),
        severe_miss_rate=severe_misses / int(severe_mask.sum()) if severe_mask.any() else 0.0,
        severe_miss_support=int(severe_mask.sum()),
        over_block_rate=over_blocks / int(over_block_mask.sum()) if over_block_mask.any() else 0.0,
        over_block_support=int(over_block_mask.sum()),
        per_level_support=tuple(int(value) for value in supports),
        per_level_recall=tuple(float(value) for value in recall),
        per_level_precision=tuple(float(value) for value in precision),
        confusion=tuple(tuple(int(value) for value in row) for row in matrix),
    )


def labels_from_frame(frame: Any) -> np.ndarray:
    """Extract label indices from an HF dataset split, erroring early on a bad column."""
    if "labels" not in frame.column_names:
        raise ValueError(f"dataset split is missing the 'labels' column; has {frame.column_names}")
    return np.asarray(frame["labels"], dtype=np.int64)


def format_level_table() -> str:
    """Markdown table of the taxonomy, for reports and the README."""
    lines = ["| index | level | slug | decision | meaning |", "| ---: | ---: | --- | --- | --- |"]
    for level in LEVELS:
        lines.append(f"| {level.index} | {int(level)} | {level.slug} | {level.decision} | {level.description} |")
    return "\n".join(lines)


def class_counts(labels: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    """Counts per label index, including zero-count levels."""
    labels = np.asarray(labels, dtype=np.int64).ravel()
    return tuple(int((labels == index).sum()) for index in range(NUM_LABELS))


__all__ = [
    "LABELS",
    "Metrics",
    "class_counts",
    "compute_metrics",
    "expected_calibration_error",
    "format_level_table",
    "labels_from_frame",
    "softmax",
]
