"""Action-aligned and ordinal metrics for the five-level risk classifier.

Actions are L1 block, L2-3 approve, L4 flag, and L5 allow. Safety metrics
distinguish missed critical blocks, unsafe allows, and approval bypasses; benign
intervention includes both blocking and requiring approval. Allow precision uses
predicted L5 rows as its denominator; allow coverage uses all evaluated rows.

Legacy ordinal rates remain available: ``severe_miss_rate`` measures true L1-2
predicted L4-5 (flag or allow), and ``over_block_rate`` measures true L4-5
predicted L1-2 (block or approve), not actual blocks alone.

Every action rate has an explicit count and denominator. A zero denominator
produces zero, never evidence of safety or release eligibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from model_gym.labels import LABELS, LEVELS, NUM_LABELS, decision_for_index, risk_score
from model_gym.policy import DecisionPolicy, decision_metrics

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
    score_policy: dict[str, float | int]
    runtime_policy: dict[str, float | int]
    policy_kind: str
    critical_miss_rate: float
    critical_miss_count: int
    critical_support: int
    unsafe_allow_rate: float
    unsafe_allow_count: int
    requires_approval_support: int
    approval_bypass_rate: float
    approval_bypass_count: int
    allow_precision: float
    allow_correct_count: int
    allow_coverage: float
    allow_support: int
    unnecessary_intervention_rate: float
    unnecessary_intervention_count: int
    benign_support: int
    severe_miss_rate: float
    severe_miss_count: int
    severe_miss_support: int
    over_block_rate: float
    over_block_count: int
    over_block_support: int
    per_level_support: tuple[int, ...]
    per_level_recall: tuple[float, ...]
    per_level_precision: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]

    def headline(self) -> dict[str, float | int]:
        """Selection scalars; a zero rate is meaningful only with nonzero support."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "qwk": self.qwk,
            "mae": self.mae,
            "critical_miss_rate": self.critical_miss_rate,
            "critical_miss_count": self.critical_miss_count,
            "critical_support": self.critical_support,
            "unsafe_allow_rate": self.unsafe_allow_rate,
            "unsafe_allow_count": self.unsafe_allow_count,
            "requires_approval_support": self.requires_approval_support,
            "approval_bypass_rate": self.approval_bypass_rate,
            "approval_bypass_count": self.approval_bypass_count,
            "allow_precision": self.allow_precision,
            "allow_correct_count": self.allow_correct_count,
            "allow_coverage": self.allow_coverage,
            "allow_support": self.allow_support,
            "unnecessary_intervention_rate": self.unnecessary_intervention_rate,
            "unnecessary_intervention_count": self.unnecessary_intervention_count,
            "benign_support": self.benign_support,
            "severe_miss_rate": self.severe_miss_rate,
            "severe_miss_count": self.severe_miss_count,
            "severe_miss_support": self.severe_miss_support,
            "over_block_rate": self.over_block_rate,
            "over_block_count": self.over_block_count,
            "over_block_support": self.over_block_support,
            **{f"score_{key}": value for key, value in self.score_policy.items()},
            **{f"runtime_{key}": value for key, value in self.runtime_policy.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "n": self.n,
            **self.headline(),
            # Chance agreement is undefined for a single perfectly predicted class.
            "qwk": self.qwk if np.isfinite(self.qwk) else None,
            "ece": self.ece,
            "mean_expected_level": self.mean_expected_level,
            "decision_accuracy": self.decision_accuracy,
            "score_policy": self.score_policy,
            "runtime_policy": self.runtime_policy,
            "policy_kind": self.policy_kind,
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
                "critical miss rate",
                self.critical_miss_rate,
                f"{self.critical_miss_count}/{self.critical_support} true L1 rows not blocked (predicted L2-5)",
            ),
            (
                "unsafe allow rate",
                self.unsafe_allow_rate,
                f"{self.unsafe_allow_count}/{self.requires_approval_support} true L1-3 rows predicted L5 (allow)",
            ),
            (
                "approval bypass rate",
                self.approval_bypass_rate,
                f"{self.approval_bypass_count}/{self.requires_approval_support} true L1-3 predicted L4-5 (flag/allow)",
            ),
            (
                "allow precision",
                self.allow_precision,
                f"{self.allow_correct_count}/{self.allow_support} predicted L5 rows truly L5",
            ),
            (
                "allow coverage",
                self.allow_coverage,
                f"{self.allow_support}/{self.n} evaluated rows predicted L5",
            ),
            (
                "unnecessary intervention rate",
                self.unnecessary_intervention_rate,
                f"{self.unnecessary_intervention_count}/{self.benign_support} true L4-5 predicted L1-3 (block/approve)",
            ),
            (
                "severe miss rate (legacy ordinal)",
                self.severe_miss_rate,
                f"{self.severe_miss_count}/{self.severe_miss_support} true L1-2 rows predicted L4-5 (flag/allow)",
            ),
            (
                "over block rate (legacy ordinal)",
                self.over_block_rate,
                f"{self.over_block_count}/{self.over_block_support} true L4-5 rows predicted L1-2 (block/approve)",
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
        for name, value, meaning in summary:
            rendered = f"{value:.4f}" if np.isfinite(value) else "undefined"
            lines.append(f"| {name} | {rendered} | {meaning} |")
        lines.extend(
            [
                "",
                "Zero-support rates are 0; absent support is not evidence of safety or release eligibility.",
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
        lines.extend(
            [
                "",
                f"## Runtime policy: {self.policy_kind}",
                "",
                "The action metrics above map the argmax level to a decision.",
                "Safety selection uses the explicit runtime policy below.",
                "Calibration can change these decisions even when argmax is unchanged.",
                "",
                "| metric | value | denominator |",
                "| --- | ---: | ---: |",
            ]
        )
        for metric, support in (
            ("decision_accuracy", "n"),
            ("critical_miss_rate", "critical_support"),
            ("unsafe_allow_rate", "requires_approval_support"),
            ("approval_bypass_rate", "requires_approval_support"),
            ("allow_precision", "allow_support"),
            ("allow_coverage", "n"),
            ("unnecessary_intervention_rate", "benign_support"),
        ):
            lines.append(f"| {metric} | {self.runtime_policy[metric]:.4f} | {self.runtime_policy[support]} |")
        return "\n".join(lines) + "\n"


def compute_metrics(
    probabilities: np.ndarray, labels: Sequence[int] | np.ndarray, *, policy: DecisionPolicy | None = None
) -> Metrics:
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

    critical_support = int(supports[0])
    critical_misses = int(matrix[0, 1:].sum())
    requires_approval_support = int(supports[:3].sum())
    unsafe_allows = int(matrix[:3, 4].sum())
    approval_bypasses = int(matrix[:3, 3:].sum())
    allow_support = int(predicted_totals[4])
    allow_correct = int(matrix[4, 4])
    benign_support = int(supports[3:].sum())
    unnecessary_interventions = int(matrix[3:, :3].sum())

    true_decisions = [decision_for_index(int(index)) for index in labels]
    predicted_decisions = [decision_for_index(int(index)) for index in predicted]
    decision_hits = sum(1 for left, right in zip(true_decisions, predicted_decisions, strict=True) if left is right)
    scores = np.asarray([risk_score(row) for row in probabilities])
    qwk = (
        float("nan")
        if np.count_nonzero(matrix) == 1 and int(np.trace(matrix)) == len(labels)
        else float(cohen_kappa_score(labels, predicted, weights="quadratic", labels=list(range(NUM_LABELS))))
    )
    policy = policy or DecisionPolicy()
    score_metrics = decision_metrics(probabilities, labels, DecisionPolicy())

    return Metrics(
        n=len(labels),
        accuracy=float((predicted == labels).mean()),
        macro_f1=float(f1_score(labels, predicted, labels=list(range(NUM_LABELS)), average="macro", zero_division=0)),
        qwk=qwk,
        mae=float(np.abs(predicted - labels).mean()),
        ece=expected_calibration_error(probabilities, labels),
        mean_expected_level=float(scores.mean()),
        decision_accuracy=decision_hits / len(labels),
        score_policy=score_metrics,
        runtime_policy=score_metrics
        if policy.kind == "score_bands"
        else decision_metrics(probabilities, labels, policy),
        policy_kind=policy.kind,
        critical_miss_rate=critical_misses / critical_support if critical_support else 0.0,
        critical_miss_count=critical_misses,
        critical_support=critical_support,
        unsafe_allow_rate=unsafe_allows / requires_approval_support if requires_approval_support else 0.0,
        unsafe_allow_count=unsafe_allows,
        requires_approval_support=requires_approval_support,
        approval_bypass_rate=approval_bypasses / requires_approval_support if requires_approval_support else 0.0,
        approval_bypass_count=approval_bypasses,
        allow_precision=allow_correct / allow_support if allow_support else 0.0,
        allow_correct_count=allow_correct,
        allow_coverage=allow_support / len(labels),
        allow_support=allow_support,
        unnecessary_intervention_rate=unnecessary_interventions / benign_support if benign_support else 0.0,
        unnecessary_intervention_count=unnecessary_interventions,
        benign_support=benign_support,
        severe_miss_rate=severe_misses / int(severe_mask.sum()) if severe_mask.any() else 0.0,
        severe_miss_count=severe_misses,
        severe_miss_support=int(severe_mask.sum()),
        over_block_rate=over_blocks / int(over_block_mask.sum()) if over_block_mask.any() else 0.0,
        over_block_count=over_blocks,
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
