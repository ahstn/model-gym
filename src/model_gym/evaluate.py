"""Checkpoint evaluation: metrics, reports, and the mistakes that matter."""

from __future__ import annotations

import dataclasses
import hashlib
import json

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

from model_gym.calibration import CALIBRATION_NAME, fit_temperature, load_calibration
from model_gym.config import ModelConfig
from model_gym.data.schema import RiskRecord
from model_gym.labels import decision_for_score, index_level, level_of, risk_score
from model_gym.metrics import Metrics, compute_metrics, softmax
from model_gym.modeling import configure_runtime, load_split, predict_logits, predict_proba, resolve_device
from model_gym.policy import POLICY_NAME, DecisionPolicy, load_policy


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Probabilities, truth, and metrics for one split."""

    split: str
    metrics: Metrics
    probabilities: np.ndarray
    labels: np.ndarray
    records: tuple[RiskRecord, ...]
    policy: DecisionPolicy = dataclasses.field(default_factory=DecisionPolicy)

    def predicted_indices(self) -> np.ndarray:
        return self.probabilities.argmax(axis=1)

    def predictions(self) -> list[dict[str, Any]]:
        """Per-row predictions, in corpus order."""
        predicted = self.predicted_indices()
        decisions = self.policy.decide(self.probabilities)
        rows: list[dict[str, Any]] = []
        for record, truth, guess, probabilities, decision in zip(
            self.records, self.labels, predicted, self.probabilities, decisions, strict=True
        ):
            expected = risk_score(probabilities)
            rows.append(
                {
                    "command": record.command,
                    "group": record.group,
                    "shell": record.shell,
                    "source": record.source,
                    "category": record.category,
                    "true_level": int(index_level(int(truth))),
                    "true_slug": record.slug,
                    "predicted_level": level_of(int(guess)),
                    "predicted_slug": index_level(int(guess)).slug,
                    "argmax_decision": str(index_level(int(guess)).decision),
                    "score_decision": str(decision_for_score(expected)),
                    "runtime_decision": str(decision),
                    "confidence": float(probabilities.max()),
                    "expected_level": expected,
                    "probabilities": [float(value) for value in probabilities],
                }
            )
        return rows

    def severe_misses(self, limit: int = 25) -> list[dict[str, Any]]:
        """Unsafe commands classified as low risk, worst first."""
        rows = [row for row in self.predictions() if row["true_level"] <= 2 and row["predicted_level"] >= 4]
        rows.sort(key=lambda row: (-row["confidence"], row["true_level"]))
        return rows[:limit]

    def over_blocks(self, limit: int = 25) -> list[dict[str, Any]]:
        """Low-risk commands classified as critical or dangerous, worst first."""
        rows = [row for row in self.predictions() if row["true_level"] >= 4 and row["predicted_level"] <= 2]
        rows.sort(key=lambda row: (-row["confidence"], row["predicted_level"]))
        return rows[:limit]

    def policy_mistakes(self, limit: int = 25) -> dict[str, list[dict[str, Any]]]:
        """Broader decision errors, including level 3 approval bypasses."""
        mistakes: dict[str, list[dict[str, Any]]] = {
            "critical_misses": [],
            "unsafe_allows": [],
            "approval_bypasses": [],
            "unnecessary_interventions": [],
            "score_critical_misses": [],
            "score_unsafe_allows": [],
            "score_approval_bypasses": [],
            "score_unnecessary_interventions": [],
            "runtime_critical_misses": [],
            "runtime_unsafe_allows": [],
            "runtime_approval_bypasses": [],
            "runtime_unnecessary_interventions": [],
        }
        for row in self.predictions():
            truth, prediction = row["true_level"], row["predicted_level"]
            if truth == 1 and prediction != 1:
                mistakes["critical_misses"].append(row)
            if truth <= 3 and prediction == 5:
                mistakes["unsafe_allows"].append(row)
            if truth <= 3 and prediction >= 4:
                mistakes["approval_bypasses"].append(row)
            if truth >= 4 and prediction <= 3:
                mistakes["unnecessary_interventions"].append(row)
            decision = row["score_decision"]
            if truth == 1 and decision != "block":
                mistakes["score_critical_misses"].append(row)
            if truth <= 3 and decision == "allow":
                mistakes["score_unsafe_allows"].append(row)
            if truth <= 3 and decision in {"flag", "allow"}:
                mistakes["score_approval_bypasses"].append(row)
            if truth >= 4 and decision in {"block", "approve"}:
                mistakes["score_unnecessary_interventions"].append(row)
            decision = row["runtime_decision"]
            if truth == 1 and decision != "block":
                mistakes["runtime_critical_misses"].append(row)
            if truth <= 3 and decision == "allow":
                mistakes["runtime_unsafe_allows"].append(row)
            if truth <= 3 and decision in {"flag", "allow"}:
                mistakes["runtime_approval_bypasses"].append(row)
            if truth >= 4 and decision in {"block", "approve"}:
                mistakes["runtime_unnecessary_interventions"].append(row)
        for name, rows in mistakes.items():
            rows.sort(key=lambda row: (-row["confidence"], row["true_level"]))
            mistakes[name] = rows[:limit]
        return mistakes

    def breakdowns(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Category/source metrics with explicit sample and policy denominators."""
        breakdowns = {}
        for field in ("category", "source"):
            groups: dict[str, list[int]] = {}
            for index, record in enumerate(self.records):
                groups.setdefault(getattr(record, field), []).append(index)
            breakdowns[field] = {
                name: {
                    "support": len(indices),
                    "metrics": compute_metrics(
                        self.probabilities[indices], self.labels[indices], policy=self.policy
                    ).to_dict(),
                }
                for name, indices in sorted(groups.items())
            }
        return breakdowns


def load_checkpoint(
    model_dir: Path | str,
    *,
    device: torch.device | None = None,
) -> tuple[PreTrainedModel, Any]:
    """Load a checkpoint; ``predict_proba`` applies its validated saved temperature."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    calibration = load_calibration(model_dir)
    device = device or resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model._model_gym_temperature = calibration["temperature"] if calibration is not None else 1.0
    policy_path = model_dir / POLICY_NAME
    model._model_gym_policy = load_policy(policy_path) if policy_path.exists() else DecisionPolicy()
    model.eval()
    model.to(device)
    return model, tokenizer


def calibrate_checkpoint(
    model_dir: Path | str,
    processed_dir: Path | str,
    *,
    max_seq_length: int = 256,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Fit from raw validation logits only and atomically save ``calibration.json``.

    Repeating the fit never compounds an earlier temperature. Test records are
    neither opened nor used to select the temperature.
    """
    configure_runtime()
    model_dir = Path(model_dir)
    validation_path = Path(processed_dir) / "validation.jsonl"
    records = load_split(processed_dir, "validation")
    validation_sha256 = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    device = device or resolve_device()
    model, tokenizer = load_checkpoint(model_dir, device=device)
    logits = predict_logits(
        model,
        tokenizer,
        [record.command for record in records],
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        device=device,
    )
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    calibration = fit_temperature(logits, labels, validation_sha256=validation_sha256)
    calibration["max_seq_length"] = max_seq_length
    calibration["decision_policy"] = model._model_gym_policy.to_dict()
    calibration["validation_metrics"] = compute_metrics(
        softmax(logits / calibration["temperature"]), labels, policy=model._model_gym_policy
    ).to_dict()
    path = model_dir / CALIBRATION_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return calibration


def evaluate_model(
    model_dir: Path | str,
    processed_dir: Path | str,
    *,
    split: str = "test",
    batch_size: int = 64,
    max_seq_length: int = 256,
    model_config: ModelConfig | None = None,
    device: torch.device | None = None,
) -> EvaluationResult:
    """Score one split with a saved checkpoint."""
    configure_runtime()
    if model_config is not None and model_config.attn_implementation == "flash_attention_2":
        max_seq_length = min(max_seq_length, model_config.max_seq_length)
    device = device or resolve_device()
    model, tokenizer = load_checkpoint(model_dir, device=device)
    records = load_split(processed_dir, split)
    probabilities = predict_proba(
        model,
        tokenizer,
        [record.command for record in records],
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        device=device,
    )
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    return EvaluationResult(
        split=split,
        metrics=compute_metrics(probabilities, labels, policy=model._model_gym_policy),
        probabilities=probabilities,
        labels=labels,
        records=tuple(records),
        policy=model._model_gym_policy,
    )


def _error_table(rows: Sequence[dict[str, Any]], empty: str) -> list[str]:
    if not rows:
        return [empty, ""]
    lines = [
        "| true | predicted | runtime decision | confidence | category | source | command |",
        "| ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        command = row["command"].replace("|", "\\|")
        category = row["category"].replace("|", "\\|")
        source = row["source"].replace("|", "\\|")
        lines.append(
            f"| {row['true_level']} {row['true_slug']} | {row['predicted_level']} {row['predicted_slug']} "
            f"| {row['runtime_decision']} | {row['confidence']:.3f} | {category} | {source} | `{command}` |"
        )
    lines.append("")
    return lines


def _breakdown_table(field: str, groups: dict[str, dict[str, Any]]) -> list[str]:
    policies = (
        ("critical_miss_rate", "critical_support"),
        ("unsafe_allow_rate", "requires_approval_support"),
        ("approval_bypass_rate", "requires_approval_support"),
        ("allow_precision", "allow_support"),
        ("allow_coverage", "n"),
        ("unnecessary_intervention_rate", "benign_support"),
    )
    headers = [field, "support", "macro F1", *(metric for metric, _ in policies)]
    lines = [
        f"## By {field}\n",
        "Policy cells use the saved runtime policy: value (denominator). Zero support is not release evidence.\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for name, group in groups.items():
        metrics = group["metrics"]
        cells = [name.replace("|", "\\|"), str(group["support"]), f"{metrics['macro_f1']:.4f}"]
        policy = metrics["runtime_policy"]
        cells.extend(f"{policy[metric]:.4f} ({policy[support]})" for metric, support in policies)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def format_report(result: EvaluationResult, title: str | None = None) -> str:
    """Metrics, category/source support, and high-confidence ordinal/policy errors."""
    title = title or f"{result.split} split"
    lines = [result.metrics.format(title=title)]
    lines.append("## Severe misses\n")
    lines.extend(_error_table(result.severe_misses(), "No level 1-2 command was classified as level 4-5."))
    lines.append("## Over blocks\n")
    lines.extend(_error_table(result.over_blocks(), "No level 4-5 command was classified as level 1-2."))
    for name, rows in result.policy_mistakes().items():
        lines.append(f"## {name.replace('_', ' ').capitalize()}\n")
        lines.extend(_error_table(rows, "No observed errors of this policy type."))
    for field, groups in result.breakdowns().items():
        lines.extend(_breakdown_table(field, groups))
    return "\n".join(lines)


def write_report(result: EvaluationResult, report_dir: Path | str, title: str | None = None) -> tuple[Path, Path]:
    """Write ``<split>.md`` and ``<split>.json`` into ``report_dir``."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / f"{result.split}.md"
    json_path = report_dir / f"{result.split}.json"
    markdown_path.write_text(format_report(result, title=title), encoding="utf-8")
    payload = {
        "split": result.split,
        "metrics": result.metrics.to_dict(),
        "decision_policy": result.policy.to_dict(),
        "severe_misses": result.severe_misses(),
        "over_blocks": result.over_blocks(),
        "policy_mistakes": result.policy_mistakes(),
        "breakdowns": result.breakdowns(),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return markdown_path, json_path


def write_predictions(result: EvaluationResult, path: Path | str) -> int:
    """Write per-row predictions as JSONL for error analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = result.predictions()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return len(rows)
