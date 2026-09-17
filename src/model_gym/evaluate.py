"""Checkpoint evaluation: metrics, reports, and the mistakes that matter."""

from __future__ import annotations

import json

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

from model_gym.config import ModelConfig
from model_gym.data.schema import RiskRecord
from model_gym.labels import LEVELS, index_level, level_of
from model_gym.metrics import Metrics, compute_metrics
from model_gym.modeling import configure_runtime, load_split, predict_proba, resolve_device


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Probabilities, truth, and metrics for one split."""

    split: str
    metrics: Metrics
    probabilities: np.ndarray
    labels: np.ndarray
    records: tuple[RiskRecord, ...]

    def predicted_indices(self) -> np.ndarray:
        return self.probabilities.argmax(axis=1)

    def predictions(self) -> list[dict[str, Any]]:
        """Per-row predictions, in corpus order."""
        predicted = self.predicted_indices()
        rows: list[dict[str, Any]] = []
        for record, truth, guess, probabilities in zip(
            self.records, self.labels, predicted, self.probabilities, strict=True
        ):
            expected = float(sum(prob * int(level) for prob, level in zip(probabilities, LEVELS, strict=True)))
            rows.append(
                {
                    "command": record.command,
                    "group": record.group,
                    "shell": record.shell,
                    "source": record.source,
                    "true_level": int(index_level(int(truth))),
                    "true_slug": record.slug,
                    "predicted_level": level_of(int(guess)),
                    "predicted_slug": index_level(int(guess)).slug,
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


def load_checkpoint(
    model_dir: Path | str,
    *,
    device: torch.device | None = None,
) -> tuple[PreTrainedModel, Any]:
    """Load a fine-tuned checkpoint and its tokenizer."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    device = device or resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    return model, tokenizer


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
        metrics=compute_metrics(probabilities, labels),
        probabilities=probabilities,
        labels=labels,
        records=tuple(records),
    )


def _error_table(rows: Sequence[dict[str, Any]], empty: str) -> list[str]:
    if not rows:
        return [empty, ""]
    lines = [
        "| true | predicted | confidence | command |",
        "| ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        command = row["command"].replace("|", "\\|")
        lines.append(
            f"| {row['true_level']} {row['true_slug']} | {row['predicted_level']} {row['predicted_slug']} "
            f"| {row['confidence']:.3f} | `{command}` |"
        )
    lines.append("")
    return lines


def format_report(result: EvaluationResult, title: str | None = None) -> str:
    """Markdown report: metrics, worst severe misses, worst over-blocks."""
    title = title or f"{result.split} split"
    lines = [result.metrics.format(title=title)]
    lines.append("## Severe misses\n")
    lines.extend(_error_table(result.severe_misses(), "No level 1-2 command was classified as level 4-5."))
    lines.append("## Over blocks\n")
    lines.extend(_error_table(result.over_blocks(), "No level 4-5 command was classified as level 1-2."))
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
        "severe_misses": result.severe_misses(),
        "over_blocks": result.over_blocks(),
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
