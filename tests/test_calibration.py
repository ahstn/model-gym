"""Calibration survives checkpoint reloads without ever fitting on held-out rows."""

from __future__ import annotations

import hashlib
import json

from pathlib import Path

import numpy as np
import pytest
import torch

from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from model_gym.calibration import CALIBRATION_NAME, TEMPERATURE_BOUNDS, fit_temperature
from model_gym.data.schema import RiskRecord, write_jsonl
from model_gym.evaluate import EvaluationResult, calibrate_checkpoint, load_checkpoint
from model_gym.metrics import compute_metrics
from model_gym.modeling import predict_logits, predict_proba


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A real local checkpoint with known overconfident logits, no downloads."""
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    vocabulary = directory / "vocab.txt"
    vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncmd\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary))
    tokenizer.save_pretrained(directory)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=6,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            num_labels=5,
        )
    )
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([6.0, 0.0, 0.0, 0.0, 0.0]))
    model.save_pretrained(directory)
    return directory


def test_checkpoint_fit_reload_and_repeat_use_raw_validation_logits(checkpoint: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    records = [RiskRecord(command=f"cmd {index}", level=1 if index < 4 else 2) for index in range(5)]
    write_jsonl(processed / "validation.jsonl", records)
    # Opening test data during fitting would fail; it must remain completely untouched.
    (processed / "test.jsonl").write_text("not valid JSON\n", encoding="utf-8")
    commands = [record.command for record in records]
    cpu = torch.device("cpu")
    model, tokenizer = load_checkpoint(checkpoint, device=cpu)
    uncalibrated = predict_proba(model, tokenizer, commands, max_seq_length=16)
    assert uncalibrated[:, 0] == pytest.approx(np.full(5, np.exp(6) / (np.exp(6) + 4)))

    fitted = calibrate_checkpoint(checkpoint, processed, max_seq_length=16, batch_size=2, device=cpu)
    assert fitted["temperature"] == pytest.approx(6 / np.log(16))
    assert fitted["nll_after"] < fitted["nll_before"]
    assert fitted["validation_sha256"] == hashlib.sha256((processed / "validation.jsonl").read_bytes()).hexdigest()
    assert fitted["sample_count"] == 5
    model, tokenizer = load_checkpoint(checkpoint, device=cpu)
    calibrated = predict_proba(model, tokenizer, commands, max_seq_length=16)
    np.testing.assert_allclose(calibrated, np.tile([0.8, 0.05, 0.05, 0.05, 0.05], (5, 1)), atol=1e-7)
    np.testing.assert_allclose(
        predict_logits(model, tokenizer, commands, max_seq_length=16), np.tile([6, 0, 0, 0, 0], (5, 1))
    )
    repeated = calibrate_checkpoint(checkpoint, processed, max_seq_length=16, batch_size=2, device=cpu)
    assert repeated == fitted
    assert json.loads((checkpoint / CALIBRATION_NAME).read_text(encoding="utf-8")) == fitted


def test_malformed_calibration_fails_instead_of_silently_loading(checkpoint: Path) -> None:
    payload = fit_temperature(np.zeros((1, 5)), np.array([0]), validation_sha256="a" * 64)
    payload["temperature"] = 0
    (checkpoint / CALIBRATION_NAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="temperature"):
        load_checkpoint(checkpoint, device=torch.device("cpu"))


def test_temperature_fit_is_bounded_and_stable_for_extreme_logits() -> None:
    logits = np.asarray([[10000.0, -10000.0, 0.0, 0.0, 0.0]])
    fitted = fit_temperature(logits, np.array([1]), validation_sha256="b" * 64)
    assert fitted["temperature"] == pytest.approx(TEMPERATURE_BOUNDS[1])
    assert np.isfinite(fitted["nll_after"])
    assert fitted["nll_after"] < fitted["nll_before"]


def test_policy_errors_and_breakdown_denominators_include_level_three() -> None:
    records = (
        RiskRecord(command="cmd approve", level=3, source="approval"),
        RiskRecord(command="cmd benign", level=4, source="benign"),
    )
    labels = np.array([2, 3])
    probabilities = np.eye(5)[[4, 2]]
    result = EvaluationResult("validation", compute_metrics(probabilities, labels), probabilities, labels, records)
    mistakes = result.policy_mistakes()
    assert [row["command"] for row in mistakes["unsafe_allows"]] == ["cmd approve"]
    assert [row["command"] for row in mistakes["approval_bypasses"]] == ["cmd approve"]
    assert [row["command"] for row in mistakes["unnecessary_interventions"]] == ["cmd benign"]
    assert result.severe_misses() == []
    assert result.over_blocks() == []
    breakdowns = result.breakdowns()
    approval = breakdowns["source"]["approval"]
    assert approval["support"] == 1
    assert approval["metrics"]["unsafe_allow_rate"] == 1.0
    assert approval["metrics"]["requires_approval_support"] == 1
    assert breakdowns["source"]["benign"]["metrics"]["requires_approval_support"] == 0
    category = breakdowns["category"][records[0].category]
    assert category["support"] == 2
    assert category["metrics"]["requires_approval_support"] == 1
    assert category["metrics"]["benign_support"] == 1
