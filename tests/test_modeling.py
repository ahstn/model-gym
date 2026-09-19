"""Pretrained head choices, frozen encoders, and calibrated validation selection."""

from __future__ import annotations

import hashlib

from pathlib import Path

import numpy as np
import pytest
import torch

from transformers import (
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    EvalPrediction,
    ModernBertConfig,
    ModernBertForSequenceClassification,
)

from model_gym.config import ModelConfig
from model_gym.data.schema import RiskRecord, write_jsonl
from model_gym.modeling import build_compute_metrics, build_model, freeze_encoder_layers
from model_gym.policy import DecisionPolicy


@pytest.fixture(params=["modernbert", "deberta-v2"])
def classifier(request):
    common = {
        "vocab_size": 16,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 32,
        "num_labels": 5,
        "pad_token_id": 0,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        if request.param == "modernbert":
            config = ModernBertConfig(
                **common, cls_token_id=1, sep_token_id=2, local_attention=16, reference_compile=False
            )
            config._attn_implementation = "eager"
            model = ModernBertForSequenceClassification(config)
        else:
            config = DebertaV2Config(**common, relative_attention=True, position_buckets=8, conv_kernel_size=3)
            config._attn_implementation = "eager"
            model = DebertaV2ForSequenceClassification(config)
        yield model


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 3, 4, 2, 0], [1, 5, 6, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]),
    }


@pytest.mark.parametrize("count", [1, -1])
def test_freezing_blocks_keeps_the_classifier_learning(classifier, count: int) -> None:
    model = classifier
    freeze_encoder_layers(model, count)
    encoder = model.base_model
    blocks = encoder.layers if model.config.model_type == "modernbert" else encoder.encoder.layer
    frozen_modules = [encoder] if count == -1 else [encoder.embeddings, blocks[0]]
    if count == 1 and model.config.model_type == "deberta-v2":
        frozen_modules.append(encoder.encoder.conv)
    frozen = [(parameter, parameter.detach().clone()) for module in frozen_modules for parameter in module.parameters()]
    head_before = model.classifier.weight.detach().clone()
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.1)
    loss = model(**_inputs(), labels=torch.tensor([0, 4])).loss
    loss.backward()
    assert all(parameter.grad is None for parameter, _ in frozen)
    if count == 1:
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in blocks[1].parameters()
        )
    optimizer.step()
    assert all(torch.equal(parameter, before) for parameter, before in frozen)
    assert not torch.equal(model.classifier.weight, head_before)


def test_freezing_more_blocks_than_exist_rejects_without_partial_freezing(classifier) -> None:
    with pytest.raises(ValueError, match="cannot freeze 3 encoder layers"):
        freeze_encoder_layers(classifier, 3)
    assert all(parameter.requires_grad for parameter in classifier.parameters())


def test_modernbert_pooling_and_dropout_survive_pretrained_reload(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        config = ModernBertConfig(
            vocab_size=16,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            num_labels=5,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            reference_compile=False,
            attention_dropout=0.0,
            embedding_dropout=0.0,
            mlp_dropout=0.0,
        )
        config._attn_implementation = "eager"
        source = ModernBertForSequenceClassification(config)
        source.save_pretrained(tmp_path / "source")
        model = build_model(
            ModelConfig(
                name=str(tmp_path / "source"), pooling="mean", classifier_dropout=0.4, attn_implementation="eager"
            )
        )
        model.save_pretrained(tmp_path / "saved")
        reloaded = build_model(ModelConfig(name=str(tmp_path / "saved"), attn_implementation="eager"))
        inputs = _inputs()
        mask = inputs["attention_mask"].unsqueeze(-1)
        for candidate in (model, reloaded):
            candidate.eval()
            with torch.no_grad():
                hidden = candidate.model(**inputs).last_hidden_state
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                head = candidate.head(pooled)
                expected = candidate.classifier(head)
                torch.testing.assert_close(candidate(**inputs).logits, expected)
                # Encoder dropout is zero; only the configured classifier dropout changes this output.
                torch.manual_seed(41)
                dropped = torch.nn.functional.dropout(head, p=0.4, training=True)
                expected = candidate.classifier(dropped)
                torch.manual_seed(41)
                candidate.train()
                torch.testing.assert_close(candidate(**inputs).logits, expected)


def test_deberta_pooling_override_is_rejected_instead_of_ignored(tmp_path: Path) -> None:
    config = DebertaV2Config(
        vocab_size=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=5
    )
    config.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="pooling overrides are not supported"):
        build_model(ModelConfig(name=str(tmp_path), pooling="mean", attn_implementation="eager"))


def test_calibrated_callback_changes_runtime_decisions_and_reports_temperature(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(
        validation_path,
        [RiskRecord(command=f"cmd {index}", level=1 if index < 4 else 2) for index in range(5)],
    )
    digest = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    prediction = EvalPrediction(
        predictions=np.tile([6.0, 0.0, 0.0, 0.0, 0.0], (5, 1)),
        label_ids=np.array([0, 0, 0, 0, 1]),
    )
    policy = DecisionPolicy(kind="risk_thresholds", block_threshold=0.9)
    raw = build_compute_metrics(policy=policy)(prediction)
    calibrated = build_compute_metrics(policy=policy, calibrate=True, validation_sha256=digest)(prediction)
    assert raw["runtime_critical_miss_rate"] == 0.0
    assert calibrated["runtime_critical_miss_rate"] == 1.0
    assert calibrated["calibration_temperature"] == pytest.approx(6 / np.log(16))
    assert calibrated["qwk"] == raw["qwk"]
    with pytest.raises(ValueError, match="validation file"):
        build_compute_metrics(policy=policy, calibrate=True)
