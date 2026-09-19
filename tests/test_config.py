"""Configuration loading, overrides, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_gym.config import ConfigError, load_config


def test_yaml_values_are_loaded_and_overrides_win(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "model:\n  name: answerdotai/ModernBERT-large\n  max_seq_length: 512\ntrain:\n  learning_rate: 1.0e-5\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.model.name == "answerdotai/ModernBERT-large"
    assert config.model.max_seq_length == 512
    assert config.train.learning_rate == 1e-5

    overridden = load_config(path, ["train.learning_rate=4e-5", "train.epochs=2"])
    assert overridden.train.learning_rate == 4e-5
    assert overridden.train.epochs == 2
    assert overridden.model.max_seq_length == 512


def test_tuple_override_accepts_a_list_or_a_comma_separated_string() -> None:
    from_list = load_config(None, ['data.ratios="[0.8, 0.1, 0.1]"'])
    from_string = load_config(None, ["data.ratios=0.8,0.1,0.1"])
    assert from_list.data.ratios == (0.8, 0.1, 0.1)
    assert from_string.data.ratios == (0.8, 0.1, 0.1)


def test_boolean_and_numeric_overrides_are_coerced() -> None:
    config = load_config(None, ["train.bf16=false", "train.batch_size=8"])
    assert config.train.bf16 is False
    assert config.train.batch_size == 8


@pytest.mark.parametrize(
    "override",
    ["train.learning_rate", "nope.key=1", "train.nope=1", "train.bf16=maybe"],
)
def test_bad_overrides_are_rejected(override: str) -> None:
    with pytest.raises(ConfigError):
        load_config(None, [override])


def test_unknown_config_sections_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("trian:\n  epochs: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown config sections"):
        load_config(path)


def test_missing_config_file_is_reported() -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config("configs/does-not-exist.yaml")


def test_mutually_exclusive_precisions_are_rejected() -> None:
    with pytest.raises(ConfigError, match="mutually exclusive"):
        load_config(None, ["train.bf16=true", "train.fp16=true"])


@pytest.mark.parametrize(
    "override",
    [
        "train.learning_rate=0",
        "train.gradient_accumulation_steps=0",
        "model.max_seq_length=4",
        "model.attn_implementation=bogus",
        "model.pooling=max",
        "model.classifier_dropout=-0.5",
        "model.classifier_dropout=1",
        "model.classifier_dropout=.nan",
        "train.class_weighting=bogus",
        "train.freeze_encoder_layers=-2",
        "train.freeze_encoder_layers=1.5",
        "train.ordinal_loss_weight=-0.1",
        "train.ordinal_loss_weight=.inf",
        "train.ordinal_loss_weight=.nan",
        "export.precision=int4",
        "export.quant_target=mips",
        "data.ratios=0.9,0.2,0.2",
        "train.selection_metric=accuracy",
        "train.max_critical_miss_rate=-0.1",
        "train.max_unsafe_allow_rate=1",
        "train.max_unnecessary_intervention_rate=.nan",
    ],
)
def test_invalid_values_are_rejected(override: str) -> None:
    with pytest.raises(ConfigError):
        load_config(None, [override])
