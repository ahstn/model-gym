"""Training configuration: YAML files plus dotted ``--set key=value`` overrides."""

from __future__ import annotations

import json
import math

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from model_gym.data.build import SPLITS

MODEL_PRESETS: dict[str, str] = {
    "base": "answerdotai/ModernBERT-base",
    "large": "answerdotai/ModernBERT-large",
}

ATTENTION_IMPLEMENTATIONS: tuple[str, ...] = ("sdpa", "flash_attention_2", "eager")
CLASS_WEIGHTING_SCHEMES: tuple[str, ...] = ("none", "sqrt", "balanced")
EXPORT_PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "int8")
QUANT_TARGETS: tuple[str, ...] = ("arm64", "avx2", "avx512", "avx512_vnni")


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the corpus comes from and how it is split."""

    sources: tuple[str, ...] = ("data/seed",)
    output_dir: str = "data/processed"
    synthetic_output: str = "data/synthetic/templates_v1.jsonl"
    include_synthetic: bool = True
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 13

    def validate(self) -> None:
        if not self.sources:
            raise ConfigError("data.sources must list at least one JSONL file or directory")
        if len(self.ratios) != len(SPLITS):
            raise ConfigError(f"data.ratios must have {len(SPLITS)} values, got {list(self.ratios)}")
        if any(ratio < 0 for ratio in self.ratios) or sum(self.ratios) <= 0:
            raise ConfigError(f"data.ratios must be non-negative and sum to a positive value, got {list(self.ratios)}")
        if abs(sum(self.ratios) - 1.0) > 0.01:
            raise ConfigError(f"data.ratios must sum to 1.0, got {list(self.ratios)}")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Base checkpoint and tokenization settings."""

    name: str = MODEL_PRESETS["base"]
    max_seq_length: int = 256
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    pooling: str = "default"
    classifier_dropout: float = -1.0

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("model.name must be set")
        if self.max_seq_length < 8:
            raise ConfigError(f"model.max_seq_length must be at least 8, got {self.max_seq_length}")
        if self.attn_implementation not in ATTENTION_IMPLEMENTATIONS:
            valid = list(ATTENTION_IMPLEMENTATIONS)
            raise ConfigError(f"model.attn_implementation must be one of {valid}, got {self.attn_implementation!r}")
        if self.pooling not in {"default", "mean", "cls"}:
            raise ConfigError("model.pooling must be default, mean, or cls")
        if self.classifier_dropout != -1.0 and not 0 <= self.classifier_dropout < 1:
            raise ConfigError("model.classifier_dropout must be -1 or in [0, 1)")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Everything the trainer needs. Defaults target one RTX 3090."""

    output_dir: str = "runs/modernbert-base-v2"
    run_name: str = ""
    epochs: float = 4.0
    max_steps: int = -1
    learning_rate: float = 3e-5
    batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    freeze_encoder_layers: int = 0
    class_weighting: str = "none"
    ordinal_loss_weight: float = 0.0
    optim: str = "auto"
    logging_steps: int = 25
    save_total_limit: int = 2
    early_stopping_patience: int = 0
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    report_to: tuple[str, ...] = ("none",)
    resume_from_checkpoint: str = ""
    seed: int = 13
    selection_metric: str = "safety"
    decision_policy_path: str = ""
    calibrate_selection: bool = False
    # Experimental validation gates, not a production safety certification.
    max_critical_miss_rate: float = 0.05
    max_unsafe_allow_rate: float = 0.01
    max_unnecessary_intervention_rate: float = 0.25
    # Hold test data out of both tokenization and scoring during model tuning.
    evaluate_test: bool = False

    def validate(self) -> None:
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ConfigError("train.batch_size and train.eval_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ConfigError("train.gradient_accumulation_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigError("train.learning_rate must be > 0")
        if self.epochs <= 0 and self.max_steps <= 0:
            raise ConfigError("set train.epochs > 0 or train.max_steps > 0")
        if self.bf16 and self.fp16:
            raise ConfigError("train.bf16 and train.fp16 are mutually exclusive")
        if self.class_weighting not in CLASS_WEIGHTING_SCHEMES:
            raise ConfigError(
                f"train.class_weighting must be one of {list(CLASS_WEIGHTING_SCHEMES)}, got {self.class_weighting!r}"
            )
        if (
            isinstance(self.freeze_encoder_layers, bool)
            or not isinstance(self.freeze_encoder_layers, int)
            or self.freeze_encoder_layers < -1
        ):
            raise ConfigError("train.freeze_encoder_layers must be -1 or a nonnegative integer")
        if not math.isfinite(self.ordinal_loss_weight) or self.ordinal_loss_weight < 0:
            raise ConfigError("train.ordinal_loss_weight must be finite and nonnegative")
        if self.selection_metric not in {"safety", "qwk"}:
            raise ConfigError("train.selection_metric must be safety or qwk")
        for key in ("max_critical_miss_rate", "max_unsafe_allow_rate", "max_unnecessary_intervention_rate"):
            if not 0 <= getattr(self, key) < 1:
                raise ConfigError(f"train.{key} must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """ONNX export and quantization settings."""

    output_dir: str = "artifacts"
    precision: str = "int8"
    quant_target: str = "arm64"
    opset: int = 18
    do_validation: bool = True
    parity_samples: int = 32
    # INT8 only; fp32 and fp16 exports must agree exactly. Rows whose top two
    # probabilities are closer than parity_tie_epsilon are excluded from the gate,
    # because their argmax is a coin flip in either runtime. A broken graph lands
    # near chance agreement (0.2), so 0.95 still fails loudly.
    parity_min_argmax_agreement: float = 0.95
    parity_tie_epsilon: float = 0.05
    # Gates the mean probability delta, not the worst row: one shifted row with a
    # stable argmax is noise, systematic drift is not.
    parity_mean_prob_delta: float = 0.05

    def validate(self) -> None:
        if self.precision not in EXPORT_PRECISIONS:
            raise ConfigError(f"export.precision must be one of {list(EXPORT_PRECISIONS)}, got {self.precision!r}")
        if self.quant_target not in QUANT_TARGETS:
            raise ConfigError(f"export.quant_target must be one of {list(QUANT_TARGETS)}, got {self.quant_target!r}")
        if not 11 <= self.opset <= 21:
            raise ConfigError(f"export.opset must be between 11 and 21, got {self.opset}")
        if not 0.0 < self.parity_min_argmax_agreement <= 1.0:
            raise ConfigError("export.parity_min_argmax_agreement must be in (0, 1]")
        if not 0.0 <= self.parity_tie_epsilon < 1.0:
            raise ConfigError("export.parity_tie_epsilon must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class Config:
    """Root configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def validate(self) -> None:
        for section in (self.data, self.model, self.train, self.export):
            section.validate()


def _section_types() -> dict[str, type]:
    return {"data": DataConfig, "model": ModelConfig, "train": TrainingConfig, "export": ExportConfig}


def _coerce(section: str, key: str, value: Any, current: Any) -> Any:
    """Coerce an override value to the declared field type."""
    try:
        if isinstance(current, bool):
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                return value.strip().lower() == "true"
            raise ConfigError(f"{section}.{key} expects true/false, got {value!r}")
        if isinstance(current, tuple):
            element = type(current[0]) if current else str
            if isinstance(value, str):
                # Accept both `0.8,0.1,0.1` and the YAML-list spelling `[0.8, 0.1, 0.1]`,
                # which reaches us as a string because the shell already consumed the quotes.
                text = value.strip().removeprefix("[").removesuffix("]")
                items = [item.strip() for item in text.split(",") if item.strip()]
            elif isinstance(value, list | tuple):
                items = list(value)
            else:
                items = [value]
            return tuple(_scalar(section, key, item, element) for item in items)
        return _scalar(section, key, value, type(current))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{section}.{key}: cannot use {value!r} ({exc})") from exc


def _scalar(section: str, key: str, value: Any, target: type) -> Any:
    """Coerce one scalar, catching YAML quirks such as ``4e-5`` parsing as a string."""
    if isinstance(value, target) and not isinstance(value, bool):
        return value
    if target is float:
        return float(value)
    if target is int:
        if isinstance(value, bool):
            raise TypeError("expected an integer, got a boolean")
        if isinstance(value, float) and not value.is_integer():
            raise TypeError("expected an integer, got a fractional or nonfinite number")
        return int(value)
    if target is str:
        return str(value)
    raise TypeError(f"{section}.{key} has unsupported type {getattr(target, '__name__', target)}")


def load_config(path: Path | str | None = None, overrides: Sequence[str] = ()) -> Config:
    """Load a YAML config, apply ``section.key=value`` overrides, and validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"{path}: expected a mapping at the top level")
        raw = {str(section): dict(values or {}) for section, values in loaded.items()}
    for override in overrides:
        section, key, value = _parse_override(override)
        raw.setdefault(section, {})[key] = value

    sections: dict[str, Any] = {}
    for section, section_type in _section_types().items():
        provided = raw.pop(section, {})
        if not isinstance(provided, Mapping):
            raise ConfigError(f"{section}: expected a mapping")
        defaults = asdict(section_type())
        unknown = sorted(set(provided) - set(defaults))
        if unknown:
            raise ConfigError(f"{section}: unknown keys {unknown}; valid keys are {sorted(defaults)}")
        merged = {
            name: _coerce(section, name, provided.get(name, default), default) for name, default in defaults.items()
        }
        sections[section] = section_type(**merged)
    if raw:
        raise ConfigError(f"unknown config sections {sorted(raw)}; valid sections are {sorted(_section_types())}")

    config = Config(**sections)
    config.validate()
    return config


def _parse_override(override: str) -> tuple[str, str, Any]:
    if "=" not in override:
        raise ConfigError(f"override {override!r} must look like section.key=value")
    dotted, _, raw_value = override.partition("=")
    section, _, key = dotted.partition(".")
    if not section or not key:
        raise ConfigError(f"override {override!r} must look like section.key=value")
    if section not in _section_types():
        raise ConfigError(f"override {override!r}: unknown section {section!r}")
    keys = {f.name for f in fields(_section_types()[section])}
    if key not in keys:
        raise ConfigError(f"override {override!r}: unknown key {key!r}; valid keys are {sorted(keys)}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"override {override!r}: invalid value: {exc}") from exc
    return section, key, value


def config_to_dict(config: Config) -> dict[str, Any]:
    """Plain JSON-serializable dict, with tuples rendered as lists."""
    return json.loads(json.dumps(asdict(config), default=list))


def describe_types() -> dict[str, dict[str, Any]]:
    """Declared defaults per section, used by ``model-gym config --print``."""
    return {section: asdict(section_type()) for section, section_type in _section_types().items()}
