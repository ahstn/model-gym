"""Tokenizer, dataset, and model construction."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from model_gym.calibration import fit_temperature, validate_temperature
from model_gym.config import ModelConfig
from model_gym.data.build import SPLITS
from model_gym.data.schema import RiskRecord, read_jsonl
from model_gym.labels import ID2LABEL, LABEL2ID, NUM_LABELS

if TYPE_CHECKING:
    from transformers import EvalPrediction

    from model_gym.policy import DecisionPolicy


def configure_runtime() -> None:
    """Enable TF32 matmuls on Ampere and newer, which the 3090 benefits from."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick a device: explicit ``cuda``/``mps``/``cpu`` wins, otherwise best available."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(config: ModelConfig) -> PreTrainedTokenizerBase:
    """Load the checkpoint's tokenizer."""
    return AutoTokenizer.from_pretrained(config.name, trust_remote_code=config.trust_remote_code)


def load_split(processed_dir: Path | str, split: str) -> list[RiskRecord]:
    """Read one split from the processed corpus directory."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
    path = Path(processed_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `model-gym build-data` first")
    return read_jsonl(path)


def encode_records(
    records: Sequence[RiskRecord],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_seq_length: int,
) -> Dataset:
    """Tokenize records into a dataset with ``input_ids``, ``attention_mask``, ``labels``.

    Padding is left to the collator so each batch is padded to its own longest row.
    """
    if not records:
        raise ValueError("cannot encode an empty record list")
    dataset = Dataset.from_list(
        [{"command": record.command, "labels": record.label, "level": int(record.level)} for record in records]
    )

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["command"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    encoded = dataset.map(tokenize, batched=True, desc="tokenizing", remove_columns=["command"])
    return encoded


def load_dataset_dict(
    processed_dir: Path | str,
    tokenizer: PreTrainedTokenizerBase,
    config: ModelConfig,
    *,
    splits: Sequence[str] = SPLITS,
) -> DatasetDict:
    """Encode every requested split into a :class:`DatasetDict`."""
    return DatasetDict(
        {
            split: encode_records(load_split(processed_dir, split), tokenizer, max_seq_length=config.max_seq_length)
            for split in splits
        }
    )


def build_model(config: ModelConfig) -> PreTrainedModel:
    """Load a pretrained encoder or saved classifier with the five-level label mapping."""
    config.validate()
    pretrained = AutoConfig.from_pretrained(
        config.name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=config.trust_remote_code,
    )
    if config.pooling != "default":
        if pretrained.model_type != "modernbert":
            raise ValueError(f"pooling overrides are not supported for {pretrained.model_type!r}")
        pretrained.classifier_pooling = config.pooling
    if config.classifier_dropout != -1.0:
        if pretrained.model_type == "modernbert":
            pretrained.classifier_dropout = config.classifier_dropout
        elif pretrained.model_type == "deberta-v2":
            pretrained.cls_dropout = config.classifier_dropout
        else:
            raise ValueError(f"classifier dropout overrides are not supported for {pretrained.model_type!r}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.name,
            config=pretrained,
            attn_implementation=config.attn_implementation,
            trust_remote_code=config.trust_remote_code,
        )
    except (ImportError, ValueError) as exc:
        if config.attn_implementation == "flash_attention_2":
            raise RuntimeError(
                "attn_implementation=flash_attention_2 needs a matching flash-attn build; "
                "install `flash-attn` for this torch/CUDA pair or set model.attn_implementation=sdpa"
            ) from exc
        raise
    return model


def freeze_encoder_layers(model: PreTrainedModel, count: int) -> None:
    """Freeze the encoder, or embeddings and its first ``count`` blocks, not the head."""
    if isinstance(count, bool) or not isinstance(count, int) or count < -1:
        raise ValueError("freeze_encoder_layers must be -1 or a nonnegative integer")
    if count == 0:
        return
    if model.config.model_type == "modernbert":
        encoder = model.model
        layers = encoder.layers
    elif model.config.model_type == "deberta-v2":
        encoder = model.deberta
        layers = encoder.encoder.layer
    else:
        raise ValueError(f"encoder freezing is not supported for {model.config.model_type!r}")
    if count > len(layers):
        raise ValueError(f"cannot freeze {count} encoder layers; {model.config.model_type} has {len(layers)}")
    if count == -1:
        encoder.requires_grad_(requires_grad=False)
        return
    encoder.embeddings.requires_grad_(requires_grad=False)
    for layer in layers[:count]:
        layer.requires_grad_(requires_grad=False)
    if model.config.model_type == "deberta-v2" and encoder.encoder.conv is not None:
        # DeBERTa's convolution is part of the first block's output.
        encoder.encoder.conv.requires_grad_(requires_grad=False)


def build_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    """Dynamic padding, which ModernBERT benefits from on short, varied inputs."""
    return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)


def build_compute_metrics(
    *,
    policy: DecisionPolicy | None = None,
    calibrate: bool = False,
    validation_sha256: str | None = None,
) -> Any:
    """Score validation logits with the runtime policy and optional fitted temperature."""
    from model_gym.metrics import compute_metrics as score

    if calibrate and (
        not isinstance(validation_sha256, str)
        or len(validation_sha256) != 64
        or any(char not in "0123456789abcdef" for char in validation_sha256)
    ):
        raise ValueError("calibrated selection requires the validation file's SHA-256 digest")

    def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
        logits = np.asarray(eval_prediction.predictions)
        labels = np.asarray(eval_prediction.label_ids).ravel()
        temperature = 1.0
        if calibrate:
            assert validation_sha256 is not None
            temperature = fit_temperature(logits, labels, validation_sha256=validation_sha256)["temperature"]
        probabilities = torch.softmax(torch.as_tensor(logits, dtype=torch.float64) / temperature, dim=-1).numpy()
        metrics = score(probabilities, labels, policy=policy).headline()
        if calibrate:
            metrics["calibration_temperature"] = temperature
        return metrics

    return compute_metrics


def _logit_batches(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    commands: Sequence[str],
    *,
    max_seq_length: int,
    batch_size: int,
    device: torch.device | None,
) -> Iterator[torch.Tensor]:
    if not commands:
        raise ValueError("no commands to score")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = device or next(model.parameters()).device
    model.eval()
    for start in range(0, len(commands), batch_size):
        batch = list(commands[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
            padding=True,
        ).to(device)
        yield model(**encoded).logits.float()


@torch.no_grad()
def predict_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    commands: Sequence[str],
    *,
    max_seq_length: int,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    """Raw checkpoint logits, deliberately ignoring saved calibration when fitting."""
    chunks = [
        logits.cpu().numpy()
        for logits in _logit_batches(
            model, tokenizer, commands, max_seq_length=max_seq_length, batch_size=batch_size, device=device
        )
    ]
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def predict_proba(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    commands: Sequence[str],
    *,
    max_seq_length: int,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    """Probabilities in input order, applying the checkpoint's saved temperature once."""
    temperature = validate_temperature(getattr(model, "_model_gym_temperature", 1.0))
    chunks = [
        torch.softmax(logits / temperature, dim=-1).cpu().numpy()
        for logits in _logit_batches(
            model, tokenizer, commands, max_seq_length=max_seq_length, batch_size=batch_size, device=device
        )
    ]
    return np.concatenate(chunks, axis=0)
