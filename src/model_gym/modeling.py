"""Tokenizer, dataset, and model construction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from model_gym.config import ModelConfig
from model_gym.data.build import SPLITS
from model_gym.data.schema import RiskRecord, read_jsonl
from model_gym.labels import ID2LABEL, LABEL2ID, NUM_LABELS

if TYPE_CHECKING:
    from transformers import EvalPrediction


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
    """Load the base checkpoint with a freshly initialized 5-way classification head."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.name,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
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


def build_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    """Dynamic padding, which ModernBERT benefits from on short, varied inputs."""
    return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)


def build_compute_metrics() -> Any:
    """``Trainer`` callback that scores predictions with the ordinal metrics."""
    from model_gym.metrics import compute_metrics as score

    def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
        logits = np.asarray(eval_prediction.predictions)
        labels = np.asarray(eval_prediction.label_ids).ravel()
        probabilities = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=-1).numpy()
        return score(probabilities, labels).headline()

    return compute_metrics


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
    """Probabilities for raw command strings, in input order.

    Uses the same truncation settings as training so reported metrics and runtime
    behaviour describe the same model.
    """
    if not commands:
        raise ValueError("no commands to score")
    device = device or next(model.parameters()).device
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(commands), batch_size):
        batch = list(commands[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
            padding=True,
        ).to(device)
        logits = model(**encoded).logits
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)
