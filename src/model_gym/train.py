"""Training entrypoint: full fine-tune of ModernBERT on the command-risk corpus.

One GPU is enough (the paste's target is an RTX 3090, 24 GiB). Everything that
matters for that box is configurable: precision, gradient checkpointing, batch
size, gradient accumulation, optimizer, and attention implementation.
"""

from __future__ import annotations

import json
import logging
import time

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from model_gym.config import Config, config_to_dict
from model_gym.data.build import SPLITS, level_counts
from model_gym.env import environment_info
from model_gym.evaluate import EvaluationResult, evaluate_model, write_report
from model_gym.labels import NUM_LABELS
from model_gym.metrics import class_counts
from model_gym.modeling import (
    build_collator,
    build_compute_metrics,
    build_model,
    configure_runtime,
    load_dataset_dict,
    load_split,
    load_tokenizer,
    resolve_device,
)

LOGGER = logging.getLogger("model_gym.train")

BEST_MODEL_DIRNAME = "best"


def resolve_precision(config: Config, device: torch.device) -> tuple[bool, bool]:
    """Return ``(bf16, fp16)``, downgrading to fp32 on devices without 16-bit support."""
    bf16, fp16 = config.train.bf16, config.train.fp16
    if not (bf16 or fp16):
        return False, False
    if device.type != "cuda":
        LOGGER.warning(
            "%s training was requested but the device is %s; running in fp32", "bf16" if bf16 else "fp16", device.type
        )
        return False, False
    if fp16:
        LOGGER.warning("fp16 needs loss scaling and is less stable than bf16 on Ampere; prefer train.bf16=true")
    return bf16, fp16


def class_weights_for(scheme: str, counts: Sequence[int], device: torch.device) -> torch.Tensor | None:
    """Inverse-frequency class weights, normalized to mean 1.

    ``balanced`` uses ``total / (k * count)``; ``sqrt`` damps that ratio, which
    keeps rare levels visible without letting them dominate the loss.
    """
    if scheme == "none":
        return None
    if len(counts) != NUM_LABELS:
        raise ValueError(f"expected {NUM_LABELS} class counts, got {len(counts)}")
    empty = [index + 1 for index, count in enumerate(counts) if count == 0]
    if empty:
        raise ValueError(
            f"class weighting needs a training row for every level; level(s) {empty} have none. "
            "Add data or set train.class_weighting=none"
        )
    tensor = torch.tensor(counts, dtype=torch.float64)
    if scheme == "balanced":
        weights = int(tensor.sum()) / (len(counts) * tensor)
    elif scheme == "sqrt":
        weights = tensor.rsqrt()
    else:
        raise ValueError(f"unknown class weighting scheme {scheme!r}")
    weights = weights / weights.mean()
    return weights.to(device=device, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """``Trainer`` with optional class-weighted cross entropy.

    Rare levels matter here: a corpus dominated by destructive commands would
    otherwise make level 5 nearly unlearnable.
    """

    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]  # noqa: FBT002
        if self.class_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=self.class_weights)
        return (loss, outputs) if return_outputs else loss


@dataclass(frozen=True, slots=True)
class RunResult:
    """What a training run produced."""

    output_dir: Path
    best_model_dir: Path
    train_seconds: float
    global_step: int
    trainable_parameters: int
    validation: EvaluationResult
    test: EvaluationResult | None
    class_weights: tuple[float, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "best_model_dir": str(self.best_model_dir),
            "train_seconds": round(self.train_seconds, 2),
            "global_step": self.global_step,
            "trainable_parameters": self.trainable_parameters,
            "class_weights": list(self.class_weights) if self.class_weights else None,
            "validation": self.validation.metrics.to_dict(),
            "test": self.test.metrics.to_dict() if self.test else None,
        }


def build_training_arguments(config: Config, *, device: torch.device, bf16: bool, fp16: bool) -> TrainingArguments:
    """Translate the config into ``TrainingArguments`` for this device."""
    train = config.train
    output_dir = Path(train.output_dir)
    stepped = train.max_steps > 0
    eval_steps = max(1, train.max_steps // 4) if stepped else None
    optim = train.optim
    if optim == "auto":
        optim = "adamw_torch_fused" if device.type == "cuda" else "adamw_torch"
    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=train.run_name or output_dir.name,
        num_train_epochs=train.epochs,
        max_steps=train.max_steps,
        learning_rate=train.learning_rate,
        per_device_train_batch_size=train.batch_size,
        per_device_eval_batch_size=train.eval_batch_size,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        warmup_ratio=train.warmup_ratio,
        weight_decay=train.weight_decay,
        lr_scheduler_type=train.lr_scheduler,
        optim=optim,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=train.gradient_checkpointing,
        logging_steps=train.logging_steps,
        logging_dir=str(output_dir / "logs"),
        eval_strategy="steps" if stepped else "epoch",
        save_strategy="steps" if stepped else "epoch",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=train.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
        seed=train.seed,
        data_seed=train.seed,
        dataloader_num_workers=train.dataloader_num_workers,
        dataloader_pin_memory=train.dataloader_pin_memory and device.type == "cuda",
        report_to=list(train.report_to),
        save_safetensors=True,
    )


def run_training(config: Config) -> RunResult:
    """Fine-tune the base checkpoint and write metrics, reports, and metadata."""
    configure_runtime()
    device = resolve_device()
    set_seed(config.train.seed)
    bf16, fp16 = resolve_precision(config, device)

    output_dir = Path(config.train.output_dir)
    tokenizer = load_tokenizer(config.model)
    datasets = load_dataset_dict(config.data.output_dir, tokenizer, config.model)
    train_records = load_split(config.data.output_dir, "train")
    counts = class_counts([record.label for record in train_records])
    weights = class_weights_for(config.train.class_weighting, counts, device)
    model = build_model(config.model)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    arguments = build_training_arguments(config, device=device, bf16=bf16, fp16=fp16)
    LOGGER.info(
        "training %s on %s | %.2fM/%.2fM trainable params | batch %d x accum %d | levels %s",
        config.model.name,
        device,
        trainable / 1e6,
        total / 1e6,
        config.train.batch_size,
        config.train.gradient_accumulation_steps,
        counts,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": arguments,
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "data_collator": build_collator(tokenizer),
        "compute_metrics": build_compute_metrics(),
        "processing_class": tokenizer,
        "class_weights": weights,
    }
    if config.train.early_stopping_patience > 0 and config.train.max_steps <= 0:
        trainer_kwargs["callbacks"] = [
            EarlyStoppingCallback(early_stopping_patience=config.train.early_stopping_patience)
        ]
    trainer = WeightedTrainer(**trainer_kwargs)

    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=config.train.resume_from_checkpoint or None)
    train_seconds = time.perf_counter() - started

    best_model_dir = output_dir / BEST_MODEL_DIRNAME
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    tokenizer.save_pretrained(str(output_dir))

    validation = evaluate_model(
        best_model_dir,
        config.data.output_dir,
        split="validation",
        batch_size=config.train.eval_batch_size,
        max_seq_length=config.model.max_seq_length,
        model_config=config.model,
        device=device,
    )
    test = (
        evaluate_model(
            best_model_dir,
            config.data.output_dir,
            split="test",
            batch_size=config.train.eval_batch_size,
            max_seq_length=config.model.max_seq_length,
            model_config=config.model,
            device=device,
        )
        if Path(config.data.output_dir, "test.jsonl").is_file()
        else None
    )
    report_dir = output_dir / "reports"
    write_report(validation, report_dir, title=f"{config.model.name} validation")
    if test is not None:
        write_report(test, report_dir, title=f"{config.model.name} test")

    result = RunResult(
        output_dir=output_dir,
        best_model_dir=best_model_dir,
        train_seconds=train_seconds,
        global_step=int(trainer.state.global_step),
        trainable_parameters=trainable,
        validation=validation,
        test=test,
        class_weights=tuple(float(value) for value in weights.cpu()) if weights is not None else None,
    )
    _write_run_metadata(config, result, trainer, train_records, datasets, device)
    return result


def _write_run_metadata(
    config: Config,
    result: RunResult,
    trainer: Trainer,
    train_records: Sequence[Any],
    datasets: Any,
    device: torch.device,
) -> None:
    """Record everything needed to explain or reproduce this run."""
    output_dir = result.output_dir
    corpus_card = Path(config.data.output_dir) / "dataset_card.json"
    metadata = {
        "run": result.to_dict(),
        "config": config_to_dict(config),
        "device": str(device),
        "corpus": {
            "split_sizes": {split: len(datasets[split]) for split in datasets},
            "train_levels": {str(level): count for level, count in level_counts(train_records).items()},
            "dataset_card": json.loads(corpus_card.read_text(encoding="utf-8")) if corpus_card.is_file() else None,
        },
        "environment": environment_info(),
        "trainer": {
            "log_history": trainer.state.log_history,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
        },
        "splits": list(SPLITS),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
