"""Training entrypoint for pretrained command-risk classifiers.

One GPU is enough (the paste's target is an RTX 3090, 24 GiB). Everything that
matters for that box is configurable: precision, gradient checkpointing, batch
size, gradient accumulation, optimizer, and attention implementation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from model_gym.config import Config, TrainingConfig, config_to_dict
from model_gym.data.build import level_counts
from model_gym.env import environment_info
from model_gym.evaluate import EvaluationResult, calibrate_checkpoint, evaluate_model, write_report
from model_gym.labels import NUM_LABELS
from model_gym.metrics import class_counts
from model_gym.modeling import (
    build_collator,
    build_compute_metrics,
    build_model,
    configure_runtime,
    freeze_encoder_layers,
    load_dataset_dict,
    load_split,
    load_tokenizer,
    resolve_device,
)
from model_gym.policy import POLICY_NAME, DecisionPolicy, load_policy, save_policy

LOGGER = logging.getLogger("model_gym.train")

BEST_MODEL_DIRNAME = "best"


def checkpoint_selection(
    metrics: Mapping[str, float],
    config: TrainingConfig,
    *,
    policy_kind: str = "score_bands",
) -> dict[str, Any]:
    """Rank validation checkpoints without trading safety gates for higher QWK.

    Feasible models rank by QWK. Otherwise minimize the summed excess rates,
    scaled by the remaining [limit, 1] range; QWK only breaks numerical ties.
    Missing support cannot satisfy a gate or outrank fully evaluated models.
    """
    gates = {
        "critical_miss_rate": config.max_critical_miss_rate,
        "unsafe_allow_rate": config.max_unsafe_allow_rate,
        "unnecessary_intervention_rate": config.max_unnecessary_intervention_rate,
    }
    supports = {
        key: metrics[f"runtime_{key}"] for key in ("critical_support", "requires_approval_support", "benign_support")
    }
    rates = {key: float(metrics[f"runtime_{key}"]) for key in gates}
    violations = {key: max(0.0, rates[key] - limit) for key, limit in gates.items()}
    supported = all(value > 0 for value in supports.values())
    eligible = supported and not any(violations.values())
    qwk = float(metrics["qwk"])
    quality = (qwk + 1.0) / 2.0 if math.isfinite(qwk) else 0.0
    if not supported:
        score = -4.0
    elif eligible:
        score = 2.0 + quality
    else:
        score = -sum(violations[key] / (1.0 - limit) for key, limit in gates.items()) + quality * 1e-9
    return {
        "score": score,
        "eligible": eligible,
        "decision_policy": policy_kind,
        "limits": gates,
        "rates": rates,
        "violations": violations,
        "supports": supports,
    }


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
    """``Trainer`` with class-weighted CE and an optional ordinal CDF penalty.

    Rare levels matter here: a corpus dominated by destructive commands would
    otherwise make level 5 nearly unlearnable.
    """

    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor | None = None,
        ordinal_loss_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if not math.isfinite(ordinal_loss_weight) or ordinal_loss_weight < 0:
            raise ValueError("ordinal_loss_weight must be finite and nonnegative")
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.ordinal_loss_weight = ordinal_loss_weight
        if class_weights is not None or ordinal_loss_weight:
            # This loss averages each batch; Trainer must scale gradient accumulation.
            self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]  # noqa: FBT002
        if self.class_weights is None and not self.ordinal_loss_weight:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        weights = self.class_weights.to(logits) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        if self.ordinal_loss_weight:
            predicted_cdf = logits.softmax(dim=-1).cumsum(dim=-1)[:, :-1]
            boundaries = torch.arange(logits.shape[-1] - 1, device=logits.device)
            target_cdf = (labels[:, None] <= boundaries).to(logits.dtype)
            loss = loss + self.ordinal_loss_weight * torch.nn.functional.mse_loss(predicted_cdf, target_cdf)
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
        gradient_checkpointing_kwargs={"use_reentrant": False} if train.freeze_encoder_layers else None,
        logging_steps=train.logging_steps,
        logging_dir=str(output_dir / "logs"),
        eval_strategy="steps" if stepped else "epoch",
        save_strategy="steps" if stepped else "epoch",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=train.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="selection_score" if train.selection_metric == "safety" else "qwk",
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
    if output_dir.exists() and any(output_dir.iterdir()) and not config.train.resume_from_checkpoint:
        raise ValueError(f"training output is not empty: {output_dir}; choose a new train.output_dir or --resume")
    policy = load_policy(config.train.decision_policy_path) if config.train.decision_policy_path else DecisionPolicy()
    validation_sha256 = (
        hashlib.sha256((Path(config.data.output_dir) / "validation.jsonl").read_bytes()).hexdigest()
        if config.train.calibrate_selection
        else None
    )
    tokenizer = load_tokenizer(config.model)
    splits = ("train", "validation", "test") if config.train.evaluate_test else ("train", "validation")
    datasets = load_dataset_dict(config.data.output_dir, tokenizer, config.model, splits=splits)
    train_records = load_split(config.data.output_dir, "train")
    counts = class_counts([record.label for record in train_records])
    weights = class_weights_for(config.train.class_weighting, counts, device)
    model = build_model(config.model)
    freeze_encoder_layers(model, config.train.freeze_encoder_layers)

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

    score_predictions = build_compute_metrics(
        policy=policy,
        calibrate=config.train.calibrate_selection,
        validation_sha256=validation_sha256,
    )

    def score_with_policy(prediction: Any) -> dict[str, float]:
        metrics = score_predictions(prediction)
        metrics["selection_score"] = checkpoint_selection(metrics, config.train, policy_kind=policy.kind)["score"]
        return metrics

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": arguments,
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "data_collator": build_collator(tokenizer),
        "compute_metrics": score_with_policy,
        "processing_class": tokenizer,
        "class_weights": weights,
        "ordinal_loss_weight": config.train.ordinal_loss_weight,
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
    save_policy(best_model_dir / POLICY_NAME, policy)
    if config.train.calibrate_selection:
        calibrate_checkpoint(
            best_model_dir,
            config.data.output_dir,
            batch_size=config.train.eval_batch_size,
            max_seq_length=config.model.max_seq_length,
            device=device,
        )

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
        if config.train.evaluate_test
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
            "selection_metric": config.train.selection_metric,
        },
        "validation_selection": checkpoint_selection(
            result.validation.metrics.headline(), config.train, policy_kind=result.validation.policy.kind
        ),
        "splits": list(datasets),
        "split_sha256": {
            split: hashlib.sha256((Path(config.data.output_dir) / f"{split}.jsonl").read_bytes()).hexdigest()
            for split in datasets
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
