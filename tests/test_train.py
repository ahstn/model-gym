"""Training argument wiring: device-specific choices that only show up on the GPU host."""

from __future__ import annotations

import pytest
import torch

from model_gym.config import load_config
from model_gym.train import build_training_arguments, class_weights_for, resolve_precision

CUDA = torch.device("cuda")
CPU = torch.device("cpu")


def test_optimizer_follows_the_device() -> None:
    config = load_config(None)
    on_cuda = build_training_arguments(config, device=CUDA, bf16=True, fp16=False)
    on_cpu = build_training_arguments(config, device=CPU, bf16=False, fp16=False)
    assert on_cuda.optim == "adamw_torch_fused"
    assert on_cpu.optim == "adamw_torch"


def test_explicit_optimizer_is_respected() -> None:
    config = load_config(None, ["train.optim=adamw_torch"])
    assert build_training_arguments(config, device=CUDA, bf16=True, fp16=False).optim == "adamw_torch"


def test_stepped_runs_evaluate_and_save_on_the_same_schedule() -> None:
    config = load_config(None, ["train.max_steps=20"])
    arguments = build_training_arguments(config, device=CUDA, bf16=True, fp16=False)
    assert arguments.eval_strategy == "steps"
    assert arguments.save_strategy == "steps"
    assert arguments.eval_steps == arguments.save_steps == 5
    # load_best_model_at_end is only valid when the strategies match.
    assert arguments.load_best_model_at_end is True
    assert arguments.metric_for_best_model == "qwk"


def test_epoch_runs_use_epoch_schedules() -> None:
    config = load_config(None)
    arguments = build_training_arguments(config, device=CUDA, bf16=True, fp16=False)
    assert arguments.eval_strategy == "epoch"
    assert arguments.save_strategy == "epoch"


def test_pin_memory_is_only_enabled_on_cuda() -> None:
    config = load_config(None)
    assert build_training_arguments(config, device=CUDA, bf16=True, fp16=False).dataloader_pin_memory is True
    assert build_training_arguments(config, device=CPU, bf16=False, fp16=False).dataloader_pin_memory is False


def test_batch_and_accumulation_reach_the_arguments() -> None:
    config = load_config("configs/large.yaml")
    arguments = build_training_arguments(config, device=CUDA, bf16=True, fp16=False)
    assert arguments.per_device_train_batch_size == 16
    assert arguments.gradient_accumulation_steps == 2
    assert arguments.gradient_checkpointing is True
    assert arguments.bf16 is True and arguments.fp16 is False


def test_precision_is_downgraded_off_cuda() -> None:
    config = load_config(None)
    assert resolve_precision(config, CUDA) == (True, False)
    assert resolve_precision(config, CPU) == (False, False)
    fp16_config = load_config(None, ["train.bf16=false", "train.fp16=true"])
    assert resolve_precision(fp16_config, CUDA) == (False, True)


def test_balanced_weighting_favours_rare_levels() -> None:
    weights = class_weights_for("balanced", [100, 50, 25, 5, 5], CPU)
    assert weights is not None
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(weights[0]) < float(weights[4])
    assert float(weights[3]) == pytest.approx(float(weights[4]))


def test_sqrt_weighting_is_gentler_than_balanced() -> None:
    counts = [100, 50, 25, 5, 5]
    balanced = class_weights_for("balanced", counts, CPU)
    sqrt = class_weights_for("sqrt", counts, CPU)
    assert balanced is not None and sqrt is not None
    assert float(sqrt[4]) < float(balanced[4])


def test_no_weighting_returns_none() -> None:
    assert class_weights_for("none", [1, 1, 1, 1, 1], CPU) is None


def test_weighting_refuses_a_missing_level() -> None:
    with pytest.raises(ValueError, match=r"level\(s\) \[4, 5\]"):
        class_weights_for("balanced", [10, 10, 10, 0, 0], CPU)
