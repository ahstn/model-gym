"""Checkpoint safety selection and training loss boundaries."""

from __future__ import annotations

import hashlib
import json

from pathlib import Path

import numpy as np
import pytest
import torch

from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, TrainingArguments

from model_gym.calibration import CALIBRATION_NAME
from model_gym.config import load_config
from model_gym.data.schema import RiskRecord, write_jsonl
from model_gym.metrics import compute_metrics
from model_gym.policy import POLICY_NAME, DecisionPolicy, load_policy, save_policy
from model_gym.train import WeightedTrainer, checkpoint_selection, class_weights_for, resolve_precision, run_training

CUDA = torch.device("cuda")
CPU = torch.device("cpu")


def test_eligible_checkpoint_beats_higher_qwk_with_unsafe_allows() -> None:
    truth = np.repeat([0, 2, 4], 100)
    conservative = truth.copy()
    conservative[:4] = 1
    conservative[-20:] = 2
    unsafe = truth.copy()
    unsafe[100:103] = 4
    conservative_metrics = compute_metrics(np.eye(5)[conservative], truth).headline()
    unsafe_metrics = compute_metrics(np.eye(5)[unsafe], truth).headline()
    assert unsafe_metrics["qwk"] > conservative_metrics["qwk"]
    config = load_config().train
    accepted = checkpoint_selection(conservative_metrics, config)
    rejected = checkpoint_selection(unsafe_metrics, config)
    assert accepted["eligible"]
    assert not rejected["eligible"]
    assert accepted["score"] > rejected["score"]


def test_unsafe_allow_limit_is_inclusive() -> None:
    truth = np.repeat([0, 2, 4], 100)
    prediction = truth.copy()
    prediction[100:102] = 4  # Two misses out of 200 approval-required commands: 1%.
    config = load_config().train
    assert checkpoint_selection(compute_metrics(np.eye(5)[prediction], truth).headline(), config)["eligible"]
    prediction[102] = 4
    assert not checkpoint_selection(compute_metrics(np.eye(5)[prediction], truth).headline(), config)["eligible"]


def test_missing_critical_examples_cannot_pass_safety_gates() -> None:
    truth = np.array([2, 4])
    metrics = compute_metrics(np.eye(5)[truth], truth).headline()
    assert not checkpoint_selection(metrics, load_config().train)["eligible"]


def test_blocking_everything_cannot_pass_intervention_gate() -> None:
    truth = np.repeat([0, 2, 4], 10)
    metrics = compute_metrics(np.eye(5)[np.zeros(len(truth), dtype=int)], truth).headline()
    selection = checkpoint_selection(metrics, load_config().train)
    assert not selection["eligible"]
    assert selection["violations"]["unnecessary_intervention_rate"] > 0


def test_selection_uses_configured_runtime_policy_not_score_diagnostics() -> None:
    probabilities = np.asarray([[0.4, 0.35, 0.15, 0.1, 0.0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])
    config = load_config().train
    score_metrics = compute_metrics(probabilities, [0, 2, 4]).headline()
    policy = DecisionPolicy(kind="argmax")
    argmax_metrics = compute_metrics(probabilities, [0, 2, 4], policy=policy).headline()
    assert argmax_metrics["score_critical_miss_rate"] == score_metrics["score_critical_miss_rate"] == 1.0
    assert not checkpoint_selection(score_metrics, config)["eligible"]
    accepted = checkpoint_selection(argmax_metrics, config, policy_kind=policy.kind)
    assert accepted["eligible"]
    assert accepted["decision_policy"] == "argmax"


def test_ordinal_loss_penalizes_farther_errors_with_the_same_cross_entropy(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = BertForSequenceClassification(
            BertConfig(
                vocab_size=8,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                num_labels=5,
            )
        )
    model.eval()
    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
        ordinal_loss_weight=2.0,
    )
    labels = torch.tensor([0, 4])
    inputs = torch.tensor([[1, 2], [2, 3]])
    near = torch.tensor([0.1, 0.85, 0.02, 0.02, 0.01])
    # Keep p(true=0) fixed, move mass from level 2 to level 5.
    far = torch.tensor([0.1, 0.01, 0.02, 0.02, 0.85])
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0])

    def loss_for(probabilities: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(probabilities.log())
        return trainer.compute_loss(model, {"input_ids": inputs[:1], "labels": labels[:1]})

    near_loss = loss_for(near).detach()
    far_loss = loss_for(far)
    assert far_loss.item() - near_loss.item() == pytest.approx(2.0 * 0.567, abs=1e-6)
    far_loss.backward()
    assert model.classifier.bias.grad is not None
    assert model.classifier.bias.grad.abs().sum() > 0

    trainer.class_weights = weights
    weighted_loss = trainer.compute_loss(model, {"input_ids": inputs, "labels": labels})
    logits = far.log().expand(2, -1)
    expected_ce = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
    target_cdf = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    expected_ordinal = torch.nn.functional.mse_loss(far.cumsum(-1)[:-1].expand(2, -1), target_cdf)
    assert weighted_loss.item() == pytest.approx((expected_ce + 2.0 * expected_ordinal).item(), abs=1e-6)


def test_training_persists_calibrated_runtime_policy_without_reading_test_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    vocabulary = checkpoint / "vocab.txt"
    vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncmd\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocabulary)).save_pretrained(checkpoint)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
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
    model.save_pretrained(checkpoint)
    processed = tmp_path / "processed"
    records = [RiskRecord(command=f"cmd {index}", level=1 if index < 4 else 2) for index in range(5)]
    for split in ("train", "validation"):
        write_jsonl(processed / f"{split}.jsonl", records)
    (processed / "test.jsonl").write_text("must never be parsed\n", encoding="utf-8")
    policy = DecisionPolicy(kind="risk_thresholds", block_threshold=0.9)
    policy_path = tmp_path / POLICY_NAME
    save_policy(policy_path, policy)
    config = load_config(
        None,
        [
            f"model.name={checkpoint}",
            "model.attn_implementation=eager",
            "model.max_seq_length=16",
            f"data.output_dir={processed}",
            f"train.output_dir={tmp_path / 'run'}",
            f"train.decision_policy_path={policy_path}",
            "train.calibrate_selection=true",
            "train.max_steps=1",
            "train.batch_size=5",
            "train.eval_batch_size=5",
            "train.learning_rate=1e-9",
            "train.bf16=false",
            "train.dataloader_num_workers=0",
        ],
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("model_gym.train.resolve_device", lambda: CPU)
    with torch.random.fork_rng(devices=[]):
        result = run_training(config)
    assert result.test is None
    assert load_policy(result.best_model_dir / POLICY_NAME) == policy
    calibration = json.loads((result.best_model_dir / CALIBRATION_NAME).read_text(encoding="utf-8"))
    assert calibration["validation_sha256"] == hashlib.sha256((processed / "validation.jsonl").read_bytes()).hexdigest()
    assert calibration["temperature"] == pytest.approx(6 / np.log(16), abs=1e-6)
    assert result.validation.metrics.headline()["runtime_critical_miss_rate"] == 1.0
    metadata = json.loads((result.output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    evaluations = [row for row in metadata["trainer"]["log_history"] if "eval_selection_score" in row]
    selected = max(evaluations, key=lambda row: row["eval_selection_score"])
    assert selected["eval_runtime_critical_miss_rate"] == 1.0
    assert selected["eval_calibration_temperature"] == pytest.approx(calibration["temperature"], abs=1e-6)
    assert metadata["validation_selection"]["decision_policy"] == "risk_thresholds"
    assert metadata["validation_selection"]["score"] == pytest.approx(selected["eval_selection_score"])


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
