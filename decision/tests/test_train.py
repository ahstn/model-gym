import math

import pytest
import torch

from decision.train import TrainConfig, soft_cross_entropy


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        TrainConfig.from_dict({"name": "x", "lr": 1e-4, "learning_rate": 1e-4})
    with pytest.raises(ValueError, match="unknown lora keys"):
        TrainConfig.from_dict({"name": "x", "lr": 1e-4, "lora": {"rank": 8}})


@pytest.mark.parametrize(
    "override",
    [{"lr": 0.0}, {"rows_per_step": 0}, {"batch_tokens": 1024, "max_seq_tokens": 2048}, {"lora": {"r": 0}}],
)
def test_config_rejects_bad_values(override: dict) -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"name": "x", "lr": 1e-4, **override})


def test_soft_ce_masks_invalid_slots_and_matches_hand_value() -> None:
    logits = torch.tensor([[0.0, math.log(3.0), -math.inf, -math.inf]])
    targets = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    loss = soft_cross_entropy(logits, targets)
    # p = [1/4, 3/4]; loss = -(0.5*log(1/4) + 0.5*log(3/4))
    expected = -(0.5 * math.log(0.25) + 0.5 * math.log(0.75))
    assert not torch.isnan(loss).any()
    assert loss.item() == pytest.approx(expected, rel=1e-6)
