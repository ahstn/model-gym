import math

import pytest
import torch

from decision.train import TrainConfig, _inverse, soft_cross_entropy, symmetric_kl


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


def _view(orig_logits: list[float], order: list[int], width: int = 6) -> torch.Tensor:
    """Logits of a view whose slot p holds original option order[p], padded with -inf."""
    row = [orig_logits[o] for o in order] + [-math.inf] * (width - len(order))
    return torch.tensor([row])


def test_symmetric_kl_compares_options_not_slots() -> None:
    orig = [2.0, 0.5, -1.0, 0.0]
    a, b = [2, 0, 3, 1], [1, 3, 0, 2]

    def inv(order: list[int]) -> torch.Tensor:
        return torch.tensor([[*_inverse(order, 4), 0, 0]])

    n = torch.tensor([4])
    same = symmetric_kl(_view(orig, a), _view(orig, b), inv(a), inv(b), n)
    assert same.item() == pytest.approx(0.0, abs=1e-6)
    # Same slot scores under different orders mean different option beliefs: the loss must see that.
    slots = _view(orig, a)
    moved = symmetric_kl(slots, slots, inv(a), inv(b), n)
    assert moved.item() > 0.1
    assert torch.isfinite(moved).all()
