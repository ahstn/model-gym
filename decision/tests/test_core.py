from pathlib import Path

import pytest

from decision.modeling import collate, token_budget_batches
from decision.prompt import MAX_OPTIONS, Readout
from decision.schema import NOUL_OPTIONS, Decision, read_jsonl, write_jsonl


def make(options: tuple[str, ...], target: tuple[float, ...], *, kind: str = "choice") -> Decision:
    return Decision(
        id="x",
        dataset="d",
        source="s",
        kind=kind,  # type: ignore[arg-type]
        state="state",
        question="question?",
        options=options,
        target=target,
        group_id="g",
    )


@pytest.mark.parametrize(
    ("options", "target", "kind"),
    [
        (("a", "b"), (0.5, 0.6), "choice"),
        (("a", "b"), (1.0,), "choice"),
        (("a", "b"), (1.5, -0.5), "choice"),
        (tuple(str(i) for i in range(256)), (1.0,) + (0.0,) * 255, "choice"),
        (("a",), (1.0,), "choice"),
        (("Yes", "No"), (0.5, 0.5), "noul"),
        (("a", "b"), (0.5, 0.5), "other"),
    ],
)
def test_decision_rejects_invalid(options: tuple[str, ...], target: tuple[float, ...], kind: str) -> None:
    with pytest.raises(ValueError):
        make(options, target, kind=kind)


def test_permuted_keeps_option_target_pairs() -> None:
    d = make(("a", "b", "c"), (0.2, 0.3, 0.5))
    p = d.permuted([2, 0, 1])
    assert p.options == ("c", "a", "b")
    assert dict(zip(p.options, p.target, strict=True)) == dict(zip(d.options, d.target, strict=True))


def test_jsonl_round_trip(tmp_path: Path) -> None:
    rows = [make(("a", "b"), (0.25, 0.75)), make(NOUL_OPTIONS, (1.0, 0.0), kind="noul")]
    path = tmp_path / "nested" / "rows.jsonl"
    assert write_jsonl(path, rows) == 2
    assert list(read_jsonl(path)) == rows


@pytest.mark.parametrize("shuffle", [False, True])
def test_token_budget_batches_cover_all_within_budget(*, shuffle: bool) -> None:
    lengths = [5, 17, 3, 40, 40, 9, 12, 1, 33, 20] * 7
    batches = token_budget_batches(lengths, max_tokens=100, max_rows=4, shuffle=shuffle, seed=3)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(lengths)))
    for b in batches:
        assert len(b) <= 4
        assert len(b) * max(lengths[i] for i in b) <= 100


def test_token_budget_batches_oversized_row_alone() -> None:
    batches = token_budget_batches([500, 2, 2], max_tokens=100, max_rows=8, shuffle=False, seed=0)
    assert [0] in batches
    assert sorted(i for b in batches for i in b) == [0, 1, 2]


def test_collate_pads_inputs_and_targets() -> None:
    decisions = [make(("a", "b", "c"), (0.2, 0.3, 0.5)), make(("a", "b"), (1.0, 0.0))]
    batch = collate([[7, 8, 9, 10], [5, 6]], decisions, pad_id=1)
    assert batch.input_ids.tolist() == [[7, 8, 9, 10], [5, 6, 1, 1]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch.last_index.tolist() == [3, 1]
    assert batch.num_options.tolist() == [3, 2]
    assert batch.targets.shape == (2, MAX_OPTIONS)
    assert batch.targets[0, :3].tolist() == pytest.approx([0.2, 0.3, 0.5])
    assert batch.targets[:, 3:].abs().sum().item() == 0
    assert batch.targets[1, :2].tolist() == [1.0, 0.0]


def test_readout_codes_are_single_answer_tokens() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("openbmb/MiniCPM5-2B", local_files_only=True)
    except OSError:
        pytest.skip("MiniCPM5-2B tokenizer not in the local HF cache")
    readout = Readout(tokenizer)
    assert len(readout.codes) == MAX_OPTIONS
    assert len(set(readout.code_token_ids)) == MAX_OPTIONS
    d = make(tuple(f"option {i}" for i in range(MAX_OPTIONS)), (1.0,) + (0.0,) * (MAX_OPTIONS - 1))
    prompt = readout.render(d)
    assert prompt.endswith("Answer:")
    ids = readout.encode(d)
    assert ids is not None
    for code, token_id in zip(readout.codes, readout.code_token_ids, strict=True):
        assert tokenizer.encode(prompt + " " + code, add_special_tokens=False) == [*ids, token_id]
    assert readout.encode(d, max_tokens=10) is None
