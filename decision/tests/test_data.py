import gzip
import json

from collections import Counter
from pathlib import Path

import pytest

from decision.data import (
    Candidate,
    SuiteFilter,
    allocate,
    binary_entropy_bits,
    dedup,
    dedup_key,
    is_blocklisted,
    normalize,
    read_suite_texts,
    split_of,
)
from decision.schema import NOUL_OPTIONS


def _raw(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {"id": "x1", "kind": "noul", "state": "some state", "question": "Is it?", "source": "s"}
    return base | kw


def test_noul_formats_map_to_no_yes() -> None:
    sd = normalize("sargedev", "train", _raw(options=["false", "true"], target=[0.2, 0.8]))
    oj = normalize("openjev", "test", _raw(state_json="{}", options=["no", "yes"], target=[0.3, 0.7], group_id="g"))
    ts = normalize("tasksource", "train", _raw(options=[], target=[0.9], group_id="g2"))
    assert sd.options == oj.options == ts.options == NOUL_OPTIONS
    assert sd.target == pytest.approx((0.2, 0.8))
    assert oj.target == pytest.approx((0.3, 0.7))
    assert ts.target == pytest.approx((0.1, 0.9))
    assert sd.group_id == "x1" and oj.group_id == "g"
    assert oj.id == "openjev/test/x1" and split_of(oj.id) == "test"


def test_normalize_drop_reasons_and_renormalization() -> None:
    choice = {"kind": "choice", "options": ["a", "b", "c"]}
    assert normalize("sargedev", "train", _raw(**choice, target=[1.0, 1.0, -1e-6])).target == pytest.approx(
        (0.5, 0.5, 0)
    )
    assert normalize("sargedev", "train", _raw(**choice, target=[1.0, 1.0, -0.5])) == "negative_target"
    assert (
        normalize("sargedev", "train", _raw(kind="choice", options=["A ", "a"], target=[1, 0])) == "duplicate_options"
    )
    assert normalize("sargedev", "train", _raw(**choice, target=[0, 0, 0])) == "zero_target"
    assert normalize("sargedev", "train", _raw(**choice, target=[float("nan"), 0, 1])) == "nonfinite_target"
    assert normalize("sargedev", "train", _raw(kind="choice", options=["a"], target=[1])) == "option_count"
    assert normalize("sargedev", "train", _raw(**choice, target=[1, 0, 0], question=" ")) == "empty_text"


def test_blocklist_matching() -> None:
    for s in ("allenai/ai2_arc", "arc-challenge", "tasksource/mmlu", "Hellaswag", "anli/r1", "bigbench/logic"):
        assert is_blocklisted(s), s
    for s in ("search_qa", "research-papers", "brightness", "banlist", "canvas"):
        assert not is_blocklisted(s), s


def test_entropy() -> None:
    assert binary_entropy_bits(0.5) == pytest.approx(1.0)
    assert binary_entropy_bits(0.45) > 0.97
    assert binary_entropy_bits(0.3) < 0.97


def test_dedup_priority() -> None:
    a = normalize("sargedev", "train", _raw(options=["false", "true"], target=[0.5, 0.5]))
    b = normalize("tasksource", "train", _raw(state="  SOME   state", options=[], target=[0.1]))
    drops: Counter[str] = Counter()
    out = dedup({"sargedev": [Candidate(a, dedup_key(a))], "tasksource": [Candidate(b, dedup_key(b))]}, drops, "train")
    assert [c.decision.dataset for c in out["tasksource"]] == ["tasksource"]
    assert out["sargedev"] == []
    assert drops["sargedev/train/duplicate"] == 1


def test_allocate_redistributes_short_stream() -> None:
    shares = {"a": 0.5, "b": 0.3, "c": 0.2}
    alloc = allocate(100, shares, {"a": 1000, "b": 10, "c": 1000})
    assert alloc["b"] == 10
    assert sum(alloc.values()) == 100
    assert alloc["a"] / alloc["c"] == pytest.approx(2.5, rel=0.05)
    assert allocate(100, shares, {"a": 5, "b": 5, "c": 5}) == {"a": 5, "b": 5, "c": 5}


def test_suite_ngram_filter(tmp_path: Path) -> None:
    words = [f"w{i}" for i in range(20)]
    path = tmp_path / "suite.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(json.dumps({"context": " ".join(words), "options": ["x"]}) + "\n")
    f = SuiteFilter(read_suite_texts(path))
    assert f.hits("prefix " + " ".join(words[3:16]).upper() + " suffix")
    assert not f.hits(" ".join(words[3:15]))  # only 12 shared words
    assert not SuiteFilter([]).hits(" ".join(words))
