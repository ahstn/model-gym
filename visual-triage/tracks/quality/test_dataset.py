"""Check leakage boundaries and native-label rules before paid validation."""

import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import pytest

TRACK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quality_dataset_builder", TRACK / "build_dataset.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_exact_decimal_label_boundaries():
    assert [builder.band(x) for x in ("1", "1.799", "1.8", "2.6", "3.4", "4.2", "5")] == [
        "bad",
        "bad",
        "poor",
        "fair",
        "good",
        "excellent",
        "excellent",
    ]
    with pytest.raises(ValueError):
        builder.band("5.01")


def test_selection_has_disjoint_content_and_balanced_low_quality_classes():
    # Exercise the source-shape and sampling rules without the archived rating CSV.
    rows = [
        {
            "ref_img": f"parent-{parent:02d}.png",
            "dist_img": f"parent-{parent:02d}-distortion-{distortion:03d}.png",
            "dmos": ("1.4", "2.2", "3.0", "3.8", "4.6")[distortion % 5],
        }
        for parent in range(81)
        for distortion in range(125)
    ]
    selected = builder.select(rows)
    by_split = defaultdict(list)
    for row in selected:
        by_split[row["split"]].append(row)
    group_sets = [{r["ref_img"] for r in by_split[split]} for split in ("train", "dev", "test")]
    assert [len(s) for s in group_sets] == [20, 20, 40]
    assert not group_sets[0] & group_sets[1]
    assert not group_sets[0] & group_sets[2]
    assert not group_sets[1] & group_sets[2]
    assert len({r["dist_img"] for r in selected}) == 160
    assert Counter(row["sampling_band"] for row in selected) == {
        "bad": 40,
        "poor": 40,
        "fair": 40,
        "good": 20,
        "excellent": 20,
    }
    assert all(builder.band(row["dmos"]) == row["sampling_band"] for row in selected)
    for split, count in (("train", 40), ("dev", 40), ("test", 80)):
        chosen = by_split[split]
        assert len(chosen) == count
        assert sum(float(r["dmos"]) < 2.6 for r in chosen) == count // 2
        assert set(Counter(r["ref_img"] for r in chosen).values()) == {2}
    assert builder.select(list(reversed(rows))) == selected


def test_selection_rejects_incomplete_source_metadata():
    with pytest.raises(ValueError, match="all 81 parents"):
        builder.select([])
