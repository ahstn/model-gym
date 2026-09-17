"""Group-aware splitting and corpus assembly."""

from __future__ import annotations

import json

from collections import defaultdict
from pathlib import Path

import pytest

from model_gym.data.build import SPLITS, build_corpus, split_records
from model_gym.data.schema import RiskRecord
from model_gym.labels import NUM_LABELS

SEED_CORPUS = Path("data/seed/seed_commands.jsonl")


def make_records(count: int, *, groups: int = 10, levels: int = 5) -> list[RiskRecord]:
    records = []
    for index in range(count):
        group = f"family:{index % groups}"
        level = index % levels + 1
        records.append(RiskRecord(command=f"cmd-{index} --group {group}", level=level, group=group, source="test"))
    return records


def test_groups_never_straddle_splits() -> None:
    records = make_records(120, groups=12)
    splits = split_records(records, ratios=(0.7, 0.15, 0.15), seed=3)
    assignment: dict[str, set[str]] = defaultdict(set)
    for split, rows in splits.items():
        for record in rows:
            assignment[record.group].add(split)
    assert all(len(splits_for_group) == 1 for splits_for_group in assignment.values())
    assert sum(len(rows) for rows in splits.values()) == len(records)
    assert all(splits.values())


def test_splitting_is_deterministic() -> None:
    records = make_records(60, groups=8)
    first = split_records(records, ratios=(0.7, 0.15, 0.15), seed=7)
    second = split_records(records, ratios=(0.7, 0.15, 0.15), seed=7)
    assert {split: [record.command for record in rows] for split, rows in first.items()} == {
        split: [record.command for record in rows] for split, rows in second.items()
    }


def test_train_gets_the_largest_share() -> None:
    splits = split_records(make_records(200, groups=20), ratios=(0.7, 0.15, 0.15), seed=1)
    assert len(splits["train"]) > len(splits["validation"])
    assert len(splits["train"]) > len(splits["test"])


def test_empty_splits_are_backfilled_when_possible() -> None:
    records = [
        RiskRecord(command=f"cmd-{index}", level=index + 1, group=f"g{index}", source="test") for index in range(3)
    ]
    splits = split_records(records, ratios=(0.7, 0.15, 0.15), seed=5)
    assert all(splits[split] for split in SPLITS)


def test_split_records_rejects_bad_ratios() -> None:
    with pytest.raises(ValueError, match="split ratios"):
        split_records(make_records(10), ratios=(0.7, 0.3), seed=1)
    with pytest.raises(ValueError, match="split ratios"):
        split_records(make_records(10), ratios=(0.0, 0.0, 0.0), seed=1)


def test_build_corpus_writes_splits_and_card(tmp_path: Path) -> None:
    summary = build_corpus(
        sources=[SEED_CORPUS],
        output_dir=tmp_path,
        synthetic_output=tmp_path / "synthetic.jsonl",
        seed=13,
        include_synthetic=True,
    )
    for split in SPLITS:
        assert (tmp_path / f"{split}.jsonl").is_file()
    card = json.loads((tmp_path / "dataset_card.json").read_text(encoding="utf-8"))
    assert card["totals"]["records"] == summary.total
    assert card["split_totals"]["train"] == summary.split_counts["train"]
    assert set(card["source_files"]) == {str(SEED_CORPUS), "<generated:synthetic:templates-v1>"}
    assert summary.synthetic_rows > 0
    assert sum(summary.split_counts.values()) > 200
    for split in SPLITS:
        levels = summary.split_levels[split]
        assert sum(levels.values()) == summary.split_counts[split]
    assert not summary.warnings, summary.warnings


def test_seed_only_corpus_warns_about_missing_levels(tmp_path: Path) -> None:
    summary = build_corpus(
        sources=[SEED_CORPUS],
        output_dir=tmp_path,
        include_synthetic=False,
        seed=13,
    )
    assert summary.total == 50
    assert summary.synthetic_rows == 0
    assert any("level 4" in warning and "training rows" in warning for warning in summary.warnings)
    # The seed set has no level 4 or 5 rows, so unbalanced splits must be reported.
    assert any("no examples for level" in warning for warning in summary.warnings)
    assert all(0 <= summary.split_levels[split][level] for split in SPLITS for level in range(1, NUM_LABELS + 1))
