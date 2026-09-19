"""Group-aware splitting and corpus assembly."""

from __future__ import annotations

import hashlib
import json

from collections import defaultdict
from pathlib import Path

import pytest

from model_gym.categories import CATEGORIES
from model_gym.data.build import SPLITS, build_corpus, load_records, split_records
from model_gym.data.schema import RiskRecord, read_jsonl
from model_gym.labels import NUM_LABELS

SEED_DIR = Path("data/seed")


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


@pytest.mark.parametrize(
    "commands",
    [
        ("git reset --hard HEAD~1", "git reset --mixed HEAD~1", "git reset --soft HEAD~1"),
        ("docker volume prune -f", "docker volume prune -a -f"),
        (
            "aws ec2 terminate-instances --instance-ids i-0a1b2c3d4e5f00001",
            "aws ec2 terminate-instances --instance-ids i-0a1b2c3d4e5f00002",
            "aws ec2 terminate-instances --instance-ids i-0a1b2c3d4e5f00001 --dry-run",
        ),
        (
            "terraform -chdir=environments/production destroy -auto-approve",
            "terraform -chdir=environments/production plan -destroy",
        ),
        (
            'op item share "Production Database" --vault Production',
            'op item get "Production Database" --vault Production --fields title',
            'op item edit "Production Database" --vault Production --generate-password=32 --dry-run',
        ),
    ],
)
def test_reviewed_and_generated_near_families_cannot_leak(commands: tuple[str, ...]) -> None:
    records, _, _ = load_records([SEED_DIR], include_synthetic=True)
    by_command = {record.command: record for record in records}
    assert len({by_command[command].group for command in commands}) == 1
    for seed in (3, 7, 13):
        splits = split_records(records, ratios=(0.7, 0.15, 0.15), seed=seed)
        assignments = {record.command: split for split, rows in splits.items() for record in rows}
        assert len({assignments[command] for command in commands}) == 1


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
        sources=[SEED_DIR],
        output_dir=tmp_path,
        synthetic_output=tmp_path / "synthetic.jsonl",
        seed=13,
        include_synthetic=True,
    )
    for split in SPLITS:
        assert (tmp_path / f"{split}.jsonl").is_file()
    card = json.loads((tmp_path / "dataset_card.json").read_text(encoding="utf-8"))
    assert card["split_policy"] == {"seed": 13, "ratios": [0.7, 0.15, 0.15], "unit": "group"}
    for split in SPLITS:
        split_path = tmp_path / f"{split}.jsonl"
        assert card["split_sha256"][split] == hashlib.sha256(split_path.read_bytes()).hexdigest()
        records = read_jsonl(split_path)
        assert set(card["split_groups"][split]) == {record.group for record in records}
        for category in CATEGORIES:
            assert card["category_groups"][category][split] == len(
                {record.group for record in records if record.category == category}
            )
        for source, coverage in card["sources"].items():
            source_records = [record for record in records if record.source == source]
            assert coverage[split]["rows"] == len(source_records)
            assert coverage[split]["groups"] == len({record.group for record in source_records})
    assert card["totals"]["records"] == summary.total
    assert card["split_totals"]["train"] == summary.split_counts["train"]
    seed_files = {str(path) for path in sorted(SEED_DIR.glob("*.jsonl"))}
    assert set(card["source_files"]) == seed_files | {"<generated:synthetic:templates-v1>"}
    assert summary.synthetic_rows > 0
    for split in SPLITS:
        levels = summary.split_levels[split]
        assert sum(levels.values()) == summary.split_counts[split]
    assert set(card["categories"]) == set(CATEGORIES)
    assert sum(row["total"] for row in card["categories"].values()) == summary.total
    for category in CATEGORIES:
        expected = sum(summary.split_categories[split][category] for split in SPLITS)
        assert card["categories"][category]["total"] == expected


def test_thin_corpora_are_reported(tmp_path: Path) -> None:
    # A corpus with one level and one category must not be reported as ready.
    corpus = tmp_path / "thin.jsonl"
    corpus.write_text(
        "".join(
            json.dumps({"command": f"kubectl get pod p{index}", "level": 5, "category": "kubernetes"}) + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    summary = build_corpus(sources=[corpus], output_dir=tmp_path / "out", include_synthetic=False, seed=13)
    assert summary.total == 6
    assert summary.synthetic_rows == 0
    assert any("level 1" in warning and "training rows" in warning for warning in summary.warnings)
    assert any("no examples for level" in warning for warning in summary.warnings)
    assert any(warning.startswith("no rows for") and "azure" in warning for warning in summary.warnings)
    assert any("kubernetes" in warning and "validation" in warning for warning in summary.warnings)
    assert any("kubernetes" in warning and "test" in warning for warning in summary.warnings)
    assert all(0 <= summary.split_levels[split][level] for split in SPLITS for level in range(1, NUM_LABELS + 1))


def test_empty_sources_are_reported(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match=r"no \*\.jsonl corpus files found"):
        build_corpus(sources=[empty], output_dir=tmp_path / "out", include_synthetic=False, seed=13)
