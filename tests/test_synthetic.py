"""The templated bootstrap corpus: uniqueness, level sanity, and seed alignment."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from model_gym.data import synthetic
from model_gym.data.schema import RiskRecord, dedupe, read_jsonl
from model_gym.labels import Level

SEED_DIR = Path("data/seed")


def seed_records() -> list[RiskRecord]:
    """Every hand-scored row in the reviewed corpus, across category files."""
    return [record for path in sorted(SEED_DIR.glob("*.jsonl")) for record in read_jsonl(path)]


@pytest.fixture(scope="module")
def generated() -> list[RiskRecord]:
    return synthetic.generate()


def test_generated_commands_are_unique(generated: list[RiskRecord]) -> None:
    unique, dropped = dedupe(generated)
    duplicates = sorted({record.command for record in dropped})
    assert duplicates == [], f"the generator emits duplicate commands: {duplicates[:5]}"
    assert len(unique) == len(generated)


def test_levels_stay_inside_the_taxonomy(generated: list[RiskRecord]) -> None:
    for record in generated:
        assert Level.CRITICAL <= record.level <= Level.LOW_RISK


def test_every_level_is_covered(generated: list[RiskRecord]) -> None:
    counts = Counter(int(record.level) for record in generated)
    assert set(counts) == {1, 2, 3, 4, 5}
    assert counts[5] >= 5, "level 5 examples are needed or the classifier can never allow anything"


@pytest.mark.parametrize(
    ("command", "level"),
    [
        ("rm -rf /", 1),
        ("rm -rf /tmp/scratch/", 3),
        ("sudo rm -rf /", 1),
        ("cat /etc/shadow", 4),
        ("ls -la /var/log", 5),
        ("kubectl delete namespace production", 1),
        ("kubectl delete --dry-run=server namespace production", 5),
        ("git clean -ndx", 5),
        ("git clean -fd", 2),
        ("git reset --hard origin/main", 3),
        ("git branch -D main", 3),
        ("git push --force origin main", 2),
        ("git push origin feature/x", 5),
        ("docker container prune -f", 2),
        ("docker network prune -f", 4),
        ("aws ec2 terminate-instances --instance-ids i-0a1b2c3d4e5f00002", 2),
        ("aws ec2 terminate-instances --instance-ids i-0a1b2c3d4e5f00001 --dry-run", 5),
        ("DROP DATABASE production;", 1),
        ("REVOKE ALL PRIVILEGES ON DATABASE production FROM app_user;", 3),
        ("redis-cli -h redis-prod.internal FLUSHDB", 2),
        ("redis-cli -h localhost FLUSHALL", 4),
    ],
)
def test_pinned_levels(generated: list[RiskRecord], command: str, level: int) -> None:
    by_command = {record.command: int(record.level) for record in generated}
    assert by_command[command] == level


def test_seed_groups_stay_inside_one_category() -> None:
    # A group is one near-duplicate family; seeing it in two category files means a
    # row was filed wrong and its near-duplicates may straddle splits.
    categories_by_group: dict[str, set[str]] = defaultdict(set)
    for record in seed_records():
        categories_by_group[record.group].add(record.category)
    split_groups = {group: sorted(categories) for group, categories in categories_by_group.items()}
    assert [group for group, categories in split_groups.items() if len(categories) > 1] == []


def test_every_seed_category_has_coverage_and_controls() -> None:
    by_category: dict[str, list[RiskRecord]] = defaultdict(list)
    for record in seed_records():
        by_category[record.category].append(record)
    assert by_category, "the reviewed corpus is empty"
    for category, rows in sorted(by_category.items()):
        levels = {int(record.level) for record in rows}
        assert len(levels) >= 3, f"{category} only has level(s) {sorted(levels)}"
        assert any(level >= 4 for level in levels), f"{category} has no level 4-5 control rows"


def test_seed_rows_agree_with_the_generated_category(generated: list[RiskRecord]) -> None:
    # Where a hand-scored row shares a group with a generated row, they must agree.
    by_group = {record.group: record.category for record in generated}
    for record in seed_records():
        if record.group in by_group:
            assert by_group[record.group] == record.category, f"{record.command} is filed under {record.category}"


def test_duplicate_commands_have_one_annotation(generated: list[RiskRecord]) -> None:
    annotations: dict[str, set[tuple[int, str, str, str]]] = defaultdict(set)
    for record in seed_records() + generated:
        annotations[record.command].add((int(record.level), record.group, record.category, record.shell))
    conflicts = {command: values for command, values in annotations.items() if len(values) != 1}
    assert conflicts == {}
