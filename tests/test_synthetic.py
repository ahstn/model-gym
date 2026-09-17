"""The templated bootstrap corpus: uniqueness, level sanity, and seed alignment."""

from __future__ import annotations

from collections import Counter

import pytest

from model_gym.data import synthetic
from model_gym.data.schema import RiskRecord, dedupe, read_jsonl
from model_gym.labels import Level

SEED_CORPUS = "data/seed/seed_commands.jsonl"


@pytest.fixture(scope="module")
def generated() -> list[RiskRecord]:
    return synthetic.generate()


def test_families_are_structurally_complete() -> None:
    for family in synthetic.FAMILIES:
        assert family.scopes, f"{family.name} has no scopes"
        assert family.variants, f"{family.name} has no variants"
        assert "{" in family.template
        assert family.note
        assert len({scope.key for scope in family.scopes}) == len(family.scopes), (
            f"{family.name} has duplicate scope keys"
        )
        assert len({variant.name for variant in family.variants}) == len(family.variants)


def test_generated_commands_are_unique(generated: list[RiskRecord]) -> None:
    unique, dropped = dedupe(generated)
    duplicates = sorted({record.command for record in dropped})
    assert duplicates == [], f"the generator emits duplicate commands: {duplicates[:5]}"
    assert len(unique) == len(generated) > 200


def test_generated_groups_name_their_family(generated: list[RiskRecord]) -> None:
    family_names = {family.name for family in synthetic.FAMILIES}
    for record in generated:
        family, _, key = record.group.partition(":")
        assert family in family_names
        assert key
        assert record.source == synthetic.SOURCE
        assert record.reason.endswith(".")
        assert "\n" not in record.command


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
        ("git push --force origin main", 2),
        ("git push origin feature/x", 5),
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


def test_seed_groups_are_covered_by_the_generator(generated: list[RiskRecord]) -> None:
    # Seed rows and generated rows for the same operation must share a group, so a
    # split never separates `rm -rf X` from its variants.
    seed_groups = {record.group for record in read_jsonl(SEED_CORPUS)}
    generated_groups = {record.group for record in generated}
    assert seed_groups <= generated_groups, f"unmatched seed groups: {sorted(seed_groups - generated_groups)}"
