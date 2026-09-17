"""Corpus record validation: normalization, grouping, dedupe, and JSONL I/O."""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from model_gym.categories import CATEGORIES
from model_gym.data.schema import (
    RecordError,
    RiskRecord,
    category_counts,
    command_digest,
    dedupe,
    derive_group,
    level_counts,
    normalize_command,
    read_jsonl,
    write_jsonl,
)
from model_gym.labels import Level


def make_record(
    command: str = "rm -rf /srv/application-data/",
    level: int = 2,
    group: str = "rm_recursive:app_data",
    **kwargs: object,
) -> RiskRecord:
    return RiskRecord(command=command, level=level, group=group, **kwargs)  # type: ignore[arg-type]


def test_command_is_normalized_and_indexed() -> None:
    record = make_record(command="  kubectl   delete   namespace    production \n")
    assert record.command == "kubectl delete namespace production"
    assert record.level is Level.DANGEROUS
    assert record.label == 1
    assert record.slug == "dangerous"
    assert record.id == command_digest(record.command)


def test_whitespace_normalization_collapses_newlines() -> None:
    record = make_record(command="kubectl apply -f -\n", level=3)
    assert record.command == "kubectl apply -f -"


def test_group_is_derived_when_omitted() -> None:
    record = RiskRecord(command="aws s3 rm s3://bucket/key --recursive", level=1, group="")
    assert record.group == "aws.s3.rm"
    assert derive_group("kubectl delete pod -n prod") == "kubectl.delete.pod"
    assert normalize_command("  a   b ") == "a b"


@pytest.mark.parametrize(
    ("command", "level", "shell"),
    [
        ("", 3, "bash"),
        ("   ", 3, "bash"),
        ("rm -rf /", 0, "bash"),
        ("rm -rf /", 6, "bash"),
        ("rm -rf /", 3, "nushell"),
        ("# rm -rf /", 3, "bash"),
        ("rm -rf /\x00", 3, "bash"),
    ],
)
def test_invalid_records_are_rejected(command: str, level: int, shell: str) -> None:
    with pytest.raises((RecordError, ValueError, TypeError)):
        RiskRecord(command=command, level=level, group="g", shell=shell)


@pytest.mark.parametrize("category", ["kubectl", "seed_commands", "", "scripts"])
def test_unknown_categories_are_rejected(category: str) -> None:
    with pytest.raises((RecordError, ValueError, TypeError)):
        RiskRecord(command="rm -rf /", level=1, category=category)


def test_category_defaults_to_shell_and_is_kept_through_jsonl(tmp_path: Path) -> None:
    default = RiskRecord(command="rm -rf /", level=1)
    assert default.category == "shell"
    explicit = RiskRecord(command="DROP TABLE users;", level=1, category="SQL", shell="sql")
    assert explicit.category == "sql"
    path = tmp_path / "corpus.jsonl"
    write_jsonl(path, [default, explicit])
    loaded = read_jsonl(path)
    assert [record.category for record in loaded] == ["shell", "sql"]
    assert category_counts(loaded) == {**dict.fromkeys(CATEGORIES, 0), "shell": 1, "sql": 1}


def test_a_category_named_file_rejects_foreign_rows(tmp_path: Path) -> None:
    path = tmp_path / "aws.jsonl"
    write_jsonl(path, [RiskRecord(command="aws s3 rb s3://prod-backups", level=1, category="aws")])
    assert [record.category for record in read_jsonl(path)] == ["aws"]
    write_jsonl(path, [RiskRecord(command="rm -rf /", level=1, category="shell")])
    with pytest.raises(RecordError, match="does not match the file name"):
        read_jsonl(path)


def test_jsonl_roundtrip_preserves_records(tmp_path: Path) -> None:
    records = [
        make_record(command="rm -rf ./build", level=3),
        make_record(command="ls -la /var/log", level=5, group="read_only_inspection:ls_logs"),
    ]
    path = tmp_path / "corpus.jsonl"
    assert write_jsonl(path, records) == 2
    loaded = read_jsonl(path)
    assert [record.command for record in loaded] == [record.command for record in records]
    assert [record.level for record in loaded] == [record.level for record in records]
    assert [record.group for record in loaded] == [record.group for record in records]
    assert level_counts(loaded) == {1: 0, 2: 0, 3: 1, 4: 0, 5: 1}


def test_invalid_jsonl_reports_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"command": "ls", "level": 5}\n{"command": "ls", "level": 9}\n', encoding="utf-8")
    with pytest.raises(RecordError, match=r":2:"):
        read_jsonl(path)


def test_from_dict_requires_command_and_level() -> None:
    with pytest.raises(RecordError, match="missing 'level'"):
        RiskRecord.from_dict({"command": "ls"}, origin="row 1")
    with pytest.raises(RecordError, match="missing 'command'"):
        RiskRecord.from_dict({"level": 5}, origin="row 1")


def test_written_rows_roundtrip_with_their_label_slug(tmp_path: Path) -> None:
    record = make_record(
        command="terraform destroy -auto-approve", level=1, reason="destroys infrastructure", source="seed:test"
    )
    path = tmp_path / "one.jsonl"
    write_jsonl(path, [record])
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["label"] == "critical"
    assert payload["level"] == 1
    assert payload["reason"] == "destroys infrastructure"
    reloaded = RiskRecord.from_dict(payload)
    assert reloaded == record


def test_dedupe_keeps_the_first_occurrence() -> None:
    curated = make_record(command="rm -rf /", level=1, source="seed")
    generated = make_record(command="rm   -rf  /", level=1, source="synthetic")
    other = make_record(command="ls -la", level=5, group="read_only_inspection:ls_logs")
    unique, dropped = dedupe([curated, generated, other])
    assert [record.source for record in unique] == ["seed", "unknown"]
    assert [record.source for record in dropped] == ["synthetic"]
    assert normalize_command(generated.command) == normalize_command(curated.command)
