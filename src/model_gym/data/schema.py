"""JSONL corpus schema.

One JSON object per line, one command per record::

    {"command": "rm -rf /srv/application-data/", "level": 2, "category": "shell", "group": "rm_recursive:app", ...}

``group`` names the near-duplicate family a command belongs to (template + scope).
Splits are made group-wise, so ``rm -r ./build`` and ``rm -rf ./build/`` never land
in different splits.

The reviewed corpus is stored one file per ``category`` (``data/seed/aws.jsonl``);
``read_jsonl`` rejects a row whose category contradicts its file name.
"""

from __future__ import annotations

import hashlib
import json
import re

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from model_gym.categories import CATEGORIES, DEFAULT_CATEGORY, coerce_category, is_category
from model_gym.labels import Level, coerce_level

SHELLS: Final[tuple[str, ...]] = ("bash", "sql", "redis", "http", "powershell")

MAX_COMMAND_CHARS: Final[int] = 8000

_WHITESPACE = re.compile(r"\s+")
_COMMENT_PREFIXES: Final[tuple[str, ...]] = ("#", "-- ")


class RecordError(ValueError):
    """Raised when a corpus record fails validation."""


def normalize_command(command: str) -> str:
    """Collapse whitespace so near-identical commands compare equal."""
    return _WHITESPACE.sub(" ", command).strip()


def command_digest(command: str) -> str:
    """Stable short digest of a normalized command, used as a record id."""
    return hashlib.sha256(normalize_command(command).encode("utf-8")).hexdigest()[:12]


def derive_group(command: str) -> str:
    """Derive a near-duplicate group from the leading program words.

    ``aws s3 rm s3://bucket/key --recursive`` becomes ``aws.s3.rm``. Only used
    when a record omits ``group``.
    """
    tokens = [token for token in normalize_command(command).split(" ") if token and not token.startswith("-")]
    if not tokens:
        return "unknown"
    return ".".join(tokens[:3]).lower()


@dataclass(frozen=True)
class RiskRecord:
    """A single labelled command."""

    command: str
    level: Level
    category: str = DEFAULT_CATEGORY
    group: str = ""
    shell: str = "bash"
    reason: str = ""
    source: str = "unknown"

    def __post_init__(self) -> None:
        command = self.command
        if not isinstance(command, str):
            raise TypeError(f"command must be a str, got {type(command).__name__}")
        if "\x00" in command:
            raise RecordError("command contains a NUL byte")
        normalized = normalize_command(command)
        if not normalized:
            raise RecordError("command is empty")
        if len(command) > MAX_COMMAND_CHARS:
            raise RecordError(f"command exceeds {MAX_COMMAND_CHARS} characters")
        if any(normalized.startswith(prefix) for prefix in _COMMENT_PREFIXES):
            raise RecordError(f"command looks like a comment: {normalized[:40]!r}")
        object.__setattr__(self, "command", normalized)
        object.__setattr__(self, "level", coerce_level(self.level))
        object.__setattr__(self, "category", coerce_category(self.category))
        shell = str(self.shell).strip().lower()
        if shell not in SHELLS:
            raise RecordError(f"unknown shell {self.shell!r}; expected one of {list(SHELLS)}")
        object.__setattr__(self, "shell", shell)
        object.__setattr__(self, "group", str(self.group).strip() or derive_group(normalized))
        object.__setattr__(self, "reason", _WHITESPACE.sub(" ", str(self.reason)).strip())
        object.__setattr__(self, "source", str(self.source).strip() or "unknown")

    @property
    def id(self) -> str:
        return command_digest(self.command)

    @property
    def label(self) -> int:
        """Zero-based label index (severity order)."""
        return self.level.index

    @property
    def slug(self) -> str:
        return self.level.slug

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "level": int(self.level),
            "label": self.slug,
            "category": self.category,
            "group": self.group,
            "shell": self.shell,
            "reason": self.reason,
            "source": self.source,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Any, *, origin: str = "<memory>") -> RiskRecord:
        """Build a record from a parsed JSON object, with a located error on failure."""
        if not isinstance(raw, dict):
            raise RecordError(f"{origin}: expected a JSON object, got {type(raw).__name__}")
        if "command" not in raw:
            raise RecordError(f"{origin}: missing 'command'")
        if "level" not in raw:
            raise RecordError(f"{origin}: missing 'level'")
        try:
            return cls(
                command=raw["command"],
                level=raw["level"],
                category=raw.get("category", DEFAULT_CATEGORY),
                group=raw.get("group", ""),
                shell=raw.get("shell", "bash"),
                reason=raw.get("reason", ""),
                source=raw.get("source", "unknown"),
            )
        except (RecordError, TypeError, ValueError) as exc:
            raise RecordError(f"{origin}: {exc}") from exc


def read_jsonl(path: Path | str) -> list[RiskRecord]:
    """Read a JSONL corpus, reporting the offending line on failure.

    When the file is named after a category, every row must carry that category.
    """
    path = Path(path)
    expected_category = path.stem if is_category(path.stem) else None
    records: list[RiskRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            origin = f"{path}:{lineno}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordError(f"{origin}: invalid JSON: {exc}") from exc
            record = RiskRecord.from_dict(raw, origin=origin)
            if expected_category is not None and record.category != expected_category:
                raise RecordError(f"{origin}: category {record.category!r} does not match the file name {path.stem!r}")
            records.append(record)
    return records


def write_jsonl(path: Path | str, records: Iterable[RiskRecord]) -> int:
    """Write records as JSONL, creating parent directories. Returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json())
            handle.write("\n")
            count += 1
    return count


def dedupe(records: Iterable[RiskRecord]) -> tuple[list[RiskRecord], list[RiskRecord]]:
    """Split records into (unique, dropped), keeping the first occurrence.

    Callers should pass curated records before generated ones so that a hand-scored
    command wins over a templated duplicate.
    """
    seen: set[str] = set()
    unique: list[RiskRecord] = []
    dropped: list[RiskRecord] = []
    for record in records:
        key = normalize_command(record.command)
        if key in seen:
            dropped.append(record)
            continue
        seen.add(key)
        unique.append(record)
    return unique, dropped


def level_counts(records: Iterable[RiskRecord]) -> dict[int, int]:
    """Count records per severity level, including zeros."""
    counts = dict.fromkeys(range(1, len(Level) + 1), 0)
    for record in records:
        counts[int(record.level)] += 1
    return counts


def category_counts(records: Iterable[RiskRecord]) -> dict[str, int]:
    """Count records per tool category, including categories with no rows."""
    counts = dict.fromkeys(CATEGORIES, 0)
    for record in records:
        counts[record.category] += 1
    return counts


def iter_paths(paths: Sequence[Path | str]) -> Iterator[Path]:
    """Expand files, directories, and globs into ``*.jsonl`` paths, sorted."""
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        found = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for candidate in found:
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate
