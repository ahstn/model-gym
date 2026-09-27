"""The Decision record shared by data building, training and evaluation."""

from __future__ import annotations

import json
import math

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Kind = Literal["choice", "score", "noul"]
KINDS: tuple[Kind, ...] = ("choice", "score", "noul")
NOUL_OPTIONS: tuple[str, str] = ("No", "Yes")

_MAX_OPTIONS = 255
_SUM_TOL = 1e-3


@dataclass(frozen=True, slots=True)
class Decision:
    id: str
    dataset: str
    source: str
    kind: Kind
    state: str
    question: str
    options: tuple[str, ...]
    target: tuple[float, ...]
    group_id: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: unknown kind {self.kind!r}")
        if not 2 <= len(self.options) <= _MAX_OPTIONS:
            raise ValueError(f"{self.id}: need 2..{_MAX_OPTIONS} options, got {len(self.options)}")
        if len(self.target) != len(self.options):
            raise ValueError(f"{self.id}: {len(self.target)} targets for {len(self.options)} options")
        if self.kind == "noul" and tuple(self.options) != NOUL_OPTIONS:
            raise ValueError(f"{self.id}: noul options must be {NOUL_OPTIONS}")
        if any(not math.isfinite(t) or t < 0 for t in self.target):
            raise ValueError(f"{self.id}: targets must be finite and >= 0")
        total = math.fsum(self.target)
        if abs(total - 1.0) > _SUM_TOL:
            raise ValueError(f"{self.id}: targets sum to {total}, not 1")

    def permuted(self, order: Sequence[int]) -> Decision:
        if sorted(order) != list(range(len(self.options))):
            raise ValueError(f"{self.id}: order is not a permutation of {len(self.options)} options")
        return Decision(
            id=self.id,
            dataset=self.dataset,
            source=self.source,
            kind=self.kind,
            state=self.state,
            question=self.question,
            options=tuple(self.options[i] for i in order),
            target=tuple(self.target[i] for i in order),
            group_id=self.group_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "source": self.source,
            "kind": self.kind,
            "state": self.state,
            "question": self.question,
            "options": list(self.options),
            "target": list(self.target),
            "group_id": self.group_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Decision:
        return cls(
            id=str(d["id"]),
            dataset=str(d["dataset"]),
            source=str(d["source"]),
            kind=d["kind"],
            state=str(d["state"]),
            question=str(d["question"]),
            options=tuple(str(o) for o in d["options"]),
            target=tuple(float(t) for t in d["target"]),
            group_id=str(d["group_id"]),
        )


def write_jsonl(path: Path, rows: Iterable[Decision]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Decision]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Decision.from_dict(json.loads(line))
