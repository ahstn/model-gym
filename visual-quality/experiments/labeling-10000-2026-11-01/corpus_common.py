"""Paths and deterministic serialization for the 10,000-view corpus expansion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
DATA_ROOT = Path("/home/ahstn/git/model-gym/visual-quality/data")
DATA = DATA_ROOT / "labeling-10000-2026-11-01"
PREVIOUS = HERE.parent / "rehearsal-4000-2026-11-01"
SEED = 2026110110000


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_lines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def stable_key(kind: str, identity: str) -> str:
    return hashlib.sha256(f"{SEED}:{kind}:{identity}".encode()).hexdigest()
