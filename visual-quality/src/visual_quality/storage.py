"""Atomic local records and bounded, cached HTTP downloads."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests


def atomic_json(path: Path, obj: Any) -> None:
    _atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    _atomic_text(path, "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, max_bytes: int = 64 * 1024 * 1024) -> Path:
    """Reuse complete files; retry failed streams without exposing partial bytes."""
    path = Path(path)
    if path.is_file() and path.stat().st_size:
        return path
    if not url.startswith("https://"):
        raise ValueError("Acquisition requires an HTTPS source URL")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                if int(response.headers.get("Content-Length", 0)) > max_bytes:
                    raise ValueError(f"Download exceeds {max_bytes} bytes: {url}")
                size = 0
                with temporary.open("wb") as output:
                    for block in response.iter_content(256 * 1024):
                        size += len(block)
                        if size > max_bytes:
                            raise ValueError(f"Download exceeds {max_bytes} bytes: {url}")
                        output.write(block)
                if size == 0:
                    raise ValueError(f"Empty download: {url}")
            os.replace(temporary, path)
            return path
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError("Unreachable download state")
