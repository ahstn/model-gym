"""Acquire additional public training-source photos without changing prior manifests."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from corpus_common import DATA, DATA_ROOT, SEED, digest, read_lines, stable_key, write_json, write_lines

from visual_quality import megalith, vizwiz


def bounded_download(url: str, path: Path, max_bytes: int):
    """Bound remote-source stalls while retaining the adapter's byte/decode checks."""
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".expansion-download")
    for attempt in range(2):
        started = time.monotonic()
        try:
            with requests.get(url, stream=True, timeout=(10, 20)) as response:
                response.raise_for_status()
                count = 0
                with temporary.open("wb") as output:
                    for block in response.iter_content(256 * 1024):
                        count += len(block)
                        if count > max_bytes or time.monotonic() - started > 60:
                            raise ValueError("Declared size or 60-second transfer resource limit exceeded")
                        output.write(block)
            if count == 0:
                raise ValueError("Empty image download")
            temporary.replace(path)
            return path
        except requests.RequestException:
            if attempt:
                raise
        finally:
            temporary.unlink(missing_ok=True)


def megalith_candidates():
    directory = DATA_ROOT / "sources/megalith"
    original = read_lines(DATA_ROOT / "acquisition/megalith.jsonl")
    excluded = {row["source_id"] for row in original}
    selected = read_lines(directory / "selection-20260927.jsonl")
    pool = [row for row in selected if row["id"] not in excluded and megalith.eligible(row)]
    pool.sort(key=lambda row: stable_key("extra-megalith", row["id"]))
    megalith.download = bounded_download
    return pool, lambda row: megalith._fetch_image(directory, row), directory / "selection-20260927.jsonl"


def vizwiz_candidates():
    directory = DATA_ROOT / "sources/vizwiz"
    annotation = directory / "train.json"
    if digest(annotation) != vizwiz.ANNOTATION_SHA256:
        raise ValueError("VizWiz annotation revision changed")
    old = {row["source_id"] for row in read_lines(DATA_ROOT / "acquisition/vizwiz.jsonl")}
    pool = [
        row
        for row in vizwiz.select_candidates(json.loads(annotation.read_text()), SEED)
        if row["image"] not in old
    ]
    index = vizwiz._archive_index(directory)
    evidence = directory / "evidence/provenance.json"
    return pool, lambda row: vizwiz._acquire_image(row, index[row["image"]], directory, evidence), annotation


def acquire(source: str, count: int, workers: int):
    output = DATA / "acquisition" / f"extra-{source}.jsonl"
    failures_path = output.with_name(f"extra-{source}-failures.jsonl")
    rows, failures = read_lines(output), read_lines(failures_path)
    seen = {r["source_id"] for r in rows}
    failed = {r["source_id"] for r in failures}
    pool, fetch, source_metadata = megalith_candidates() if source == "megalith" else vizwiz_candidates()
    identity = (lambda row: row["id"]) if source == "megalith" else (lambda row: row["image"])
    pool = [r for r in pool if identity(r) not in seen | failed]

    def attempt(row):
        try:
            record = fetch(row)
            record["component"] = source
            return record, None
        except (requests.RequestException, ValueError, OSError, SyntaxError) as error:
            return None, {"source_id": identity(row), "reason": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(workers) as executor:
        for start in range(0, len(pool), 512):
            if len(rows) >= count:
                break
            for offset, (record, error) in enumerate(
                executor.map(attempt, pool[start : start + min(512, count - len(rows))]), 1
            ):
                if error:
                    failures.append(error)
                elif record["source_id"] not in seen:
                    rows.append(record)
                    seen.add(record["source_id"])
                if offset % 64 == 0:
                    write_lines(output, rows)
                    write_lines(failures_path, failures)
                    print(f"{source}: {len(rows)}/{count}, failures={len(failures)}", flush=True)
            write_lines(output, rows)
            write_lines(failures_path, failures)
            print(f"{source}: {len(rows)}/{count}, failures={len(failures)}", flush=True)
    write_json(
        output.with_suffix(".provenance.json"),
        {
            "source": source,
            "requested": count,
            "acquired": len(rows),
            "seed": SEED,
            "selection": "Pinned training-source metadata; fixed seeded ordering, no quality model scores or captions used",
            "source_metadata_path": str(source_metadata),
            "source_metadata_sha256": digest(source_metadata),
            "script_sha256": digest(__file__),
            "records_sha256": digest(output),
            "prior_acquisition_unchanged": True,
        },
    )
    if len(rows) < count:
        raise RuntimeError(f"Insufficient additional {source}: {len(rows)}/{count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("megalith", "vizwiz"))
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    acquire(args.source, args.count, args.workers)
