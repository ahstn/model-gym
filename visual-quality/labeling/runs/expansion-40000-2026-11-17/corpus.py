# ruff: noqa: I001
# Imports follow the explicit legacy run-context bootstrap order.
"""Prepare fresh expansion candidates using the frozen source/rendering adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/home/ahstn/git/model-gym/visual-quality/data")
DATA = DATA_ROOT / "labeling-40000-2026-11-17"
OLD = DATA_ROOT / "labeling-20000-2026-11-16"
INITIAL = DATA_ROOT / "labeling-10000-2026-11-01"
SEED = 2026111740000
os.environ.setdefault(
    "VISUAL_QUALITY_HISTORY", str(DATA_ROOT / "experiment-history-2026-11-04/tree/visual-quality/experiments")
)
sys.path[:0] = [str(PACKAGE / "src"), str(PACKAGE / "experiments/labeling-10000-2026-11-01")]
# These legacy adapters share a single run context. Configure it before imports;
# the original run's modules and data are never edited.
import corpus_common as common

common.DATA, common.SEED = DATA, SEED
from corpus_common import digest, read_lines, stable_key, write_json, write_lines
import corpus_acquire as acquisition
import corpus_prepare as preparation
from visual_quality.validation import _HashIndex


def old_cases():
    rows = json.loads((OLD / "combined-manifest.json").read_text())["cases"]
    if len(rows) != 20000 or len({r["case_id"] for r in rows}) != 20000:
        raise ValueError("Expected exactly 20,000 unique previously labeled cases")
    return rows


def pin_initial_evidence():
    """Freeze hashes once, then verify rather than replacing historical evidence."""
    target = DATA / "initial-evidence-pins.json"
    if target.exists():
        pins = json.loads(target.read_text())
    else:
        paths = [
            OLD / name
            for name in (
                "manifest.json",
                "combined-manifest.json",
                "stats.jsonl",
                "cluster-prototypes.npz",
                "corpus-embeddings.npz",
                "content-coverage.json",
                "initial-evidence-pins.json",
            )
        ]
        paths += sorted(p for p in (OLD / "exports").rglob("*") if p.is_file())
        paths += [
            INITIAL / name
            for name in (
                "manifest.json",
                "references-all.jsonl",
                "references-freeze.json",
                "stats.jsonl",
                "exports/labelled-images.jsonl",
                "exports/scores-long.jsonl",
                "exports/summary.json",
            )
        ]
        paths += sorted(p for p in (PACKAGE / "labeling/records").iterdir() if p.is_file())
        pins = {str(path): digest(path) for path in paths}
        write_json(target, pins)
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError(f"Historical evidence changed: {path}")
    return pins


def acquire(source, count, workers, megalith_pool=None):
    if megalith_pool is not None and source != "megalith":
        raise ValueError("--megalith-pool is only valid for Megalith acquisition")
    if source not in ("megalith", "vizwiz") or not count or count < 1:
        raise ValueError("Acquisition requires a supported source and positive count")
    if (DATA / "manifest.json").exists():
        raise ValueError("Selection already frozen; acquisition must not change")
    pin_initial_evidence()
    excluded = {r["source_id"] for r in old_cases()}
    protected = read_lines(DATA_ROOT / "pilot-10k-v1/development.jsonl") + read_lines(
        DATA_ROOT / "pilot-10k-v1/test.jsonl"
    )
    excluded.update(r["source_id"] for r in protected)
    output = DATA / "acquisition" / f"extra-{source}.jsonl"
    cached, seen = [], set()
    paths = [output, DATA_ROOT / f"acquisition/{source}.jsonl"]
    paths += [root / f"acquisition/extra-{source}.jsonl" for root in (INITIAL, OLD)]
    for path in paths:
        for row in read_lines(path):
            if row["source_id"] not in excluded | seen:
                cached.append(row)
                seen.add(row["source_id"])
    write_lines(output, cached)
    factory_name = source + "_candidates"
    original = getattr(acquisition, factory_name)

    def candidates():
        if megalith_pool is not None:
            from acquire_pool import candidates as expanded_candidates

            pool, fetch, evidence = expanded_candidates(Path(megalith_pool))
        else:
            pool, fetch, evidence = original()
        key = "id" if source == "megalith" else "image"
        return [r for r in pool if r[key] not in excluded], fetch, evidence

    setattr(acquisition, factory_name, candidates)
    try:
        acquisition.acquire(source, count, workers)
    finally:
        setattr(acquisition, factory_name, original)


def prepare(workers):
    if (DATA / "manifest.json").exists():
        raise ValueError("Selection already frozen; preparation must not change")
    pin_initial_evidence()
    old = old_cases()
    protected = read_lines(DATA_ROOT / "pilot-10k-v1/development.jsonl") + read_lines(
        DATA_ROOT / "pilot-10k-v1/test.jsonl"
    )
    ids = {r["image_id"] for r in old} | {r["id"] for r in protected}
    families = {r["family_id"] for r in old} | {r["split_group"] for r in protected}
    pixels = {r["decoded_sha256"] for r in old} | {r["decoded_sha256"] for r in protected}
    byte_hashes = {r["source_sha256"] for r in old} | {r["sha256"] for r in protected}
    views = {r["view_sha256"] for r in old}
    index = _HashIndex()
    for row in old + protected:
        if row.get("dhash"):
            index.add(int(row["dhash"], 16), row.get("image_id", row.get("id")))
    inputs = read_lines(DATA / "acquisition/extra-megalith.jsonl") + read_lines(
        DATA / "acquisition/extra-vizwiz.jsonl"
    )
    inputs += [
        r for r in read_lines(DATA_ROOT / "pilot-10k-v1/train.jsonl") if r["component"] in ("clevr", "plotqa")
    ]
    inputs = [r for r in inputs if r["id"] not in ids]
    inputs.sort(key=lambda r: stable_key("prepare", r["id"]))
    rows, rejects = [], []
    with ThreadPoolExecutor(workers) as pool:
        for i, row in enumerate(pool.map(preparation.prepare, inputs), 1):
            reason = row.get("error")
            if not reason:
                reason = (
                    "existing/protected/exact duplicate"
                    if (
                        row["image_id"] in ids
                        or row["family_id"] in families
                        or row["source_sha256"] in byte_hashes
                        or row["decoded_sha256"] in pixels
                        or row["view_sha256"] in views
                    )
                    else None
                )
            if not reason:
                nearby = index.nearby(int(row["dhash"], 16), 2)
                if nearby:
                    reason = "conservative perceptual duplicate exclusion"
                    row["nearby_ids"] = nearby
            if reason:
                rejects.append(
                    {"image_id": row["image_id"], "reason": reason, "nearby_ids": row.get("nearby_ids", [])}
                )
            else:
                row["candidate_id"] = "N" + hashlib.sha256(row["image_id"].encode()).hexdigest()[:16]
                row["case_id"] = row["candidate_id"]
                row["historically_fresh"] = True
                row["benchmark_order"] = row["publication_order"] = None
                rows.append(row)
                ids.add(row["image_id"])
                families.add(row["family_id"])
                pixels.add(row["decoded_sha256"])
                byte_hashes.add(row["source_sha256"])
                views.add(row["view_sha256"])
                index.add(int(row["dhash"], 16), row["image_id"])
            if i % 100 == 0:
                print(f"Prepared {i}/{len(inputs)}; accepted={len(rows)} rejected={len(rejects)}", flush=True)
    write_json(
        DATA / "candidate-manifest.json",
        {
            "schema_version": "expansion-40000-candidates-v1",
            "seed": SEED,
            "old_manifest_sha256": digest(OLD / "combined-manifest.json"),
            "counts": dict(Counter(r["category"] for r in rows)),
            "cases": rows,
        },
    )
    write_lines(DATA / "candidate-rejections.jsonl", rejects)
    print(dict(Counter(r["category"] for r in rows)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["acquire", "prepare"])
    parser.add_argument("--source", choices=["megalith", "vizwiz"])
    parser.add_argument("--count", type=int)
    parser.add_argument("--megalith-pool", type=Path, help="Hash-pinned expanded source metadata")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.source, args.count, args.workers, args.megalith_pool)
    else:
        prepare(args.workers)
