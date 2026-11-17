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
DATA = DATA_ROOT / "labeling-20000-2026-11-16"
OLD = DATA_ROOT / "labeling-10000-2026-11-01"
SEED = 2026111620000
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
    return json.loads((OLD / "manifest.json").read_text())["cases"]


def acquire(source, count, workers):
    excluded = {r["source_id"] for r in old_cases()}
    output = DATA / "acquisition" / f"extra-{source}.jsonl"
    cached = read_lines(output)
    seen = {r["source_id"] for r in cached}
    for path in [DATA_ROOT / f"acquisition/{source}.jsonl", OLD / f"acquisition/extra-{source}.jsonl"]:
        for row in read_lines(path):
            if row["source_id"] not in excluded | seen:
                cached.append(row)
                seen.add(row["source_id"])
    write_lines(output, cached)
    factory_name = source + "_candidates"
    original = getattr(acquisition, factory_name)

    def candidates():
        pool, fetch, evidence = original()
        key = "id" if source == "megalith" else "image"
        return [r for r in pool if r[key] not in excluded], fetch, evidence

    setattr(acquisition, factory_name, candidates)
    acquisition.acquire(source, count, workers)


def prepare(workers):
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
            "schema_version": "expansion-20000-candidates-v1",
            "seed": SEED,
            "old_manifest_sha256": digest(OLD / "manifest.json"),
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
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.source, args.count, args.workers)
    else:
        prepare(args.workers)
