"""Embed prepared views through the pinned GPU worker, after CPU/GPU overlap checks."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from corpus_common import DATA, DATA_ROOT, PREVIOUS, digest, read_lines, stable_key, write_json

SOURCES = {}
for source_file in [DATA_ROOT / "acquisition/megalith.jsonl", DATA / "acquisition/extra-megalith.jsonl"]:
    for record in read_lines(source_file):
        SOURCES[record["source_id"]] = record["download_url"]
VIZ_INDEX = json.loads((DATA_ROOT / "sources/vizwiz/archive-index.json").read_text())


def remote_source(row):
    if row["source"] == "Spawning/megalith-cc0" and row["source_id"] in SOURCES:
        return {
            "url": SOURCES[row["source_id"]],
            "sha256": row["source_sha256"],
            "target_edge": row["target_edge"],
        }
    if row["source"] == "vizwiz-quality-issues":
        return {
            "kind": "vizwiz",
            "url": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
            "sha256": row["source_sha256"],
            "target_edge": row["target_edge"],
            "member": VIZ_INDEX["members"][row["source_id"]],
            "etag": VIZ_INDEX["etag"],
            "archive_size": 11298421598,
        }
    return None


def old_vectors():
    records = json.loads((PREVIOUS / "candidate-manifest.json").read_text())["cases"]
    vectors = np.load(PREVIOUS / "candidate-embeddings.npz", allow_pickle=False)
    by_id = dict(zip(vectors["case_ids"].tolist(), vectors["embeddings"], strict=True))
    return {row["view_sha256"]: by_id[row["candidate_id"]] for row in records}, records


def encode(rows, endpoint):
    images = []
    for row in rows:
        if digest(row["image_path"]) != row["view_sha256"]:
            raise ValueError("Prepared view changed")
        source = remote_source(row)
        if source:
            images.append({"sha256": row["view_sha256"], "source": source})
        else:
            payload = Path(row["image_path"]).read_bytes()
            images.append({"sha256": row["view_sha256"], "data": base64.b64encode(payload).decode("ascii")})
    started = time.monotonic()
    response = requests.post(endpoint, json={"images": images}, timeout=(15, 180))
    if not response.ok:
        print(response.text[:1000], flush=True)
        response.raise_for_status()
    result = response.json()
    if result["sha256"] != [r["view_sha256"] for r in rows]:
        raise ValueError("Response image identities differ")
    values = np.asarray(result["embeddings"], dtype=np.float32)
    assert values.shape == (len(rows), 768) and np.isfinite(values).all()
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)
    return values, {"wall_seconds": time.monotonic() - started, "server_seconds": result["seconds"]}


def check_overlap(endpoint, prior, rows):
    path = DATA / "embedding-operations/cpu-gpu-overlap.json"
    identity = requests.get(endpoint, timeout=30).json()
    expected = {
        "model": "google/embeddinggemma-2",
        "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
        "dtype": "float32",
        "dimensions": 768,
        "max_soft_tokens": 280,
        "audio_config": None,
        "tf32": False,
        "parameters": 438760448,
        "weights_sha256": "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79",
    }
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Embedding worker identity differs from pinned contract: {identity}")
    if path.exists():
        saved = json.loads(path.read_text())
        old_identity = {k: v for k, v in saved["worker_identity"].items() if k != "server_sha256"}
        new_identity = {k: v for k, v in identity.items() if k != "server_sha256"}
        if old_identity != new_identity or not saved["passed"]:
            raise ValueError("Worker changed since overlap validation")
        if saved["worker_identity"] != identity:
            saved.setdefault("transport_revisions", []).append(identity["server_sha256"])
            write_json(path, saved)
        return saved
    chosen = sorted(rows, key=lambda r: stable_key("cpu-gpu-overlap", r["image_id"]))[:64]
    comparisons = []
    for start in range(0, len(chosen), 4):
        batch = chosen[start : start + 4]
        vectors, _timing = encode(batch, endpoint)
        for row, gpu in zip(batch, vectors, strict=True):
            cpu = prior[row["view_sha256"]]
            comparisons.append(
                {
                    "view_sha256": row["view_sha256"],
                    "cosine": float(cpu @ gpu),
                    "max_absolute_difference": float(np.abs(cpu - gpu).max()),
                }
            )
    minimum = min(r["cosine"] for r in comparisons)
    maximum = max(r["max_absolute_difference"] for r in comparisons)
    result = {
        "worker_identity": identity,
        "compared": len(comparisons),
        "rows": comparisons,
        "minimum_cosine": minimum,
        "maximum_coordinate_difference": maximum,
        "predeclared_acceptance": {"minimum_cosine": 0.9999, "maximum_coordinate_difference": 0.001},
        "passed": minimum >= 0.9999 and maximum <= 0.001,
        "purpose": "Check numerical compatibility before reusing prior CPU vectors alongside GPU embeddings",
    }
    write_json(path, result)
    if not result["passed"]:
        raise ValueError("CPU/GPU drift exceeds acceptance; do not mix embeddings")
    return result


def main(endpoint, limit, frame=None):
    prior, old = old_vectors()
    metadata = check_overlap(endpoint, prior, old)
    cases = json.loads((DATA / "candidate-manifest.json").read_text())["cases"]
    if frame:
        cases = json.loads(Path(frame).read_text())["cases"]
    output = DATA / "embedding-cache"
    output.mkdir(exist_ok=True)
    ledger = DATA / "embedding-operations/batches.jsonl"
    missing = []
    for row in cases:
        target = output / f"{row['view_sha256']}.npy"
        if target.exists():
            continue
        if row["view_sha256"] in prior:
            np.save(target, prior[row["view_sha256"]], allow_pickle=False)
        else:
            missing.append(row)
    # Benchmark-photo candidates first; fault photos are already plentiful in acquisition.
    missing.sort(
        key=lambda r: (
            not r.get("fresh_for_expansion_benchmark"),
            r["category"] != "megalith",
            stable_key("embed", r["image_id"]),
        )
    )
    if limit:
        missing = missing[:limit]
    print(
        f"Embedding {len(missing)} missing views; reusing prior CPU vectors after drift validation",
        flush=True,
    )
    batches = [missing[start : start + 4] for start in range(0, len(missing), 4)]
    with ThreadPoolExecutor(4) as pool:
        for i, (batch, (values, timing)) in enumerate(
            zip(batches, pool.map(lambda batch: encode(batch, endpoint), batches), strict=True)
        ):
            for row, vector in zip(batch, values, strict=True):
                target = output / f"{row['view_sha256']}.npy"
                temporary = target.with_suffix(f".{os.getpid()}.tmp.npy")
                np.save(temporary, vector, allow_pickle=False)
                temporary.replace(target)
            with ledger.open("a") as stream:
                stream.write(json.dumps({"views": [r["view_sha256"] for r in batch], **timing}) + "\n")
            if i % 10 == 0:
                print(f"Embedded {i * 4 + len(batch)}/{len(missing)} new views", flush=True)
    write_json(
        DATA / "embedding-operations/provenance.json",
        {
            "worker_identity": metadata["worker_identity"],
            "overlap_sha256": digest(DATA / "embedding-operations/cpu-gpu-overlap.json"),
            "prior_vector_path": str(PREVIOUS / "candidate-embeddings.npz"),
            "prior_vector_sha256": digest(PREVIOUS / "candidate-embeddings.npz"),
            "script_sha256": digest(__file__),
            "cache_count": len(list(output.glob("*.npy"))),
            "candidate_manifest_sha256": digest(DATA / "candidate-manifest.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8921")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frame", type=Path)
    args = parser.parse_args()
    main(args.endpoint, args.limit, args.frame)
