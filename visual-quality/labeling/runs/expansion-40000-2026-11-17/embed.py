# ruff: noqa: I001
# Imports follow the explicit legacy run-context bootstrap order.
"""Embed an expansion frame, reusing only exact-view pinned vectors."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import corpus
from corpus import DATA, OLD, INITIAL, digest, read_lines, write_json
import corpus_embed as legacy


def validate_vector(value):
    if value.shape != (768,) or not np.isfinite(value).all():
        raise ValueError("Invalid embedding dimensions or nonfinite values")
    np.testing.assert_allclose(np.linalg.norm(value), 1, atol=1e-5)
    return value


def previous_vectors():
    """Reuse only exact-view vectors; never key a variant by its parent view."""
    prior, records = {}, {}
    for root in (INITIAL, OLD):
        previous = json.loads((root / "candidate-manifest.json").read_text())["cases"]
        for row in previous:
            sha = row["view_sha256"]
            file = root / "embedding-cache" / (sha + ".npy")
            if file.exists():
                value = validate_vector(np.load(file, allow_pickle=False))
                if sha in prior:
                    np.testing.assert_allclose(prior[sha], value, atol=1e-5)
                prior[sha], records[sha] = value, row
        # Published corpora retain exact vectors even when candidate caches were pruned.
        manifest_name = "combined-manifest.json" if root == OLD else "manifest.json"
        rows = json.loads((root / manifest_name).read_text())["cases"]
        with np.load(root / "corpus-embeddings.npz", allow_pickle=False) as archive:
            if archive["case_ids"].tolist() != [row["case_id"] for row in rows]:
                raise ValueError("Historical embedding order differs from manifest")
            for row, value, sha in zip(
                rows, archive["embeddings"], archive["embedding_view_sha256"], strict=True
            ):
                if sha != row["view_sha256"]:
                    continue
                if sha not in prior:
                    prior[sha], records[sha] = validate_vector(value.copy()), row
    return prior, list(records.values())


def encode_local(batch, endpoint):
    import base64
    import time

    started = time.monotonic()
    items = []
    for row in batch:
        path = corpus.Path(row["image_path"])
        if digest(path) != row["view_sha256"]:
            raise ValueError("Local delivered view changed")
        items.append(
            {"sha256": row["view_sha256"], "data": base64.b64encode(path.read_bytes()).decode("ascii")}
        )
    response = requests.post(endpoint, json={"images": items}, timeout=(15, 180))
    response.raise_for_status()
    result = response.json()
    if result["sha256"] != [row["view_sha256"] for row in batch]:
        raise ValueError("Embedding response view identities differ")
    values = np.asarray(result["embeddings"], dtype=np.float32)
    if values.shape != (len(batch), 768):
        raise ValueError("Invalid embedding response dimensions")
    for value in values:
        validate_vector(value)
    return values, {
        "wall_seconds": time.monotonic() - started,
        "server_seconds": result["seconds"],
        "verified_local_bytes": True,
    }


def main(endpoint):
    corpus.pin_initial_evidence()
    frame_bytes = (DATA / "candidate-manifest.json").read_bytes()
    frame_sha = hashlib.sha256(frame_bytes).hexdigest()
    rows = json.loads(frame_bytes)["cases"]
    snapshot = DATA / "embedding-operations/frames" / (frame_sha + ".json")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and snapshot.read_bytes() != frame_bytes:
        raise ValueError("Candidate snapshot changed")
    snapshot.write_bytes(frame_bytes)
    prior, previous = previous_vectors()
    (DATA / "embedding-operations").mkdir(exist_ok=True)
    for root in (INITIAL, OLD, DATA):
        for row in read_lines(root / "acquisition/extra-megalith.jsonl"):
            legacy.SOURCES[row["source_id"]] = row["download_url"]
    # Verify transported local bytes against old exact-view embeddings, independent
    # of whether the original public image host still serves identical bytes.
    original_encode = legacy.encode
    legacy.encode = encode_local
    try:
        overlap = legacy.check_overlap(endpoint, prior, previous)
    finally:
        legacy.encode = original_encode
    cache = DATA / "embedding-cache"
    cache.mkdir(exist_ok=True)
    missing, reused = [], 0
    for row in rows:
        path = cache / (row["view_sha256"] + ".npy")
        if path.exists():
            value = np.load(path, allow_pickle=False)
            validate_vector(value)
            continue
        if row["view_sha256"] in prior:
            np.save(path, prior[row["view_sha256"]], allow_pickle=False)
            reused += 1
        else:
            missing.append(row)
    print(f"Exact-view reuse={reused}; need embeddings={len(missing)}", flush=True)
    batches = [missing[i : i + 4] for i in range(0, len(missing), 4)]

    def encode(batch):
        try:
            return legacy.encode(batch, endpoint)
        except (requests.RequestException, ValueError, AssertionError, OSError):
            return encode_local(batch, endpoint)

    with ThreadPoolExecutor(4) as pool:
        for i, (batch, (values, timing)) in enumerate(
            zip(batches, pool.map(encode, batches), strict=True), 1
        ):
            for row, vector in zip(batch, values, strict=True):
                target = cache / (row["view_sha256"] + ".npy")
                temporary = target.with_suffix(f".{os.getpid()}.tmp.npy")
                np.save(temporary, vector, allow_pickle=False)
                temporary.replace(target)
            with (DATA / "embedding-operations/batches.jsonl").open("a") as stream:
                stream.write(json.dumps({"views": [r["view_sha256"] for r in batch], **timing}) + "\n")
            if i % 10 == 0:
                print(f"Embedded {i * 4}/{len(missing)}", flush=True)
    write_json(
        DATA / "embedding-operations/provenance.json",
        {
            "worker_identity": overlap["worker_identity"],
            "overlap_sha256": digest(DATA / "embedding-operations/cpu-gpu-overlap.json"),
            "candidate_manifest_sha256": frame_sha,
            "candidate_snapshot": str(snapshot),
            "candidate_count": len(rows),
            "previous_provenance_sha256": {
                str(root): digest(root / "embedding-operations/provenance.json") for root in (INITIAL, OLD)
            },
            "exact_view_reused_this_execution": reused,
            "prior_exact_view_cache_count": len(prior),
            "cache_count": len(list(cache.glob("*.npy"))),
            "script_sha256": digest(__file__),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18766")
    args = parser.parse_args()
    main(args.endpoint)
