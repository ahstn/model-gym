# ruff: noqa: I001
# Imports follow the explicit legacy run-context bootstrap order.
"""Embed an expansion frame, reusing only exact-view pinned vectors."""

from __future__ import annotations
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import corpus
from corpus import DATA, OLD, digest, read_lines, write_json
import corpus_embed as legacy


def main(endpoint):
    rows = json.loads((DATA / "candidate-manifest.json").read_text())["cases"]
    previous = json.loads((OLD / "candidate-manifest.json").read_text())["cases"]
    prior = {}
    for row in previous:
        file = OLD / "embedding-cache" / (row["view_sha256"] + ".npy")
        if file.exists():
            prior[row["view_sha256"]] = np.load(file, allow_pickle=False)
    (DATA / "embedding-operations").mkdir(exist_ok=True)
    overlap = legacy.check_overlap(endpoint, prior, [r for r in previous if r["view_sha256"] in prior])
    for row in read_lines(DATA / "acquisition/extra-megalith.jsonl"):
        legacy.SOURCES[row["source_id"]] = row["download_url"]
    cache = DATA / "embedding-cache"
    cache.mkdir(exist_ok=True)
    missing, reused = [], 0
    for row in rows:
        path = cache / (row["view_sha256"] + ".npy")
        if path.exists():
            value = np.load(path, allow_pickle=False)
            assert value.shape == (768,) and np.isfinite(value).all()
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
            # Source hosts may change or disappear. Retry with verified local bytes,
            # preserving the exact delivered view rather than substituting an image.
            import base64
            import time

            started = time.monotonic()
            items = []
            for row in batch:
                path = corpus.Path(row["image_path"])
                assert digest(path) == row["view_sha256"]
                items.append(
                    {
                        "sha256": row["view_sha256"],
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    }
                )
            response = requests.post(endpoint, json={"images": items}, timeout=(15, 180))
            response.raise_for_status()
            result = response.json()
            assert result["sha256"] == [r["view_sha256"] for r in batch]
            values = np.asarray(result["embeddings"], dtype=np.float32)
            assert values.shape == (len(batch), 768) and np.isfinite(values).all()
            np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)
            return values, {
                "wall_seconds": time.monotonic() - started,
                "server_seconds": result["seconds"],
                "local_bytes_fallback": True,
            }

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
            "candidate_manifest_sha256": digest(DATA / "candidate-manifest.json"),
            "previous_provenance_sha256": digest(OLD / "embedding-operations/provenance.json"),
            "cache_count": len(list(cache.glob("*.npy"))),
            "script_sha256": digest(__file__),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18766")
    args = parser.parse_args()
    main(args.endpoint)
