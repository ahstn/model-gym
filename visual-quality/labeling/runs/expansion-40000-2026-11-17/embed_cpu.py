"""Offline pinned CPU fallback with a 64-view compatibility check before reuse."""

# Imports configure offline execution before loading ML libraries.
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

# The pinned ML environment intentionally omits HTTP dependencies used by the
# shared corpus adapters. Fall back to the existing labeling environment while
# retaining this environment's torch/transformers/numpy versions first.
sys.path.append(
    str(
        Path("/home/ahstn/git/model-gym/visual-quality/.venv/lib")
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
)

import corpus
from corpus import DATA, DATA_ROOT, INITIAL, OLD, SEED, digest, stable_key, write_json
from embed import previous_vectors, validate_vector

CACHE = DATA_ROOT / "embeddinggemma2-clustering-2026-10-31/.hf_cache"
REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
WEIGHTS_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
os.environ.update(
    HF_HOME=str(CACHE),
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
    TOKENIZERS_PARALLELISM="false",
    HF_HUB_DISABLE_PROGRESS_BARS="1",
)

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


def load_model(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    path = CACHE / "hub/models--google--embeddinggemma-2/snapshots" / REVISION
    if digest(path / "model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("EmbeddingGemma2 pinned weights changed")
    model = SentenceTransformer(
        str(path),
        device="cpu",
        config_kwargs={"audio_config": None},
        model_kwargs={"dtype": torch.float32},
    )
    model.eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 438760448:
        raise ValueError(f"Pinned vision/text parameter count changed: {parameters}")
    identity = {
        "model": "google/embeddinggemma-2",
        "revision": REVISION,
        "dtype": "float32",
        "device": "cpu",
        "dimensions": 768,
        "audio_config": None,
        "max_soft_tokens": 280,
        "normalization": "L2",
        "tf32": False,
        "parameters": parameters,
        "weights_sha256": WEIGHTS_SHA256,
        "threads": threads,
        "batch_size": 4,
        "script_sha256": digest(__file__),
        "software": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "transformers", "sentence-transformers", "numpy", "Pillow")
        },
    }
    print(json.dumps({"worker_identity": identity}), flush=True)
    return model, identity


def encode(model, rows):
    images = []
    for row in rows:
        if digest(row["image_path"]) != row["view_sha256"]:
            raise ValueError("Local delivered view changed")
        with Image.open(row["image_path"]) as image:
            images.append({"image": image.convert("RGB")})
    started = time.monotonic()
    with torch.inference_mode():
        values = model.encode(
            images,
            batch_size=4,
            processing_kwargs={"image": {"max_soft_tokens": 280}},
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        values = torch.nn.functional.normalize(values.float(), p=2, dim=-1).cpu().numpy()
    if values.shape != (len(rows), 768):
        raise ValueError("Invalid embedding response dimensions")
    for value in values:
        validate_vector(value)
    return values, time.monotonic() - started


def check_overlap(model, identity, prior, rows):
    path = DATA / "embedding-operations/cpu-overlap.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["worker_identity"] == identity and saved["passed"] and saved["compared"] == 64:
            return saved
    chosen = sorted(rows, key=lambda row: stable_key("cpu-gpu-overlap", row["image_id"]))[:64]
    if len(chosen) != 64:
        raise ValueError("CPU compatibility requires 64 prior exact-view vectors")
    comparisons, batches = [], []
    started = time.monotonic()
    for start in range(0, len(chosen), 4):
        batch = chosen[start : start + 4]
        values, seconds = encode(model, batch)
        batches.append({"count": len(batch), "seconds": seconds})
        for row, value in zip(batch, values, strict=True):
            previous = prior[row["view_sha256"]]
            comparisons.append(
                {
                    "view_sha256": row["view_sha256"],
                    "cosine": float(previous @ value),
                    "max_absolute_difference": float(np.abs(previous - value).max()),
                }
            )
        print(f"CPU overlap {start + len(batch)}/64; batch {seconds:.2f}s", flush=True)
    minimum = min(row["cosine"] for row in comparisons)
    maximum = max(row["max_absolute_difference"] for row in comparisons)
    elapsed = time.monotonic() - started
    result = {
        "worker_identity": identity,
        "compared": 64,
        "rows": comparisons,
        "batches": batches,
        "minimum_cosine": minimum,
        "maximum_coordinate_difference": maximum,
        "predeclared_acceptance": {"minimum_cosine": 0.9999, "maximum_coordinate_difference": 0.001},
        "passed": minimum >= 0.9999 and maximum <= 0.001,
        "wall_seconds": elapsed,
        "seconds_per_image": elapsed / 64,
        "purpose": "Verify pinned CPU inference against prior exact-view vectors before mixing caches",
    }
    write_json(path, result)
    if not result["passed"]:
        raise ValueError("CPU numerical drift exceeds acceptance; cache unchanged")
    return result


def main(threads=8, benchmark_only=False):
    corpus.pin_initial_evidence()
    model, identity = load_model(threads)
    prior, previous = previous_vectors()
    overlap = check_overlap(model, identity, prior, previous)
    if benchmark_only:
        print(
            json.dumps(
                {
                    "benchmark_passed": True,
                    "seconds_per_image": overlap["seconds_per_image"],
                    "projected_20000_hours": overlap["seconds_per_image"] * 20000 / 3600,
                }
            ),
            flush=True,
        )
        return
    frame_bytes = (DATA / "candidate-manifest.json").read_bytes()
    frame_sha = hashlib.sha256(frame_bytes).hexdigest()
    rows = json.loads(frame_bytes)["cases"]
    snapshot = DATA / "embedding-operations/frames" / (frame_sha + ".json")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(frame_bytes)
    cache = DATA / "embedding-cache"
    cache.mkdir(exist_ok=True)
    missing, reused = [], 0
    for row in rows:
        file = cache / (row["view_sha256"] + ".npy")
        if file.exists():
            validate_vector(np.load(file, allow_pickle=False))
        elif row["view_sha256"] in prior:
            np.save(file, prior[row["view_sha256"]], allow_pickle=False)
            reused += 1
        else:
            missing.append(row)
    print(f"CPU exact-view reuse={reused}; need embeddings={len(missing)}", flush=True)
    receipt = DATA / "embedding-operations/cpu-batches.jsonl"
    for start in range(0, len(missing), 4):
        batch = [
            row for row in missing[start : start + 4] if not (cache / (row["view_sha256"] + ".npy")).exists()
        ]
        if not batch:
            continue
        values, seconds = encode(model, batch)
        for row, value in zip(batch, values, strict=True):
            file = cache / (row["view_sha256"] + ".npy")
            temporary = file.with_suffix(f".cpu-{os.getpid()}.tmp.npy")
            np.save(temporary, value, allow_pickle=False)
            temporary.replace(file)
        with receipt.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "views": [row["view_sha256"] for row in batch],
                        "seconds": seconds,
                        "candidate_manifest_sha256": frame_sha,
                        "worker_identity_sha256": hashlib.sha256(
                            json.dumps(identity, sort_keys=True).encode()
                        ).hexdigest(),
                    }
                )
                + "\n"
            )
        if start % 40 == 0:
            print(f"CPU embedded {start + len(batch)}/{len(missing)}", flush=True)
    write_json(
        DATA / "embedding-operations/provenance.json",
        {
            "worker_identity": identity,
            "overlap_sha256": digest(DATA / "embedding-operations/cpu-overlap.json"),
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
            "transport": "local_cpu",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--benchmark-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    main(arguments.threads, arguments.benchmark_only)
