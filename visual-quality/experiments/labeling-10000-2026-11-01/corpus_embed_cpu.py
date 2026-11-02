"""Pinned CPU fallback for the benchmark-critical image embeddings."""

import argparse
import importlib.metadata
import json
import os
import time

from corpus_common import DATA, DATA_ROOT, PREVIOUS, SEED, digest, stable_key, write_json

CACHE = DATA_ROOT / "embeddinggemma2-clustering-2026-10-31/.hf_cache"
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


def main(shard):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    candidates = json.loads((DATA / "candidate-manifest.json").read_text())["cases"]
    chosen = []
    for category, count in [("megalith", 1400), ("vizwiz", 400)]:
        pool = [r for r in candidates if r.get("fresh_for_expansion_benchmark") and r["category"] == category]
        chosen.extend(sorted(pool, key=lambda r: stable_key("fresh-benchmark", r["image_id"]))[:count])
    chosen = chosen[shard::2]
    output = DATA / "embedding-cache"
    output.mkdir(exist_ok=True)
    software = {
        n: importlib.metadata.version(n)
        for n in ["torch", "torchvision", "transformers", "sentence-transformers", "numpy", "Pillow"]
    }
    prior = json.loads((PREVIOUS / "embedding-shard-0.json").read_text())
    assert software == prior["software"]
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        revision="f9972ac22e696a105c482202646979e33e4b5a1e",
        device="cpu",
        cache_folder=str(CACHE / "hub"),
        config_kwargs={"audio_config": None},
        model_kwargs={"dtype": torch.float32},
    )
    model.eval()
    metadata = {
        "software": software,
        "device": "cpu",
        "dtype": "float32",
        "shard": shard,
        "shards": 2,
        "checkpoint_revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
        "max_soft_tokens": 280,
        "audio_config": None,
        "dimensions": 768,
        "threads": 4,
        "batch_size": 4,
        "normalization": "L2",
        "script_sha256": digest(__file__),
        "batches": [],
        "status": "running",
    }
    print(f"CPU shard{shard} ready for {len(chosen)} benchmark-priority views", flush=True)
    for start in range(0, len(chosen), 4):
        batch = [r for r in chosen[start : start + 4] if not (output / f"{r['view_sha256']}.npy").exists()]
        if not batch:
            continue
        images = []
        for row in batch:
            assert digest(row["image_path"]) == row["view_sha256"]
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
        for row, value in zip(batch, values, strict=True):
            target = output / f"{row['view_sha256']}.npy"
            if not target.exists():
                temporary = target.with_suffix(f".cpu{shard}.tmp.npy")
                np.save(temporary, value, allow_pickle=False)
                temporary.replace(target)
        metadata["batches"].append(
            {"views": [r["view_sha256"] for r in batch], "seconds": time.monotonic() - started}
        )
        write_json(DATA / f"embedding-operations/cpu-shard-{shard}.json", metadata)
        if start % 40 == 0:
            print(f"CPU shard{shard} traversed{start + len(batch)}/{len(chosen)}", flush=True)
    metadata["status"] = "complete"
    write_json(DATA / f"embedding-operations/cpu-shard-{shard}.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    main(parser.parse_args().shard)
