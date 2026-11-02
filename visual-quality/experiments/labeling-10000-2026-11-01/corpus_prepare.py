"""Prepare incremental, hash-pinned candidates; all full records remain outside Git."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus_common import DATA, DATA_ROOT, PREVIOUS, digest, read_lines, stable_key, write_json
from PIL import Image

from visual_quality.validation import inspect_image
from visual_quality.views import _render


def prepare(row: dict) -> dict:
    cache = DATA / "prepared" / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")
    if cache.exists():
        saved = json.loads(cache.read_text())
        if not saved.get("error") and Path(saved["image_path"]).exists():
            if (
                digest(saved["image_path"]) != saved["view_sha256"]
                or digest(saved["source_path"]) != saved["source_sha256"]
            ):
                raise ValueError("Prepared cache source or view bytes changed")
            return saved
        if saved.get("error"):
            return saved
    try:
        verified = inspect_image(row, DATA_ROOT)
        path = DATA_ROOT / verified["image_path"]
        payload = path.read_bytes()
        edge = 1536
        while True:
            data, preprocessing = _render(payload, edge)
            if len(data) <= 2650000:
                break
            edge = edge * 9 // 10
            if edge < 256:
                raise ValueError("Cannot satisfy common PNG request budget")
        width, height = preprocessing["output_dimensions"]
        if min(width, height) < 256 or width * height < 78400:
            raise ValueError("Outside common model image dimension limits")
        view_sha = hashlib.sha256(data).hexdigest()
        view = DATA / "views" / f"{view_sha}.png"
        view.parent.mkdir(parents=True, exist_ok=True)
        if not view.exists():
            view.write_bytes(data)
        with Image.open(path) as image:
            source_metadata = {
                "format": image.format,
                "frames": getattr(image, "n_frames", 1),
                "icc": bool(image.info.get("icc_profile")),
                "alpha": "A" in image.getbands() or "transparency" in image.info,
            }
        category = "controls" if row["component"] in ("clevr", "plotqa") else row["component"]
        votes = row.get("source_labels", {}).get("human_flaw_votes", {})
        result = {
            "image_id": row["id"],
            "source": row["source"],
            "source_id": row["source_id"],
            "family_id": row.get("split_group", row.get("source_group", row["id"])),
            "source_group": row.get("source_group", row["id"]),
            "source_revision": row["source_revision"],
            "source_url": row.get("source_url"),
            "license": row["license"],
            "source_path": str(path),
            "source_sha256": verified["sha256"],
            "image_path": str(view),
            "view_sha256": view_sha,
            "decoded_sha256": verified["decoded_sha256"],
            "dhash": verified["dhash"],
            "width": width,
            "height": height,
            "delivered_bytes": len(data),
            "preprocessing": preprocessing,
            "target_edge": edge,
            "mime_type": "image/png",
            "source_metadata": source_metadata,
            "source_format": source_metadata["format"],
            "component": row["component"],
            "category": category,
            "stratum": category,
            "split": "train",
            "split_group": row.get("split_group", row.get("source_group", row["id"])),
            "known_nonphoto_source": category == "controls",
            "controlled_parent": False,
            "operator": None,
            "severity": None,
            "role": "standalone",
            "vizwiz_stratum": min(votes, key=lambda key: (-votes[key], key)) if votes else None,
            "benchmark_partition": "rehearsal",
            "task_masks": {"technical_quality": False, "aesthetics": False, "photo_domain": False},
        }
    except (OSError, ValueError) as error:
        result = {"image_id": row["id"], "error": f"{type(error).__name__}: {error}"}
    write_json(cache, result)
    return result


def main(workers: int):
    old = json.loads((PREVIOUS / "candidate-manifest.json").read_text())["cases"]
    old_ids = {r["image_id"] for r in old}
    protected = read_lines(DATA_ROOT / "pilot-10k-v1/development.jsonl") + read_lines(
        DATA_ROOT / "pilot-10k-v1/test.jsonl"
    )
    protected_ids = {r["id"] for r in protected}
    protected_families = {r["split_group"] for r in protected}
    protected_pixels = {r["decoded_sha256"] for r in protected}
    protected_bytes = {r["sha256"] for r in protected}
    protected_hashes = {int(r["dhash"], 16) for r in protected if r.get("dhash")}
    train = read_lines(DATA_ROOT / "pilot-10k-v1/train.jsonl")
    extra = read_lines(DATA / "acquisition/extra-megalith.jsonl") + read_lines(
        DATA / "acquisition/extra-vizwiz.jsonl"
    )
    new_ids = {r["id"] for r in extra}
    inputs = [r for r in train if r["component"] in ("megalith", "vizwiz", "clevr", "plotqa")]
    inputs += extra
    inputs = [r for r in inputs if r["id"] not in old_ids | protected_ids]
    inputs.sort(key=lambda r: (r["id"] not in new_ids, stable_key("render", r["id"])))
    candidates, rejects = [], []
    seen_ids = set()
    for row in old:
        saved = dict(row)
        saved["original_candidate_id"] = saved["candidate_id"]
        saved["candidate_id"] = "C" + hashlib.sha256(saved["image_id"].encode()).hexdigest()[:16]
        saved["benchmark_candidate"] = False
        saved["fresh_for_expansion_benchmark"] = False
        candidates.append(saved)
        seen_ids.add(saved["image_id"])
    with ThreadPoolExecutor(workers) as executor:
        for i, row in enumerate(executor.map(prepare, inputs), 1):
            if row.get("error"):
                rejects.append(row)
                continue
            blocked = (
                row["image_id"] in seen_ids
                or row["family_id"] in protected_families
                or row["source_sha256"] in protected_bytes
                or row["decoded_sha256"] in protected_pixels
            )
            d = int(row["dhash"], 16)
            if blocked or any((d ^ h).bit_count() <= 2 for h in protected_hashes):
                rejects.append(
                    {
                        "image_id": row["image_id"],
                        "error": "protected or duplicate identity/perceptual candidate",
                    }
                )
                continue
            row = dict(row)
            row["candidate_id"] = "C" + hashlib.sha256(row["image_id"].encode()).hexdigest()[:16]
            row["case_id"] = row["candidate_id"]
            row["original_candidate_id"] = None
            row["fresh_for_expansion_benchmark"] = row["image_id"] in new_ids
            row["historically_fresh"] = row["fresh_for_expansion_benchmark"]
            row["benchmark_candidate"] = row["fresh_for_expansion_benchmark"]
            candidates.append(row)
            seen_ids.add(row["image_id"])
            if i % 100 == 0:
                print(f"Prepared {i}/{len(inputs)}; candidates={len(candidates)}", flush=True)
    # Existing candidates retain their original hash-pinned records and embeddings.
    candidates.sort(
        key=lambda r: (not r["fresh_for_expansion_benchmark"], stable_key("candidate", r["image_id"]))
    )
    write_json(
        DATA / "candidate-manifest.json",
        {
            "schema_version": 1,
            "cases": candidates,
            "counts": dict(Counter(r["category"] for r in candidates)),
            "fresh_counts": dict(
                Counter(r["category"] for r in candidates if r["fresh_for_expansion_benchmark"])
            ),
            "inputs": {
                "prior_candidates_sha256": digest(PREVIOUS / "candidate-manifest.json"),
                "prepare_script_sha256": digest(__file__),
            },
        },
    )
    write_json(
        DATA / "candidate-preparation-audit.json",
        {
            "rejections": rejects,
            "counts": dict(Counter(r["category"] for r in candidates)),
            "protected_families": len(protected_families),
            "candidate_count": len(candidates),
        },
    )
    print(f"Prepared snapshot {len(candidates)} candidates; rejected {len(rejects)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args().workers)
