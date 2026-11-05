"""Acquire caption-matched Synthbuster photo pairs for the AI detection track."""

import argparse
import hashlib
import io
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
TRACK = Path(__file__).resolve().parent
DATA = ROOT / "data/tracks/ai_detection"
CACHE = DATA / "source-assets"
MANIFEST = TRACK / "dataset.json"
REPO = "marco-willi/synthbuster-plus"
REVISION = "dbfb72f1ee96e953ee5cff80c58832fe89e1d2b5"
GENERATORS = ("dalle3", "midjourney-v5", "stable-diffusion-xl")
SEED = "20261011-ai-matched-v1"
PRIOR = ("evidence/samples.json", "evidence/benchmarks/quality.json", "evidence/benchmarks/ai.json")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def get(url, **kwargs):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=120, **kwargs)
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def page(offset):
    target = CACHE / f"catalogue-test-{offset:04d}.json"
    if not target.exists():
        result = get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": REPO, "config": "default", "split": "test", "offset": offset, "length": 100},
        ).json()
        target.write_text(json.dumps(result) + "\n")
    result = json.loads(target.read_text())
    for item in result["rows"]:
        if f"/--/{REVISION}/--/" not in item["row"]["image"]["src"]:
            raise ValueError("Dataset Viewer revision changed")
    return result["rows"]


def acquire(item):
    row = item["row"]
    target = CACHE / f"{row['source']}-{row['image_id']}.bin"
    if not target.exists():
        response = requests.get(row["image"]["src"], timeout=120)
        if response.status_code == 403:
            fresh = get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": REPO,
                    "config": "default",
                    "split": "test",
                    "offset": item["row_idx"],
                    "length": 1,
                },
            ).json()["rows"][0]["row"]
            if (fresh["source"], fresh["image_id"]) != (row["source"], row["image_id"]):
                raise ValueError("Refreshed source row changed")
            if f"/--/{REVISION}/--/" not in fresh["image"]["src"]:
                raise ValueError("Refreshed source revision changed")
            response = get(fresh["image"]["src"])
        response.raise_for_status()
        target.write_bytes(response.content)
    return target.read_bytes()


def deliver(raw):
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    original_size = image.size
    image = ImageOps.fit(image, (512, 512), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=92, subsampling=0, optimize=True)
    encoded = stream.getvalue()
    return encoded, list(original_size), list(image.size)


def pixel_hash(raw):
    with Image.open(io.BytesIO(raw)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return sha(str(rgb.size).encode() + rgb.tobytes())


def prior_references():
    hashes, pixels, ids = set(), set(), set()
    for name in PRIOR:
        for sample in json.loads((ROOT / name).read_text())["samples"]:
            hashes.update(sample[key] for key in ("original_sha256", "delivered_sha256") if key in sample)
            pixels.add(pixel_hash((ROOT / sample["image"]).read_bytes()))
            source = sample["source"]
            ids.update(str(source[key]) for key in ("id", "row_id", "image_id", "group_id") if key in source)
    return hashes, pixels, ids


def check(document):
    eligibility_raw = (TRACK / "eligibility.json").read_bytes()
    if document["eligibility_sha256"] != sha(eligibility_raw):
        raise ValueError("Frozen medium-eligibility evidence changed")
    eligible = {
        (entry["generator"], entry["image_id"])
        for entry in json.loads(eligibility_raw)["decisions"]
        if entry["eligible"]
    }
    prior_hashes, prior_pixels, prior_ids = prior_references()
    groups = defaultdict(list)
    seen_hashes, seen_pixels = set(), set()
    for sample in document["samples"]:
        raw = (ROOT / sample["image"]).read_bytes()
        pixel = pixel_hash(raw)
        if sha(raw) != sample["delivered_sha256"] or pixel != sample["decoded_rgb_sha256"]:
            raise ValueError(f"Delivered hash mismatch: {sample['id']}")
        if sample["source"]["image_id"] in prior_ids:
            raise ValueError("Prior source ID reused")
        if sha(raw) in prior_hashes | seen_hashes or pixel in prior_pixels | seen_pixels:
            raise ValueError("Exact encoded or decoded duplicate found")
        if sample["original_sha256"] in prior_hashes:
            raise ValueError("Prior acquired image reused")
        with Image.open(io.BytesIO(raw)) as delivered:
            if delivered.size != (512, 512) or delivered.mode != "RGB":
                raise ValueError("Delivered view must be RGB512x512 for both classes")
        if (
            sample["labels"]["ai_generated"]
            and (sample["source"]["generator"], sample["source"]["image_id"]) not in eligible
        ):
            raise ValueError("Selected pair lacks photographic-medium eligibility")
        groups[sample["group_id"]].append(sample)
        seen_hashes.add(sha(raw))
        seen_pixels.add(pixel)
    if len(groups) != 60 or len(document["samples"]) != 120:
        raise ValueError("Expected 60 unique matched pairs")
    counts = Counter()
    for pair in groups.values():
        if len(pair) != 2 or {x["labels"]["ai_generated"] for x in pair} != {False, True}:
            raise ValueError("Each source group must contain one real and one generated image")
        if len({x["split"] for x in pair}) != 1:
            raise ValueError("Source group crosses a split")
        if len({x["source"]["image_id"] for x in pair}) != 1:
            raise ValueError("Pair members must share the same published source image ID")
        fake = next(x for x in pair if x["labels"]["ai_generated"])
        counts[(fake["source"]["generator"], fake["split"])] += 1
    expected = {
        (generator, split): count
        for generator in GENERATORS
        for split, count in (("train", 5), ("dev", 5), ("test", 10))
    }
    if dict(counts) != expected:
        raise ValueError("Generator or split balance changed")
    print("Verified 120 unique images / 60 pairs; 30 train, 30 dev, 60 test; no exact prior-150 duplicates")


def choose_pairs(families, decisions):
    """Replace rejected whole pairs in slot order using unused hash-ranked families."""
    eligible = [
        key for key, family in families.items() if all(kind in family for kind in ("raise1k", *GENERATORS))
    ]
    ranked = sorted(eligible, key=lambda key: sha(f"{SEED}:{key}".encode()))
    if len(ranked) < 60:
        raise ValueError("Not enough complete source families")
    reviewed = {(row["generator"], row["image_id"]): row["eligible"] for row in decisions}
    selected, pending = [], []
    cursor = 60
    for index, original_key in enumerate(ranked[:60]):
        generator = GENERATORS[index // 20]
        position = index % 20
        split = "train" if position < 5 else "dev" if position < 10 else "test"
        key = original_key
        while (generator, key) in reviewed and not reviewed[(generator, key)]:
            if cursor >= len(ranked):
                raise ValueError("Source pool exhausted by photographic-medium exclusions")
            key = ranked[cursor]
            cursor += 1
        if (generator, key) not in reviewed:
            pending.append({"image_id": key, "generator": generator, "split": split})
        for kind in ("raise1k", generator):
            selected.append((families[key][kind], split, generator))
    return selected, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="Verify frozen delivered files without network calls"
    )
    parser.add_argument(
        "--prepare-candidates",
        action="store_true",
        help="Cache pending medium-review pairs; do not freeze a dataset",
    )
    args = parser.parse_args()
    if args.verify:
        check(json.loads(MANIFEST.read_text()))
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    (DATA / "images").mkdir(exist_ok=True)
    if get(f"https://huggingface.co/api/datasets/{REPO}").json()["sha"] != REVISION:
        raise ValueError("Source revision changed; review before repinning")
    with ThreadPoolExecutor(max_workers=6) as pool:
        catalogue = [item for batch in pool.map(page, range(0, 2800, 100)) for item in batch]
    families = defaultdict(dict)
    for item in catalogue:
        row = item["row"]
        families[row["image_id"]][row["source"]] = item
    eligibility = json.loads((TRACK / "eligibility.json").read_text())
    selected, pending = choose_pairs(families, eligibility["decisions"])
    if args.prepare_candidates:
        pending_keys = {(entry["generator"], entry["image_id"]) for entry in pending}
        candidates = [entry for entry in selected if (entry[2], entry[0]["row"]["image_id"]) in pending_keys]
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda entry: acquire(entry[0]), candidates))
        (DATA / "pending-eligibility.json").write_text(json.dumps(pending, indent=2) + "\n")
        print(f"Cached {len(pending)} pairs awaiting medium-only eligibility review")
        return
    if pending:
        raise ValueError("Medium eligibility review is incomplete; run --prepare-candidates")
    selected.sort(
        key=lambda entry: sha(f"{SEED}:{entry[0]['row']['image_id']}:{entry[0]['row']['source']}".encode())
    )
    with ThreadPoolExecutor(max_workers=6) as pool:
        acquired = list(pool.map(lambda entry: acquire(entry[0]), selected))
    samples, deliveries = [], []
    for index, ((item, split, paired_generator), raw) in enumerate(zip(selected, acquired)):
        row = item["row"]
        generated = row["source"] != "raise1k"
        if row["label"] != int(generated):
            raise ValueError("Source and published binary label disagree")
        encoded, original_size, delivered_size = deliver(raw)
        image_id = f"ai3-{index:03d}"
        path = f"data/tracks/ai_detection/images/{image_id}.jpg"
        samples.append(
            {
                "id": image_id,
                "image": path,
                "group_id": f"raise1k:{row['image_id']}",
                "split": split,
                "original_sha256": sha(raw),
                "delivered_sha256": sha(encoded),
                "decoded_rgb_sha256": pixel_hash(encoded),
                "original_size": original_size,
                "delivered_size": delivered_size,
                "delivered_bytes": len(encoded),
                "source": {
                    "dataset": REPO,
                    "revision": REVISION,
                    "config": "default",
                    "split": "test",
                    "row_index": item["row_idx"],
                    "image_id": row["image_id"],
                    "source_kind": row["source"],
                    "generator": row["source"] if generated else None,
                    "paired_generator": paired_generator,
                    "asset_url": row["image"]["src"].split("?")[0],
                    "source_url": f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
                    "representation": "Hugging Face Dataset Viewer JPEG; acquired hash is not a native camera/generator file hash",
                    "matching_basis": "Exact shared RAISE image_id; synthetic image generated from edited descriptions of this source photograph; semantic rather than pixel alignment",
                    "license": {
                        "id": "CC-BY-NC-SA-4.0"
                        if generated
                        else "RAISE non-commercial research and education",
                        "url": "https://zenodo.org/records/10066460"
                        if generated
                        else "http://loki.disi.unitn.it/RAISE/download.html",
                        "attribution": "Quentin Bammey, Synthbuster (2023)"
                        if generated
                        else "Dang-Nguyen et al., RAISE (2015)",
                        "use": "Non-commercial research; image binaries excluded from version control",
                    },
                },
                "labels": {
                    "ai_generated": generated,
                    "source_label": row["label"],
                    "basis": "Published dataset provenance; not a model judgement",
                },
            }
        )
        deliveries.append((ROOT / path, encoded))
    document = {
        "version": 2,
        "track": "ai_detection",
        "task": "ai",
        "created": "2026-10-11",
        "selection_seed": SEED,
        "selection": "Hash-rank complete families from200 native test families; initial60 slots assign20 per generator with5 train,5 dev,10 test. Exclude whole non-photographic pairs by independent pre-scoring medium review. Replace rejected slots in order with the next unused hash-ranked source family, retaining slot generator and split. No source family crosses local splits",
        "eligibility_sha256": sha((TRACK / "eligibility.json").read_bytes()),
        "eligibility_reviewed_pairs": len(eligibility["decisions"]),
        "eligibility_excluded_pairs": sum(not row["eligible"] for row in eligibility["decisions"]),
        "delivery": "EXIF transpose, RGB, aspect-preserving resize and square centre crop512x512; JPEG92, subsampling0, optimize; metadata and pair partners are never supplied to inference",
        "licence_scope": "Only original Synthbuster generators used; later Synthbuster+ generators excluded because their additional data licence is not explicit",
        "limitations": [
            "A small matched convenience panel, not the full Synthbuster benchmark or a deployment prevalence estimate",
            "Caption matching reduces semantic confounding but does not make composition, lighting, quality or resolution identical",
            "Original Synthbuster generators predate current image models; generalisation to newer generators is untested",
            "Viewer and delivery re-encoding remove native-file forensic properties; this tests visible-photo triage",
            "Square centre crops remove aspect-ratio class cues but can remove scene context; prior encoding histories are not identical",
            "Exact hash exclusion does not prove all semantically related scenes absent from previous data or pretraining",
        ],
        "samples": samples,
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != document:
        raise ValueError("Frozen manifest differs; no delivered files overwritten")
    for path, encoded in deliveries:
        path.write_bytes(encoded)
    check(document)
    MANIFEST.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
