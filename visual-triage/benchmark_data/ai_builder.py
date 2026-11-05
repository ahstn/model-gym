"""Build a balanced, provenance-labelled AI-origin panel from pinned OpenFake v2."""

import argparse
import hashlib
import io
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REPO = "ComplexDataLab/OpenFake"
REVISION = "3247b942f18d39d1fd476b1077852af033d35d8c"
GENERATORS = (
    "flux.2-klein-9b",
    "gpt-image-1.5",
    "midjourney-7",
    "z-image-turbo",
    "ernie-image-turbo",
    "recraft-v3",
)
SEED = 20261011
# This is an origin benchmark, so keep explicit-content testing separate.
EXCLUDED = re.compile(r"\b(nude|naked|nudity|porn|sexual|sex|erotic|topless|nipple|genital)\b", re.IGNORECASE)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def get(url, **kwargs):
    response = requests.get(url, timeout=90, **kwargs)
    response.raise_for_status()
    return response


def dhash(image):
    gray = image.convert("L").resize((9, 8))
    pixels = [gray.getpixel((x, y)) for y in range(8) for x in range(9)]
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def deliver(raw):
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    original_size = image.size
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=92, subsampling=0, optimize=True)
    return stream.getvalue(), original_size, image.size, dhash(image)


def catalogue_page(offset, cache):
    path = cache / f"rows-{offset:05d}.json"
    if path.exists():
        document = json.loads(path.read_text())
    else:
        document = get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": REPO, "config": "core", "split": "test", "offset": offset, "length": 100},
        ).json()
        path.write_text(json.dumps(document, indent=2) + "\n")
    for item in document["rows"]:
        assert f"/--/{REVISION}/--/" in item["row"]["image"]["src"], "Dataset Viewer revision changed"
    return document["rows"]


def fetch_asset(item, cache):
    row = item["row"]
    target = cache / (row["row_id"] + ".bin")
    if not target.exists():
        # Refresh signed URL by exact row if the cached catalog link has expired.
        response = requests.get(row["image"]["src"], timeout=90)
        if response.status_code == 403:
            fresh = get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": REPO,
                    "config": "core",
                    "split": "test",
                    "offset": item["row_idx"],
                    "length": 1,
                },
            ).json()["rows"][0]
            assert fresh["row"]["row_id"] == row["row_id"]
            assert f"/--/{REVISION}/--/" in fresh["row"]["image"]["src"]
            response = get(fresh["row"]["image"]["src"])
        response.raise_for_status()
        target.write_bytes(response.content)
    return target.read_bytes()


def blind_sheets(samples, destination):
    for start in range(0, len(samples), 6):
        canvas = Image.new("RGB", (1024, 3 * 550), "white")
        draw = ImageDraw.Draw(canvas)
        for index, sample in enumerate(samples[start : start + 6]):
            image = Image.open(ROOT / sample["image"])
            x, y = index % 2 * 512, index // 2 * 550
            canvas.paste(image, (x + (512 - image.width) // 2, y + (512 - image.height) // 2))
            draw.text((x + 12, y + 520), sample["id"], fill="black")
        canvas.save(destination / f"blind-{start // 6:02d}.jpg", quality=95)


def verify_frozen_samples(previous, samples):
    """Compare the full sample records before replacing any frozen evidence."""
    normalized = json.loads(json.dumps(samples))
    if previous["samples"] != normalized:
        raise ValueError("Reproduction differs from the frozen sample records; no delivered files changed")


def cached_deliveries(samples, cache):
    """Validate every cached source and reproduced image before allowing writes."""
    pending = []
    for sample in samples:
        raw = (cache / (sample["source"]["row_id"] + ".bin")).read_bytes()
        if digest(raw) != sample["original_sha256"]:
            raise ValueError(f"Cached source hash mismatch: {sample['id']}")
        encoded, original_size, delivered_size, _ = deliver(raw)
        if (
            digest(encoded) != sample["delivered_sha256"]
            or list(original_size) != sample["original_size"]
            or list(delivered_size) != sample["delivered_size"]
            or len(encoded) != sample["delivered_bytes"]
        ):
            raise ValueError(f"Cached reproduction differs: {sample['id']}")
        pending.append((ROOT / sample["image"], encoded))
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Verify frozen files without network access")
    mode.add_argument(
        "--rebuild-from-cache",
        action="store_true",
        help="Recreate the exact frozen delivered images from verified cached assets, without network access",
    )
    args = parser.parse_args()
    target = ROOT / "evidence/benchmarks/ai.json"
    if args.verify:
        samples = json.loads(target.read_text())["samples"]
        assert len(samples) == 60
        assert sum(sample["labels"]["ai_generated"] for sample in samples) == 30
        for sample in samples:
            raw = (ROOT / sample["image"]).read_bytes()
            assert digest(raw) == sample["delivered_sha256"], sample["id"]
        print("Verified 60 images, balanced 30 real / 30 AI, and all delivered hashes")
        return

    destination = ROOT / "data/benchmarks/ai"
    cache = destination / "source-assets"
    if args.rebuild_from_cache:
        samples = json.loads(target.read_text())["samples"]
        pending = cached_deliveries(samples, cache)
        for path, encoded in pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
        blind_sheets(samples, destination)
        print(f"Rebuilt {len(samples)} frozen delivered images; manifest and labels unchanged")
        return
    previous = json.loads(target.read_text()) if target.exists() else None
    cache.mkdir(parents=True, exist_ok=True)
    revision = get(f"https://huggingface.co/api/datasets/{REPO}").json()["sha"]
    assert revision == REVISION, "Dataset changed; use frozen manifest, or review and repin the source"
    models = ("docci", *GENERATORS)
    with ThreadPoolExecutor(max_workers=4) as executor:
        pages = list(executor.map(lambda offset: catalogue_page(offset, cache), range(0, 1000, 100)))
    candidates = [item for page in pages for item in page]
    catalogues = {
        model: sorted(
            (item for item in candidates if item["row"]["model"] == model),
            key=lambda item: digest(f"{SEED}:{item['row']['row_id']}".encode()),
        )
        for model in models
    }

    old = json.loads((ROOT / "evidence/samples.json").read_text())["samples"]
    old_hashes = {sample["delivered_sha256"] for sample in old}
    old_dhashes = [dhash(Image.open(ROOT / sample["image"])) for sample in old]
    collected, rejected, accepted_hashes = [], [], set()
    accepted_dhashes = []
    for model in models:
        goal = 30 if model == "docci" else 5
        count = 0
        for item in catalogues[model]:
            row = item["row"]
            expected_label = "real" if model == "docci" else "fake"
            assert row["label"] == expected_label, f"Unexpected source label for {model}"
            if EXCLUDED.search(row.get("prompt") or ""):
                rejected.append(
                    {"row_id": row["row_id"], "reason": "explicit-content keyword in source caption/prompt"}
                )
                continue
            raw = fetch_asset(item, cache)
            encoded, original_size, size, perceptual_hash = deliver(raw)
            encoded_hash = digest(encoded)
            distances = [
                (perceptual_hash ^ previous).bit_count() for previous in (*old_dhashes, *accepted_dhashes)
            ]
            if encoded_hash in old_hashes | accepted_hashes or min(distances) <= 5:
                rejected.append({"row_id": row["row_id"], "reason": "exact duplicate or dHash distance <= 5"})
                continue
            accepted_hashes.add(encoded_hash)
            accepted_dhashes.append(perceptual_hash)
            collected.append((item, raw, encoded, original_size, size, perceptual_hash))
            count += 1
            if count == goal:
                break
        if count != goal:
            raise ValueError(f"Only {count} eligible images for {model}; need {goal}")
        print(f"Acquired source stratum {model}: {count}", flush=True)

    random.Random(SEED).shuffle(collected)
    samples = []
    pending_images = []
    for index, (item, raw, encoded, original_size, size, perceptual_hash) in enumerate(collected):
        row = item["row"]
        identifier = f"ai-{index:03d}"
        relative = f"data/benchmarks/ai/{identifier}.jpg"
        pending_images.append((ROOT / relative, encoded))
        actual_source_hash = digest(raw)
        samples.append(
            {
                "id": identifier,
                "image": relative,
                "original_sha256": actual_source_hash,
                "delivered_sha256": digest(encoded),
                "original_size": original_size,
                "delivered_size": size,
                "delivered_bytes": len(encoded),
                "dhash": f"{perceptual_hash:016x}",
                "source": {
                    "dataset": REPO,
                    "revision": REVISION,
                    "config": "core",
                    "split": "test",
                    "row_index": item["row_idx"],
                    "row_id": row["row_id"],
                    "group_id": "prompt-"
                    + digest(" ".join((row.get("prompt") or row["row_id"]).split()).encode())[:24],
                    "group_basis": "Exact normalized source prompt or caption; related scenes with different captions can remain",
                    "source_url": f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
                    "asset_url": row["image"]["src"].split("?")[0],
                    "source_kind": row["model"],
                    "generator_type": row["type"],
                    "release_date_as_published": row["release_date"],
                    "published_original_sha256": row["image_sha256"],
                    "asset_matches_published_original_sha256": actual_source_hash == row["image_sha256"],
                    "representation": "Hugging Face Dataset Viewer image asset; original_sha256 hashes acquired bytes, which may be re-encoded from upstream",
                    "license": {
                        "id": "CC-BY-NC-4.0",
                        "url": f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md",
                        "upstream_real_photos": "DOCCI: CC-BY-4.0, Google LLC",
                        "use": "non-commercial evaluation; no image redistribution in this repository",
                    },
                },
                "labels": {
                    "ai_generated": row["label"] == "fake",
                    "basis": "Dataset generation provenance for fake; DOCCI camera capture provenance for real. No model prediction is ground truth.",
                    "source_label": row["label"],
                },
            }
        )
    manifest = {
        "benchmark": "ai",
        "version": 1,
        "created": "2026-10-11",
        "selection_seed": SEED,
        "selection": "Hash-rank source strata within the first 1000 published core/test rows; 30 DOCCI real plus five per six generators; keyword exclusion and dHash<=5 duplicate rejection; shuffle fixed seed before assigning blind IDs.",
        "delivery": "Full frame; EXIF transpose; RGB; maximum edge 512 without upscaling; JPEG quality 92, subsampling 0, optimize=True; metadata never sent to models.",
        "limitations": [
            "A small stratified convenience panel, not a population estimate or full OpenFake benchmark.",
            "Real photos come from DOCCI only; source/content/camera confounding remains. Scenes are not paired.",
            "Fixed modern-generator subset does not establish performance on unseen generators.",
            "Viewer assets can contain upstream re-encoding; published and acquired hashes are separate.",
            "dHash screening catches only some near duplicates and does not prove semantic scene independence.",
        ],
        "excluded_candidates": rejected,
        "samples": samples,
    }
    if previous is not None:
        verify_frozen_samples(previous, samples)
    for path, encoded in pending_images:
        path.write_bytes(encoded)
    target.parent.mkdir(parents=True, exist_ok=True)
    if previous is None:
        target.write_text(json.dumps(manifest, indent=2) + "\n")
    blind_sheets(samples, destination)
    print(f"Saved {len(samples)} samples and 10 blind contact sheets", flush=True)


if __name__ == "__main__":
    main()
