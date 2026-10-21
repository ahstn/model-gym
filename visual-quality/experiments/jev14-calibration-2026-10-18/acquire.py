"""Freeze a separate, family-disjoint MS COCOAI calibration and validation cohort."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/jev14-calibration-2026-10-18"
OLD = Path("/Users/ahstn/.codex/worktrees/d187/model-gym/visual-triage")
REPO = "Rajarshi-Roy-research/Defactify_Image_Dataset"
REVISION = "787334f7857fa54f29027a7f09c30e895ad486ef"
SEED = "jev14-independent-calibration-20261018-v1"
GENERATORS = {1: "SD2.1", 2: "SDXL", 3: "SD3", 4: "DALL-E3", 5: "Midjourney-v6"}
TOTALS = {"validation": 9000, "test": 45000}
DELIVERY = "EXIF transpose; RGB; centred ImageOps.fit 512x512 LANCZOS; JPEG quality95 subsampling0; no source metadata copied"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def native_files():
    tree_path = DATA / "research/native-tree.json"
    if not tree_path.exists():
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data?recursive=true&expand=false"
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        if len(response.content) > 1024 * 1024:
            raise ValueError("Unexpectedly large pinned source tree")
        save(tree_path, response.json())
    tree = read(tree_path)
    chosen = [row for row in tree if any(row["path"].startswith(f"data/{split}-") for split in TOTALS)]
    if len(chosen) != 10 or sum(row["size"] for row in chosen) > 5_000_000_000:
        raise ValueError("Native acquisition exceeds the frozen 5 GB cap")
    return sorted(chosen, key=lambda row: (not row["path"].startswith("data/validation"), row["path"]))


def capture_documents():
    documents = {
        "dataset-card.md": f"https://huggingface.co/datasets/{REPO}/raw/{REVISION}/README.md",
        "paper.html": "https://arxiv.org/html/2601.00553v1",
        "coco-terms.html": "https://cocodataset.org/dataset/termsofuse.htm",
    }
    records = []
    for filename, url in documents.items():
        path = DATA / "research" / filename
        if not path.exists():
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            if len(response.content) > 2 * 1024 * 1024:
                raise ValueError("Source document exceeds capture limit")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        records.append({"url": url, "path": str(path.relative_to(DATA)), "sha256": file_sha(path)})
    save(DATA / "research/source-documents.json", records)
    return records


def download_native(row):
    destination = DATA / "native" / Path(row["path"]).name
    expected = row["lfs"]["oid"]
    if destination.exists():
        if destination.stat().st_size != row["size"] or file_sha(destination) != expected:
            raise ValueError("Cached native parquet does not match the published LFS hash")
        return
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row['path']}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    digest, size, last_progress = hashlib.sha256(), 0, 0
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        if response.status_code == 429:
            save(DATA / "research/native-rate-limit.json", {"url": url, "headers": dict(response.headers)})
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                size += len(chunk)
                if size > row["size"]:
                    raise ValueError("Native source exceeded published length")
                stream.write(chunk)
                digest.update(chunk)
                if size - last_progress >= 128 * 1024 * 1024:
                    print(
                        f"Native {destination.name}: {size / 1e6:.0f}/{row['size'] / 1e6:.0f} MB", flush=True
                    )
                    last_progress = size
    if size != row["size"] or digest.hexdigest() != expected:
        raise ValueError("Native source failed the published LFS SHA256 check")
    temporary.replace(destination)
    print(f"Verified native file {destination.name}", flush=True)


def metadata():
    capture_documents()
    jobs = native_files()
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        list(pool.map(download_native, jobs))
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    for split, total in TOTALS.items():
        rows = []
        for info in jobs:
            if not info["path"].startswith(f"data/{split}-"):
                continue
            filename = Path(info["path"]).name
            parquet = pq.ParquetFile(DATA / "native" / filename)
            for group in range(parquet.num_row_groups):
                group_rows = parquet.read_row_group(
                    group, columns=["Caption", "Label_A", "Label_B"]
                ).to_pylist()
                for local_index, row in enumerate(group_rows):
                    if row["Label_A"] != int(row["Label_B"] != 0) or row["Label_B"] not in range(6):
                        raise ValueError("Unexpected native source label")
                    rows.append(
                        {
                            "row_idx": len(rows),
                            "row": row,
                            "source_split": split,
                            "native_file": filename,
                            "native_row_group": group,
                            "native_group_row": local_index,
                        }
                    )
        if len(rows) != total:
            raise ValueError("Native split count differs from pinned source")
        save(DATA / "metadata" / f"native-{split}.json", rows)
    print("Native metadata acquisition complete", flush=True)


def normalized(text):
    return " ".join(text.casefold().split())


def family_id(text):
    return "caption-" + sha(normalized(text).encode())


def phash(image):
    """64-bit DCT perceptual hash, with the DC term excluded from the median."""
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    positions = np.arange(32)
    cosine = np.cos(np.pi * np.arange(8)[:, None] * (2 * positions + 1) / 64)
    coefficients = (cosine @ pixels @ cosine.T).ravel()
    return sum(int(value > np.median(coefficients[1:])) << i for i, value in enumerate(coefficients))


def image_record(raw):
    with Image.open(io.BytesIO(raw)) as source:
        original_dimensions = list(source.size)
        image = ImageOps.exif_transpose(source).convert("RGB")
    original_phash = phash(image)
    decoded_sha = sha(str(image.size).encode() + image.tobytes())
    delivered = ImageOps.fit(image, (512, 512), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    output = io.BytesIO()
    delivered.save(output, format="JPEG", quality=95, subsampling=0)
    return output.getvalue(), {
        "source_sha256": sha(raw),
        "source_dimensions": original_dimensions,
        "decoded_rgb_sha256": decoded_sha,
        "source_phash": f"{original_phash:016x}",
        "delivered_phash": f"{phash(delivered):016x}",
        "delivered_sha256": sha(output.getvalue()),
    }


def prior_records():
    """Read all published prior origin cohorts, plus acquisition eligibility images."""
    manifests = [OLD / "evidence/samples.json", OLD / "evidence/benchmarks/ai.json"]
    manifests += [
        OLD / f"tracks/{track}/dataset.json" for track in ("ai_detection", "ai_detection_v2", "metadata_v3")
    ]
    assets = set()
    captions = set()
    source_pins = []
    for path in manifests:
        document = read(path)
        source_pins.append({"path": str(path), "sha256": sha(path.read_bytes())})
        for row in document["samples"]:
            assets.add(OLD / row["image"])
    for path in (OLD / "data/benchmarks/ai/source-assets").glob("rows-*.json"):
        for item in read(path)["rows"]:
            captions.add(family_id(item["row"].get("prompt") or ""))
    for relative, field in (
        ("data/metadata-v3/research/ai/prompts.csv", "Prompt"),
        ("data/agiqa-data.csv", "prompt"),
    ):
        path = OLD / relative
        source_pins.append({"path": str(path), "sha256": file_sha(path)})
        with path.open(newline="") as stream:
            captions.update(family_id(row[field]) for row in csv.DictReader(stream) if row.get(field))
    # Include every locally acquired candidate from the origin prompt-development tracks.
    for track in ("ai_detection", "ai_detection_v2"):
        for path in (OLD / f"data/tracks/{track}/source-assets").glob("*.bin"):
            assets.add(path)
    cache = DATA / "prior-fingerprints.json"
    previous = read(cache).get("records", {}) if cache.exists() else {}
    records = {}
    for path in sorted(assets):
        raw = path.read_bytes()
        raw_sha = sha(raw)
        key = str(path)
        if key in previous and previous[key]["source_sha256"] == raw_sha:
            record = previous[key]
        else:
            _, record = image_record(raw)
        records[key] = record
    save(cache, {"records": records, "source_manifests": source_pins})
    pilot_hashes = set()
    pilot_sources = Counter()
    for split in ("train", "development", "test"):
        path = DATA.parent / "pilot-10k-v1" / f"{split}.jsonl"
        source_pins.append({"path": str(path), "sha256": sha(path.read_bytes())})
        for line in path.read_text().splitlines():
            row = json.loads(line)
            pilot_sources[row["component"]] += 1
            pilot_hashes.update(row[key] for key in ("sha256", "source_sha256") if row.get(key))
    print(
        f"Prior screening: {len(records)} origin images; {sum(pilot_sources.values())} pilot hashes",
        flush=True,
    )
    return records, captions, pilot_hashes, source_pins, dict(pilot_sources)


def families(split):
    groups = defaultdict(lambda: defaultdict(list))
    for item in read(DATA / "metadata" / f"native-{split}.json"):
        groups[family_id(item["row"]["Caption"])][item["row"]["Label_B"]].append(item)
    return groups


@lru_cache(maxsize=4)
def native_image_group(filename, group):
    return pq.ParquetFile(DATA / "native" / filename).read_row_group(group, columns=["Image"]).to_pylist()


def source_image(item):
    split, index = item["source_split"], item["row_idx"]
    destination = DATA / "source-assets" / f"{split}-{index:05d}.bin"
    if not destination.exists():
        row = native_image_group(item["native_file"], item["native_row_group"])[item["native_group_row"]]
        raw = row["Image"]["bytes"]
        if not raw or len(raw) > 16 * 1024 * 1024:
            raise ValueError("Invalid or oversized embedded native image")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return destination.read_bytes()


def collisions(record, comparisons, pilot_hashes):
    if record["source_sha256"] in pilot_hashes:
        return {"kind": "pilot_exact_source_hash"}
    for other_id, other in comparisons.items():
        if any(
            record[key] == other[key] for key in ("source_sha256", "decoded_rgb_sha256", "delivered_sha256")
        ):
            return {"kind": "exact", "other": other_id}
        # Check full-frame and common delivered-view hashes for crop robustness.
        for key in ("source_phash", "delivered_phash"):
            distance = (int(record[key], 16) ^ int(other[key], 16)).bit_count()
            if distance <= 6:
                return {"kind": "phash", "view": key, "distance": distance, "other": other_id}
    return None


def acquire_pair(pair):
    return [(item, *image_record(source_image(item))) for item in pair]


def build():
    if (HERE / "manifest.json").exists():
        verify()
        return
    metadata()
    priors, prior_captions, pilot_hashes, source_pins, pilot_sources = prior_records()
    comparisons = dict(priors)
    used_families, cases, exclusions = set(), [], []
    available = {}
    for source_split, split, count in (("validation", "calibration", 300), ("validation", "validation", 500)):
        groups = families(source_split)
        ranked = sorted(groups, key=lambda fid: sha(f"{SEED}:{split}:{fid}".encode()))
        available[split] = {
            "source_families": len(groups),
            "families_with_real": sum(0 in g for g in groups.values()),
        }
        accepted = Counter()
        offered = set()
        while sum(accepted.values()) < count:
            batch = []
            for generator in range(1, 6):
                needed = count // 5 - accepted[generator]
                if needed <= 0:
                    continue
                for fid in ranked:
                    if fid in offered or fid in used_families or fid in prior_captions:
                        continue
                    group = groups[fid]
                    if 0 not in group or generator not in group:
                        continue
                    pair = [min(group[label], key=lambda item: item["row_idx"]) for label in (0, generator)]
                    batch.append((fid, generator, pair))
                    offered.add(fid)
                    if sum(g == generator for _, g, _ in batch) >= min(10, needed):
                        break
            if not batch:
                raise ValueError(f"Insufficient unused real/generated caption pairs for {split}: {accepted}")
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = pool.map(acquire_pair, [pair for _, _, pair in batch])
                for (fid, generator, _), measured in zip(batch, results, strict=True):
                    collision = None
                    pair_records = {}
                    for item, raw, record in measured:
                        collision = collisions(record, comparisons, pilot_hashes) or collisions(
                            record, pair_records, set()
                        )
                        if collision:
                            break
                        pair_records[f"{source_split}:{item['row_idx']}"] = record
                    if collision:
                        exclusions.append(
                            {"family_id": fid, "split": split, "generator": generator, **collision}
                        )
                        save(DATA / "exclusions-in-progress.json", exclusions)
                        continue
                    pair_index = sum(accepted.values())
                    for item, raw, record in measured:
                        label = item["row"]["Label_B"]
                        case_id = f"{split}/mscocoai-{pair_index:04d}-{'ai' if label else 'real'}"
                        image = f"delivered/{case_id.replace('/', '-')}.jpg"
                        destination = DATA / image
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists() and destination.read_bytes() != raw:
                            raise ValueError("Acquisition resume changed delivered bytes")
                        destination.write_bytes(raw)
                        case = {
                            "row_id": case_id,
                            "case_id": case_id,
                            "split": split,
                            "task": "ai",
                            "sample": case_id.split("/")[-1],
                            "image": image,
                            "image_width": 512,
                            "image_height": 512,
                            "image_bytes": len(raw),
                            "image_mime": "image/jpeg",
                            "eligible": True,
                            "expected_ai": bool(label),
                            "family_id": fid,
                            "generator": GENERATORS.get(label),
                            "paired_generator": GENERATORS[generator],
                            "source_dataset": REPO,
                            "source_revision": REVISION,
                            "source_split": source_split,
                            "source_row_index": item["row_idx"],
                            "source_url": f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
                            "source_asset_url": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{item['native_file']}",
                            "native_row_group": item["native_row_group"],
                            "native_group_row": item["native_group_row"],
                            "source_labels": {"Label_A": item["row"]["Label_A"], "Label_B": label},
                            "representation": "Native embedded bytes from pinned parquet verified against published LFS SHA256; no claim of untouched camera originals",
                            **record,
                        }
                        cases.append(case)
                        comparisons[case_id] = record
                    used_families.add(fid)
                    accepted[generator] += 1
                    if len(cases) % 100 == 0:
                        print(
                            f"Accepted images {len(cases)}/1600; {split} generators {dict(accepted)}",
                            flush=True,
                        )
            save(DATA / "acquisition-progress.json", {"cases": cases, "exclusions": exclusions})
        available[split]["selected_generators"] = dict(accepted)
    dimensions = defaultdict(Counter)
    for case in cases:
        dimensions[case["generator"] or "real"]["x".join(map(str, case["source_dimensions"]))] += 1
    manifest = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "source_dataset": REPO,
        "source_revision": REVISION,
        "delivery": DELIVERY,
        "selection": "Hash-ranked normalized caption families within official validation; one real plus one generated image; five generators balanced per split; calibration selected first; whole families rejected on overlap or near-duplicate collision before inference",
        "cases": cases,
    }
    audit = {
        "source_revision": REVISION,
        "available": available,
        "source_dimensions": dict(dimensions),
        "source_manifests": source_pins,
        "prior_origin_images": len(priors),
        "pilot_component_counts": pilot_sources,
        "prior_caption_exclusions": len(prior_captions),
        "excluded_families": exclusions,
        "selected_families": len(used_families),
        "selected_images": len(cases),
        "duplicate_method": "SHA256 of acquired bytes, RGB pixels with dimensions and delivered bytes; 64-bit DCT perceptual hash, full-frame and standard delivered crop, Hamming distance at most6. Compared against prior origin cohorts, acquired origin-development candidates, all accepted images and each paired member. Pilot10k checked by source-byte hashes only.",
        "limitations": [
            "Caption normalization cannot prove capture-session or photographer independence; related scenes with different captions can remain.",
            "Perceptual hash can reject similar but distinct scenes and miss transformed duplicates.",
            "Both cohorts use distinct families from official validation. The official test split repeats each caption/label five times and its first block is brightness-reduced; it was excluded before model inference.",
            "Source resolutions and compression histories differ by generator; a shared delivery transform cannot remove these upstream cues.",
            "Dataset licence is not specified; paper CC BY4.0 is not an image licence. Keep image assets local and out of Git; commercial or redistribution rights are not established.",
            "Distributed parquet bytes are verified; this does not establish untouched camera or generator-output identity.",
            "No model outcomes, visual quality filters or detector scores were used for selection.",
        ],
        "native_files": [
            {"path": row["path"], "bytes": row["size"], "sha256": row["lfs"]["oid"]} for row in native_files()
        ],
        "source_document_snapshots": read(DATA / "research/source-documents.json"),
        "acquisition_issue": "Dataset Viewer returned HTTP429 after 62 metadata pages; stopped Viewer requests and used ten pinned native parquet files instead. Before inference, the initial official-test cohort was archived and replaced by unused official-validation families because test rows contain five variant blocks and minimum-index selection chose reduced brightness.",
        "source_documents": [
            f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md",
            "https://arxiv.org/html/2601.00553v1",
            "https://cocodataset.org/dataset/termsofuse.htm",
        ],
    }
    save(HERE / "manifest.json", manifest)
    save(HERE / "acquisition-audit.json", audit)
    verify()


def verify():
    document = read(HERE / "manifest.json")
    groups = defaultdict(set)
    counts = Counter()
    hashes = set()
    for case in document["cases"]:
        raw = (DATA / case["image"]).read_bytes()
        if sha(raw) != case["delivered_sha256"] or len(raw) != case["image_bytes"]:
            raise ValueError("Frozen delivered asset changed")
        if case["delivered_sha256"] in hashes:
            raise ValueError("Duplicate delivered bytes")
        hashes.add(case["delivered_sha256"])
        groups[case["split"]].add(case["family_id"])
        counts[(case["split"], case["expected_ai"])] += 1
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (512, 512) or image.getexif() or image.format != "JPEG":
                raise ValueError("Frozen delivered image contract changed")
    if groups["calibration"] & groups["validation"]:
        raise ValueError("Cross-split caption family leakage")
    expected = {
        ("calibration", False): 300,
        ("calibration", True): 300,
        ("validation", False): 500,
        ("validation", True): 500,
    }
    if dict(counts) != expected:
        raise ValueError("Incomplete or imbalanced frozen cohort")
    print(f"Verified {len(hashes)} delivered images and disjoint caption families", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.metadata_only:
        metadata()
    elif args.verify:
        verify()
    else:
        build()
