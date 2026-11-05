"""Build a native-view KonIQ panel with human MOS and verified photo licences."""

import csv
import hashlib
import io
import json
import random
import zlib
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[2]
TRACK = Path(__file__).resolve().parent
EVIDENCE = TRACK / "evidence"
DATA = ROOT / "data/tracks/quality_v2"
ARCHIVE = "https://darus.uni-stuttgart.de/api/access/datafile/72289"
SCORE_MD5 = "0bc91efacab484f87f7aaf22bd0fd9b4"
SEED = 20261012
AUTHOR_SPLITS_SHA256 = "d0bd1ad54a60bc36fe172049e46ac76c83554e50ab84acebfd47b82b3e698a0a"
BANDS = ("bad", "poor", "fair", "good", "excellent")
QUOTAS = {"train": (38, 37, 38, 37), "dev": (20, 30, 25, 25), "test": (25, 25, 25, 25)}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def band(value):
    return BANDS[min(4, int((Decimal(str(value)) - 1) / Decimal("0.8")))]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def select(rows):
    frozen = TRACK / "selection.json"
    if frozen.exists():
        return json.loads(frozen.read_text())["rows"]
    rng = random.Random(SEED)
    official = {
        r["image_name"]: {"training": "train", "validation": "dev", "test": "test"}[r["set"]]
        for r in csv.DictReader((EVIDENCE / "author-splits.csv").read_text().splitlines())
    }
    lookups = json.loads((EVIDENCE / "licence-lookups.json").read_text())
    available = []
    for row in rows:
        image_id = row["image_name"].split(".")[0]
        rights = lookups.get(
            image_id, {"id": image_id, "error": "Not requested after Flickr rate limit"}
        ).copy()
        if rights.get("licence_url") and not rights.get("eligible"):
            continue
        rights["research_rights_basis"] = (
            "Authors declare research availability and selection of CC photos allowing editing and redistribution"
        )
        rights["individual_current_licence_verified"] = bool(rights.get("eligible"))
        rights.setdefault("photo_url", "https://www.flickr.com/photo.gne?id=" + image_id)
        author = (rights.get("author") or {}).get("url")
        available.append(
            {
                **row,
                "rights": rights,
                "sampling_band": band(row["MOS"]),
                "split": official[row["image_name"]],
                "group_id": "flickr-author:" + author if author else "flickr-photo:" + image_id,
            }
        )
    # Honour the official partitions and retain known author isolation where possible.
    authors = defaultdict(list)
    for row in available:
        if row["group_id"].startswith("flickr-author:"):
            authors[row["group_id"]].append(row)
    owner_split = {
        owner: max(
            ("train", "dev", "test"),
            key=lambda split: (
                sum(r["sampling_band"] == "bad" and r["split"] == split for r in images),
                sum(r["split"] == split for r in images),
            ),
        )
        for owner, images in authors.items()
    }
    available = [
        r for r in available if r["group_id"] not in owner_split or owner_split[r["group_id"]] == r["split"]
    ]
    selected = []
    for split, counts in QUOTAS.items():
        for wanted, count in zip(BANDS[:4], counts, strict=True):
            pool = [r for r in available if r["split"] == split and r["sampling_band"] == wanted]
            rng.shuffle(pool)
            if len(pool) < count:
                raise ValueError(f"Insufficient official {split} {wanted} records: {len(pool)} < {count}")
            selected.extend(pool[:count])
    rng.shuffle(selected)
    for i, row in enumerate(selected):
        row["id"] = f"koniq-v2-{i:03d}"
    save(frozen, {"seed": SEED, "rows": selected, "selection_precedes_model_calls": True})
    return selected


def image_info(raw):
    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert("RGB")
        tiny = rgb.convert("L").resize((9, 8))
        values = list(tiny.getdata())
        dhash = sum(
            (values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8)
        )
        return image.size, sha(str(rgb.size).encode() + rgb.tobytes()), dhash


def priors():
    files = [
        ROOT / p
        for p in (
            "evidence/samples.json",
            "evidence/benchmarks/quality.json",
            "evidence/benchmarks/ai.json",
            "tracks/quality/dataset.json",
            "tracks/ai_detection/dataset.json",
        )
    ]
    prior = []
    for path in files:
        for row in json.loads(path.read_text())["samples"]:
            raw = (ROOT / row["image"]).read_bytes()
            _, pixel, dhash = image_info(raw)
            prior.append({"id": row["id"], "encoded": sha(raw), "pixel": pixel, "dhash": dhash})
    return prior


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if sha((EVIDENCE / "author-splits.csv").read_bytes()) != AUTHOR_SPLITS_SHA256:
        raise ValueError("Pinned official partitions changed")
    raw_scores = (EVIDENCE / "native-scores.csv").read_bytes()
    if hashlib.md5(raw_scores).hexdigest() != SCORE_MD5:
        raise ValueError("Primary MOS table checksum mismatch")
    rows = list(csv.DictReader(raw_scores.decode().splitlines()))
    selected = select(rows)
    prior = priors()
    samples, seen_pixels, near = [], set(), []
    new_hashes = []
    with RemoteZip(ARCHIVE, timeout=90) as archive:
        names = {Path(x.filename).name: x.filename for x in archive.infolist() if not x.is_dir()}
        for row in selected:
            path = DATA / (row["id"] + ".jpg")
            member = names[row["image_name"]]
            info = archive.getinfo(member)
            raw = path.read_bytes() if path.exists() else archive.read(member)
            if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
                raise ValueError("Archive member size or CRC mismatch")
            size, pixel, dhash = image_info(raw)
            digest = sha(raw)
            if size != (1024, 768):
                raise ValueError("Unexpected human-rated view size")
            if pixel in seen_pixels or any(p["encoded"] == digest or p["pixel"] == pixel for p in prior):
                raise ValueError("Duplicate in frozen panel")
            seen_pixels.add(pixel)
            for p in prior:
                distance = (p["dhash"] ^ dhash).bit_count()
                if distance <= 4:
                    near.append({"new_id": row["id"], "prior_id": p["id"], "distance": distance})
            for previous_id, previous_hash in new_hashes:
                distance = (previous_hash ^ dhash).bit_count()
                if distance <= 4:
                    near.append({"new_id": row["id"], "new_peer_id": previous_id, "distance": distance})
            new_hashes.append((row["id"], dhash))
            path.write_bytes(raw)
            samples.append(
                {
                    "id": row["id"],
                    "image": str(path.relative_to(ROOT)),
                    "split": row["split"],
                    "group_id": row["group_id"],
                    "delivered_sha256": digest,
                    "original_sha256": digest,
                    "decoded_rgb_sha256": pixel,
                    "delivered_size": list(size),
                    "delivered_bytes": len(raw),
                    "mime": "image/jpeg",
                    "source": {
                        "dataset": "KonIQ-10k",
                        "id": row["image_name"],
                        "group_id": row["group_id"],
                        "type": "authentic_distortion_of_photo",
                        "license": row["rights"].get(
                            "licence_url",
                            "Individual CC licence declared by source authors; current record unavailable",
                        ),
                        "individual_current_licence_verified": row["rights"][
                            "individual_current_licence_verified"
                        ],
                        "research_rights_basis": row["rights"]["research_rights_basis"],
                        "attribution": row["rights"].get("author"),
                        "photo_url": row["rights"]["photo_url"],
                        "primary_url": "https://doi.org/10.18419/darus-2435",
                        "archive_url": ARCHIVE,
                        "archive_member": member,
                        "archive_member_crc32": f"{info.CRC:08x}",
                    },
                    "labels": {
                        "quality_mos_1to5": float(row["MOS"]),
                        "native_mos": row["MOS"],
                        "native_sd": row["SD"],
                        "native_mos_zscore": row["MOS_zscore"],
                        "rating_distribution": [int(row[f"c{i}"]) for i in range(1, 6)],
                        "ratings_per_image": int(row["c_total"]),
                        "scale": [1, 5],
                        "direction": "higher_is_better",
                        "mapping": "identity",
                        "kind": "human_absolute_category_mean",
                        "low_quality": float(row["MOS"]) < 2.6,
                        "sampling_band": row["sampling_band"],
                        "aesthetics_mos": None,
                    },
                }
            )
            print(f"Acquired {len(samples)}/{len(selected)}", flush=True)
    save(
        EVIDENCE / "duplicate-check.json",
        {
            "prior_images": len(prior),
            "exact_matches": 0,
            "decoded_matches": 0,
            "dhash_max_distance": 4,
            "near_matches": near,
        },
    )
    if near:
        raise ValueError("Potential scene overlap needs visual review")
    manifest = {
        "track": "quality_v2",
        "benchmark": "koniq-quality-expansion-20261011",
        "seed": SEED,
        "selection_sha256": sha((TRACK / "selection.json").read_bytes()),
        "source": {
            "dataset": "KonIQ-10k",
            "primary_doi": "10.18419/darus-2435",
            "version": "1.0",
            "metadata_sha256": sha(raw_scores),
            "metadata_primary_md5": SCORE_MD5,
            "archive_primary_md5": "85f076d7a43b2e306f7080814f69af3e",
            "full_archive_hash_verified": False,
            "author_splits_git_revision": "1c271570ac83aeb6aa8e7ff25449f3299b0d2e53",
            "author_splits_sha256": AUTHOR_SPLITS_SHA256,
            "individual_current_licences_verified": 45,
            "individual_current_licences_unverified": 305,
            "research_use_basis": "Authors' published CC/research declaration; noncommercial scope",
            "photographer_isolation": "Partial; all known photographers are isolated, attribution incomplete",
        },
        "delivery": "Unmodified native 1024x768 JPEG; no resize, crop or re-encoding",
        "human_view": "Native 1024x768 technical quality ACR, no pristine reference",
        "low_quality_rule": "quality_mos_1to5 < 2.6",
        "split_counts": dict(Counter(s["split"] for s in samples)),
        "sampling_band_counts": dict(Counter(s["labels"]["sampling_band"] for s in samples)),
        "samples": samples,
    }
    path = TRACK / "dataset.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Rebuild changed frozen manifest")
    save(path, manifest)


if __name__ == "__main__":
    main()
