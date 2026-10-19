"""Freeze a group-separated KADID-10k quality stress test, then fetch native PNGs."""

import csv
import hashlib
import io
import json
import random
import zlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[2]
TRACK = Path(__file__).resolve().parent
DATA = ROOT / "data/tracks/quality"
REPO = "myzhao1999/kadid10k"
REVISION = "251f5882d7c6f5072b37dae3f16a0ba6babf6c75"
ARCHIVE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/kadid10k.zip"
ARCHIVE_SHA256 = "fe59ace86a2525d5785ff011a2119fa88839e0329f5029f7b23994727efd185c"
METADATA_SHA256 = "573e2ed98fdaa2a5aed7b50ad716906e125e1001ed0e97c8e93e79d7b997ba99"
SEED = 20261011
BANDS = ("bad", "poor", "fair", "good", "excellent")
DISTORTIONS = (
    "gaussian_blur",
    "lens_blur",
    "motion_blur",
    "color_diffusion",
    "color_shift",
    "color_quantization",
    "color_saturation_1",
    "color_saturation_2",
    "jpeg2000",
    "jpeg",
    "white_noise",
    "white_noise_color",
    "impulse_noise",
    "multiplicative_noise",
    "denoise",
    "brighten",
    "darken",
    "mean_shift",
    "jitter",
    "non_eccentricity_patch",
    "pixelate",
    "quantization",
    "color_block",
    "high_sharpen",
    "contrast_change",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def band(value):
    """Use exact left-closed decimal boundaries, including 5.0 in the last band."""
    value = Decimal(str(value))
    if not Decimal(1) <= value <= Decimal(5):
        raise ValueError("Invalid native KADID quality value")
    return BANDS[min(4, int((value - 1) / Decimal("0.8")))]


def select(rows):
    parents = defaultdict(list)
    for row in rows:
        parents[row["ref_img"]].append(row)
    if len(rows) != 10125 or len(parents) != 81:
        raise ValueError("Expected all 81 parents and 10,125 human-rated distortions")
    rng = random.Random(SEED)
    groups = sorted(parents)
    rng.shuffle(groups)
    selected = []
    for split, chosen in (("train", groups[:20]), ("dev", groups[20:40]), ("test", groups[40:80])):
        for i, parent in enumerate(chosen):
            wanted = ["bad" if i % 2 == 0 else "poor", ("fair", "good", "fair", "excellent")[i % 4]]
            for target in wanted:
                eligible = sorted(
                    (r for r in parents[parent] if band(r["dmos"]) == target), key=lambda r: r["dist_img"]
                )
                if not eligible:
                    raise ValueError(f"Missing {target} band in {parent}")
                selected.append({**rng.choice(eligible), "split": split, "sampling_band": target})
    rng.shuffle(selected)
    for i, row in enumerate(selected):
        row["id"] = f"kadid-{i:03d}"
    return selected


def pixel_sha(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        return sha(str(image.size).encode() + image.tobytes())


def prior_hashes():
    encoded, pixels = set(), set()
    for relative in (
        "evidence/samples.json",
        "evidence/benchmarks/quality.json",
        "evidence/benchmarks/ai.json",
    ):
        for sample in json.loads((ROOT / relative).read_text())["samples"]:
            raw = (ROOT / sample["image"]).read_bytes()
            encoded.add(sha(raw))
            pixels.add(pixel_sha(raw))
    return encoded, pixels


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    metadata = (TRACK / "evidence/dmos.csv").read_bytes()
    if sha(metadata) != METADATA_SHA256:
        raise ValueError("Human-score source changed")
    selected = select(list(csv.DictReader(io.StringIO(metadata.decode()))))
    primary_checksums = {
        row["name"]: row for row in json.loads((TRACK / "evidence/osf-image-checksums.json").read_text())
    }
    selection = {"seed": SEED, "rows": selected, "metadata_sha256": METADATA_SHA256}
    selection_path = TRACK / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Frozen sample selection changed")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    # Selection is written before any image retrieval or inference outcome is read.
    prior_encoded, prior_pixels = prior_hashes()
    manifest_path = TRACK / "dataset.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    frozen = {s["id"]: s for s in existing["samples"]} if existing else {}
    samples = []
    seen = set()
    with RemoteZip(ARCHIVE, timeout=90) as archive:
        if sha(archive.read("kadid10k/dmos.csv")) != METADATA_SHA256:
            raise ValueError("Archive labels differ from the author's OSF score table")
        for row in selected:
            member = "kadid10k/images/" + row["dist_img"]
            path = DATA / (row["id"] + ".png")
            raw = path.read_bytes() if path.exists() else archive.read(member)
            info = archive.getinfo(member)
            if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
                raise ValueError("Archive size or CRC mismatch: " + row["id"])
            digest, pixels = sha(raw), pixel_sha(raw)
            primary = primary_checksums[row["dist_img"]]
            if digest != primary["sha256"] or len(raw) != primary["size"]:
                raise ValueError("Image differs from the author's OSF checksum: " + row["id"])
            if row["id"] in frozen and digest != frozen[row["id"]]["delivered_sha256"]:
                raise ValueError("Frozen image changed: " + row["id"])
            if digest in prior_encoded or pixels in prior_pixels or pixels in seen:
                raise ValueError("Duplicate image: " + row["id"])
            seen.add(pixels)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.size != (512, 384):
                    raise ValueError("Unexpected native image view")
            path.write_bytes(raw)
            distortion = int(row["dist_img"].split("_")[1])
            level = int(row["dist_img"].split("_")[2].split(".")[0])
            samples.append(
                {
                    "id": row["id"],
                    "image": str(path.relative_to(ROOT)),
                    "split": row["split"],
                    "group_id": "kadid:" + row["ref_img"],
                    "delivered_sha256": digest,
                    "original_sha256": digest,
                    "decoded_rgb_sha256": pixels,
                    "delivered_size": [512, 384],
                    "delivered_bytes": len(raw),
                    "mime": "image/png",
                    "source": {
                        "dataset": "KADID-10k",
                        "id": row["dist_img"],
                        "parent_image": row["ref_img"],
                        "group_id": "kadid:" + row["ref_img"],
                        "type": "synthetic_distortion_of_photo",
                        "distortion_id": distortion,
                        "distortion_name": DISTORTIONS[distortion - 1],
                        "distortion_level": level,
                        "license": "CC-BY-4.0 (authors' OSF repository)",
                        "primary_url": "https://osf.io/xkqjh/",
                        "archive_url": ARCHIVE,
                        "archive_member": member,
                        "archive_member_crc32": f"{info.CRC:08x}",
                    },
                    "labels": {
                        "quality_mos_1to5": float(row["dmos"]),
                        "native_dmos": row["dmos"],
                        "native_variance": row["var"],
                        "scale": [1, 5],
                        "direction": "higher_is_better",
                        "mapping": "identity",
                        "ratings_per_image": 30,
                        "kind": "human_degradation_category_mean",
                        "low_quality": Decimal(row["dmos"]) < Decimal("2.6"),
                        "sampling_band": row["sampling_band"],
                        "aesthetics_mos": None,
                    },
                }
            )
            print(f"{len(samples)}/160 {row['id']}", flush=True)
    manifest = {
        "track": "quality",
        "benchmark": "kadid-quality-validation-20261011",
        "created_at": existing["created_at"] if existing else datetime.now(UTC).isoformat(),
        "seed": SEED,
        "selection_sha256": sha(selection_path.read_bytes()),
        "source": {
            "dataset": "KADID-10k",
            "hf_repo": REPO,
            "hf_revision": REVISION,
            "archive_sha256_advertised_by_hf": ARCHIVE_SHA256,
            "full_archive_downloaded_or_hash_verified": False,
            "metadata_sha256": METADATA_SHA256,
            "metadata_primary_checksum_url": "https://api.osf.io/v2/files/5eafe61069d3e100cade8671/",
        },
        "delivery": "Unmodified native 512x384 PNG bytes; no resizing or re-encoding",
        "human_view": "Same PNG pixels; human DCR included a pristine reference, withheld from model",
        "low_quality_rule": "quality_mos_1to5 < 2.6; equality is not low quality",
        "split_counts": dict(Counter(s["split"] for s in samples)),
        "sampling_band_counts": dict(Counter(s["labels"]["sampling_band"] for s in samples)),
        "prior_exact_duplicate_check": {"prior_images": 150, "encoded_and_decoded_matches": 0},
        "samples": samples,
    }
    if existing and existing != manifest:
        raise ValueError("Rebuild changed the frozen dataset manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "samples": len(samples),
                "splits": manifest["split_counts"],
                "bands": manifest["sampling_band_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
