"""Select 60 held-out UHD-IQA photos by native MOS, before any model inference."""

import argparse
import csv
import hashlib
import io
import json
import random
import warnings
import zipfile
import zlib
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFile, ImageOps
from remotezip import RemoteZip
from urllib3.exceptions import InsecureRequestWarning

# Pillow needs a larger JPEG encoder buffer for some detailed 512px photos.
ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, 2**20)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/benchmarks/quality"
EVIDENCE = ROOT / "evidence/benchmarks/quality.json"
SOURCE = "https://datasets.vqa.mmsp-kn.de/archives/UHD-IQA/"
ARCHIVE = SOURCE + "UHD-IQA-database.zip"
ETAG = '"67093f54-27d6c14ac"'
ARCHIVE_BYTES = 10_694_169_772
METADATA_SHA256 = "a020225f5972776c66249eac316e0dd5e605e90e9addd5fd4f0ca4df52cd0850"
SEED = 20261011


def sha(data):
    return hashlib.sha256(data).hexdigest()


def select_rows(rows):
    """Take six seeded rows from each native-MOS decile, then mask selection order."""
    rows = sorted(
        (r for r in rows if r["set"] == "test"),
        key=lambda r: (float(r["quality_mos"]), r["image_name"]),
    )
    if len(rows) != 900 or len({r["image_name"] for r in rows}) != 900:
        raise ValueError("Expected the original 900 unique UHD-IQA test images")
    rng = random.Random(SEED)
    selected = []
    for decile in range(10):
        for row in rng.sample(rows[decile * 90 : (decile + 1) * 90], 6):
            selected.append({**row, "sampling_decile": decile + 1})
    rng.shuffle(selected)
    return selected


def deliver(raw):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    original = image.size
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92, subsampling=0, optimize=True)
    return output.getvalue(), original, image.size


def dhash(raw):
    image = Image.open(io.BytesIO(raw)).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(image.get_flattened_data())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return result


def contact_sheets(samples):
    target = DATA / "sheets"
    target.mkdir(exist_ok=True)
    for start in range(0, len(samples), 6):
        sheet = Image.new("RGB", (1024, 3 * 544), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for i, sample in enumerate(samples[start : start + 6]):
            image = Image.open(ROOT / sample["image"])
            x, y = (i % 2) * 512, (i // 2) * 544
            sheet.paste(image, (x + (512 - image.width) // 2, y + 28))
            draw.text((x + 10, y + 8), sample["id"], fill="black")
        sheet.save(target / f"quality-{start:03d}-{start + 5:03d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-expired-source-certificate",
        action="store_true",
        help="Limit a certificate exception to this public, credential-free source archive.",
    )
    args = parser.parse_args()
    verify = not args.allow_expired_source_certificate
    if not verify:
        warnings.simplefilter("ignore", InsecureRequestWarning)
    (DATA / "source").mkdir(parents=True, exist_ok=True)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    headers = requests.head(ARCHIVE, timeout=60, verify=verify)
    headers.raise_for_status()
    if headers.headers.get("ETag") != ETAG or int(headers.headers["Content-Length"]) != ARCHIVE_BYTES:
        raise ValueError("The pinned source archive changed; review it before selecting new media")
    metadata_path = DATA / "source/uhd-iqa-metadata.zip"
    if not metadata_path.exists():
        response = requests.get(SOURCE + metadata_path.name, timeout=60, verify=verify)
        response.raise_for_status()
        metadata_path.write_bytes(response.content)
    metadata_zip = metadata_path.read_bytes()
    if sha(metadata_zip) != METADATA_SHA256:
        raise ValueError("The frozen source metadata changed")
    with zipfile.ZipFile(io.BytesIO(metadata_zip)) as metadata_archive:
        metadata = metadata_archive.read("uhd-iqa-metadata.csv")
    (DATA / "source/uhd-iqa-metadata.csv").write_bytes(metadata)
    selected = select_rows(list(csv.DictReader(io.StringIO(metadata.decode()))))
    prior = json.loads((ROOT / "evidence/samples.json").read_text())["samples"]
    prior_hashes = {s["delivered_sha256"] for s in prior}
    prior_dhashes = [dhash((ROOT / s["image"]).read_bytes()) for s in prior]
    expected = {}
    if EVIDENCE.exists():
        expected = {s["id"]: s for s in json.loads(EVIDENCE.read_text())["samples"]}
    samples = []
    seen_hashes = set()
    near_pairs = []
    with RemoteZip(ARCHIVE, timeout=90, verify=verify) as archive:
        for i, row in enumerate(selected):
            sample_id = f"quality-{i:03d}"
            member = f"UHD-IQA-database/test/{row['image_name']}"
            source_path = DATA / "source" / row["image_name"]
            raw = source_path.read_bytes() if source_path.exists() else archive.read(member)
            frozen = expected.get(sample_id)
            if frozen and sha(raw) != frozen["original_sha256"]:
                raise ValueError(f"Frozen source hash mismatch: {sample_id}")
            info = archive.getinfo(member)
            if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
                raise ValueError(f"Source size mismatch: {member}")
            delivered, original_size, delivered_size = deliver(raw)
            digest = sha(delivered)
            if frozen and digest != frozen["delivered_sha256"]:
                raise ValueError(f"Frozen delivered hash mismatch: {sample_id}")
            if not source_path.exists():
                source_path.write_bytes(raw)
            if digest in prior_hashes or digest in seen_hashes:
                raise ValueError(f"Exact duplicate selected: {sample_id}")
            seen_hashes.add(digest)
            fingerprint = dhash(delivered)
            for j, value in enumerate(prior_dhashes):
                if (fingerprint ^ value).bit_count() <= 2:
                    near_pairs.append({"image": sample_id, "other": prior[j]["id"], "kind": "prior"})
            for other in samples:
                if (fingerprint ^ int(other["dhash"], 16)).bit_count() <= 2:
                    near_pairs.append({"image": sample_id, "other": other["id"], "kind": "within"})
            relative_path = f"data/benchmarks/quality/{sample_id}.jpg"
            (ROOT / relative_path).write_bytes(delivered)
            mos = float(row["quality_mos"])
            samples.append(
                {
                    "id": sample_id,
                    "image": relative_path,
                    "original_sha256": sha(raw),
                    "delivered_sha256": digest,
                    "original_size": original_size,
                    "delivered_size": delivered_size,
                    "delivered_bytes": len(delivered),
                    "dhash": f"{fingerprint:016x}",
                    "source": {
                        "dataset": "UHD-IQA",
                        "id": row["image_name"],
                        "group_id": f"uhd-iqa:{row['image_name']}",
                        "split": "test",
                        "source_url": "https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html",
                        "archive_url": ARCHIVE,
                        "archive_etag": ETAG,
                        "archive_member": member,
                        "archive_member_crc32": f"{info.CRC:08x}",
                        "page_url": row["page_url"],
                        "author": row["user_name"],
                        "license": "CC0-1.0 (photos; primary dataset page)",
                        "source_original_size": [int(row["orig_width"]), int(row["orig_height"])],
                        "subset": row["subset"],
                    },
                    "labels": {
                        "technical_quality": {
                            "value": mos,
                            "scale": [0.01, 1],
                            "normalized_1_to_5": 1 + 4 * (mos - 0.01) / 0.99,
                            "kind": "human_mos",
                            "caveat": "Human image quality at 3840px; delivered 512px; affine normalized, not calibrated technical-quality gold",
                        },
                        "quality_mos": mos,
                        "quality_mos_native_scale": [0.01, 1.0],
                        "quality_mos_1_5": 1 + 4 * (mos - 0.01) / 0.99,
                        "quality_mos_1_5_mapping": "1 + 4 * (native_mos - 0.01) / 0.99; affine, not fitted",
                        "quality_ratings_per_image": 20,
                        "quality_raters": 10,
                        "quality_reference_width": 3840,
                        "aesthetics_mos": None,
                        "sampling_decile": row["sampling_decile"],
                    },
                }
            )
            print(f"Prepared {sample_id}", flush=True)
    output = {
        "version": 1,
        "benchmark": "quality",
        "dataset": "UHD-IQA",
        "selection_seed": SEED,
        "selection": "Official test only; six seeded samples from each MOS decile; shuffle IDs; no model output used",
        "delivery": "EXIF transpose, RGB, full frame, max edge 512, no upscaling, JPEG92 subsampling0 optimize",
        "metadata_zip_sha256": sha(metadata_zip),
        "metadata_csv_sha256": sha(metadata),
        "archive": {"url": ARCHIVE, "etag": ETAG, "bytes": ARCHIVE_BYTES},
        "source_certificate_verified": verify,
        "source_certificate_note": "Official archive server certificate expired on acquisition; request-only exception, no credentials",
        "near_duplicate_candidates_dhash_distance_lte2": near_pairs,
        "limitations": [
            "Human MOS was collected at width3840; delivered512 judgments are a different view",
            "Native human aesthetic scores are unavailable; assistant ratings are a separate model reference",
            "Dataset favors high-aesthetic stock photos; low-MOS strata are relative to this curated population",
            "No exact matches to prior30; dHash is only a candidate screen, not proof of independent scenes",
        ],
        "samples": samples,
    }
    if EVIDENCE.exists():
        previous = json.loads(EVIDENCE.read_text())
        for old, new in zip(previous["samples"], samples, strict=True):
            if any(old[key] != new[key] for key in ("id", "original_sha256", "delivered_sha256")):
                raise ValueError("Reproduction differs from the frozen manifest")
    EVIDENCE.write_text(json.dumps(output, indent=2) + "\n")
    contact_sheets(samples)
    print(f"Prepared {len(samples)} images and {len(near_pairs)} near-duplicate candidates")


if __name__ == "__main__":
    main()
