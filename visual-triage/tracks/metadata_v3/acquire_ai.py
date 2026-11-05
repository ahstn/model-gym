"""Acquire unchanged Synthbuster member bytes with bounded archive range reads."""

import argparse
import csv
import hashlib
import io
import json
import re
import struct
import time
import unicodedata
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/metadata-v3"
CACHE = DATA / "research/ai"
IMAGES = DATA / "candidates/ai"
URL = "https://zenodo.org/api/records/10066460/files/synthbuster.zip/content"
SEED = "20261012-native-ai-v3"
GENERATORS = ("firefly", "dalle2", "dalle3", "midjourney-v5")
PRIOR = (
    "evidence/samples.json",
    "evidence/benchmarks/ai.json",
    "tracks/ai_detection/dataset.json",
    "tracks/ai_detection/eligibility.json",
    "tracks/ai_detection/evidence/acquisition-v1-not-used.json",
    "tracks/ai_detection_v2/dataset.json",
    "tracks/ai_detection_v2/eligibility.json",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def get_range(start, end):
    for attempt in range(8):
        response = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
        if response.status_code in (429, 500, 502, 503, 504):
            retry = response.headers.get("Retry-After", "60")
            time.sleep(min(60, float(retry) if retry.isdigit() else 60))
            continue
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != end - start + 1:
            raise ValueError("Original archive range response mismatch")
        return response.content
    raise ValueError("Archive retry budget exhausted")


class RemoteArchive(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.size = 12372557226

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size - 1 if n < 0 else min(self.size - 1, self.pos + n - 1)
        if end < self.pos:
            return b""
        start = self.pos
        self.pos = end + 1
        return get_range(start, end)


def catalogue():
    path = CACHE / "archive-index.json"
    if path.exists():
        return json.loads(path.read_text())
    with zipfile.ZipFile(RemoteArchive()) as archive:
        result = {
            i.filename: {
                "offset": i.header_offset,
                "compressed_bytes": i.compress_size,
                "native_bytes": i.file_size,
                "compression": i.compress_type,
                "crc32": i.CRC,
            }
            for i in archive.infolist()
            if not i.is_dir()
        }
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def member_bytes(name, info):
    # Local filename and extra fields are parsed; no image is decoded or re-encoded.
    end = info["offset"] + 30 + len(name.encode()) + info["compressed_bytes"] + 4096 - 1
    raw = get_range(info["offset"], min(12372557225, end))
    header = struct.unpack("<4s5H3I2H", raw[:30])
    if header[0] != b"PK\x03\x04" or header[3] != info["compression"]:
        raise ValueError("ZIP local header conflict")
    start = 30 + header[-2] + header[-1]
    payload = raw[start : start + info["compressed_bytes"]]
    output = zlib.decompress(payload, -15) if info["compression"] == 8 else payload
    if len(output) != info["native_bytes"] or zlib.crc32(output) != info["crc32"]:
        raise ValueError("Native member CRC or length mismatch")
    return output


def prompt_table(index):
    path = CACHE / "prompts.csv"
    if not path.exists():
        path.write_bytes(member_bytes("synthbuster/prompts.csv", index["synthbuster/prompts.csv"]))
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    result = {}
    for row in rows:
        identifier = row["image name (matching Raise-1k)"]
        canonical = " ".join(unicodedata.normalize("NFKC", row["Prompt"]).casefold().split())
        result[identifier] = {
            "prompt_sha256": sha(canonical.encode()),
            "prompt_raw_sha256": sha(row["Prompt"].encode()),
        }
    return result


def prior_ids():
    identifiers = set()
    for relative in PRIOR:
        path = ROOT / relative
        if path.exists():
            identifiers.update(re.findall(r"r[0-9a-f]{8}t", path.read_text()))
    return identifiers


def candidates(index, prompts, decisions):
    excluded = prior_ids()
    previous_prompt_hashes = {prompts[i]["prompt_sha256"] for i in excluded if i in prompts}
    complete = [
        i
        for i in prompts
        if i not in excluded
        and prompts[i]["prompt_sha256"] not in previous_prompt_hashes
        and all(f"synthbuster/{gen}/{i}.png" in index for gen in GENERATORS)
    ]
    ranked = sorted(complete, key=lambda i: sha(f"{SEED}:{i}".encode()))
    staged = CACHE / "candidates.json"
    if staged.exists():
        families = [x["image_id"] for x in json.loads(staged.read_text())["samples"][:40]]
    else:
        used = set()
        prompt_used = set()
        families = []
        for slot in range(40):
            identifier = next(
                i for i in ranked if i not in used and prompts[i]["prompt_sha256"] not in prompt_used
            )
            used.add(identifier)
            prompt_used.add(prompts[identifier]["prompt_sha256"])
            families.append(identifier)
    used_ids = set(families)
    used_prompts = {prompts[i]["prompt_sha256"] for i in families}
    used_ids.update(key.removeprefix("family:") for key in decisions)
    used_prompts.update(prompts[i]["prompt_sha256"] for i in used_ids if i in prompts)
    for slot, identifier in enumerate(families):
        if decisions.get(f"family:{identifier}") is not False:
            continue
        replacement = next(
            i
            for i in ranked
            if i not in used_ids
            and prompts[i]["prompt_sha256"] not in used_prompts
            and decisions.get(f"family:{i}") is not False
        )
        used_ids.add(replacement)
        used_prompts.add(prompts[replacement]["prompt_sha256"])
        families[slot] = replacement
    return [
        {
            "image_id": identifier,
            "generator": gen,
            "split": "train" if slot < 10 else "dev" if slot < 20 else "test",
            "family_slot": slot,
            "slot": gen_index * 40 + slot,
        }
        for gen_index, gen in enumerate(GENERATORS)
        for slot, identifier in enumerate(families)
    ]


def acquire(entry, index, prompts):
    name = f"synthbuster/{entry['generator']}/{entry['image_id']}.png"
    suffix = ".jpg" if entry["generator"] == "firefly" else ".png"
    cache = IMAGES / f"{entry['generator']}-{entry['image_id']}{suffix}"
    if not cache.exists():
        cache.write_bytes(member_bytes(name, index[name]))
    raw = cache.read_bytes()
    if zlib.crc32(raw) != index[name]["crc32"]:
        raise ValueError("Cached native CRC mismatch")
    with Image.open(io.BytesIO(raw)) as image:
        decoded = ImageOps.exif_transpose(image).convert("RGB")
        exif = image.getexif()
        tag_ids = set(exif.keys())
        sub = exif.get_ifd(34665) if 34665 in exif else {}
        tag_ids.update(sub)
        camera = {
            field: True
            for field, tag in [
                ("iso", 34855),
                ("exposure_time", 33434),
                ("f_number", 33437),
                ("focal_length", 37386),
            ]
            if tag in tag_ids
        }
        record = {
            **entry,
            "file": str(cache.relative_to(ROOT)),
            "group_id": f"raise1k:{entry['image_id']}",
            "original_sha256": sha(raw),
            "decoded_rgb_sha256": sha(str(decoded.size).encode() + decoded.tobytes()),
            "native_format": image.format,
            "native_dimensions": list(image.size),
            "native_bytes": len(raw),
            "exif_present": bool(exif),
            "camera_fields_present": sorted(camera),
            "c2pa_marker_present": b"c2pa" in raw.lower(),
            "jumbf_marker_present": b"jumb" in raw.lower(),
            "xmp_marker_present": b"xmp" in raw.lower(),
            "archive_member": name,
            "archive_url": URL,
            "zenodo_record": "10066460",
            "member_crc32": index[name]["crc32"],
            **prompts[entry["image_id"]],
            "labels": {"ai_generated": True, "basis": "Original published generator directory provenance"},
            "license": {
                "id": "CC-BY-NC-SA-4.0",
                "url": "https://zenodo.org/records/10066460",
                "attribution": "Quentin Bammey, Synthbuster (2023)",
            },
            "native_scope": "Unchanged original author ZIP member bytes, verified by ZIP CRC. Author readme states no resampling or recompression; no provider-export byte comparison was available.",
        }
    return record


def contact_sheets(records):
    for start in range(0, len(records), 20):
        sheet = Image.new("RGB", (1000, 5 * 220), "white")
        draw = ImageDraw.Draw(sheet)
        for n, entry in enumerate(records[start : start + 20]):
            with Image.open(ROOT / entry["file"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((240, 190))
                x = (n % 4) * 250
                y = (n // 4) * 220
                sheet.paste(image, (x, y + 25))
                draw.text((x, y), f"{start + n} {entry['generator']}", fill="black")
        sheet.save(IMAGES / f"contact-{start:03d}.jpg")


def verify(records, decisions):
    prior_hashes = set()
    for relative in PRIOR:
        path = ROOT / relative
        if path.exists():

            def collect(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in ("original_sha256", "delivered_sha256", "decoded_rgb_sha256"):
                            prior_hashes.add(item)
                        collect(item)
                elif isinstance(value, list):
                    for item in value:
                        collect(item)

            collect(json.loads(path.read_text()))
    families = {}
    hashes = set()
    pixels = set()
    for record in records:
        raw = (ROOT / record["file"]).read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            decoded = ImageOps.exif_transpose(image).convert("RGB")
            pixel = sha(str(decoded.size).encode() + decoded.tobytes())
        if sha(raw) != record["original_sha256"] or pixel != record["decoded_rgb_sha256"]:
            raise ValueError("Native hash mismatch")
        if zlib.crc32(raw) != record["member_crc32"]:
            raise ValueError("Native CRC mismatch")
        if sha(raw) in prior_hashes | hashes or pixel in prior_hashes | pixels:
            raise ValueError("Exact native or decoded duplicate")
        if record["image_id"] in prior_ids():
            raise ValueError("Prior source ID reused")
        if decisions.get(f"family:{record['image_id']}") is not True:
            raise ValueError("Family medium review incomplete")
        hashes.add(sha(raw))
        pixels.add(pixel)
        families.setdefault(record["image_id"], []).append(record)
    if len(records) != 160 or len(families) != 40:
        raise ValueError("Expected160images/40families")
    prompt_splits = {}
    for family in families.values():
        if (
            len(family) != 4
            or {x["generator"] for x in family} != set(GENERATORS)
            or len({x["split"] for x in family}) != 1
        ):
            raise ValueError("Generator balance or family split leakage")
        for row in family:
            prompt_splits.setdefault(row["prompt_sha256"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in prompt_splits.values()):
        raise ValueError("Prompt split leakage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    index = catalogue()
    prompts = prompt_table(index)
    decisions_path = CACHE / "eligibility.json"
    decisions = json.loads(decisions_path.read_text())["decisions"] if decisions_path.exists() else {}
    selected = candidates(index, prompts, decisions)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda entry: acquire(entry, index, prompts), selected))
    pending = {
        "version": 1,
        "seed": SEED,
        "prior_source_ids_excluded": len(prior_ids()),
        "prompt_group_rule": "40 source families with four generator variants each; source IDs and canonical prompt hashes stay within one split",
        "source": "Original Synthbuster Zenodo10066460",
        "samples": records,
    }
    (CACHE / "candidates.json").write_text(json.dumps(pending, indent=2) + "\n")
    contact_sheets(records)
    if not args.prepare:
        if any(decisions.get(f"family:{x['image_id']}") is not True for x in records):
            raise ValueError("Medium eligibility not complete")
        verify(records, decisions)
        target = CACHE / "source.json"
        pending["eligibility_sha256"] = sha(decisions_path.read_bytes())
        if target.exists() and json.loads(target.read_text()) != pending:
            raise ValueError("Frozen source panel differs")
        target.write_text(json.dumps(pending, indent=2) + "\n")
    print(f"Prepared {len(records)} native AI images; {len(prior_ids())} prior source IDs excluded")


if __name__ == "__main__":
    main()
