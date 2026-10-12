"""Acquire native VISION photographs with device/model-disjoint splits."""

import argparse
import hashlib
import json
import math
import statistics
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/metadata-v3"
RESEARCH = DATA / "research/real"
DESTINATION = DATA / "candidates/real"
SOURCE_LIST = "https://lesc.dinfo.unifi.it/VISION/VISION_files.txt"
RIGHTS_URL = "https://lesc.dinfo.unifi.it/VISION/README.txt"
SEED = "metadata-v3:real:20261012"
CAMERA_TAGS = {"iso": 34855, "exposure_time": 33434, "aperture": 33437, "focal_length": 37386}
COSINES = [
    [math.cos(math.pi * (2 * position + 1) * frequency / 64) for position in range(32)]
    for frequency in range(8)
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def rank(value):
    return sha256((SEED + ":" + value).encode())


def perceptual_hash(image):
    """64-bit low-frequency DCT hash; exclude the DC term from the median."""
    pixels = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS).tobytes()
    horizontal = [
        [sum(pixels[y * 32 + x] * COSINES[u][x] for x in range(32)) for u in range(8)] for y in range(32)
    ]
    coefficients = [
        sum(horizontal[y][u] * COSINES[v][y] for y in range(32)) for v in range(8) for u in range(8)
    ]
    midpoint = statistics.median(coefficients[1:])
    return sum((value > midpoint) << position for position, value in enumerate(coefficients))


def fingerprints(path):
    raw = path.read_bytes()
    with Image.open(path) as original:
        rgb = ImageOps.exif_transpose(original).convert("RGB")
        return {
            "original_sha256": sha256(raw),
            "decoded_rgb_sha256": sha256(rgb.tobytes()),
            "sized_rgb_sha256": sha256(str(rgb.size).encode() + rgb.tobytes()),
            "phash": f"{perceptual_hash(rgb):016x}",
        }


def fetch(url, destination):
    if not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "model-gym metadata research"})
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read(25_000_001)
        if len(raw) > 25_000_000:
            raise ValueError(f"Asset exceeds the 25 MB per-file limit: {url}")
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(raw)
        temporary.replace(destination)
    return destination


def source_devices(urls):
    devices = defaultdict(list)
    for url in urls:
        if "/images/nat/" in url and url.endswith(".jpg"):
            devices[url.split("/dataset/", 1)[1].split("/", 1)[0]].append(url)
    # Repeated devices of the same model cannot appear on opposite sides of a split.
    unique_models = {}
    for device in sorted(devices, key=rank):
        model = device.split("_", 1)[1]
        unique_models.setdefault(model, device)
    chosen = sorted(unique_models.values(), key=rank)[:20]
    assignments = {
        device: "train" if i < 5 else "dev" if i < 10 else "test" for i, device in enumerate(chosen)
    }
    return devices, assignments


def prior_records():
    manifests = sorted((ROOT / "tracks").glob("*/dataset.json"))
    manifests += [
        ROOT / "evidence/samples.json",
        ROOT / "evidence/benchmarks/quality.json",
        ROOT / "evidence/benchmarks/ai.json",
    ]
    records, missing = [], []
    seen = set()
    for manifest in manifests:
        if "metadata_v3" in manifest.parts:
            continue
        for row in json.loads(manifest.read_text()).get("samples", []):
            names = [row.get("image"), row.get("source", {}).get("native_image")]
            for name in names:
                if not name or name in seen:
                    continue
                seen.add(name)
                path = ROOT / name
                if not path.exists():
                    missing.append(name)
                    continue
                try:
                    records.append({"id": row["id"], "path": name, **fingerprints(path)})
                except (OSError, ValueError):
                    missing.append(name)
    return records, missing


def inspect(url):
    path = fetch(url, DESTINATION / url.rsplit("/", 1)[1])
    with Image.open(path) as image:
        exif = image.getexif()
        inner = exif.get_ifd(34665)
        camera = {key: inner.get(tag, exif.get(tag)) for key, tag in CAMERA_TAGS.items()}
        camera = {key: float(value) if value is not None else None for key, value in camera.items()}
        return {
            "id": "vision-" + path.stem,
            "image": str(path.relative_to(ROOT)),
            "label": "real",
            "source": "VISION",
            "source_url": url,
            "group_id": "vision-device:" + url.split("/dataset/", 1)[1].split("/", 1)[0],
            "camera_make": str(exif.get(271, "")).rstrip("\x00"),
            "camera_model": str(exif.get(272, "")).rstrip("\x00"),
            "native_fields": camera,
            "original_size": list(image.size),
            "bytes": path.stat().st_size,
            "license": "CC-BY-SA-4.0",
            "license_url": RIGHTS_URL,
            "credit": "CSP Lab, Department of Information Engineering, University of Florence (2017); Shullani et al., VISION",
            "label_basis": "Original natural-camera JPEG in the authors' VISION native-image directory",
            **fingerprints(path),
        }


def duplicate(row, previous):
    for other in previous:
        for field in ("original_sha256", "decoded_rgb_sha256", "sized_rgb_sha256"):
            if row[field] == other[field]:
                return {"kind": field, "other": other["id"]}
        distance = (int(row["phash"], 16) ^ int(other["phash"], 16)).bit_count()
        if distance <= 6:
            return {"kind": "phash", "distance": distance, "other": other["id"]}
    return None


def acquire(exclusions_path=None, output=None, replacements_path=None):
    RESEARCH.mkdir(parents=True, exist_ok=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    fetch(SOURCE_LIST, RESEARCH / "vision-files.txt")
    fetch(RIGHTS_URL, RESEARCH / "vision-readme.txt")
    devices, assignments = source_devices((RESEARCH / "vision-files.txt").read_text().splitlines())
    replacements_sha256 = None
    if replacements_path:
        replacement_raw = replacements_path.read_bytes()
        replacements_sha256 = sha256(replacement_raw)
        replacements = {
            item["old_device"]: item["new_device"] for item in json.loads(replacement_raw)["replacements"]
        }
        if not set(replacements) <= set(assignments) or any(
            item not in devices for item in replacements.values()
        ):
            raise ValueError("Unknown replacement device")
        replaced = {replacements.get(device, device): split for device, split in assignments.items()}
        if len(replaced) != len(assignments) or len({device.split("_", 1)[1] for device in replaced}) != len(
            replaced
        ):
            raise ValueError("Replacement merges device or model groups")
        assignments = replaced
    previous, missing = prior_records()
    print(f"Prior screening: {len(previous)} assets; {len(missing)} missing", flush=True)
    if missing:
        raise ValueError("Acquire all prior benchmark images before selection; missing " + str(len(missing)))
    flags = {}
    exclusions_sha256 = None
    if exclusions_path:
        exclusions_raw = exclusions_path.read_bytes()
        exclusions_sha256 = sha256(exclusions_raw)
        flags = {
            item["id"]: item
            for item in json.loads(exclusions_raw)["flags"]
            if item["decision"].startswith("replace_")
        }
    selected, excluded = [], []
    for device, split in assignments.items():
        candidates = sorted(devices[device], key=rank)
        eligible_urls = []
        for url in candidates:
            image_id = "vision-" + url.rsplit("/", 1)[1].removesuffix(".jpg")
            if image_id in flags:
                excluded.append(
                    {
                        "id": image_id,
                        "source_url": url,
                        "kind": flags[image_id]["decision"],
                        "reason": flags[image_id]["reason"],
                    }
                )
            else:
                eligible_urls.append(url)
        candidates = eligible_urls
        accepted = 0
        while accepted < 8:
            urls, candidates = candidates[: 8 - accepted], candidates[8 - accepted :]
            if not urls:
                raise ValueError(f"No eligible candidates remain for {device}")
            with ThreadPoolExecutor(max_workers=4) as pool:
                for row in pool.map(inspect, urls):
                    issue = duplicate(row, previous + selected)
                    if issue:
                        excluded.append({"id": row["id"], "source_url": row["source_url"], **issue})
                        continue
                    row["split"] = split
                    selected.append(row)
                    accepted += 1
        print(f"{device}: {split} {accepted} accepted; {len(selected)} total", flush=True)
    result = {
        "version": 2 if exclusions_path or replacements_path else 1,
        "seed": SEED,
        "selection": "Hash-ranked unique camera models; 5 train, 5 dev and 10 test devices; 8 URL-ranked images per device",
        "limitations": "Perceptual screening does not prove semantic scene independence; EXIF fields remain unverified camera claims",
        "source_manifest_sha256": sha256((RESEARCH / "vision-files.txt").read_bytes()),
        "license_evidence_sha256": sha256((RESEARCH / "vision-readme.txt").read_bytes()),
        "device_assignments": assignments,
        "prior_screening": {"assets": len(previous), "missing": missing, "phash_max_distance": 6},
        "excluded": excluded,
        "visual_exclusions": {
            "path": str(exclusions_path.relative_to(ROOT)),
            "sha256": exclusions_sha256,
        }
        if exclusions_path
        else None,
        "device_replacements": {
            "path": str(replacements_path.relative_to(ROOT)),
            "sha256": replacements_sha256,
        }
        if replacements_path
        else None,
        "samples": selected,
        "coverage": {
            "samples": len(selected),
            "bytes": sum(row["bytes"] for row in selected),
            "splits": dict(Counter(row["split"] for row in selected)),
            "camera_complete": sum(
                all(value is not None for value in row["native_fields"].values()) for row in selected
            ),
            "models": len({row["camera_model"] for row in selected}),
        },
    }
    output = output or RESEARCH / "vision-native-160.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["coverage"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclusions", type=Path, help="Visual eligibility JSON; replace flagged candidates")
    parser.add_argument("--output", type=Path, help="New source manifest path; preserve previous snapshots")
    parser.add_argument(
        "--device-replacements", type=Path, help="Reviewed whole-device replacements for location-bound pools"
    )
    args = parser.parse_args()
    acquire(
        args.exclusions.resolve() if args.exclusions else None,
        args.output,
        args.device_replacements.resolve() if args.device_replacements else None,
    )
