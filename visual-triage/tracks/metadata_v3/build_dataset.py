"""Assemble a frozen, source-balanced native metadata cohort without API calls."""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_protocol import digest
from evidence_archive import require_restored
from tracks.metadata_v3 import acquire_real, features, provenance

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/metadata-v3"
TRACK = ROOT / "tracks/metadata_v3"
EXPECTED = {"train": 80, "dev": 80, "test": 160}
DELIVERY = "EXIF transpose, RGB conversion, centred ImageOps.fit crop to 512x512 with LANCZOS; JPEG quality95, subsampling0, no source metadata copied"


def file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def _path(value):
    path = Path(value)
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("Cohort paths must remain within visual-triage")
    return resolved


def _relative(path):
    return str(_path(path).relative_to(ROOT))


def _write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        # Compact records reduce frozen-evidence line churn.
        handle.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def _normalise(real, ai):
    rows = []
    for item in real["samples"]:
        rows.append(
            {
                "id": item["id"],
                "split": item["split"],
                "group_id": item["group_id"],
                "native_image": item["image"],
                "native_sha256": item["original_sha256"],
                "gold": False,
                "source": {
                    "dataset": "VISION",
                    "id": item["id"],
                    "camera_group": item["group_id"],
                    "generator": None,
                    "license": item["license"],
                    "url": item["source_url"],
                    "license_url": item.get("license_url"),
                    "credit": item.get("credit"),
                },
                "input_fingerprints": {
                    key: item.get(key) for key in ("decoded_rgb_sha256", "sized_rgb_sha256", "phash")
                },
            }
        )
    families = defaultdict(list)
    prompts = {}
    for item in ai["samples"]:
        family = item["group_id"]
        split = item["split"]
        if family in prompts and prompts[family] != (split, item["prompt_sha256"]):
            raise ValueError("Native AI family crosses splits or prompt identities")
        prompts[family] = (split, item["prompt_sha256"])
        families[(split, family)].append(item)
    grouped = {}
    for split in EXPECTED:
        names = sorted(family for owner, family in families if owner == split)
        if len(names) % 2:
            raise ValueError("AI grouping requires two whole prompt families per group")
        for index, family in enumerate(names):
            if len(families[(split, family)]) != 4:
                raise ValueError("AI family must contain four distinct generator variants")
            if len({item["generator"] for item in families[(split, family)]}) != 4:
                raise ValueError("Duplicate generator in native AI family")
            grouped[(split, family)] = f"ai-family-bundle:{split}:{index // 2:03d}"
    for (split, family), items in sorted(families.items()):
        for item in sorted(items, key=lambda value: value["generator"]):
            rows.append(
                {
                    "id": f"ai-{item['generator']}-{item['image_id']}",
                    "split": split,
                    "group_id": grouped[(split, family)],
                    "native_image": item["file"],
                    "native_sha256": item["original_sha256"],
                    "gold": True,
                    "source": {
                        "dataset": "Synthbuster",
                        "id": item["image_id"],
                        "camera_group": None,
                        "generator": item["generator"],
                        "license": item["license"],
                        "url": item["archive_url"],
                        "archive_member": item["archive_member"],
                        "native_family_id": family,
                        "prompt_sha256": item["prompt_sha256"],
                    },
                    "input_fingerprints": {"sized_rgb_sha256": item["decoded_rgb_sha256"]},
                }
            )
    _validate_cohort(rows)
    return sorted(
        rows,
        key=lambda value: (
            ("train", "dev", "test").index(value["split"]),
            value["gold"],
            value["group_id"],
            value["id"],
        ),
    )


def _validate_cohort(rows):
    if any(
        not isinstance(row["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,150}", row["id"])
        for row in rows
    ):
        raise ValueError("Cohort IDs must be safe bounded filename identifiers")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Cohort IDs must be unique")
    if Counter(row["split"] for row in rows) != EXPECTED:
        raise ValueError("Cohort requires 80 train, 80 dev and 160 test images")
    groups, prompt_owners = defaultdict(list), {}
    for row in rows:
        groups[row["group_id"]].append(row)
        prompt = row["source"].get("prompt_sha256")
        if prompt is not None:
            if prompt in prompt_owners and prompt_owners[prompt] != row["split"]:
                raise ValueError("Canonical AI prompt crosses splits")
            prompt_owners[prompt] = row["split"]
    for members in groups.values():
        if len(members) != 8 or len({(row["split"], row["gold"]) for row in members}) != 1:
            raise ValueError("Every GEPA group requires eight images in one split and class")
    for split, count in EXPECTED.items():
        if Counter(row["gold"] for row in rows if row["split"] == split) != {
            False: count // 2,
            True: count // 2,
        }:
            raise ValueError("Each split must have equal real and generated class sizes")


def _fingerprints(path):
    native_sha = file_sha(path)
    helper_sha = file_sha(acquire_real.__file__)
    cache_key = digest({"native_sha256": native_sha, "helper_sha256": helper_sha})
    cache = DATA / "cache/fingerprints" / f"{cache_key}.json"
    if cache.exists():
        record = json.loads(cache.read_bytes())
        value = record["fingerprints"]
        if (
            value["original_sha256"] != native_sha
            or record["helper_sha256"] != helper_sha
            or record["fingerprints_sha256"] != digest(value)
        ):
            raise ValueError("Cached native fingerprint hash changed")
        return value
    value = acquire_real.fingerprints(path)
    _write_once(
        cache, {"fingerprints": value, "fingerprints_sha256": digest(value), "helper_sha256": helper_sha}
    )
    return value


def _prior_fingerprints():
    """Read every locally restored prior asset; never treat missing evidence as empty."""
    require_restored(ROOT)
    manifests = sorted((ROOT / "tracks").glob("*/dataset.json"))
    manifests += [
        ROOT / "evidence/samples.json",
        ROOT / "evidence/benchmarks/quality.json",
        ROOT / "evidence/benchmarks/ai.json",
    ]
    records, seen = [], set()
    for manifest in manifests:
        if "metadata_v3" in manifest.parts:
            continue
        if not manifest.exists():
            raise ValueError(f"Prior manifest is absent; restore evidence before assembling: {manifest}")
        for sample in json.loads(manifest.read_bytes()).get("samples", []):
            for name in (sample.get("image"), sample.get("source", {}).get("native_image")):
                if not name or name in seen:
                    continue
                seen.add(name)
                path = _path(name)
                if not path.exists():
                    raise ValueError(f"Prior image is absent; restore/acquire it before assembling: {name}")
                records.append({"id": sample["id"], "path": name, **_fingerprints(path)})
    return records


def _check_collisions(rows, fingerprints, priors):
    seen = []
    for row in rows:
        current = {"id": row["id"], **fingerprints[row["id"]]}
        issue = acquire_real.duplicate(current, priors)
        if issue:
            raise ValueError(f"Prior-image collision; contact the experiment owner: {row['id']} {issue}")
        for previous in seen:
            exact = any(
                current[key] == previous[key]
                for key in ("original_sha256", "decoded_rgb_sha256", "sized_rgb_sha256")
            )
            distance = (int(current["phash"], 16) ^ int(previous["phash"], 16)).bit_count()
            if exact or (row["split"] != previous["split"] and distance <= 6):
                raise ValueError(
                    f"Exact or cross-split perceptual collision; contact the experiment owner: {row['id']} vs {previous['id']} (distance {distance})"
                )
        seen.append({**current, "split": row["split"]})
    return {
        "prior_assets": len(priors),
        "cohort_images": len(rows),
        "exact_rgb_duplicates": 0,
        "cross_split_phash_distance_at_most_6": 0,
        "perceptual_method": "64-bit DCT hash from acquire_real; threshold6; not semantic scene proof",
    }


def _deliver(native, destination):
    with Image.open(native) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.fit(image, (512, 512), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    if destination.exists():
        # Produce bytes again to verify the existing cache under this exact policy.
        import io

        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=95, subsampling=0)
        if destination.read_bytes() != stream.getvalue():
            raise ValueError("Delivered cache differs from the fixed policy")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            image.save(handle, format="JPEG", quality=95, subsampling=0)
    with Image.open(destination) as checked:
        if checked.size != (512, 512) or checked.getexif():
            raise ValueError("Delivered JPEG dimensions or metadata differ")


def _measure(row, helper_hashes):
    started = perf_counter()
    native = _path(row["native_image"])
    delivered = DATA / "delivered" / f"{row['id']}.jpg"
    _deliver(native, delivered)
    delivered_sha = file_sha(delivered)
    cache_key = digest({"native": row["native_sha256"], "delivered": delivered_sha, **helper_hashes})
    cache = DATA / "cache/features" / f"{cache_key}.json"
    if cache.exists():
        measured = json.loads(cache.read_bytes())
        features.validate_features(measured["features"])
        if digest(measured["features"]) != measured["features_sha256"]:
            raise ValueError("Frozen feature cache hash changed")
    else:
        value = features.extract_features(native, delivered, provenance.extract_c2pa(native))
        measured = {
            "features": value,
            "features_sha256": digest(value),
            "preprocess_seconds": perf_counter() - started,
        }
        _write_once(cache, measured)
    return {key: value for key, value in row.items() if key != "input_fingerprints"} | {
        "image": _relative(delivered),
        "delivered_sha256": delivered_sha,
        **measured,
    }


def assemble(real_manifest, ai_manifest, output=None, *, staging=False):
    """Build after explicit medium review; staging can inspect unreviewed candidates only."""
    real_path, ai_path = _path(real_manifest), _path(ai_manifest)
    real, ai = json.loads(real_path.read_bytes()), json.loads(ai_path.read_bytes())
    if not staging:
        for name in ("visual_exclusions", "device_replacements"):
            if name in real:
                receipt = real[name]
                if file_sha(_path(receipt["path"])) != receipt["sha256"]:
                    raise ValueError(f"Frozen real-source review receipt changed: {name}")
        eligibility = ai_path.parent / "eligibility.json"
        if not ai.get("eligibility_sha256") or file_sha(eligibility) != ai["eligibility_sha256"]:
            raise ValueError("Final assembly requires a frozen manual AI eligibility record")
        decisions = json.loads(eligibility.read_bytes())["decisions"]
        if any(decisions.get("family:" + item["image_id"]) is not True for item in ai["samples"]):
            raise ValueError("AI family lacks manual photographic eligibility")
    target = _path(output or (DATA / "assembled.json" if staging else TRACK / "dataset.json"))
    if target.exists():
        raise ValueError("Cohort manifest already exists; preserve frozen evidence")
    rows = _normalise(real, ai)
    fingerprints = {}
    for row in rows:
        native = _path(row["native_image"])
        if file_sha(native) != row["native_sha256"]:
            raise ValueError(f"Native source SHA256 mismatch: {row['id']}")
        fingerprint = _fingerprints(native)
        for key, expected in row["input_fingerprints"].items():
            if expected is not None and fingerprint[key] != expected:
                raise ValueError(f"Source RGB or perceptual fingerprint changed: {row['id']} {key}")
        fingerprints[row["id"]] = fingerprint
    priors = _prior_fingerprints()
    audit = _check_collisions(rows, fingerprints, priors)
    helper_hashes = {
        "feature_helper_sha256": file_sha(features.__file__),
        "provenance_helper_sha256": file_sha(provenance.__file__),
    }
    samples = [_measure(row, helper_hashes) for row in rows]
    delivered_fingerprints = {sample["id"]: _fingerprints(_path(sample["image"])) for sample in samples}
    delivered_audit = _check_collisions(samples, delivered_fingerprints, priors)
    audit["delivered_views"] = delivered_audit
    statuses = Counter((sample["gold"], sample["features"]["camera"]["status"]) for sample in samples)
    result = {
        "version": 3,
        "track": "metadata_v3",
        "created_at": datetime.now(UTC).isoformat(),
        "manual_eligibility_complete": not staging,
        "delivery": DELIVERY,
        "grouping": "Eight images per real device; pair lexically sorted AI prompt families within each split into eight-image groups; native family IDs remain in source audit fields",
        "source_manifests": [
            {"path": _relative(path), "sha256": file_sha(path)} for path in (real_path, ai_path)
        ],
        **helper_hashes,
        "native_pixel_limit": features.NATIVE_MAX_PIXELS_V3,
        "collision_audit": audit,
        "extraction_status": [
            {"gold": gold, "camera_status": status, "count": count}
            for (gold, status), count in sorted(statuses.items())
        ],
        "samples": samples,
    }
    _write_once(target, result)
    return {
        "path": _relative(target),
        "sha256": file_sha(target),
        "samples": len(samples),
        "extraction_status": result["extraction_status"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=DATA / "research/real/vision-native-160.json")
    parser.add_argument("--ai", type=Path, default=DATA / "research/ai/source.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--staging", action="store_true")
    args = parser.parse_args()
    print(json.dumps(assemble(args.real, args.ai, args.output, staging=args.staging), indent=2))


if __name__ == "__main__":
    main()
