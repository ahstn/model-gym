"""Build explicit test-only metadata interventions without changing native gold."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image

from bench_protocol import digest
from tracks.metadata_v3.features import validate_features
from tracks.metadata_v3.provenance import MAX_INPUT_BYTES, SDK_POLICY, extract_c2pa

VT_ROOT = Path(__file__).resolve().parents[2]
DONOR_SEED = "metadata-v3-camera-donors-v1"


def _segments(data: bytes) -> tuple[list[tuple[int, int, int]], int]:
    """Return JPEG header segments and the untouched scan-data offset."""
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("C2PA byte controls require native JPEG bytes")
    segments, position = [], 2
    while position + 4 <= len(data):
        if data[position] != 255:
            raise ValueError("Invalid JPEG segment")
        marker = data[position + 1]
        if marker == 218:
            return segments, position
        length = int.from_bytes(data[position + 2 : position + 4], "big")
        end = position + length + 2
        if length < 2 or end > len(data):
            raise ValueError("Invalid JPEG segment length")
        segments.append((marker, position, end))
        position = end
    raise ValueError("JPEG scan not found")


def _credential_segments(data: bytes) -> list[tuple[int, int]]:
    segments, _ = _segments(data)
    selected = []
    for marker, start, end in segments:
        payload = data[start + 4 : end]
        if marker == 235 and len(payload) >= 16 and payload[:2] == b"JP" and payload[12:16] == b"jumb":
            selected.append((start, end))
    if not selected:
        raise ValueError("Embedded JPEG JUMBF segments not found")
    return selected


def strip_credentials(data: bytes) -> bytes:
    """Remove the identified JPEG JUMBF header segments; preserve image coding."""
    selected = set(_credential_segments(data))
    segments, scan = _segments(data)
    return (
        b"\xff\xd8"
        + b"".join(data[start:end] for _, start, end in segments if (start, end) not in selected)
        + data[scan:]
    )


def copy_credentials(asset: bytes, donor: bytes) -> bytes:
    """Copy donor JUMBF bytes onto another asset as an invalid-binding control."""
    credentials = b"".join(donor[start:end] for start, end in _credential_segments(donor))
    return b"\xff\xd8" + credentials + strip_credentials(asset)[2:]


def _cbor_header(data: bytes, position: int) -> tuple[int, int, int]:
    if position >= len(data):
        raise ValueError("Truncated CBOR")
    code = data[position]
    major, extra, position = code >> 5, code & 31, position + 1
    if extra < 24:
        return major, extra, position
    size = {24: 1, 25: 2, 26: 4, 27: 8}.get(extra)
    if size is None or position + size > len(data):
        raise ValueError("Unsupported or truncated CBOR length")
    return major, int.from_bytes(data[position : position + size], "big"), position + size


def _skip_cbor(data: bytes, position: int, depth: int = 0) -> int:
    if depth > 64:
        raise ValueError("CBOR nesting limit")
    major, length, position = _cbor_header(data, position)
    if major in (0, 1, 7):
        return position
    if major in (2, 3):
        if position + length > len(data):
            raise ValueError("Truncated CBOR string")
        return position + length
    if major == 6:
        return _skip_cbor(data, position, depth + 1)
    if length > len(data) or major not in (4, 5):
        raise ValueError("Invalid CBOR collection")
    for _ in range(length * (2 if major == 5 else 1)):
        position = _skip_cbor(data, position, depth + 1)
    return position


def tamper_signature(data: bytes) -> bytes:
    """Flip one COSE signature byte, including across fragmented APP11 storage."""
    parts, byte_offsets, first_header = [], [], None
    instance = None
    for start, end in _credential_segments(data):
        payload = data[start + 4 : end]
        if instance is None:
            instance = payload[:4]
            first_header = payload[8:16]
            offset = start + 12
        else:
            if payload[:4] != instance or payload[8:16] != first_header:
                raise ValueError("Multiple JUMBF instances are unsupported for signature controls")
            offset = start + 20
        parts.append(data[offset:end])
        byte_offsets.extend(range(offset, end))
    joined = b"".join(parts)
    label = b"c2pa.signature\x00"
    position = joined.rfind(label)
    if position < 0:
        raise ValueError("C2PA signature box not found")
    position += len(label)
    if joined[position + 4 : position + 8] != b"cbor":
        raise ValueError("C2PA signature is not a CBOR box")
    major, tag, position = _cbor_header(joined, position + 8)
    if major != 6 or tag != 18:
        raise ValueError("Expected tagged COSE Sign1")
    major, count, position = _cbor_header(joined, position)
    if major != 4 or count != 4:
        raise ValueError("Invalid COSE Sign1")
    for _ in range(3):
        position = _skip_cbor(joined, position)
    major, length, position = _cbor_header(joined, position)
    if major != 2 or length not in (64, 96, 128, 256, 384, 512) or position + length > len(joined):
        raise ValueError("Unsupported COSE signature byte string")
    changed = bytearray(data)
    changed[byte_offsets[position]] ^= 1
    return bytes(changed)


def _file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes the visual-triage root")
    return path


def _check_sample(root: Path, sample: dict) -> None:
    if type(sample["gold"]) not in (bool, int) or sample["gold"] not in (0, 1):
        raise ValueError("Control gold must be an integer or boolean binary label")
    validate_features(sample["features"])
    if digest(sample["features"]) != sample["features_sha256"]:
        raise ValueError("Frozen feature hash mismatch")
    for path_key, hash_key in (("image", "delivered_sha256"), ("native_image", "native_sha256")):
        if hashlib.sha256(_file(root, sample[path_key]).read_bytes()).hexdigest() != sample[hash_key]:
            raise ValueError("Frozen image hash mismatch")


def _match_donors(recipients: list[dict], pool: list[dict]) -> dict[int, int]:
    assigned = {}

    def assign(recipient_index: int, visited: set[int]) -> bool:
        recipient = recipients[recipient_index]
        for donor_index, donor in enumerate(pool):
            if donor_index in visited or donor["group_id"] == recipient["group_id"]:
                continue
            visited.add(donor_index)
            previous = assigned.get(donor_index)
            if previous is None or assign(previous, visited):
                assigned[donor_index] = recipient_index
                return True
        return False

    for index in range(len(recipients)):
        if not assign(index, set()):
            raise ValueError("No balanced camera donor permutation can exclude own source groups")
    return assigned


def _stratified_pools(pool: list[dict], invert: bool) -> dict[bool, list[dict]]:
    strata = {}
    for sample in pool:
        source = sample.get("source")
        if source is None:
            stratum = "test-fixture-class"
        elif sample["gold"] == 0:
            stratum = source.get("camera_group") or sample["group_id"]
        else:
            stratum = source.get("generator")
            if not stratum:
                raise ValueError("AI camera donor strata require a generator")
        strata.setdefault(str(stratum), []).append(sample)
    result = {False: [], True: []}
    for stratum in sorted(strata):
        group = sorted(
            strata[stratum],
            key=lambda sample: hashlib.sha256(
                (DONOR_SEED + "|" + stratum + "|" + sample["id"]).encode()
            ).hexdigest(),
        )
        if len(group) % 2:
            raise ValueError("Donor source strata must have even counts")
        midpoint = len(group) // 2
        result[bool(invert)].extend(group[:midpoint])
        result[not bool(invert)].extend(group[midpoint:])
    return result


def _camera_donors(samples: list[dict]) -> dict[str, dict]:
    classes = {
        gold: sorted((s for s in samples if s["gold"] == gold), key=lambda s: s["id"])
        for gold in (False, True)
    }
    size = len(classes[False])
    if size < 2 or size != len(classes[True]) or size % 2:
        raise ValueError("Camera swap requires equal even test class counts of at least two")
    donors = {}
    for donor_gold in (False, True):
        # Reversing all halves preserves stratum quotas and supports tiny fixtures.
        for invert in (False, True):
            pools = _stratified_pools(classes[donor_gold], invert)
            candidate = {}
            try:
                for recipient_gold in (False, True):
                    recipients = [
                        sample
                        for index, sample in enumerate(classes[recipient_gold])
                        if bool(index % 2) == donor_gold
                    ]
                    pool = pools[recipient_gold]
                    assigned = _match_donors(recipients, pool)
                    for donor_index, recipient_index in assigned.items():
                        candidate[recipients[recipient_index]["id"]] = pool[donor_index]
            except ValueError:
                continue
            donors.update(candidate)
            break
        else:
            raise ValueError("No source-balanced camera donor permutation can exclude own groups")
    return donors


def _context_row(base: dict, condition: str, groups: dict, arms: list[str], donor=None) -> dict:
    row = copy.deepcopy(base)
    row.update(
        {
            "id": base["id"] + "--" + condition,
            "base_sample_id": base["id"],
            "condition": condition,
            "donor_id": donor,
            "arms": arms,
            "control_kind": "context_only",
            "expected_c2pa_status": None,
            "c2pa_validation_pass": None,
        }
    )
    row["features"].update(copy.deepcopy(groups))
    validate_features(row["features"])
    row["features_sha256"] = digest(row["features"])
    return row


def _missing_camera() -> dict:
    return {
        "status": "not_supplied",
        "exif_availability": "unknown",
        "iso": None,
        "exposure_seconds": None,
        "f_number": None,
        "focal_length_mm": None,
    }


def _byte_rows(root: Path, output: Path, base: dict, donor: dict) -> list[dict]:
    source = _file(root, base["native_image"])
    raw, donor_raw = source.read_bytes(), _file(root, donor["native_image"]).read_bytes()
    if len(raw) > MAX_INPUT_BYTES or len(donor_raw) > MAX_INPUT_BYTES:
        raise ValueError("Native control input exceeds byte limit")
    interventions = {
        "c2pa_stripped": (strip_credentials(raw), "no_manifest", None),
        "c2pa_swapped": (copy_credentials(raw, donor_raw), "invalid", donor["id"]),
        "c2pa_invalid_signature": (tamper_signature(raw), "invalid", None),
    }
    rows = []
    with Image.open(source) as image:
        original = (image.size, image.convert("RGB").tobytes())
    for condition, (changed, expected, donor_id) in interventions.items():
        key = hashlib.sha256(base["id"].encode()).hexdigest()[:16]
        path = output / f"{key}-{condition}.jpg"
        path.write_bytes(changed)
        with Image.open(path) as image:
            if (image.size, image.convert("RGB").tobytes()) != original:
                raise ValueError("Metadata byte intervention changed RGB pixels")
        summary = extract_c2pa(path)
        passed = summary["status"] == expected
        if condition == "c2pa_swapped":
            passed &= summary["signature_valid"] is True and summary["hard_binding_valid"] is False
        elif condition == "c2pa_invalid_signature":
            passed &= summary["signature_valid"] is False and summary["hard_binding_valid"] is True
        if not passed:
            raise ValueError("C2PA byte control failed its validation contract")
        row = _context_row(base, condition, {"provenance": summary}, ["provenance", "all_context"], donor_id)
        row.update(
            {
                "control_kind": "native_byte_intervention",
                "native_image": str(path.relative_to(root)),
                "native_sha256": hashlib.sha256(changed).hexdigest(),
                "expected_c2pa_status": expected,
                "c2pa_validation_pass": True,
                "decoded_rgb_preserved": True,
            }
        )
        rows.append(row)
    return rows


def build_controls(dataset: dict, *, root: Path = VT_ROOT, output_dir: Path | None = None) -> dict:
    """Build derived test rows; preserve the base dataset and all delivered pixels."""
    root = root.resolve()
    output = (output_dir or root / "data/metadata-v3/controls/native").resolve()
    if not output.is_relative_to(root):
        raise ValueError("Control output must stay inside the visual-triage root")
    samples = sorted((s for s in dataset["samples"] if s["split"] == "test"), key=lambda s: s["id"])
    if len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate base sample identifiers")
    for sample in samples:
        _check_sample(root, sample)
    donors = _camera_donors(samples)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    eligible = [
        s
        for s in samples
        if s["features"]["provenance"]["status"] == "valid"
        and s["features"]["provenance"]["ai_generation_claim"] is True
    ]
    if len(eligible) == 1:
        raise ValueError("C2PA copy controls require at least two valid AI-claim test assets")
    for base in samples:
        rows.append(
            _context_row(base, "camera_stripped", {"camera": _missing_camera()}, ["camera", "all_context"])
        )
        donor = donors[base["id"]]
        rows.append(
            _context_row(
                base,
                "camera_swapped",
                {"camera": donor["features"]["camera"]},
                ["camera", "all_context"],
                donor["id"],
            )
        )
        resolution = copy.deepcopy(base["features"]["resolution"])
        resolution["native"] = {"status": "not_supplied", "width": None, "height": None, "aspect_ratio": None}
        rows.append(
            _context_row(
                base, "resolution_stripped", {"resolution": resolution}, ["resolution", "all_context"]
            )
        )
        absent = {
            "status": "no_manifest",
            "manifest_present": False,
            "signature_valid": None,
            "hard_binding_valid": None,
            "trust_status": "unknown",
            "ai_generation_claim": None,
        }
        rows.append(
            _context_row(
                base,
                "all_stripped",
                {"camera": _missing_camera(), "resolution": resolution, "provenance": absent},
                ["all_context"],
            )
        )
    for index, base in enumerate(eligible):
        rows.extend(_byte_rows(root, output, base, eligible[(index + 1) % len(eligible)]))
    return {
        "version": 3,
        "track": "metadata_v3_controls",
        "base_dataset_sha256": digest(dataset),
        "cases": rows,
        "sdk_policy": copy.deepcopy(SDK_POLICY),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "camera_donor_policy": "one-to-one test permutation; half each camera group and AI generator per recipient class; seeded within-stratum rank and deterministic matching; different source group",
        "camera_donor_seed": DONOR_SEED,
        "note": "Controlled interventions, separate from native results; context controls do not modify native EXIF.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())
    controls = build_controls(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(controls, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"control_rows": len(controls["cases"]), "sha256": digest(controls)}))


if __name__ == "__main__":
    main()
