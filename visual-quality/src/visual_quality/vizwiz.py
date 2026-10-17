"""Acquire official VizWiz training photos selected by human defect votes."""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import struct
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image
from remotezip import RemoteZip
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .storage import atomic_json, download, write_jsonl

SOURCE_PAGE = "https://vizwiz.org/tasks-and-datasets/image-quality-issues/"
MEDIA_LICENSE_PAGE = "https://vizwiz.org/tasks-and-datasets/image-captioning/"
REPO_REVISION = "736265e64805685d8595bc43c01364e8e51efae2"
REPO_RAW = f"https://raw.githubusercontent.com/chiutaiyin/VizWiz-QualityIssues/{REPO_REVISION}"
ANNOTATION_URL = f"{REPO_RAW}/annotations/quality_annotations/train.json"
ANNOTATION_SHA256 = "921e3c7fa91af8811f88111d771c40950d8f33593df34519059aa0a4e75eb70f"
ARCHIVE_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip"
ARCHIVE_ETAG = '"2a1703b5e-59afdc1e58080"'
ARCHIVE_SIZE = 11_298_421_598
DEFECTS = ("BLR", "BRT", "DRK", "FRM", "OBS", "ROT")
TRAIN_IMAGE = re.compile(r"VizWiz_train_[0-9]{8}\.jpg\Z")
_THREAD = threading.local()


def select_candidates(annotations: list[dict], seed: int) -> list[dict]:
    """Balance observed defect types; neither VQA answers nor MOS are inferred."""
    buckets: dict[str, list[dict]] = {key: [] for key in DEFECTS}
    seen: set[str] = set()
    for annotation in annotations:
        name = annotation.get("image", "")
        if not TRAIN_IMAGE.fullmatch(name) or name in seen:
            continue
        flaws = annotation.get("flaws", {})
        votes = [flaws.get(key, 0) for key in (*DEFECTS, "NON")]
        if any(type(vote) is not int or not 0 <= vote <= 5 for vote in votes):
            raise ValueError(f"Invalid five-rater defect votes: {name}")
        if flaws.get("NON", 0) >= 3 or max(votes[:-1]) < 3:
            continue
        dominant = max(DEFECTS, key=lambda key: flaws.get(key, 0))
        buckets[dominant].append(annotation)
        seen.add(name)
    rng = random.Random(seed)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: row["image"])
        rng.shuffle(bucket)
    selected = []
    while any(buckets.values()):
        for bucket in buckets.values():
            if bucket:
                selected.append(bucket.pop())
    return selected


def _session() -> requests.Session:
    if not hasattr(_THREAD, "session"):
        session = requests.Session()
        retries = Retry(total=4, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        _THREAD.session = session
    return _THREAD.session


def _archive_index(source_dir: Path) -> dict[str, dict]:
    cache = source_dir / "archive-index.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        if data["etag"] != ARCHIVE_ETAG:
            raise ValueError("Cached VizWiz archive revision does not match")
        return data["members"]
    with _session().head(ARCHIVE_URL, timeout=(15, 60)) as response:
        response.raise_for_status()
        if response.headers.get("ETag") != ARCHIVE_ETAG:
            raise ValueError("Official VizWiz media archive has changed; review and repin it")
        if int(response.headers["Content-Length"]) != ARCHIVE_SIZE:
            raise ValueError("Unexpected VizWiz media archive size")
    with RemoteZip(ARCHIVE_URL, headers={"If-Match": ARCHIVE_ETAG}, timeout=(15, 60)) as archive:
        members = {
            Path(info.filename).name: {
                "name": info.filename,
                "offset": info.header_offset,
                "compressed_size": info.compress_size,
                "size": info.file_size,
                "compression": info.compress_type,
                "crc": info.CRC,
            }
            for info in archive.infolist()
            if TRAIN_IMAGE.fullmatch(Path(info.filename).name)
        }
    atomic_json(cache, {"url": ARCHIVE_URL, "etag": ARCHIVE_ETAG, "members": members})
    return members


def _read_member(member: dict) -> bytes:
    """Read one bounded ZIP range and check the original member's length and CRC."""
    if not 0 < member["size"] <= 20_000_000 or member["compressed_size"] > 20_000_000:
        raise ValueError("VizWiz image exceeds acquisition size limit")
    start = member["offset"]
    end = min(ARCHIVE_SIZE - 1, start + member["compressed_size"] + 8192)
    with _session().get(
        ARCHIVE_URL,
        headers={"Range": f"bytes={start}-{end}", "If-Match": ARCHIVE_ETAG},
        timeout=(15, 90),
        stream=True,
    ) as response:
        response.raise_for_status()
        expected_range = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
            raise ValueError("Server did not return the requested bounded ZIP range")
        payload = response.raw.read(end - start + 2)
    if len(payload) != end - start + 1 or payload[:4] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local header or truncated range")
    filename_size, extra_size = struct.unpack_from("<HH", payload, 26)
    offset = 30 + filename_size + extra_size
    compressed = payload[offset : offset + member["compressed_size"]]
    if len(compressed) != member["compressed_size"]:
        raise ValueError("ZIP local header exceeded the bounded metadata allowance")
    if member["compression"] == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -zlib.MAX_WBITS)
    elif member["compression"] == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError("Unsupported ZIP compression")
    if len(data) != member["size"] or zlib.crc32(data) != member["crc"]:
        raise ValueError("ZIP member size or CRC mismatch")
    return data


def _evidence(source_dir: Path) -> Path:
    evidence = source_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for url, name in [
        (SOURCE_PAGE, "quality-issues.html"),
        (MEDIA_LICENSE_PAGE, "caption-media-license.html"),
        (f"{REPO_RAW}/LICENSE", "annotation-repository-LICENSE.txt"),
        (f"{REPO_RAW}/README.md", "annotation-repository-README.md"),
    ]:
        download(url, evidence / name, max_bytes=2_000_000)
    license_html = (evidence / "caption-media-license.html").read_text()
    if "creativecommons.org/licenses/by/4.0" not in license_html or ARCHIVE_URL not in license_html:
        raise ValueError("Official media license page no longer links the expected CC BY image archive")
    path = evidence / "provenance.json"
    atomic_json(
        path,
        {
            "media_license": "CC-BY-4.0",
            "media_license_url": "https://creativecommons.org/licenses/by/4.0/",
            "media_license_evidence": str((evidence / "caption-media-license.html").resolve()),
            "media_attribution": "VizWiz-Captions: Danna Gurari, Yinan Zhao, Meng Zhang, Nilavra Bhattacharya (ECCV 2020); contributing VizWiz photographers.",
            "annotation_repository": "https://github.com/chiutaiyin/VizWiz-QualityIssues",
            "annotation_repository_license": "MIT",
            "annotation_license_scope": "Official repository contains quality annotations and a top-level MIT license; preserved separately from image rights.",
            "annotation_attribution": "Tai-Yin Chiu, Yinan Zhao, Danna Gurari. Assessing Image Quality Issues for Real-World Problems, CVPR 2020.",
            "annotation_revision": REPO_REVISION,
            "annotation_sha256": ANNOTATION_SHA256,
            "archive_url": ARCHIVE_URL,
            "archive_etag": ARCHIVE_ETAG,
            "archive_size": ARCHIVE_SIZE,
            "source_pages": [SOURCE_PAGE, MEDIA_LICENSE_PAGE],
            "privacy_note": "Only official QualityIssues training IDs 00000000–00023430; no private originals or later VQA redaction additions.",
        },
    )
    return path


def _acquire_image(annotation: dict, member: dict, source_dir: Path, evidence: Path) -> dict:
    name = annotation["image"]
    path = source_dir / "images" / name
    if not path.exists():
        data = _read_member(member)
        temporary = path.with_suffix(".jpg.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    data = path.read_bytes()
    if len(data) != member["size"] or zlib.crc32(data) != member["crc"]:
        raise ValueError(f"Cached image differs from original ZIP member: {name}")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        rgb = image.convert("RGB")
        decoded_hash = hashlib.sha256(struct.pack("<II", *rgb.size) + rgb.tobytes()).hexdigest()
        width, height = rgb.size
    return {
        "id": f"vizwiz:{name}",
        "source": "vizwiz-quality-issues",
        "source_id": name,
        "source_url": SOURCE_PAGE,
        "download_url": f"{ARCHIVE_URL}#{member['name']}",
        "source_revision": f"annotations:{REPO_REVISION};media-etag:{ARCHIVE_ETAG}",
        "source_split": "train",
        "source_group": f"vizwiz:{name}",
        "license": {
            "id": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "VizWiz-Captions: Danna Gurari, Yinan Zhao, Meng Zhang, Nilavra Bhattacharya (ECCV 2020); contributing VizWiz photographers.",
            "evidence_path": str(evidence.resolve()),
        },
        "image_path": str(path.resolve()),
        "source_labels": {
            "human_flaw_votes": annotation["flaws"],
            "human_unrecognizable_votes": annotation["unrecognizable"],
            "raters": 5,
            "selection": "At least 3/5 votes for one named defect, with fewer than 3/5 no-flaw votes; balanced dominant defects.",
            "annotation_license": "MIT (official repository)",
        },
        "content_kind": "photo",
        "source_origin": "camera",
        "photo_domain_eligible": None,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "decoded_sha256": decoded_hash,
        "width": width,
        "height": height,
    }


def acquire(root: Path, count: int, seed: int) -> list[dict]:
    """Return unique training-source images; downstream code assigns pilot splits."""
    if count <= 0:
        raise ValueError("count must be positive")
    source_dir = root / "sources" / "vizwiz"
    (source_dir / "images").mkdir(parents=True, exist_ok=True)
    evidence = _evidence(source_dir)
    annotation_path = download(ANNOTATION_URL, source_dir / "train.json", max_bytes=5_000_000)
    if hashlib.sha256(annotation_path.read_bytes()).hexdigest() != ANNOTATION_SHA256:
        raise ValueError("Pinned official VizWiz training annotation checksum mismatch")
    candidates = select_candidates(json.loads(annotation_path.read_text()), seed)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible training images, requested {count}")
    index = _archive_index(source_dir)
    records, rejections, seen = [], [], set()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for batch_start in range(0, len(candidates), 64):
            batch = candidates[batch_start : batch_start + min(64, count - len(records) + 8)]
            futures = [
                (row, pool.submit(_acquire_image, row, index[row["image"]], source_dir, evidence))
                for row in batch
            ]
            for row, future in futures:
                try:
                    record = future.result()
                except (requests.RequestException, ValueError, OSError, zlib.error) as error:
                    rejections.append({"source_id": row["image"], "reason": str(error)})
                    continue
                if record["decoded_sha256"] in seen:
                    rejections.append({"source_id": row["image"], "reason": "duplicate decoded image"})
                    continue
                seen.add(record["decoded_sha256"])
                records.append(record)
            write_jsonl(source_dir / "records.jsonl", records[:count])
            if len(records) >= count:
                break
    atomic_json(
        source_dir / "acquisition.json",
        {
            "requested": count,
            "seed": seed,
            "eligible_annotations": len(candidates),
            "acquired_unique": len(records),
            "selected": min(count, len(records)),
            "rejections": rejections,
            "selection_uses": "official training QualityIssues votes, not VQA answerability or teacher scores",
        },
    )
    if len(records) < count:
        raise RuntimeError(f"Acquired only {len(records)} unique VizWiz images; inspect acquisition.json")
    write_jsonl(source_dir / "spares.jsonl", records[count:])
    return records[:count]
