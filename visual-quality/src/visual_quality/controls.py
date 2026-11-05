"""Acquire training-only chart and render controls with source evidence."""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import pairwise
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from remotezip import RemoteZip

from .storage import atomic_json, download, read_jsonl, write_jsonl

PLOTQA_REPO = "achang/plot_qa"
PLOTQA_REVISION = "e9da2cdb45c671f3faf63449e6c329f0c3281aa0"
PLOTQA_LICENSE_REVISION = "e0f5c34acbe92753f70798aafda99184e3e0cd8c"
PLOTQA_TRAIN_COUNT = 157070
CLEVR_ARCHIVE = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
CLEVR_VERSION = "B2HsawCh6GoabZXQDbLoboNXB4FrlnvC"
CLEVR_ETAG = '"3426f0a360618fcf2179966a8d1428c3-1210"'
CLEVR_REVISION = f"v1.0;s3-version={CLEVR_VERSION};etag={CLEVR_ETAG}"
CC_BY_URL = "https://creativecommons.org/licenses/by/4.0/"


def _image_info(data: bytes) -> dict:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError("Control delivery must be lossless PNG")
        decoded = image.convert("RGBA")
        pixel_digest = hashlib.sha256()
        pixel_digest.update(f"{image.width}x{image.height}:RGBA:".encode())
        pixel_digest.update(decoded.tobytes())
        return {
            "width": image.width,
            "height": image.height,
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "decoded_sha256": pixel_digest.hexdigest(),
        }


def _write_png(path: Path, data: bytes) -> dict:
    info = _image_info(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(data)
    temporary.replace(path)
    return info


def _cached(directory: Path, count: int, seed: int, revision: str) -> list[dict] | None:
    metadata = directory / "acquisition.json"
    if not metadata.exists():
        return None
    state = json.loads(metadata.read_text())
    if state.get("seed") != seed or state.get("source_revision") != revision:
        raise ValueError(f"Existing controls use a different seed or revision: {directory}")
    rows = read_jsonl(directory / "records.jsonl")
    valid = [r for r in rows if Path(r["image_path"]).is_file()]
    if state.get("complete") and len(valid) >= count:
        return valid[:count]
    return None


def _checkpoint(directory: Path, rows: list[dict], state: dict, complete: bool = False) -> None:
    write_jsonl(directory / "records.jsonl", rows)
    atomic_json(directory / "acquisition.json", {**state, "complete": complete, "acquired": len(rows)})


def _plotqa_group(text: str, row_index: int) -> tuple[str, str]:
    """Group matching series data after removing recorded rendering properties."""
    if not text:
        return f"plotqa:unknown-group:train:{row_index}", "unknown; using source row ID"
    content = re.sub(r"<s_bboxes>.*?</s_bboxes>", "", text, flags=re.DOTALL)
    content = re.sub(r"<s_(colors?|width)>.*?</s_\1>", "", content, flags=re.DOTALL)
    digest = hashlib.sha256(content.encode()).hexdigest()
    return f"plotqa:series-data:{digest}", "derived from series tokens; canonical template group unknown"


def _validate_plotqa_row(row: dict) -> None:
    index = row["row_idx"]
    if not 0 <= index < PLOTQA_TRAIN_COUNT:
        raise ValueError("PlotQA row is outside the declared training split")
    image = row["row"]["image"]
    parsed = urlparse(image["src"])
    expected = f"/{PLOTQA_REPO}/--/{PLOTQA_REVISION}/--/default/train/{index}/image/"
    if parsed.scheme != "https" or parsed.netloc != "datasets-server.huggingface.co":
        raise ValueError("Unexpected PlotQA image delivery host")
    if expected not in parsed.path or not parsed.path.endswith(".png"):
        raise ValueError("PlotQA asset is not the pinned training PNG")


def _plotqa_record(directory: Path, row: dict, window_url: str, evidence: Path) -> dict:
    _validate_plotqa_row(row)
    index = row["row_idx"]
    data = row["row"]
    path = directory / "images" / f"train-{index:06d}.png"
    download(data["image"]["src"], path, max_bytes=20 * 1024 * 1024)
    info = _image_info(path.read_bytes())
    if (info["width"], info["height"]) != (data["image"]["width"], data["image"]["height"]):
        raise ValueError("PlotQA image dimensions do not match its Viewer record")
    group, group_status = _plotqa_group(data.get("text", ""), index)
    return {
        "id": f"plotqa:train:{index}",
        "source": "plotqa",
        "source_id": f"train/row-{index}",
        "source_url": f"https://huggingface.co/datasets/{PLOTQA_REPO}/tree/{PLOTQA_REVISION}",
        "download_url": data["image"]["src"],
        "source_revision": PLOTQA_REVISION,
        "source_split": "train",
        "source_group": group,
        "license": {
            "id": "CC-BY-4.0",
            "url": CC_BY_URL,
            "attribution": "PlotQA — Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, Pratyush Kumar (2020)",
            "evidence_path": str(evidence.resolve()),
        },
        "image_path": str(path.resolve()),
        "source_labels": {"series_tokens": data.get("text"), "group_status": group_status},
        "content_kind": "chart",
        "source_origin": "render",
        "photo_domain_eligible": False,
        "delivery": {
            "method": "Hugging Face Dataset Viewer cached PNG",
            "window_url": window_url,
            "original_bytes_verified": False,
            "original_pixels_verified": False,
            "processing": "Viewer may decode/re-encode PNG; no local resizing or recompression",
            "upstream_original_image_id": None,
        },
        **info,
    }


def _plotqa_window(session: requests.Session, offset: int, length: int) -> requests.Response:
    for attempt in range(6):
        response = session.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": PLOTQA_REPO,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": length,
            },
            timeout=(15, 120),
        )
        if response.status_code not in {429, 502, 503, 504}:
            break
        delay = int(response.headers.get("Retry-After", 2 ** (attempt + 1)))
        if delay > 60:
            response.raise_for_status()
        time.sleep(delay)
    return response


def _plotqa_payload(session: requests.Session, directory: Path, offset: int, length: int) -> tuple:
    evidence = directory / "evidence" / f"window-{offset}.json"
    if evidence.exists():
        payload = json.loads(evidence.read_text())
        if all((directory / "images" / f"train-{r['row_idx']:06d}.png").exists() for r in payload["rows"]):
            url = (
                f"https://datasets-server.huggingface.co/rows?dataset={PLOTQA_REPO}"
                f"&config=default&split=train&offset={offset}&length={length}"
            )
            return payload, url, 200, True
    scan_limit = evidence.with_name(f"scan-limit-{offset}.json")
    if scan_limit.exists():
        return None, None, 500, True
    response = _plotqa_window(session, offset, length)
    if response.status_code == 500 and "Scan size limit exceeded" in response.text:
        atomic_json(scan_limit, {"url": response.url, "error": response.json()})
        return None, response.url, 500, False
    response.raise_for_status()
    payload = response.json()
    if payload["num_rows_total"] != PLOTQA_TRAIN_COUNT or payload.get("partial"):
        raise ValueError("PlotQA Viewer does not represent the complete declared train split")
    atomic_json(evidence, payload)
    return payload, response.url, 200, False


def acquire_plotqa(root: Path, count: int, seed: int) -> list[dict]:
    """Sample Viewer-accessible windows across train; declare scan-limit selection bias."""
    if not 0 < count <= 1200:
        raise ValueError("This pilot adapter supports 1–1200 PlotQA images")
    directory = Path(root) / "sources" / "plotqa"
    if (cached := _cached(directory, count, seed, PLOTQA_REVISION)) is not None:
        return cached
    evidence_dir = directory / "evidence"
    evidence = download(
        f"https://raw.githubusercontent.com/NiteshMethani/PlotQA/{PLOTQA_LICENSE_REVISION}/README.md",
        evidence_dir / "upstream-license-readme.md",
    )
    if "CC-BY-4.0" not in evidence.read_text():
        raise ValueError("Upstream PlotQA data license declaration changed")
    download(
        f"https://huggingface.co/datasets/{PLOTQA_REPO}/resolve/{PLOTQA_REVISION}/README.md",
        evidence_dir / "hub-readme.md",
    )
    download(
        f"https://huggingface.co/datasets/{PLOTQA_REPO}/resolve/{PLOTQA_REVISION}/dataset_infos.json",
        evidence_dir / "dataset-infos.json",
    )
    state = {
        "seed": seed,
        "source_revision": PLOTQA_REVISION,
        "requested": count,
        "sampling": "12 equal-width train strata; seeded contiguous windows; exact pixel dedup",
        "limitation": "Viewer rejects row groups above 300MB; sampled distribution is conditional on access",
        "attempts": [],
    }
    rng = random.Random(seed)
    records, seen = [], set()
    with requests.Session() as session, ThreadPoolExecutor(max_workers=6) as pool:
        for stratum in range(12):
            target = count // 12 + (stratum < count % 12)
            if not target:
                continue
            lower = PLOTQA_TRAIN_COUNT * stratum // 12
            upper = PLOTQA_TRAIN_COUNT * (stratum + 1) // 12
            accepted = 0
            for attempt in range(40):
                length = min(100, target - accepted + 10)
                offset = rng.randrange(lower, upper - length + 1)
                payload, window_url, status, cache_hit = _plotqa_payload(session, directory, offset, length)
                state["attempts"].append(
                    {
                        "stratum": stratum,
                        "offset": offset,
                        "length": length,
                        "status": status,
                        "cache_hit": cache_hit,
                    }
                )
                if payload is None:
                    continue
                candidates = payload["rows"]
                rng.shuffle(candidates)
                # Preserve input order while downloading independent image assets concurrently.
                make_record = partial(_plotqa_record, directory, window_url=window_url, evidence=evidence)
                rows = pool.map(make_record, candidates)
                for record in rows:
                    if record["decoded_sha256"] in seen:
                        continue
                    if accepted < target:
                        seen.add(record["decoded_sha256"])
                        records.append(record)
                        accepted += 1
                _checkpoint(directory, records, state)
                if accepted == target:
                    break
            if accepted != target:
                raise RuntimeError(f"Insufficient accessible PlotQA images in stratum {stratum}")
    _checkpoint(directory, records, state, complete=True)
    return records


def _clevr_members(names: list[str], seed: int) -> list[str]:
    members = sorted(n for n in names if re.fullmatch(r"CLEVR_v1\.0/images/train/CLEVR_train_\d{6}\.png", n))
    if len(members) != 70000:
        raise ValueError(f"CLEVR archive has {len(members)} training PNGs; expected 70000")
    random.Random(seed).shuffle(members)
    return members


def _clevr_record(directory: Path, member: str, data: bytes, evidence: Path) -> dict:
    if not re.fullmatch(r"CLEVR_v1\.0/images/train/CLEVR_train_\d{6}\.png", member):
        raise ValueError("CLEVR extraction is restricted to original training images")
    scene = Path(member).stem
    path = directory / "images" / Path(member).name
    info = _write_png(path, data)
    return {
        "id": f"clevr:{scene}",
        "source": "clevr",
        "source_id": scene,
        "source_url": "https://cs.stanford.edu/people/jcjohns/clevr/",
        "download_url": CLEVR_ARCHIVE,
        "source_revision": CLEVR_REVISION,
        "source_split": "train",
        "source_group": f"clevr:scene:{scene}",
        "license": {
            "id": "CC-BY-4.0",
            "url": CC_BY_URL,
            "attribution": "CLEVR (c) 2017 Facebook, Inc.; Johnson et al. (2017)",
            "evidence_path": str(evidence.resolve()),
        },
        "image_path": str(path.resolve()),
        "source_labels": {"scene_id": scene},
        "content_kind": "render",
        "source_origin": "render",
        "photo_domain_eligible": False,
        "delivery": {
            "method": "HTTP range extraction of original ZIP member",
            "archive_member": member,
            "original_bytes_verified": True,
            "original_pixels_verified": True,
            "processing": "ZIP decompression only; stored PNG bytes unchanged",
        },
        **info,
    }


def _decode_zip_member(payload: bytes, info: zipfile.ZipInfo) -> bytes:
    if payload[:4] != b"PK\x03\x04" or len(payload) < 30:
        raise ValueError("Invalid CLEVR local ZIP header")
    name_length, extra_length = struct.unpack_from("<HH", payload, 26)
    start = 30 + name_length + extra_length
    compressed = payload[start : start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise ValueError("Unsupported CLEVR ZIP compression")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError("CLEVR ZIP member failed its original size/CRC integrity check")
    return data


def _fetch_clevr_member(directory: Path, evidence: Path, entry: tuple[zipfile.ZipInfo, int]) -> dict:
    info, end = entry
    path = directory / "images" / Path(info.filename).name
    if path.exists():
        data = path.read_bytes()
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise ValueError(f"Cached CLEVR file does not match the original archive: {path}")
    else:
        headers = {"Range": f"bytes={info.header_offset}-{end}", "If-Match": CLEVR_ETAG}
        with requests.get(CLEVR_ARCHIVE, headers=headers, stream=True, timeout=(15, 90)) as response:
            if response.status_code != 206 or response.headers.get("ETag") != CLEVR_ETAG:
                raise ValueError("CLEVR member requires a range response from the pinned archive")
            expected_size = end - info.header_offset + 1
            if expected_size > 20 * 1024 * 1024:
                raise ValueError("CLEVR image member exceeds the pilot size limit")
            payload = response.content
            if len(payload) != expected_size:
                raise ValueError("Incomplete CLEVR ZIP member range")
        data = _decode_zip_member(payload, info)
    return _clevr_record(directory, info.filename, data, evidence)


def acquire_clevr(root: Path, count: int, seed: int) -> list[dict]:
    """Extract original, randomly selected train PNGs without downloading the archive."""
    if not 0 < count <= 70000:
        raise ValueError("CLEVR count must be in 1–70000")
    directory = Path(root) / "sources" / "clevr"
    if (cached := _cached(directory, count, seed, CLEVR_REVISION)) is not None:
        return cached
    with requests.get(CLEVR_ARCHIVE, headers={"Range": "bytes=-8"}, stream=True, timeout=30) as response:
        if response.status_code != 206:
            raise ValueError("CLEVR requires HTTP range support; refusing full archive download")
        if response.headers.get("ETag") != CLEVR_ETAG:
            raise ValueError("CLEVR archive changed from the pinned ETag")
        metadata = {
            key: response.headers.get(key)
            for key in ("ETag", "Last-Modified", "Content-Range", "x-amz-version-id")
        }
    if metadata["x-amz-version-id"] != CLEVR_VERSION:
        raise ValueError("CLEVR archive changed from the pinned object version")
    evidence_dir = directory / "evidence"
    atomic_json(evidence_dir / "archive-metadata.json", metadata)
    state = {
        "seed": seed,
        "source_revision": CLEVR_REVISION,
        "requested": count,
        "sampling": "seeded shuffle of all 70000 original train image members; exact pixel dedup",
    }
    records, seen = [], set()
    with RemoteZip(CLEVR_ARCHIVE, timeout=90, headers={"If-Match": CLEVR_ETAG}) as archive:
        for filename in ("COPYRIGHT.txt", "LICENSE.txt", "README.txt"):
            (evidence_dir / filename).write_bytes(archive.read(f"CLEVR_v1.0/{filename}"))
        evidence = evidence_dir / "COPYRIGHT.txt"
        if "Creative Commons Attribution 4.0" not in evidence.read_text():
            raise ValueError("CLEVR archive data license is not the expected CC BY 4.0")
        ordered = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
        entries = {
            entry.filename: (entry, following.header_offset - 1) for entry, following in pairwise(ordered)
        }
        entries[ordered[-1].filename] = (ordered[-1], archive.start_dir - 1)
        members = _clevr_members(archive.namelist(), seed)
    fetch = partial(_fetch_clevr_member, directory, evidence)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(members), 40):
            batch = members[start : start + 40]
            for record in pool.map(fetch, [entries[member] for member in batch]):
                if record["decoded_sha256"] in seen:
                    continue
                if len(records) < count:
                    seen.add(record["decoded_sha256"])
                    records.append(record)
            _checkpoint(directory, records, state)
            if len(records) == count:
                break
    if len(records) != count:
        raise RuntimeError("CLEVR did not contain enough distinct selected training images")
    _checkpoint(directory, records, state, complete=True)
    return records
