"""Container-level provenance scan for the mug control set: C2PA, XMP, IPTC, EXIF and text chunks.

Run with the pinned C2PA SDK (c2pa-python 0.38.0, Rust 0.91.0) so `extract_c2pa` applies the
frozen offline policy from visual-triage. Without it the C2PA summary reads "unavailable".

    python scan_provenance.py images/*  > provenance.jsonl
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import sys
import zlib

from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "visual-triage" / "tracks" / "metadata_v3"))
from provenance import SDK_POLICY, extract_c2pa

XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"
# Byte patterns that indicate provenance data even when the container walk misses it.
SIGNATURES = {
    "jumbf_box": b"jumb",
    "c2pa_label": b"c2pa",
    "trained_algorithmic_media": b"trainedAlgorithmicMedia",
    "composite_trained_algorithmic_media": b"compositeWithTrainedAlgorithmicMedia",
    "iptc_digital_source_type": b"DigitalSourceType",
    "synthid_text": b"SynthID",
}


def _jpeg(data: bytes) -> list[dict]:
    blocks, pos = [], 2
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker == 0xDA:  # start of scan: metadata segments end here
            break
        length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        body = data[pos + 4 : pos + 2 + length]
        kind = f"APP{marker - 0xE0}" if 0xE0 <= marker <= 0xEF else ("COM" if marker == 0xFE else f"0x{marker:02X}")
        ident = body[:32].split(b"\x00")[0].decode("latin-1", "replace")
        if kind == "APP1" and body.startswith(XMP_ID):
            kind, payload = "XMP", body[len(XMP_ID) :]
        elif kind == "APP1" and body.startswith(b"Exif\x00"):
            kind, payload = "EXIF", body
        elif kind == "APP11" and body[:2] == b"JP":
            kind, payload = "JUMBF", body
        elif kind == "APP13" and body.startswith(b"Photoshop 3.0"):
            kind, payload = "IPTC_IRB", body
        elif kind == "APP2" and body.startswith(b"ICC_PROFILE"):
            kind, payload = "ICC", body
        else:
            payload = body
        if kind.startswith("APP") or kind in {"XMP", "EXIF", "JUMBF", "IPTC_IRB", "ICC", "COM"}:
            blocks.append({"container": kind, "id": ident, "bytes": len(body), "payload": payload})
        pos += 2 + length
    return blocks


def _png(data: bytes) -> list[dict]:
    blocks, pos = [], 8
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        name = kind.decode("latin-1")
        if name in {"tEXt", "iTXt", "zTXt"}:
            keyword, _, rest = body.partition(b"\x00")
            if name == "zTXt":
                rest = zlib.decompress(rest[1:])
            elif name == "iTXt":
                flag, rest = rest[0], rest[2:]
                rest = rest.split(b"\x00", 2)[-1]
                rest = zlib.decompress(rest) if flag else rest
            container = "XMP" if keyword == b"XML:com.adobe.xmp" else "TEXT"
            blocks.append({"container": container, "id": keyword.decode("latin-1"), "bytes": length, "payload": rest})
        elif name in {"eXIf", "iCCP", "caBX", "sRGB", "gAMA", "cHRM", "pHYs", "tIME"}:
            container = {"eXIf": "EXIF", "iCCP": "ICC", "caBX": "JUMBF"}.get(name, name)
            blocks.append({"container": container, "id": name, "bytes": length, "payload": body})
        pos += 12 + length
    return blocks


def _webp(data: bytes) -> list[dict]:
    blocks, pos = [], 12
    while pos + 8 <= len(data):
        kind, length = struct.unpack("<4sI", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        name = kind.decode("latin-1").strip()
        container = {"EXIF": "EXIF", "XMP": "XMP", "ICCP": "ICC", "C2PA": "JUMBF"}.get(name, name)
        blocks.append({"container": container, "id": name, "bytes": length, "payload": body})
        pos += 8 + length + (length & 1)
    return blocks


def _iptc_records(irb: bytes) -> dict:
    """Photoshop IRB resource 0x0404 holds IPTC-IIM; return dataset 2:xx values."""
    records, pos = {}, irb.find(b"8BIM")
    while pos != -1 and pos + 12 <= len(irb):
        resource = struct.unpack(">H", irb[pos + 4 : pos + 6])[0]
        name_len = irb[pos + 6]
        offset = pos + 7 + name_len + ((name_len + 1) & 1)
        size = struct.unpack(">I", irb[offset : offset + 4])[0]
        block = irb[offset + 4 : offset + 4 + size]
        if resource == 0x0404:
            i = 0
            while i + 5 <= len(block) and block[i] == 0x1C:
                record, dataset, n = block[i + 1], block[i + 2], struct.unpack(">H", block[i + 3 : i + 5])[0]
                records[f"{record}:{dataset}"] = block[i + 5 : i + 5 + n].decode("utf-8", "replace")
                i += 5 + n
        pos = irb.find(b"8BIM", offset + 4 + size)
    return records


def _xmp_fields(xmp: bytes) -> dict:
    text = xmp.decode("utf-8", "replace")
    fields = {}
    for key in ("DigitalSourceType", "CreatorTool", "Credit", "Description", "Software", "History"):
        match = re.search(rf"{key}(?:=\"([^\"]*)\"|>([^<]*)<)", text)
        if match:
            fields[key] = (match.group(1) or match.group(2) or "").strip()[:200]
    fields["namespaces"] = sorted(set(re.findall(r"xmlns:(\w+)=", text)))
    return fields


def _exif_fields(image: Image.Image) -> dict:
    exif = image.getexif()
    names = {271: "Make", 272: "Model", 305: "Software", 270: "ImageDescription", 315: "Artist", 33432: "Copyright"}
    return {name: str(exif[tag])[:200] for tag, name in names.items() if tag in exif} | {"tag_count": len(exif)}


def scan(path: Path) -> dict:
    data = path.read_bytes()
    if data.startswith(b"\xff\xd8"):
        fmt, blocks = "JPEG", _jpeg(data)
    elif data.startswith(b"\x89PNG"):
        fmt, blocks = "PNG", _png(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        fmt, blocks = "WEBP", _webp(data)
    else:
        fmt, blocks = "unknown", []
    with Image.open(io.BytesIO(data)) as image:
        size, exif = image.size, _exif_fields(image)
        rgb_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    xmp = next((b["payload"] for b in blocks if b["container"] == "XMP"), None)
    irb = next((b["payload"] for b in blocks if b["container"] == "IPTC_IRB"), None)
    raw_iptc = next((b["payload"] for b in blocks if b["id"] == "Raw profile type iptc"), None)
    if irb is None and raw_iptc:  # ImageMagick-style PNG text: "IPTC profile", length, then hex of the IRB
        irb = bytes.fromhex("".join(raw_iptc.decode("latin-1").split()[3:]))
    texts = {b["id"]: b["payload"].decode("utf-8", "replace")[:200] for b in blocks if b["container"] == "TEXT"}
    return {
        "file": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "decoded_rgb_sha256": rgb_hash,
        "format_from_bytes": fmt,
        "webp_variant": next((b["id"] for b in blocks if b["id"] in {"VP8", "VP8L", "VP8X"}), None),
        "dimensions": list(size),
        "file_bytes": len(data),
        "containers": [{"container": b["container"], "id": b["id"], "bytes": b["bytes"]} for b in blocks],
        "exif": exif,
        "xmp_present": xmp is not None,
        "xmp": _xmp_fields(xmp) if xmp else None,
        "iptc_present": irb is not None,
        "iptc_iim": _iptc_records(irb) if irb else None,
        "png_text": texts or None,
        "jumbf_container_present": any(b["container"] == "JUMBF" for b in blocks),
        "byte_signatures": {name: sig in data for name, sig in SIGNATURES.items()},
        "c2pa": extract_c2pa(path),
        "c2pa_policy": SDK_POLICY["trust_policy"],
    }


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps(scan(Path(arg)), sort_keys=True))
