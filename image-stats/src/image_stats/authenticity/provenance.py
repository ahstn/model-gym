"""Explicit declarations are review evidence; absent metadata does not establish origin."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

from pathlib import Path

from PIL import Image

from image_stats.authenticity.c2pa import AI_URIS, SDK_POLICY, inspect_c2pa
from image_stats.authenticity.containers import _exif_fields, _jpeg, _png, _webp

EXPLICIT_AI = {
    "ai generated",
    "ai-generated",
    "ai_generated",
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
}


def xmp_fields(payload: bytes) -> tuple[dict, str | None]:
    if not payload:
        return {}, None
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        return {}, str(error)
    fields = {}
    for node in root.iter():
        for name, value in [(node.tag, " ".join(node.itertext()).strip()), *node.attrib.items()]:
            local = name.rsplit("}", 1)[-1]
            if local in {"DigitalSourceType", "CreatorTool", "Software", "Model", "Make"} and value:
                fields.setdefault(local, [])
                if value not in fields[local]:
                    fields[local].append(value[:2000])
    return fields, None


def explicit_declarations(exif: dict, xmps: list[dict]) -> list[dict]:
    evidence = []
    explicit = {s.casefold() for s in EXPLICIT_AI}
    for name, value in exif.items():
        text = str(value).strip()
        if text.casefold() in explicit:
            evidence.append({"source": "EXIF", "field": name, "value": text})
    for fields in xmps:
        for name, values in fields.items():
            for value in values:
                if (name == "DigitalSourceType" and value in AI_URIS) or value.casefold() in explicit:
                    evidence.append({"source": "XMP", "field": name, "value": value})
    return evidence


def signals(check: dict) -> dict:
    credential = check["c2pa"]
    return {
        "verified": credential["state"] == "verified" and credential["declared_ai"],
        "review": credential["declared_ai"] or bool(check["unsigned_ai_declarations"]),
    }


def inspect_provenance(path: str | Path) -> dict:
    source = Path(path)
    data = source.read_bytes()
    if data.startswith(b"\xff\xd8"):
        blocks = _jpeg(data)
    elif data.startswith(b"\x89PNG"):
        blocks = _png(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        blocks = _webp(data)
    else:
        blocks = []
    with Image.open(source) as image:
        exif = _exif_fields(image)
    xmp = []
    for block in blocks:
        if block["container"] == "XMP":
            fields, error = xmp_fields(block["payload"])
            xmp.append({"fields": fields, "parse_error": error})
    credential = inspect_c2pa(source)
    if any(b["container"] == "JUMBF" for b in blocks) and credential["state"] == "missing":
        credential.update(state="error", read_error="JUMBF container present, no readable active manifest")
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "c2pa": credential,
        "containers": [{k: b[k] for k in ("container", "id", "bytes")} for b in blocks],
        "exif": exif,
        "exif_present": any(b["container"] == "EXIF" for b in blocks) or exif["tag_count"] > 0,
        "xmp_present": bool(xmp),
        "xmp": xmp,
        "unsigned_ai_declarations": explicit_declarations(exif, [x["fields"] for x in xmp]),
        "policy": SDK_POLICY,
    }
