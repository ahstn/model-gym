"""JPEG/PNG/WebP provenance containers, promoted from the frozen mug scanner."""

from __future__ import annotations

import struct
import zlib

from PIL import Image

XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"


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


def _exif_fields(image: Image.Image) -> dict:
    exif = image.getexif()
    names = {271: "Make", 272: "Model", 305: "Software", 270: "ImageDescription", 315: "Artist", 33432: "Copyright"}
    return {name: str(exif[tag])[:200] for tag, name in names.items() if tag in exif} | {"tag_count": len(exif)}
