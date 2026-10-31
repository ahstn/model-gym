"""File-level facts that pixels cannot show. Read them from the original file, not from a re-encoded view."""

from __future__ import annotations

import io
import math

from pathlib import Path

from PIL import Image, JpegImagePlugin

# IJG Annex K luminance table; quality is estimated from the table's overall scale.
_STANDARD_LUMA_TABLE = (
    *(16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55),
    *(14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62),
    *(18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92),
    *(49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99),
)
_SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}
_EXIF_IFD = 0x8769
# Capture time, GPS, and serial numbers are deliberately not read.
_TAGS = {"camera_make": 271, "camera_model": 272, "software": 305}
_EXIF_TAGS = {"exposure_time_s": 33434, "f_number": 33437, "iso": 34855, "focal_length_mm": 37386}


def read_metadata(source: bytes | str | Path) -> dict:
    data = Path(source).read_bytes() if isinstance(source, str | Path) else bytes(source)
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        exif = image.getexif()
        details = exif.get_ifd(_EXIF_IFD)
        metadata = {
            "format": image.format,
            "mode": image.mode,
            "stored_dimensions": [width, height],
            "frames": int(getattr(image, "n_frames", 1)),
            "file_bytes": len(data),
            "bytes_per_megapixel": round(len(data) / (width * height / 1e6), 1),
            "has_alpha": "A" in image.getbands() or "transparency" in image.info,
            "has_icc_profile": image.info.get("icc_profile") is not None,
            "exif_present": len(exif) > 0,
            "exif_orientation": _number(exif.get(274)),
            **{key: _text(exif.get(tag)) for key, tag in _TAGS.items()},
            **{key: _number(details.get(tag)) for key, tag in _EXIF_TAGS.items()},
            "jpeg_quality_estimate": None,
            "jpeg_subsampling": None,
        }
        if image.format == "JPEG":
            metadata["jpeg_quality_estimate"] = _jpeg_quality(image)
            metadata["jpeg_subsampling"] = _SUBSAMPLING.get(JpegImagePlugin.get_sampling(image))
    return metadata


def _jpeg_quality(image: Image.Image) -> float | None:
    """Invert IJG table scaling (5000 / q below 50, 200 - 2q from 50); reads high below q20 as 8-bit tables clip."""
    table = getattr(image, "quantization", None) or {}
    if 0 not in table:
        return None
    scale = 100 * sum(table[0]) / sum(_STANDARD_LUMA_TABLE)
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return round(min(100.0, max(1.0, quality)), 1)


def _text(value: object) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        return None
    value = value.replace("\x00", "").strip()
    return value or None


def _number(value: object) -> float | None:
    if isinstance(value, tuple | list):
        value = value[0] if value else None
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) else None
