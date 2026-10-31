"""Decode images with the same orientation, colour, and resize policy as the visual-quality views."""

from __future__ import annotations

import io

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PIL import Image, ImageCms, ImageOps

DEFAULT_MAX_EDGE = 1536
MIN_EDGE = 16

ImageInput = Image.Image | bytes | str | Path | np.ndarray


@dataclass(frozen=True)
class Decoded:
    """An oriented sRGB uint8 array at native resolution, with alpha composited on white."""

    rgb: np.ndarray
    colour_action: str
    exif_orientation: int


def decode(source: ImageInput) -> Decoded:
    if isinstance(source, np.ndarray):
        return Decoded(_from_array(source), "array assumed sRGB", 1)
    if isinstance(source, str | Path):
        source = Path(source).read_bytes()
    if isinstance(source, bytes | bytearray | memoryview):
        with Image.open(io.BytesIO(source)) as image:
            image.load()
            return _from_pil(image)
    if isinstance(source, Image.Image):
        return _from_pil(source)
    raise TypeError(f"Unsupported image input: {type(source).__name__}")


def fit(rgb: np.ndarray, max_edge: int) -> np.ndarray:
    """Downscale with Pillow's LANCZOS thumbnail, matching the frozen view policy; never upscale."""
    if type(max_edge) is not int or not 64 <= max_edge <= 8192:
        raise ValueError("max_edge must be an integer from 64 to 8192")
    if max(rgb.shape[:2]) <= max_edge:
        return rgb
    image = Image.fromarray(rgb)
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return np.asarray(image)


def _from_pil(image: Image.Image) -> Decoded:
    orientation = image.getexif().get(274, 1)
    icc = image.info.get("icc_profile")
    oriented = ImageOps.exif_transpose(image)
    has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
    alpha = oriented.convert("RGBA").getchannel("A") if has_alpha else None
    if oriented.mode.startswith("I;16") or oriented.mode == "I":
        # Pillow clips 16-bit greyscale when converting to RGB; rescale instead.
        wide = np.asarray(oriented, dtype=np.float64)
        top = 65535.0 if wide.max() > 255 else 255.0
        oriented = Image.fromarray(np.clip(wide / top * 255 + 0.5, 0, 255).astype(np.uint8), "L")
    rgb, colour_action = _to_srgb(oriented, icc)
    if alpha is not None:
        rgb = Image.composite(rgb, Image.new("RGB", rgb.size, "white"), alpha)
    return Decoded(np.array(rgb, dtype=np.uint8), colour_action, int(orientation or 1))


def _to_srgb(image: Image.Image, icc: bytes | None) -> tuple[Image.Image, str]:
    if icc is None:
        return image.convert("RGB"), "missing ICC assumed sRGB"
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        cms_input = image if image.mode in ("RGB", "CMYK", "LAB", "L") else image.convert("RGB")
        return ImageCms.profileToProfile(cms_input, profile, srgb, outputMode="RGB"), "embedded ICC converted to sRGB"
    except (ImageCms.PyCMSError, OSError, ValueError):
        return image.convert("RGB"), "unreadable ICC ignored; assumed sRGB"


def _from_array(array: np.ndarray) -> np.ndarray:
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (1, 3, 4)):
        raise ValueError("Expected an HxW, HxWx1, HxWx3, or HxWx4 array")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0, 1) * 255
    elif array.dtype != np.uint8:
        raise ValueError("Expected a uint8 array or a float array in [0, 1]")
    array = array.astype(np.float64)
    if array.ndim == 2:
        array = array[..., None]
    if array.shape[2] == 4:
        alpha = array[..., 3:] / 255
        array = array[..., :3] * alpha + 255 * (1 - alpha)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)
