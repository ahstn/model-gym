"""Evidence images a multimodal teacher can inspect beside the photo (after IQA-T1's and Tool-IQA's tool outputs)."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from PIL import Image

from image_stats.loading import DEFAULT_MAX_EDGE, ImageInput, decode, fit
from image_stats.metrics import tile_size, tiles

EVIDENCE_NAMES = ("sharpness", "detail_crop", "gamma", "noise_residual", "clipping", "spectrum")
# Tile Laplacian variance drawn at full intensity; typical sharp photo tiles reach 1e3-1e4 at 1536 px.
_SHARPNESS_FULL_SCALE = 20_000.0
# Native pixels on each side of the detail crop, capped by the image's short edge.
_CROP_SIDE = 512
# The gamma view maps median luma to mid-grey, with the exponent held within these limits.
_GAMMA_TARGET = 0.5
_GAMMA_LIMITS = (0.25, 4.0)
_CROP_OUTLINE = (0, 230, 255)


class _Box(NamedTuple):
    top: int
    left: int
    side: int


def evidence_maps(source: ImageInput, *, max_edge: int = DEFAULT_MAX_EDGE) -> dict[str, Image.Image]:
    native = decode(source).rgb
    view = fit(native, max_edge)
    luma = cv2.cvtColor(view.astype(np.float32), cv2.COLOR_RGB2GRAY)
    tile = tile_size(luma.shape)
    tile_var = tiles(cv2.Laplacian(luma, cv2.CV_32F), tile).var(axis=(2, 3))
    crop = _detail_box(tile_var, tile, view.shape, native.shape)
    return {
        "sharpness": _sharpness(luma, tile_var, tile, _scale_box(crop, native.shape, view.shape)),
        "detail_crop": Image.fromarray(native[crop.top : crop.top + crop.side, crop.left : crop.left + crop.side]),
        "gamma": _gamma(view, luma),
        "noise_residual": _noise_residual(luma),
        "clipping": _clipping(luma),
        "spectrum": _spectrum(luma),
    }


def _detail_box(tile_var: np.ndarray, tile: int, view_shape: tuple[int, ...], native_shape: tuple[int, ...]) -> _Box:
    """A native-resolution square centred on the sharpest view tile, so it keeps detail the downscale removed."""
    row, col = np.unravel_index(np.argmax(tile_var), tile_var.shape)
    side = min(_CROP_SIDE, *native_shape[:2])
    y_scale, x_scale = native_shape[0] / view_shape[0], native_shape[1] / view_shape[1]
    top = round((row + 0.5) * tile * y_scale - side / 2)
    left = round((col + 0.5) * tile * x_scale - side / 2)
    return _Box(
        int(np.clip(top, 0, native_shape[0] - side)),
        int(np.clip(left, 0, native_shape[1] - side)),
        side,
    )


def _scale_box(box: _Box, native_shape: tuple[int, ...], view_shape: tuple[int, ...]) -> _Box:
    scale = view_shape[1] / native_shape[1]
    return _Box(round(box.top * scale), round(box.left * scale), max(1, round(box.side * scale)))


def _sharpness(luma: np.ndarray, tile_var: np.ndarray, tile: int, crop: _Box) -> Image.Image:
    """Per-tile Laplacian variance on a fixed log scale, so a blurred frame stays dark rather than self-normalised.

    The detail crop's region is outlined in cyan.
    """
    scaled = np.log1p(tile_var) / np.log1p(_SHARPNESS_FULL_SCALE)
    heat = cv2.resize(
        np.clip(scaled * 255, 0, 255).astype(np.uint8),
        (tile_var.shape[1] * tile, tile_var.shape[0] * tile),
        interpolation=cv2.INTER_NEAREST,
    )
    colour = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    grey = np.repeat(luma[: heat.shape[0], : heat.shape[1], None], 3, axis=2)
    blended = np.clip(0.55 * colour + 0.45 * grey, 0, 255).astype(np.uint8)
    corner = (crop.left + crop.side - 1, crop.top + crop.side - 1)
    cv2.rectangle(blended, (crop.left, crop.top), corner, _CROP_OUTLINE, thickness=2)
    return Image.fromarray(blended)


def _gamma(view: np.ndarray, luma: np.ndarray) -> Image.Image:
    """Gamma-corrected view that brings median luma to mid-grey: lifts dark frames to expose hidden noise,
    banding, and crushed detail, and darkens bright frames toward their highlights."""
    median = float(np.clip(np.median(luma) / 255, 1 / 255, 254 / 255))
    exponent = float(np.clip(np.log(_GAMMA_TARGET) / np.log(median), *_GAMMA_LIMITS))
    table = np.rint(255 * (np.arange(256) / 255) ** exponent).astype(np.uint8)
    return Image.fromarray(table[view])


def _noise_residual(luma: np.ndarray) -> Image.Image:
    """Absolute difference from a 3x3 median, amplified 8x: grain and compression texture stand out."""
    smooth = cv2.medianBlur(np.clip(np.rint(luma), 0, 255).astype(np.uint8), 3).astype(np.float32)
    return Image.fromarray(np.clip(np.abs(luma - smooth) * 8, 0, 255).astype(np.uint8), "L")


def _clipping(luma: np.ndarray) -> Image.Image:
    """Grey image with crushed shadows painted blue and blown highlights painted red."""
    grey = np.clip(np.rint(luma * 0.6 + 50), 0, 255).astype(np.uint8)
    rgb = np.repeat(grey[..., None], 3, axis=2)
    rgb[luma <= 5] = (40, 90, 255)
    rgb[luma >= 250] = (255, 40, 40)
    return Image.fromarray(rgb)


def _spectrum(luma: np.ndarray) -> Image.Image:
    """Centred log-magnitude spectrum of a 512-pixel square crop; periodic artefacts show as off-axis spots."""
    side = min(512, *luma.shape)
    top, left = (luma.shape[0] - side) // 2, (luma.shape[1] - side) // 2
    crop = luma[top : top + side, left : left + side]
    window = np.outer(np.hanning(side), np.hanning(side))
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2((crop - crop.mean()) * window))))
    peak = magnitude.max()
    scaled = magnitude / peak * 255 if peak > 0 else magnitude
    return Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8), "L")
