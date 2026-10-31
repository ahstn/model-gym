"""Evidence images a multimodal teacher can inspect beside the photo (after IQA-T1's tool outputs)."""

from __future__ import annotations

import cv2
import numpy as np

from PIL import Image

from image_stats.loading import DEFAULT_MAX_EDGE, ImageInput, decode, fit
from image_stats.metrics import tile_size, tiles

EVIDENCE_NAMES = ("sharpness", "noise_residual", "clipping", "spectrum")


def evidence_maps(source: ImageInput, *, max_edge: int = DEFAULT_MAX_EDGE) -> dict[str, Image.Image]:
    view = fit(decode(source).rgb, max_edge)
    luma = cv2.cvtColor(view.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return {
        "sharpness": _sharpness(view, luma),
        "noise_residual": _noise_residual(luma),
        "clipping": _clipping(luma),
        "spectrum": _spectrum(luma),
    }


def _sharpness(view: np.ndarray, luma: np.ndarray) -> Image.Image:
    """Per-tile Laplacian variance on a log scale, inferno colours, blended over a grey copy."""
    tile = tile_size(luma.shape)
    tile_var = tiles(cv2.Laplacian(luma, cv2.CV_32F), tile).var(axis=(2, 3))
    scaled = np.log1p(tile_var) / np.log1p(max(float(tile_var.max()), 1.0))
    heat = cv2.resize(
        np.clip(scaled * 255, 0, 255).astype(np.uint8),
        (tile_var.shape[1] * tile, tile_var.shape[0] * tile),
        interpolation=cv2.INTER_NEAREST,
    )
    colour = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    grey = np.repeat(luma[: heat.shape[0], : heat.shape[1], None], 3, axis=2)
    return Image.fromarray(np.clip(0.55 * colour + 0.45 * grey, 0, 255).astype(np.uint8))


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
