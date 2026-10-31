"""Model-free pixel statistics. Every value is a float computed deterministically on the CPU."""

from __future__ import annotations

import math

from dataclasses import dataclass

import cv2
import numpy as np

from image_stats.loading import MIN_EDGE

# Bump whenever a formula, threshold inside a formula, or analysis default changes.
ALGORITHM_VERSION = "image-stats-v1"

_EPS = 1e-9
_TILES_ON_LONG_EDGE = 16
_FLAT_TILE_STD = 2.0  # luma levels (0-255) below which a tile counts as flat
_SHADOW_LEVEL, _HIGHLIGHT_LEVEL = 5.0, 250.0
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_BLOCKINESS_MAX_PIXELS = 16_000_000


@dataclass(frozen=True)
class Metric:
    name: str
    group: str
    unit: str
    description: str


METRICS: tuple[Metric, ...] = (
    Metric("luma_mean", "exposure", "0-1", "Mean Rec. 601 luma."),
    Metric("luma_median", "exposure", "0-1", "Median luma."),
    Metric("luma_p01", "exposure", "0-1", "1st percentile of luma (black point)."),
    Metric("luma_p99", "exposure", "0-1", "99th percentile of luma (white point)."),
    Metric("shadow_clip", "exposure", "fraction", "Share of pixels with luma at or below 5/255."),
    Metric("highlight_clip", "exposure", "fraction", "Share of pixels with luma at or above 250/255."),
    Metric("channel_clip", "exposure", "fraction", "Share of pixels with any RGB channel at or above 254."),
    Metric("dark_channel_mean", "exposure", "0-1", "Mean dark channel (He et al. haze prior); high = haze or veil."),
    Metric("rms_contrast", "contrast", "0-1", "Standard deviation of luma."),
    Metric("dynamic_range", "contrast", "0-1", "luma_p99 - luma_p01."),
    Metric("michelson_contrast", "contrast", "0-1", "(p99 - p01) / (p99 + p01) of luma."),
    Metric("local_contrast", "contrast", "0-1", "Median per-tile luma standard deviation."),
    Metric("luma_entropy", "contrast", "bits", "Shannon entropy of the 256-bin luma histogram."),
    Metric("laplacian_var", "sharpness", "luma^2", "Variance of the 3x3 Laplacian over the whole frame."),
    Metric("sharpness_peak", "sharpness", "luma^2", "Laplacian variance of the sharpest tile."),
    Metric("sharpness_median", "sharpness", "luma^2", "Median tile Laplacian variance."),
    Metric(
        "focus_area",
        "sharpness",
        "fraction",
        "Share of tiles with at least a quarter of the peak tile's Laplacian variance.",
    ),
    Metric("tenengrad", "sharpness", "luma^2", "Mean squared Sobel gradient magnitude."),
    Metric(
        "high_frequency_ratio", "sharpness", "fraction", "Spectral energy above 1/4 Nyquist over all non-DC energy."
    ),
    Metric("edge_density", "sharpness", "fraction", "Share of Canny (50/150) edge pixels."),
    Metric(
        "gradient_anisotropy",
        "sharpness",
        "0-1",
        "Structure-tensor coherence of the whole frame; high = one dominant gradient direction.",
    ),
    Metric("noise_sigma", "noise", "luma levels", "Immerkaer noise sigma, 10th percentile over unclipped tiles."),
    Metric("chroma_noise_sigma", "noise", "levels", "Immerkaer sigma of Cr and Cb, 10th percentile over tiles."),
    Metric("colourfulness", "colour", "0-255", "Hasler-Suesstrunk colourfulness."),
    Metric("saturation_mean", "colour", "0-1", "Mean HSV saturation."),
    Metric("chroma_p99", "colour", "Lab", "99th percentile of Lab chroma; near 0 = monochrome."),
    Metric("cast_a", "colour", "Lab", "Mean Lab a* (green - / red +)."),
    Metric("cast_b", "colour", "Lab", "Mean Lab b* (blue - / yellow +)."),
    Metric("flat_area", "content", "fraction", "Share of tiles whose luma standard deviation is below 2/255."),
    Metric("palette_size", "content", "count", "4-bit-per-channel RGB bins needed to cover 95% of pixels."),
    Metric("top_colour_share", "content", "fraction", "Share of pixels in the most common 4-bit RGB bin."),
    Metric(
        "blockiness",
        "compression",
        "ratio",
        "Mean luma step across 8x8 block boundaries over steps inside blocks, at native resolution; 1 = none.",
    ),
)
METRIC_NAMES = tuple(metric.name for metric in METRICS)


def compute(native: np.ndarray, view: np.ndarray) -> dict[str, float]:
    """Compute every metric; `native` is the decoded image, `view` the copy fitted to the analysis edge."""
    if min(view.shape[:2]) < MIN_EDGE:
        raise ValueError(f"Images must be at least {MIN_EDGE} pixels on each side")
    rgb = view.astype(np.float32)
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tile = tile_size(luma.shape)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    values = {
        **_exposure(rgb, luma, view),
        **_contrast(luma, tile),
        **_sharpness(luma, gx, gy, tile),
        **_noise(rgb, luma, tile),
        **_colour(rgb),
        **_content(view, luma, tile),
        "blockiness": blockiness(native),
    }
    return {name: float(values[name]) for name in METRIC_NAMES}


def tiles(plane: np.ndarray, tile: int) -> np.ndarray:
    """Return a (rows, cols, tile, tile) view; partial tiles at the right and bottom are dropped."""
    rows, cols = plane.shape[0] // tile, plane.shape[1] // tile
    if rows == 0 or cols == 0:
        tile = min(plane.shape[:2])
        rows, cols = plane.shape[0] // tile, plane.shape[1] // tile
    return plane[: rows * tile, : cols * tile].reshape(rows, tile, cols, tile).swapaxes(1, 2)


def tile_size(shape: tuple[int, ...]) -> int:
    return min(max(MIN_EDGE, round(max(shape[:2]) / _TILES_ON_LONG_EDGE)), min(shape[:2]))


def _exposure(rgb: np.ndarray, luma: np.ndarray, view: np.ndarray) -> dict[str, float]:
    p01, median, p99 = np.percentile(luma, [1, 50, 99])
    small = _shrink(rgb / 255, 512)
    dark = cv2.erode(small.min(axis=2), np.ones((15, 15), np.uint8), borderType=cv2.BORDER_REFLECT)
    return {
        "luma_mean": luma.mean(dtype=np.float64) / 255,
        "luma_median": median / 255,
        "luma_p01": p01 / 255,
        "luma_p99": p99 / 255,
        "shadow_clip": np.mean(luma <= _SHADOW_LEVEL),
        "highlight_clip": np.mean(luma >= _HIGHLIGHT_LEVEL),
        "channel_clip": np.mean(view.max(axis=2) >= 254),
        "dark_channel_mean": dark.mean(dtype=np.float64),
    }


def _contrast(luma: np.ndarray, tile: int) -> dict[str, float]:
    p01, p99 = np.percentile(luma, [1, 99])
    levels = np.clip(np.rint(luma), 0, 255).astype(np.uint8)
    hist = np.bincount(levels.ravel(), minlength=256) / levels.size
    hist = hist[hist > 0]
    return {
        "rms_contrast": luma.std(dtype=np.float64) / 255,
        "dynamic_range": (p99 - p01) / 255,
        "michelson_contrast": (p99 - p01) / (p99 + p01) if p99 + p01 > _EPS else 0.0,
        "local_contrast": np.median(tiles(luma, tile).std(axis=(2, 3))) / 255,
        "luma_entropy": -np.sum(hist * np.log2(hist)),
    }


def _sharpness(luma: np.ndarray, gx: np.ndarray, gy: np.ndarray, tile: int) -> dict[str, float]:
    laplacian = cv2.Laplacian(luma, cv2.CV_32F)
    tile_var = tiles(laplacian, tile).var(axis=(2, 3), dtype=np.float64)
    peak = tile_var.max()
    jxx, jyy, jxy = (float(np.sum(a * b, dtype=np.float64)) for a, b in ((gx, gx), (gy, gy), (gx, gy)))
    trace = jxx + jyy
    return {
        "laplacian_var": laplacian.var(dtype=np.float64),
        "sharpness_peak": peak,
        "sharpness_median": np.median(tile_var),
        "focus_area": np.mean(tile_var >= 0.25 * peak) if peak > _EPS else 0.0,
        "tenengrad": np.mean(gx * gx + gy * gy, dtype=np.float64),
        "high_frequency_ratio": _high_frequency_ratio(luma),
        "edge_density": np.mean(cv2.Canny(np.clip(np.rint(luma), 0, 255).astype(np.uint8), 50, 150) > 0),
        "gradient_anisotropy": math.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / trace if trace > _EPS else 0.0,
    }


def _high_frequency_ratio(luma: np.ndarray) -> float:
    height, width = luma.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.rfft2((luma - luma.mean()) * window)) ** 2
    radius = np.hypot(np.fft.fftfreq(height)[:, None], np.fft.rfftfreq(width)[None, :]) / 0.5
    band = (radius > 0.02) & (radius <= 1.0)
    total = power[band].sum()
    return float(power[band & (radius > 0.25)].sum() / total) if total > _EPS else 0.0


def _noise(rgb: np.ndarray, luma: np.ndarray, tile: int) -> dict[str, float]:
    ycrcb = cv2.cvtColor(rgb / 255, cv2.COLOR_RGB2YCrCb) * 255
    level_tiles = tiles(luma, tile)
    clipped = ((level_tiles <= _SHADOW_LEVEL) | (level_tiles >= _HIGHLIGHT_LEVEL)).mean(axis=(2, 3))
    usable = clipped < 0.5
    return {
        "noise_sigma": _immerkaer_floor(luma, tile, usable),
        "chroma_noise_sigma": sum(
            _immerkaer_floor(np.ascontiguousarray(ycrcb[..., channel]), tile, usable) for channel in (1, 2)
        )
        / 2,
    }


def _immerkaer_floor(plane: np.ndarray, tile: int, usable: np.ndarray) -> float:
    """Immerkaer (1996) sigma per tile; the 10th percentile approximates the noise in flat regions."""
    residual = np.abs(cv2.filter2D(plane, cv2.CV_32F, _IMMERKAER, borderType=cv2.BORDER_REFLECT))
    per_tile = tiles(residual, tile).mean(axis=(2, 3), dtype=np.float64) * math.sqrt(math.pi / 2) / 6
    selected = per_tile[usable]
    return float(np.percentile(selected, 10)) if selected.size else 0.0


def _colour(rgb: np.ndarray) -> dict[str, float]:
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = red - green, 0.5 * (red + green) - blue
    unit = rgb / 255
    saturation = cv2.cvtColor(unit, cv2.COLOR_RGB2HSV)[..., 1]
    lab = cv2.cvtColor(unit, cv2.COLOR_RGB2Lab)
    return {
        "colourfulness": math.hypot(rg.std(dtype=np.float64), yb.std(dtype=np.float64))
        + 0.3 * math.hypot(rg.mean(dtype=np.float64), yb.mean(dtype=np.float64)),
        "saturation_mean": saturation.mean(dtype=np.float64),
        "chroma_p99": np.percentile(np.hypot(lab[..., 1], lab[..., 2]), 99),
        "cast_a": lab[..., 1].mean(dtype=np.float64),
        "cast_b": lab[..., 2].mean(dtype=np.float64),
    }


def _content(view: np.ndarray, luma: np.ndarray, tile: int) -> dict[str, float]:
    bins = view >> 4
    index = (bins[..., 0].astype(np.int32) << 8) | (bins[..., 1].astype(np.int32) << 4) | bins[..., 2]
    counts = np.sort(np.bincount(index.ravel(), minlength=4096))[::-1]
    covered = np.cumsum(counts) / index.size
    return {
        "flat_area": np.mean(tiles(luma, tile).std(axis=(2, 3)) < _FLAT_TILE_STD),
        "palette_size": int(np.searchsorted(covered, 0.95)) + 1,
        "top_colour_share": counts[0] / index.size,
    }


def blockiness(native: np.ndarray) -> float:
    height, width = native.shape[:2]
    if height * width > _BLOCKINESS_MAX_PIXELS:
        # A centred crop whose origin stays on the 8x8 grid keeps the JPEG block phase.
        side = int(math.sqrt(_BLOCKINESS_MAX_PIXELS)) // 8 * 8
        top, left = max(0, (height - side) // 2) // 8 * 8, max(0, (width - side) // 2) // 8 * 8
        native = native[top : top + side, left : left + side]
    if min(native.shape[:2]) < 32:
        return 1.0
    luma = cv2.cvtColor(native.astype(np.float32), cv2.COLOR_RGB2GRAY)
    across, inside = 0.0, 0.0
    for steps in (np.abs(np.diff(luma, axis=1)), np.abs(np.diff(luma, axis=0)).T):
        boundary = np.zeros(steps.shape[1], dtype=bool)
        boundary[7::8] = True
        across += steps[:, boundary].mean(dtype=np.float64)
        inside += steps[:, ~boundary].mean(dtype=np.float64)
    return across / inside if inside > 1e-3 else 1.0


def _shrink(image: np.ndarray, edge: int) -> np.ndarray:
    scale = edge / max(image.shape[:2])
    if scale >= 1:
        return image
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
