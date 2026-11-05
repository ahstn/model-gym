"""Scale-free flags. Sharpness and noise are deliberately absent: judge them against a corpus reference."""

from __future__ import annotations

import math

from collections.abc import Mapping
from dataclasses import dataclass

# Flags that justify skipping a model call and applying the caller's policy floor.
GATING_FLAGS = ("uniform_frame",)

FLAG_TEXT = {
    "uniform_frame": "The frame is nearly uniform; almost no image content is present.",
    "near_black": "Almost the whole frame is very dark.",
    "near_white": "Almost the whole frame is very bright.",
    "shadow_clipping": "A large share of the frame is crushed to black.",
    "highlight_clipping": "A large share of the frame is blown to white.",
    "graphic_like": "Few colours and large flat areas: possibly a graphic, document, or screenshot.",
    "monochrome": "The image has no meaningful colour.",
    "blocking": "Regular 8x8 block edges suggest strong JPEG compression.",
}


@dataclass(frozen=True)
class Thresholds:
    """Uniform-frame values come from the 72-image reviewed trial; the rest are initial guesses."""

    uniform_edge_density: float = 0.0005
    uniform_structure: float = 2.0  # luma levels of tile contrast left after removing noise
    near_black_p99: float = 0.08
    near_white_p01: float = 0.92
    shadow_clip: float = 0.4
    highlight_clip: float = 0.2
    graphic_palette_size: int = 24
    graphic_flat_area: float = 0.5
    graphic_min_p99: float = 0.25
    monochrome_chroma_p99: float = 4.0
    blocking: float = 1.5


DEFAULT_THRESHOLDS = Thresholds()


def evaluate(metrics: Mapping[str, float], thresholds: Thresholds = DEFAULT_THRESHOLDS) -> tuple[str, ...]:
    t = thresholds
    checks = {
        "uniform_frame": metrics["edge_density"] < t.uniform_edge_density and structure(metrics) < t.uniform_structure,
        "near_black": metrics["luma_p99"] < t.near_black_p99,
        "near_white": metrics["luma_p01"] > t.near_white_p01,
        "shadow_clipping": metrics["shadow_clip"] >= t.shadow_clip,
        "highlight_clipping": metrics["highlight_clip"] >= t.highlight_clip,
        # Dark photos also collapse into few 4-bit colour bins, so require some bright content.
        "graphic_like": metrics["palette_size"] <= t.graphic_palette_size
        and metrics["flat_area"] >= t.graphic_flat_area
        and metrics["luma_p99"] >= t.graphic_min_p99,
        "monochrome": metrics["chroma_p99"] < t.monochrome_chroma_p99,
        "blocking": metrics["blockiness"] >= t.blocking,
    }
    return tuple(name for name in FLAG_TEXT if checks[name])


def structure(metrics: Mapping[str, float]) -> float:
    """Median tile contrast in luma levels with the noise floor removed, so sensor noise is not content."""
    local = metrics["local_contrast"] * 255
    return math.sqrt(max(local * local - metrics["noise_sigma"] ** 2, 0.0))
