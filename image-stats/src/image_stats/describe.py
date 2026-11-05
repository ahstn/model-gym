"""Plain-language observations for teacher prompts. Raw values mean little to a model; percentiles do."""

from __future__ import annotations

from dataclasses import dataclass

from image_stats.analysis import Analysis
from image_stats.flags import FLAG_TEXT
from image_stats.reference import Reference

HEADER = (
    "Automatic pixel measurements{source}. They can be wrong for unusual scenes, deliberate blur, or artistic "
    "intent; use them to check what you see, not to replace it."
)

# (metric, label, kind): "rank" needs a reference; "share" is an absolute fraction.
PROMPT_METRICS = (
    ("sharpness_peak", "Detail in the sharpest region", "rank"),
    ("focus_area", "Frame area close to peak detail", "share"),
    ("sharpness_median", "Detail across the frame", "rank"),
    ("gradient_anisotropy", "Directional smear (motion blur or strong parallel lines)", "rank"),
    ("noise_sigma", "Luma noise in flat areas", "rank"),
    ("chroma_noise_sigma", "Colour noise in flat areas", "rank"),
    ("luma_mean", "Brightness", "rank"),
    ("shadow_clip", "Pixels crushed to black", "share"),
    ("highlight_clip", "Pixels blown to white", "share"),
    ("rms_contrast", "Global contrast", "rank"),
    ("dark_channel_mean", "Haze or veiling glare", "rank"),
    ("colourfulness", "Colourfulness", "rank"),
)


@dataclass(frozen=True)
class Observation:
    metric: str
    label: str
    value: float
    percentile: float | None
    band: str | None
    text: str


def band(percentile: float) -> str:
    if percentile <= 5:
        return "very low"
    if percentile <= 25:
        return "low"
    if percentile < 75:
        return "typical"
    if percentile < 95:
        return "high"
    return "very high"


def observations(analysis: Analysis, reference: Reference | None = None) -> list[Observation]:
    if reference is not None:
        reference.check(analysis.version, analysis.max_edge)
    found = []
    for metric, label, kind in PROMPT_METRICS:
        value = analysis.metrics[metric]
        if kind == "share":
            found.append(Observation(metric, label, value, None, None, f"{label}: {value:.0%}"))
        elif reference is not None:
            rank = reference.percentile(metric, value)
            text = f"{label}: {band(rank)} ({_ordinal(round(rank))} percentile)"
            found.append(Observation(metric, label, value, rank, band(rank), text))
    return found


def render(analysis: Analysis, reference: Reference | None = None, *, header: str = HEADER) -> str:
    source = f", ranked against {reference.count} images in '{reference.name}'" if reference else ""
    lines = [header.format(source=source)]
    lines += [f"- {item.text}" for item in observations(analysis, reference)]
    lines += [f"- Note: {FLAG_TEXT[flag]}" for flag in analysis.flags]
    return "\n".join(lines)


def _ordinal(number: int) -> str:
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"
