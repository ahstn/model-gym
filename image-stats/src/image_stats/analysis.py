"""The entry point shared by labelling, teacher prompts, and the inference API."""

from __future__ import annotations

import hashlib

from dataclasses import dataclass, field
from pathlib import Path

from image_stats.flags import DEFAULT_THRESHOLDS, GATING_FLAGS, Thresholds, evaluate
from image_stats.loading import DEFAULT_MAX_EDGE, ImageInput, decode, fit
from image_stats.metadata import read_metadata
from image_stats.metrics import ALGORITHM_VERSION, blockiness, compute


@dataclass(frozen=True)
class Analysis:
    metrics: dict[str, float]
    flags: tuple[str, ...]
    max_edge: int
    native_dimensions: tuple[int, int]
    analysed_dimensions: tuple[int, int]
    colour_action: str
    exif_orientation: int
    version: str = field(default=ALGORITHM_VERSION)

    @property
    def gate(self) -> str | None:
        """The first flag that justifies skipping inference, if any."""
        return next((flag for flag in self.flags if flag in GATING_FLAGS), None)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "max_edge": self.max_edge,
            "native_dimensions": list(self.native_dimensions),
            "analysed_dimensions": list(self.analysed_dimensions),
            "colour_action": self.colour_action,
            "exif_orientation": self.exif_orientation,
            "metrics": {name: _round(value) for name, value in self.metrics.items()},
            "flags": list(self.flags),
            "gate": self.gate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Analysis:
        return cls(
            metrics=dict(data["metrics"]),
            flags=tuple(data["flags"]),
            max_edge=data["max_edge"],
            native_dimensions=tuple(data["native_dimensions"]),
            analysed_dimensions=tuple(data["analysed_dimensions"]),
            colour_action=data["colour_action"],
            exif_orientation=data["exif_orientation"],
            version=data["version"],
        )


def analyse(
    source: ImageInput, *, max_edge: int = DEFAULT_MAX_EDGE, thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> Analysis:
    """Analyse pixels. Pass the image the model will see so the statistics describe the same pixels."""
    decoded = decode(source)
    view = fit(decoded.rgb, max_edge)
    metrics = compute(decoded.rgb, view)
    return Analysis(
        metrics=metrics,
        flags=evaluate(metrics, thresholds),
        max_edge=max_edge,
        native_dimensions=(decoded.rgb.shape[1], decoded.rgb.shape[0]),
        analysed_dimensions=(view.shape[1], view.shape[0]),
        colour_action=decoded.colour_action,
        exif_orientation=decoded.exif_orientation,
    )


def analyse_file(
    path: str | Path,
    *,
    max_edge: int = DEFAULT_MAX_EDGE,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    source_path: str | Path | None = None,
) -> dict:
    """Return a JSON-ready record. `source_path` supplies file metadata when `path` is a re-encoded view."""
    data = Path(path).read_bytes()
    if source_path is None:
        metadata = read_metadata(data)
    else:
        source = Path(source_path).read_bytes()
        # Resizing destroys the 8x8 block phase, so measure compression on the original too.
        metadata = {**read_metadata(source), "source_blockiness": _round(blockiness(decode(source).rgb))}
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "analysis": analyse(data, max_edge=max_edge, thresholds=thresholds).to_dict(),
        "metadata": metadata,
    }


def _round(value: float) -> float:
    return float(f"{value:.6g}")
