"""Deterministic image statistics for labelling, teacher prompts, and inference."""

from image_stats.analysis import Analysis, analyse, analyse_file
from image_stats.describe import Observation, observations, render
from image_stats.evidence import EVIDENCE_NAMES, evidence_maps
from image_stats.flags import DEFAULT_THRESHOLDS, FLAG_TEXT, GATING_FLAGS, Thresholds
from image_stats.loading import DEFAULT_MAX_EDGE
from image_stats.metadata import read_metadata
from image_stats.metrics import ALGORITHM_VERSION, METRIC_NAMES, METRICS
from image_stats.reference import Reference

__all__ = [
    "ALGORITHM_VERSION",
    "DEFAULT_MAX_EDGE",
    "DEFAULT_THRESHOLDS",
    "EVIDENCE_NAMES",
    "FLAG_TEXT",
    "GATING_FLAGS",
    "METRICS",
    "METRIC_NAMES",
    "Analysis",
    "Observation",
    "Reference",
    "Thresholds",
    "analyse",
    "analyse_file",
    "evidence_maps",
    "observations",
    "read_metadata",
    "render",
]
