"""Smaug plus explicit AI declarations, with independent diagnostic image statistics."""

from __future__ import annotations

import hashlib
import math

from pathlib import Path

from image_stats.analysis import analyse_file
from image_stats.authenticity.provenance import inspect_provenance, signals
from image_stats.authenticity.smaug import DEFAULT_THRESHOLD

CANDIDATE_ID = "smaug-ai-declarations-image-stats-v1"


def decide(probability_ai: float, provenance: dict, *, threshold: float = DEFAULT_THRESHOLD) -> dict:
    if not math.isfinite(probability_ai) or not 0 <= probability_ai <= 1:
        raise ValueError("Smaug probability must be finite and in [0, 1]")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    declaration = signals(provenance)
    pixel_flag = probability_ai >= threshold
    return {
        "candidate": CANDIDATE_ID,
        "threshold": threshold,
        "smaug_flag": pixel_flag,
        "ai_declaration_flag": declaration["review"],
        "verified_ai_declaration": declaration["verified"],
        "review_flag": pixel_flag or declaration["review"],
        "credential_state": provenance["c2pa"]["state"],
        "rule": "smaug_score >= threshold OR explicit AI declaration",
        "image_stats_role": "diagnostic_only",
    }


def inspect_image(path: str | Path, model, *, threshold: float = DEFAULT_THRESHOLD) -> dict:
    source = Path(path)
    statistics = analyse_file(source)
    provenance = inspect_provenance(source)
    prediction = model.predict(source)
    if (
        statistics["sha256"] != provenance["sha256"]
        or statistics["sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
    ):
        raise ValueError("Image bytes changed during inspection")
    return {
        "schema_version": 1,
        "image_path": str(source),
        "sha256": statistics["sha256"],
        "provenance": provenance,
        "image_stats": {"analysis": statistics["analysis"], "file_facts": statistics["metadata"]},
        "smaug": prediction,
        "decision": decide(prediction["probability_ai"], provenance, threshold=threshold),
    }


__all__ = ["CANDIDATE_ID", "decide", "inspect_image", "inspect_provenance"]
