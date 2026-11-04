"""Locations of restored historical evidence, separate from executable helpers."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PACKAGE_ROOT / "data"


def experiment_root() -> Path:
    """Return the directory containing restored experiment folders (no I/O at import)."""
    configured = os.environ.get("VISUAL_QUALITY_HISTORY")
    if configured:
        return Path(configured).expanduser().resolve()
    return DATA_ROOT / "experiment-history-2026-11-04/tree/visual-quality/experiments"


def rehearsal_root() -> Path:
    return experiment_root() / "rehearsal-4000-2026-11-01"
