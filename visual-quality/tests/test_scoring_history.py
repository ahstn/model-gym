"""Current runtime imports must not require removed experiment code or local evidence."""

import os
import subprocess
import sys
from pathlib import Path

from visual_quality.scoring import history


def test_history_root_can_be_relocated(monkeypatch, tmp_path):
    monkeypatch.setenv("VISUAL_QUALITY_HISTORY", str(tmp_path))
    assert history.rehearsal_root() == tmp_path / "rehearsal-4000-2026-11-01"


def test_current_workflow_imports_without_restored_history(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "VISUAL_QUALITY_HISTORY": str(tmp_path / "missing"),
        "PYTHONPATH": os.pathsep.join(
            [str(root / "src"), str(root / "experiments/labeling-10000-2026-11-01")]
        ),
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import api_runner, api_profile, lite_run, reference_work, corpus_common, gallery_run; "
                "from visual_quality.scoring import wire, reference_review; "
                "assert not corpus_common.PREVIOUS.exists()"
            ),
        ],
        env=env,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=30,
    )
