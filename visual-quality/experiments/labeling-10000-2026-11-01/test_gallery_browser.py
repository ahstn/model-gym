"""Check that a timed-out browser cannot leave children writing its profile."""

import json
import subprocess
import sys
from pathlib import Path

import gallery_run
import pytest

validator = gallery_run.validate_gallery


def test_browser_command_preserves_output_and_return_code():
    result = validator.run_browser(
        [
            sys.executable,
            "-c",
            "import sys; print('result'); print('diagnostic', file=sys.stderr); sys.exit(3)",
        ],
        timeout=5,
    )
    assert (result.returncode, result.stdout, result.stderr) == (3, "result\n", "diagnostic\n")


def test_browser_timeout_stops_descendants_before_returning(tmp_path):
    child_pid = tmp_path / "child.pid"
    child_script = "import time; time.sleep(60)"
    parent_script = (
        "import pathlib, subprocess, sys, time; "
        f"child=subprocess.Popen([sys.executable, '-c', {child_script!r}]); "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        validator.run_browser([sys.executable, "-c", parent_script], timeout=0.5)
    pid = int(child_pid.read_text())
    status = Path(f"/proc/{pid}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"


def test_browser_result_still_requires_success(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html><head></head><body></body></html>")
    result = {"status": "failed", "loadedPreviews": 9999, "errors": ["missing preview"]}
    monkeypatch.setattr(
        validator,
        "run_browser",
        lambda command: subprocess.CompletedProcess(command, 0, json.dumps(result), ""),
    )
    args = validator.argparse.Namespace(gallery=tmp_path, chromium=Path("unused-chromium"))
    with pytest.raises(ValueError, match="Browser validation failed"):
        validator.browser_check(args)
