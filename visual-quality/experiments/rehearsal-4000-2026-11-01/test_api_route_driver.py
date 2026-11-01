"""Phase sequencing must halt before another billable phase on any failed check."""

import copy
import sys

import api_route_driver as driver
import pytest


@pytest.mark.parametrize("first_status,verify_fails", [("complete", False), ("blocked", False), ("complete", True)])
def test_driver_requires_completed_verified_phase_before_advancing(monkeypatch, tmp_path, first_status, verify_fails):
    calls, writes = [], []
    monkeypatch.setattr(sys, "argv", ["driver", "--route", "deepseek"])
    monkeypatch.setattr(driver.runner, "directory", lambda route, phase=None: tmp_path / (phase or "route"))
    monkeypatch.setattr(driver.runner, "read", lambda path: {"phase": path.parent.name})
    monkeypatch.setattr(driver.runner, "write", lambda path, value: writes.append(copy.deepcopy(value)))

    def execute(plan):
        calls.append(("execute", plan["phase"]))
        return {"status": first_status if plan["phase"] == "rehearsal" else "complete"}

    def validate(plan, **_):
        calls.append(("validate", plan["phase"]))
        if verify_fails:
            raise ValueError("synthetic changed pinned input")

    monkeypatch.setattr(driver.runner, "execute", execute)
    monkeypatch.setattr(driver.runner, "validate_plan", validate)
    monkeypatch.setattr(driver.runner, "reconstruct", lambda plan: [])
    monkeypatch.setattr(driver.runner, "save", lambda plan, records: {"status": "complete"})
    if first_status != "complete" or verify_fails:
        with pytest.raises((RuntimeError, ValueError)):
            driver.main()
        assert ("execute", "benchmark") not in calls
        assert writes[-1]["status"] == "stopped_requires_audit"
    else:
        driver.main()
        assert calls == [("execute", "rehearsal"), ("validate", "rehearsal"),
                         ("execute", "benchmark"), ("validate", "benchmark")]
        assert writes[-1]["status"] == "complete"
