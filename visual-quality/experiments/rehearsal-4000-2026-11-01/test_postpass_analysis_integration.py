"""Final analysis requires explicit postpass provenance rather than relaxing replay checks."""

import json

import analyze
import postpass_recovery
import pytest


def test_allowlist_loader_is_explicit_and_preserves_every_historical_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, "DATA", tmp_path)
    monkeypatch.setattr(analyze.api_runner, "ROUTES", ("luna",))
    monkeypatch.setattr(analyze.api_runner, "PHASES", ("rehearsal",))
    plan_path = tmp_path / "results/api/luna/rehearsal/plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps({"created_at": "2026-11-01T13:00:00+00:00"}))
    monkeypatch.setattr(analyze.api_runner, "validate_plan", lambda *args, **kwargs: None)

    def original_loader(plan):
        raise ValueError("Frozen loader rejects postpass sequences")

    monkeypatch.setattr(analyze.api_runner, "reconstruct", original_loader)
    with pytest.raises(ValueError, match="Frozen loader"):
        analyze.load_attempts(tmp_path, tmp_path / "manifest.json", [])
    with pytest.raises(ValueError, match="both allowlist"):
        analyze.load_attempts(tmp_path, tmp_path / "manifest.json", [], postpass_allowlist=tmp_path / "allowlist.json")
    approval = {"helper_sha256": {"helper.py": "helper-hash"}, "policy_path": "policy.json", "policy_sha256": "policy-hash"}
    calls = []

    def read(path, sha):
        calls.append((path, sha))
        return approval

    def reconstruct(plan, passed):
        assert passed is approval
        return [{"status": "unknown", "price_usd": None, "attempt_id": "old"},
                {"status": "valid", "price_usd": "0.001", "attempt_id": "new"}]

    monkeypatch.setattr(postpass_recovery, "read_allowlist", read)
    monkeypatch.setattr(postpass_recovery, "reconstruct", reconstruct)
    monkeypatch.setattr(postpass_recovery, "root", lambda: tmp_path / "postpass")
    rows, pins = analyze.load_attempts(tmp_path, tmp_path / "manifest.json", [],
        postpass_allowlist=tmp_path / "allowlist.json", postpass_allowlist_sha256="external-hash")
    assert calls == [(tmp_path / "allowlist.json", "external-hash")]
    assert [row["attempt_id"] for row in rows] == ["old", "new"]
    assert rows[0]["price_usd"] is None and rows[1]["price_usd"] == "0.001"
    assert pins[str(tmp_path / "allowlist.json")] == "external-hash"
    assert pins["helper.py"] == "helper-hash"
