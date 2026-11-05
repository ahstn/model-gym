"""Verify the consolidated profile keeps evaluated model and output contracts."""

import json
import shutil
from pathlib import Path

import pytest

from visual_quality.labeling_profile import load_profile, request_payload

PROFILE = Path(__file__).resolve().parents[1] / "labeling"
PNG = b"\x89PNG\r\n\x1a\nexample"


def test_current_profile_builds_both_teacher_requests_without_losing_the_schema():
    profile = load_profile(PROFILE)
    for model in ("luna", "deepseek"):
        payload = request_payload(profile, model, PNG)
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert payload["messages"][1]["content"][0]["text"] == profile["teacher"]["user"]
    assert request_payload(profile, "deepseek", PNG)["reasoning_effort"] == "high"
    assert request_payload(profile, "luna", PNG)["reasoning"]["effort"] == "high"


def test_current_profile_preserves_two_scores_and_explicit_exclusion_decisions():
    profile = load_profile(PROFILE)
    for model in ("liquid", "jev"):
        payload = request_payload(profile, model, PNG)
        assert set(payload["questions"]) == {"technical_quality", "aesthetics", "excluded"}
        question = payload["questions"]["excluded"]
        assert question["type"] == "noul"
        assert "YES" in question["instructions"] and "NO" in question["instructions"]
        assert payload["state"][0] == profile["decision"]["common"]
    assert profile["decision"]["exclusion_threshold"] == 0.5


def test_current_profile_fails_closed_for_bad_image_and_unknown_model():
    profile = load_profile(PROFILE)
    with pytest.raises(ValueError, match="PNG"):
        request_payload(profile, "luna", b"not an image")
    with pytest.raises(ValueError, match="Unknown"):
        request_payload(profile, "gemini", PNG)


def test_current_profile_rejects_mismatched_decision_model(tmp_path):
    profile_dir = tmp_path / "labeling"
    shutil.copytree(PROFILE / "prompts", profile_dir / "prompts")
    shutil.copy2(PROFILE / "routes.json", profile_dir / "routes.json")
    routes = json.loads((profile_dir / "routes.json").read_text())
    routes["routes"]["jev"]["requested_model"] = "liquid/d1.1"
    (profile_dir / "routes.json").write_text(json.dumps(routes))
    with pytest.raises(ValueError, match="must match"):
        load_profile(profile_dir)
