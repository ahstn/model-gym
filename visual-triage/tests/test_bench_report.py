import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_report as report


def result(identifier, task, arm="decider", **overrides):
    return {
        "sample": identifier,
        "task": task,
        "arm": arm,
        "requested_reasoning": None if arm == "decider" else {"effort": "low"},
        "prediction": {
            "technical_quality": 3.4,
            "aesthetics": 2.6,
            "ai_probability": 0.8,
            "medium": "photo",
            "uncertainty": "high",
        },
        "valid": True,
        "effective_cost_usd": 0.002,
        "cost_usd": 0.0,
        "seconds": 1.23,
        "provider": "Example",
        "delivered_sha256": "hash",
        **overrides,
    }


def test_report_preserves_task_labels_reference_limits_and_separate_costs(tmp_path):
    samples = {
        "quality-000": {
            "id": "quality-000",
            "image": "data/quality-000.jpg",
            "labels": {
                "technical_quality": {
                    "value": 0.5,
                    "scale": [0.01, 1],
                    "normalized_1_to_5": 2.98,
                    "caveat": "Rated at a different view.",
                }
            },
        },
        "ai-000": {"id": "ai-000", "labels": {"ai_generated": False}},
    }
    references = {
        "quality-000": {"technical_quality": 3.1, "aesthetics": 2.8, "evidence": "Sharp | clean <scene>"},
        "ai-000": {
            "ai_probability": 0.6,
            "ai_probability_interval": [0.4, 0.8],
            "evidence": "Origin is uncertain.",
        },
    }
    rows = [result("quality-000", "quality", arm) for arm in report.MODELS] + [result("ai-000", "ai")]
    markdown = report.render_report({"per_image": rows}, references, samples, tmp_path / "report.md")
    assert "## Quality and aesthetics" in markdown
    assert "## AI-origin detection" in markdown
    assert "Dataset origin label: **non-generative**" in markdown
    assert "native 0.5000 on [0.01, 1]; 1–5 value 2.98" in markdown
    assert "Human aesthetics: not available." in markdown
    assert "Rated at a different view." in markdown
    assert "Sharp \\| clean &lt;scene&gt;" in markdown
    assert "GPT-6 Luna (low)" in markdown
    assert "$0.0020000 | $0.0000000 | 1.230 s" in markdown
    assert "Gemini 3.8 Flash — no attempt recorded" in markdown
    assert "[Open delivered image]" in markdown


def test_invalid_outputs_keep_cost_without_showing_untrusted_partial_scores():
    rows = [result("one", "quality", valid=False)]
    rendered = "\n".join(report.prediction_table(rows, "quality"))
    assert "invalid output | — | — | $0.0020000" in rendered
    assert "3.40" not in rendered
    assert "2.60" not in rendered


def test_manifest_image_and_hash_mismatch_fail_without_inspecting_pixels(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        report.image_heading({"id": "one", "image": "../outside.jpg"}, tmp_path / "report.md")
    with pytest.raises(ValueError, match="hash"):
        report.render_report(
            {"per_image": [result("one", "quality")]},
            {"one": {"delivered_sha256": "different"}},
            {"one": {"id": "one"}},
            tmp_path / "report.md",
        )
