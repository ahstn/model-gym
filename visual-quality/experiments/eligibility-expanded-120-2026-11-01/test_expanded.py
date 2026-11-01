"""Verify development selection protects against false floors and missing-result bias."""

import json

import analyze_expanded as subject


def test_select_requires_false_floor_guard_and_penalizes_missing(tmp_path, monkeypatch):
    cases = [
        {
            "case_id": f"I{i:03d}",
            "partition": "development",
            "eligibility": "eligible" if i < 60 else "excluded",
            "group": "test",
        }
        for i in range(120)
    ]

    def rows(misses, false=None, missing=None):
        result = []
        for i, c in enumerate(cases):
            eligible = i < 60
            floor = i in misses or (false is not None and i == false)
            # The excluded default is to apply the floor; listed misses are exceptions.
            if not eligible:
                floor = i not in misses
            result.append(
                {
                    "case_id": c["case_id"],
                    "group": "test",
                    "reference_eligibility": c["eligibility"],
                    "status": "missing" if i == missing else "valid",
                    "scope_floor_applied": None if i == missing else floor,
                    "technical_native": 3.0,
                    "aesthetic_native": 3.0,
                }
            )
        return result

    previous = rows({60, 61, 62})
    bad_floor = rows({60}, false=0)
    good = rows({60, 61})
    missing = rows({60, 61}, missing=5)

    def get(stage, model, cases):
        return {"original": previous, "previous": previous, "r11": bad_floor, "r12": good, "r13": missing}[
            stage
        ]

    monkeypatch.setattr(subject, "HERE", tmp_path)
    monkeypatch.setattr(subject, "data", get)
    (tmp_path / "reference-manifest.json").write_text(json.dumps({"cases": cases}))
    (tmp_path / "variants.json").write_text(json.dumps({"r11": "bad", "r12": "good", "r13": "short"}))
    (tmp_path / "trial-freeze.json").write_text(json.dumps({"inputs": {}}))
    (tmp_path / "selection-rule.json").write_text("{}")
    (tmp_path / "selection-rule-freeze.json").write_text(
        json.dumps({"sha256": subject.sha(tmp_path / "selection-rule.json")})
    )
    subject.select()
    w = json.loads((tmp_path / "winner.json").read_text())
    for model in subject.MODELS:
        assert w["models"][model]["round"] == "r12"
        assert not w["development"][model]["rounds"]["r11"]["qualifies"]
        assert w["development"][model]["rounds"]["r13"]["missing_on_previous_valid"] == 1
        assert w["development"][model]["rounds"]["r13"]["qualifies"]
