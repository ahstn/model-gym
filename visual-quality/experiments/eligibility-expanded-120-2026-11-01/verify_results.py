"""Validate frozen inputs, split integrity, attempt coverage and paired report metrics."""

from __future__ import annotations

import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path

import analyze_expanded as a

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-expanded-120-2026-11-01")


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def t(x):
    return datetime.datetime.fromisoformat(x)


def main():
    freeze = read(HERE / "trial-freeze.json")
    assert all(sha(p) == h for p, h in freeze["inputs"].items())
    assert sha(HERE / "selection-rule.json") == read(HERE / "selection-rule-freeze.json")["sha256"]
    assert sha(HERE / "analyze_expanded.py") == read(HERE / "selection-analysis-freeze.json")["sha256"]
    assert sha(HERE / "winner.json") == read(HERE / "winner-freeze.json")["sha256"]
    manifest = read(HERE / "reference-manifest.json")
    winner = read(HERE / "winner.json")
    result = read(HERE / "results.json")
    assert result["winner_sha256"] == sha(HERE / "winner.json") and result["reference_sha256"] == sha(
        HERE / "reference-manifest.json"
    )
    panels = {
        part: [x for x in manifest["cases"] if x["partition"] == part] for part in ("development", "heldout")
    }
    assert all(
        len(cases) == 120 and Counter(c["eligibility"] for c in cases) == {"eligible": 60, "excluded": 60}
        for cases in panels.values()
    )
    assert not ({x["family_id"] for x in panels["development"]} & {x["family_id"] for x in panels["heldout"]})
    for c in manifest["cases"]:
        assert sha(c["image_path"]) == c["view_sha256"]
    for rid in ["original", "previous", *[f"r{i}" for i in range(11, 21)]]:
        for model in a.MODELS:
            for c in panels["development"]:
                assert (DATA / "attempts" / rid / model / c["case_id"] / "01/request.json").exists(), (
                    rid,
                    model,
                    c["case_id"],
                )
    requests = list((DATA / "attempts").glob("holdout-*/*/*/*/request.json"))
    first = min(t(read(p)["started_at"]) for p in requests)
    assert t(read(HERE / "winner-freeze.json")["frozen_at"]) < first
    assert t(read(HERE / "boundary-sensitivity.json")["recorded_at"]) < first
    variants = read(HERE / "variants.json")
    past = read(HERE / "previous-prompts.json")
    for model in a.MODELS:
        selected = winner["models"][model]["round"]
        expected = hashlib.sha256(
            (past[model]["question"] if selected == "previous" else variants[selected]).encode()
        ).hexdigest()
        for p in (DATA / "attempts" / "holdout-selected" / model).glob("*/*/request.json"):
            assert read(p)["question_sha256"] == expected
        for arm in ("original", "previous", "selected"):
            rows = [x for x in result["case_rows"] if x["model"] == model and x["display_arm"] == arm]
            assert len(rows) == 120
            valid = [x for x in rows if x["status"] == "valid" and x["scope_floor_applied"] is not None]
            m = result["models"][model]["arms"][arm]
            assert len(valid) == m["valid"]
            assert (
                sum(x["reference_eligibility"] == "excluded" and not x["scope_floor_applied"] for x in valid)
                == m["missed_exclusions"]
            )
            assert (
                sum(x["reference_eligibility"] == "eligible" and x["scope_floor_applied"] for x in valid)
                == m["false_floors"]
            )
    out = {
        "frozen_inputs_verified": len(freeze["inputs"]),
        "development_and_holdout_cases": {p: len(v) for p, v in panels.items()},
        "family_disjoint": True,
        "all_240_input_view_hashes_verified": True,
        "all_2880_development_jobs_attempted": True,
        "holdout_first_request": first.isoformat(),
        "winner_frozen_at": read(HERE / "winner-freeze.json")["frozen_at"],
        "boundary_sensitivity_declared_at": read(HERE / "boundary-sensitivity.json")["recorded_at"],
        "selected_questions_match_frozen_winner": True,
        "independent_holdout_recount_agrees": True,
    }
    (HERE / "verification.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
