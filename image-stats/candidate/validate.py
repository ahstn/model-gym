"""Validate the portable candidate records and exact retained image bytes."""

from __future__ import annotations

import hashlib
import json

from collections import Counter
from pathlib import Path

from image_stats.authenticity import decide
from image_stats.authenticity.smaug import MODEL_ID, REVISION


def validate(root: Path) -> dict:
    index = json.loads((root / "images.json").read_text())["images"]
    generation = json.loads((root / "generation.json").read_text())["images"]
    expected_summary = json.loads((root / "capture.json").read_text())["summary"]
    assert len(index) == len({r["id"] for r in index}) == 100
    assert Counter(r["label"] for r in index) == {"real": 50, "fake": 50}
    assert len(generation) == len({r["id"] for r in generation}) == 50
    assert {r["id"] for r in generation} == {r["id"] for r in index if r["label"] == "fake"}
    for row in generation:
        assert hashlib.sha256(row["prompt"].encode()).hexdigest() == row["prompt_sha256"]
        assert not row["reference_images"]
    summary, states, views = {}, Counter(), 0
    for item in index:
        record = json.loads((root / item["record"]).read_text())
        assert record["id"] == item["id"] and record["native_sha256"] == item["native_sha256"]
        assert record["training_allowed"] is False and len(record["views"]) == 5
        for view in record["views"]:
            path = (root / view["image_path"]).resolve()
            assert path.is_relative_to(root.resolve())
            with path.open("rb") as handle:
                assert hashlib.file_digest(handle, "sha256").hexdigest() == view["sha256"]
            assert view["image_stats"]["file_facts"]["file_bytes"] == path.stat().st_size
            assert (view["smaug"]["model"], view["smaug"]["revision"]) == (MODEL_ID, REVISION)
            assert view["decision"] == decide(view["smaug"]["probability_ai"], view["provenance"])
            if view["view"] != "native":
                assert not view["decision"]["ai_declaration_flag"]
                assert not any(b["container"] in {"EXIF", "XMP", "JUMBF"} for b in view["provenance"]["containers"])
            tally = summary.setdefault(
                view["view"],
                {"real": 0, "fake": 0, "smaug_fake_caught": 0, "combined_fake_caught": 0, "combined_real_flagged": 0},
            )
            tally[record["label"]] += 1
            tally["smaug_fake_caught"] += int(record["label"] == "fake" and view["decision"]["smaug_flag"])
            tally["combined_fake_caught"] += int(record["label"] == "fake" and view["decision"]["review_flag"])
            tally["combined_real_flagged"] += int(record["label"] == "real" and view["decision"]["review_flag"])
            states[view["provenance"]["c2pa"]["state"]] += 1
            views += 1
    assert views == 500 and summary == expected_summary
    return {
        "status": "passed",
        "images": len(index),
        "views": views,
        "fake_generation_records": len(generation),
        "credential_states": dict(states),
        "summary": summary,
    }


if __name__ == "__main__":
    print(json.dumps(validate(Path(__file__).resolve().parent), indent=2))
