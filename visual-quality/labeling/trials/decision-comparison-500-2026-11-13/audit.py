"""Independently join and revalidate all requests, responses and final exports."""

import base64
import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare import DATA, sha, write

from visual_quality.scoring.api_contract import encoded, interpret


def main():
    plan = json.loads((DATA / "plan.json").read_text())
    cases = json.loads((DATA / "subset.json").read_text())
    exported = [json.loads(line) for line in (DATA / "exports/labelled-images.jsonl").open()]
    assert len(cases) == len(exported) == 500
    assert len({c["case_id"] for c in cases}) == len({c["family_id"] for c in cases}) == 500
    assert len({c["content_cluster"] for c in cases}) == 64
    assert Counter(c["selection_arm"] for c in cases) == {"high_disagreement": 375, "diversity_sample": 125}
    assert sha(DATA / "subset.json") == plan["subset_sha256"]
    for path, expected in plan["source_pins"].items():
        assert sha(path) == expected
    count = 0
    for original, result in zip(cases, exported, strict=True):
        assert original["case_id"] == result["case_id"]
        assert 1 <= original["benchmark_order"] <= 1000
        assert original["image_stats"] == result["image_stats"]
        assert result["training_ready"] is False and not any(result["task_masks"].values())
        for model, old in original["assessments"].items():
            assert result["assessments"][model] == old
        image = Path(original["trial_image_path"]).read_bytes()
        assert hashlib.sha256(image).hexdigest() == original["view_sha256"]
        for model, route in plan["routes"].items():
            body = encoded(
                {
                    "model": route["requested_model"],
                    "state": [
                        plan["common"],
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + base64.b64encode(image).decode(),
                                "detail": "high",
                            },
                        },
                    ],
                    "questions": plan["questions"],
                }
            )
            row = result["assessments"][model]
            assert row["request_sha256"] == hashlib.sha256(body).hexdigest()
            assert row["plan_sha256"] == sha(DATA / "plan.json")
            response_path = Path(row["raw_response_path"])
            assert row["raw_response_sha256"] == sha(response_path)
            response = json.loads(response_path.read_text())
            parsed = interpret(response, route)
            assert parsed == response["parsed"]
            assert all(row[key] == value for key, value in parsed.items())
            assert row["latency_seconds"] == response["latency_seconds"] > 0
            assert row["training_ready"] is False
            attempts = sorted(response_path.parent.parent.glob("*/request.json"))
            assert len(attempts) == row["attempt_count"] <= plan["max_attempts"]
            for path in attempts:
                request = json.loads(path.read_text())
                assert request["request_sha256"] == row["request_sha256"]
                assert request["view_sha256"] == original["view_sha256"]
                assert path.with_name("result.json").exists()
            count += 1
    gallery = (DATA / "gallery/index.html").read_text()
    assert gallery.count("<article ") == gallery.count('<img loading="lazy"') == 500
    assert gallery.count("<tr>") == 5000
    audit = {
        "images_verified": 500,
        "new_observations_reparsed": count,
        "historical_observations_unchanged": 3500,
        "gallery_cards": 500,
        "gallery_assessor_rows": 4500,
        "plan_sha256": sha(DATA / "plan.json"),
        "report_sha256": sha(DATA / "report.json"),
        "export_sha256": sha(DATA / "exports/labelled-images.jsonl"),
        "source_code_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
    }
    write(DATA / "audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
