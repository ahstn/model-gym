"""Apply recorded visual review decisions without changing frozen benchmark views."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime

from corpus_common import DATA, digest, write_json


def main():
    path = DATA / "manifest.json"
    manifest = json.loads(path.read_text())
    proposal = json.loads((DATA / "final-replacements-proposal.json").read_text())
    if proposal["status"] != "visually_approved":
        raise ValueError("Replacement views and neighbours require explicit visual review")
    if digest(path) != proposal["parent_manifest_sha256"]:
        raise ValueError("Replacement review belongs to a different corpus manifest")
    write_json(DATA / "review" / "corpus-before-final-replacements.json", manifest)
    old = {row["case_id"]: row for row in manifest["cases"]}
    replacements = {}
    for replacement in proposal["replacements"]:
        case_id = replacement["case_id"]
        before, after = old[case_id], replacement["new_record"]
        assert before["benchmark_order"] is None and int(case_id[1:]) > 5000
        assert after["benchmark_order"] is None and after["case_id"] == case_id
        assert before["image_id"] == replacement["old_image_id"]
        assert before["category"] == after["category"]
        assert digest(after["image_path"]) == after["view_sha256"]
        replacements[case_id] = after
    manifest["cases"] = [replacements.get(row["case_id"], row) for row in manifest["cases"]]
    assert len({row["image_id"] for row in manifest["cases"]}) == 10000
    assert len({row["view_sha256"] for row in manifest["cases"]}) == 10000
    assert Counter(row["category"] for row in manifest["cases"]) == manifest["counts"]["categories"]
    manifest["status"] = "frozen_after_visual_review_with_documented_uncertainties"
    manifest["frozen_utc"] = datetime.now(UTC).isoformat()
    manifest["final_review"] = {
        "replacement_plan_sha256": digest(DATA / "final-replacements-proposal.json"),
        "replaced_incoming_rehearsal_views": len(replacements),
        "frozen_benchmark_views_changed": 0,
        "limitation": "Reviewed neighbours and contact sheets do not establish exhaustive photographer/session independence. Distinct nearly blank views remain unresolved duplicate candidates.",
    }
    write_json(path, manifest)
    for filename, rows in [
        ("inference-rehearsal-manifest.json", [r for r in manifest["cases"] if r["benchmark_order"] is None]),
        ("new-rehearsal-manifest.json", [r for r in manifest["cases"] if int(r["case_id"][1:]) > 5000]),
    ]:
        write_json(
            DATA / filename,
            {
                "parent_manifest": str(path),
                "parent_manifest_sha256": digest(path),
                "cases": rows,
                "image_count": len(rows),
                "contains_reference_scores": False,
            },
        )
    print(json.dumps({"manifest": str(path), "replaced": len(replacements), "rehearsal": 8500}))


if __name__ == "__main__":
    main()
