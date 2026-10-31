"""Combine independent eligibility reviews with preserved or new quality judgments."""

import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "prompt-ablation-2026-10-30"
DATA = HERE.parents[1] / "data" / HERE.name
TASKS = ("technical_quality", "aesthetics")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    spec = importlib.util.spec_from_file_location("reference_scope_wire", PREVIOUS / "wire.py")
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    manifest = json.loads((HERE / "manifest.json").read_text())
    old = {r["case_id"]: r for r in rows(PREVIOUS / "references.jsonl")}
    previous_freeze = json.loads((PREVIOUS / "references-freeze.json").read_text())
    assert previous_freeze["references_sha256"] == sha(PREVIOUS / "references.jsonl")
    reviews, reviewer_paths = {}, []
    for n in (1, 2):
        path = HERE / f"review-{n}.jsonl"
        reviewer_paths.append(path)
        packet = json.loads((DATA / f"review-packets/packet-{n}.json").read_text())
        assigned = {r["case_id"]: r for r in packet}
        submitted = rows(path)
        assert len(submitted) == 36 and {r["case_id"] for r in submitted} == set(assigned)
        for row in submitted:
            case_id = row["case_id"]
            assert case_id not in reviews and row["eligibility"] in {"eligible", "excluded", "uncertain"}
            assert row["view_sha256"] == assigned[case_id]["view_sha256"]
            assert isinstance(row["eligibility_evidence"], str) and row["eligibility_evidence"].strip()
            assert sha(Path(assigned[case_id]["path"])) == row["view_sha256"]
            if assigned[case_id]["needs_quality_reference"]:
                payload = {t: {**row[t], "score": f"{row[t]['score']:.2f}"} for t in TASKS}
                parsed = wire.parse_joint(json.dumps(payload))
                assert not parsed["errors"], (case_id, parsed["errors"])
                if row["eligibility"] == "excluded":
                    assert all(row[t]["score"] == 1.0 and row[t]["limitations"] == ["unsupported_domain"] for t in TASKS)
            else:
                assert not any(t in row for t in TASKS)
            reviews[case_id] = row
    assert len(reviews) == 72
    combined = []
    for case in manifest["cases"]:
        case_id = case["case_id"]
        review = reviews[case_id]
        if case_id in old:
            result = json.loads(json.dumps(old[case_id]))
            result["reference_origin"] = "previous_frozen_quality_plus_new_eligibility_review"
            if review["eligibility"] == "excluded":
                result["prior_quality_ratings"] = {task: result[task] for task in TASKS}
                for task in TASKS:
                    result[task] = {
                        "task": task, "status": "ok", "score": 1.0,
                        "evidence": "The visible content is an excluded non-photographic reproduction, so the photography eligibility rule sets the score floor.",
                        "limitations": ["unsupported_domain"],
                    }
        else:
            result = {"case_id": case_id, "view_sha256": case["view_sha256"], **{t: review[t] for t in TASKS}}
            result["reference_origin"] = "new_blind_assistant_quality_and_eligibility_review"
        result.update(
            eligibility=review["eligibility"],
            eligibility_evidence=review["eligibility_evidence"],
            photo_domain_eligible={"eligible": True, "excluded": False, "uncertain": None}[review["eligibility"]],
            human_gold=False,
        )
        assert result["view_sha256"] == case["view_sha256"]
        combined.append(result)
    target = HERE / "references.jsonl"
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined)
    if target.exists() and target.read_text() != content:
        raise ValueError("Existing assembled references differ; never silently replace them")
    target.write_text(content)
    inputs = [*reviewer_paths, HERE / "review-instructions.md", HERE / "photo-scope.txt", Path(__file__),
              PREVIOUS / "references.jsonl", PREVIOUS / "references-freeze.json"]
    inputs += [DATA / f"review-packets/packet-{n}.json" for n in (1, 2)]
    frozen = {
        "count": 72,
        "created_at": datetime.now(UTC).isoformat(),
        "reference_kind": "hybrid_assistant_quality_with_photo_scope",
        "human_gold": False,
        "references_sha256": sha(target),
        "manifest_sha256": sha(HERE / "manifest.json"),
        "inputs": {str(path): sha(path) for path in inputs},
        "blinding": "New reviewers saw only policy, baseline rubric, anonymous image packets and originals; no student responses or previous references.",
        "timing": "Independent reference work ran alongside inference. Prior 50 quality judgments predate this trial; new references were not model prompt inputs.",
        "excluded_case_ids": [r["case_id"] for r in combined if r["eligibility"] == "excluded"],
        "uncertain_case_ids": [r["case_id"] for r in combined if r["eligibility"] == "uncertain"],
        "note": "This display extension and user-specified floor are not an independent human accuracy benchmark.",
    }
    freeze_path = HERE / "references-freeze.json"
    if not freeze_path.exists():
        freeze_path.write_text(json.dumps(frozen, indent=2) + "\n")
    else:
        saved = json.loads(freeze_path.read_text())
        assert all(saved[k] == v for k, v in frozen.items() if k != "created_at")
    print(json.dumps({k: frozen[k] for k in ("count", "excluded_case_ids", "uncertain_case_ids", "references_sha256")}))


if __name__ == "__main__":
    main()
