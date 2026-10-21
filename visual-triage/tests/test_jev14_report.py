"""Keep Jev's original views, compatibility calls and missing results separate."""

import copy
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracks.jev14 import report
from tracks.jev14.client import HEADERS, MODEL, REVISION, image_info, make_payload, normalize


def prediction(task, value):
    return (
        {"technical_quality": value, "aesthetics": 3.0}
        if task == "quality"
        else {"ai_probability": value, "medium": "photo", "uncertainty": "medium"}
    )


@pytest.fixture
def cohort():
    cases, manifests, references, baselines, records = [], {}, {}, {}, []
    for task in report.TASKS:
        manifests[task], references[task], baselines[task] = {}, {}, []
        for index in range(2):
            identifier = f"{task}-{index}"
            sha = (str(index) if task == "quality" else str(index + 2)) * 64
            sample = {
                "id": identifier,
                "delivered_sha256": sha,
                "labels": {"technical_quality": {"value": 3.0, "scale": [1, 5]}}
                if task == "quality"
                else {"ai_generated": bool(index)},
            }
            manifests[task][identifier] = sample
            references[task][identifier] = {
                "id": identifier,
                "delivered_sha256": sha,
                "technical_quality": 3.0,
                "aesthetics": 3.0,
            }
            case = {
                "phase": "primary",
                "task": task,
                "sample": identifier,
                "delivered_sha256": sha,
                "parent_delivered_sha256": sha,
                "image_width": 512,
                "image_height": 341,
                "image_bytes": 1000,
            }
            cases.append(case)
            value = 3.5 if task == "quality" else [0.1, 0.8][index]
            row = {
                **case,
                "arm": "jev14",
                "status": 200,
                "normalized": prediction(task, value),
                "seconds": 0.5 + index,
                "usage": {"cost": 0.01, "input_tokens": 500},
                "cost_usd": 0.01,
                "timestamp": "2026-10-15T12:00:00+00:00",
            }
            records.append(row)
            for arm in report.BASELINES:
                baselines[task].append({**row, "arm": arm, "ledger": "old.jsonl"})
    return cases, records, manifests, references, baselines


def test_primary_comparison_and_compact_export(cohort):
    summary, cases = report.assemble_report(*cohort)
    assert summary["status"] == "complete"
    assert summary["operations"]["attempted"] == 4
    quality = summary["primary"]["quality"]["jev14"]
    assert quality["metrics"]["assistant_reference"]["technical_quality"]["mae_valid"] == 0.5
    assert quality["metrics"]["human_reference"]["aesthetics"]["referenced"] == 0
    ai = summary["primary"]["ai"]["jev14"]["metrics"]
    assert ai["accuracy_valid_only"] == 1
    assert ai["brier_valid_only"] == pytest.approx(0.025)
    assert ai["auroc_valid_only"] == 1
    assert cases[0]["input_tokens"] == 500
    assert "response" not in cases[0] and "usage" not in cases[0]


def test_missing_and_invalid_are_not_negative_ai_predictions(cohort):
    cases, records, *rest = cohort
    records.pop()
    records[-1].update(status=502, normalized=None, validation_error="HTTP failure", cost_usd=None, usage={})
    summary, exported = report.assemble_report(cases, records, *rest)
    result = summary["primary"]["ai"]["jev14"]
    assert summary["status"] == "partial"
    assert result["unattempted"] == 1
    assert result["operations"]["invalid"] == 1
    assert result["metrics"]["accuracy_valid_only"] is None
    assert result["metrics"]["accuracy_all_labeled_invalid_as_wrong"] == 0
    assert result["metrics"]["brier_all_expected_invalid_penalty1"] == 1
    assert result["operations"]["effective_provider_inference_cost_usd"] is None
    assert [row["disposition"] for row in exported[-2:]] == ["invalid", "unattempted"]
    assert all(row["ai_probability"] is None for row in exported[-2:])
    assert summary["primary"]["ai"]["paired_valid_ids"] == []


def test_compatibility_and_size_probes_are_not_pooled(cohort):
    cases, records, *rest = cohort
    for phase in ("compatibility", "size_probe"):
        case = {**cases[0], "phase": phase, "delivered_sha256": phase[0] * 64}
        cases.append(case)
        records.append({**records[0], **case, "normalized": prediction("quality", 1.0)})
    summary, exported = report.assemble_report(cases, records, *rest)
    assert summary["expected"] == 6
    assert summary["primary"]["quality"]["expected"] == 2
    assert (
        summary["primary"]["quality"]["jev14"]["metrics"]["assistant_reference"]["technical_quality"][
            "mae_valid"
        ]
        == 0.5
    )
    assert summary["phases"]["compatibility"]["attempted"] == 1
    assert summary["operations"]["effective_provider_inference_cost_usd"] == pytest.approx(0.06)
    assert exported[-1]["technical_quality"] == 1.0


def test_paired_baselines_use_only_valid_jev_ids(cohort):
    cases, records, *rest = cohort
    records.pop()
    summary, _ = report.assemble_report(cases, records, *rest)
    ai = summary["primary"]["ai"]
    assert ai["expected"] == 2
    assert ai["paired_valid_ids"] == ["ai-0"]
    assert ai["baselines"]["decider"]["expected"] == 2
    assert ai["paired_valid_baselines"]["decider"]["expected"] == 1


def test_finished_run_keeps_invalid_output_visible(cohort):
    cohort[1][0].update(normalized=None, validation_error="Probability sum differs")
    summary, cases = report.assemble_report(*cohort)
    assert summary["status"] == "complete_with_invalid"
    assert summary["operations"]["attempted"] == 4
    assert summary["operations"]["valid"] == 3
    assert summary["failures"][0]["sample"] == "quality-0"
    assert "Probability sum differs" in report.render_markdown(summary, cases)


@pytest.mark.parametrize("target", ["cases", "records", "baselines"])
def test_duplicate_identity_rejected(cohort, target):
    cases, records, manifests, references, baselines = cohort
    if target == "cases":
        cases.append(copy.deepcopy(cases[0]))
    elif target == "records":
        records.append(copy.deepcopy(records[0]))
    else:
        baselines["quality"].append(copy.deepcopy(baselines["quality"][0]))
    with pytest.raises(ValueError):
        report.assemble_report(cases, records, manifests, references, baselines)


def test_changed_primary_pixels_rejected(cohort):
    cohort[0][0]["delivered_sha256"] = "f" * 64
    cohort[1][0]["delivered_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="view differs"):
        report.assemble_report(*cohort)


def test_changed_baseline_pixels_rejected(cohort):
    cohort[4]["quality"][0]["delivered_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="hash does not match"):
        report.assemble_report(*cohort)


def test_missing_baseline_is_rejected(cohort):
    cohort[4]["quality"].pop()
    with pytest.raises(ValueError, match="does not cover"):
        report.assemble_report(*cohort)


def test_byok_upstream_cost_is_not_reported_as_zero(cohort):
    cohort[1][0].update(
        cost_usd=0, usage={"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.02}}
    )
    summary, cases = report.assemble_report(*cohort)
    assert cases[0]["effective_cost_usd"] == 0.02
    assert summary["operations"]["effective_provider_inference_cost_usd"] == pytest.approx(0.05)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def frozen(tmp_path, cohort):
    cases, records, manifests, references, baselines = cohort
    for case in cases:
        path = tmp_path / "data" / f"{case['sample']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (320, 320)).save(path)
        case.update(
            image=str(path.relative_to(tmp_path)),
            delivered_sha256=report.sha(path),
            parent_delivered_sha256=report.sha(path),
            case_id=f"primary/{case['sample']}",
            benchmark=case["task"],
            **image_info(path),
        )
        manifests[case["task"]][case["sample"]]["delivered_sha256"] = case["delivered_sha256"]
        references[case["task"]][case["sample"]]["delivered_sha256"] = case["delivered_sha256"]
        for row in baselines[case["task"]]:
            if row["sample"] == case["sample"]:
                row["delivered_sha256"] = case["delivered_sha256"]
    sources, prompts = {}, {}
    for task in report.TASKS:
        questions = (
            {
                key: {"type": "score", "criteria": [str(i) for i in range(5)], "instructions": "Rate it"}
                for key in ("technical_quality", "aesthetics")
            }
            if task == "quality"
            else {
                "ai_probability": {"type": "noul", "instructions": "Detect AI"},
                "medium": {
                    "type": "choice",
                    "criteria": {"photo": "Photo", "graphic": "Graphic"},
                    "instructions": "Classify",
                },
                "uncertainty": {
                    "type": "choice",
                    "criteria": {"low": "Low", "medium": "Medium"},
                    "instructions": "Assess",
                },
            }
        )
        prompts[task] = {"id": task, "task": task, "context": "Judge pixels", "questions": questions}
        for relative, document in [
            (f"evidence/benchmarks/{task}.json", {"samples": list(manifests[task].values())}),
            (f"evidence/assessments/{task}-60.json", {"samples": list(references[task].values())}),
            (f"prompts/final/{task}.json", prompts[task]),
        ]:
            write_json(tmp_path / relative, document)
            sources[relative] = {
                "sha256": report.sha(tmp_path / relative),
                "bytes": (tmp_path / relative).stat().st_size,
            }
        relative = f"evidence/final/{task}.jsonl"
        write_rows(tmp_path / relative, baselines[task])
        sources[relative] = {
            "sha256": report.sha(tmp_path / relative),
            "bytes": (tmp_path / relative).stat().st_size,
        }
    write_json(tmp_path / "evidence/samples.json", {"samples": []})
    manifest_path = tmp_path / "data/jev14-v1/manifest.json"
    write_json(manifest_path, {"cases": cases, "primary_exclusions": []})
    protocol_path = tmp_path / "tracks/jev14/protocol.json"
    protocol = {
        "manifest_sha256": report.sha(manifest_path),
        "source_inputs": sources,
        "execution_sha256": {},
        "primary_counts": {"quality": 2, "ai": 2},
        "compatibility_cases": 0,
        "size_probe_cases": 0,
        "maximum_attempts": 4,
        "request_headers": HEADERS,
        "model": MODEL,
        "expected_revision": REVISION,
        "context_tokens": 262144,
    }
    write_json(protocol_path, protocol)
    for index, (row, case) in enumerate(zip(records, cases), 1):
        prompt = prompts[case["task"]]
        answers = {}
        for key, question in prompt["questions"].items():
            kind, value = question["type"], row["normalized"][key]
            answers[key] = {
                "type": kind,
                {"score": "score", "noul": "noul", "choice": "choice"}[kind]: value - 1
                if kind == "score"
                else value,
            }
        response = {"answers": answers, "model": REVISION, "provider": "TypeSafe", "usage": row["usage"]}
        payload = make_payload(tmp_path, case, prompt)
        row.update(
            case,
            attempt=index,
            response=response,
            normalized=normalize(response, prompt["questions"]),
            request_headers=HEADERS,
            endpoint=report.DECISIONS,
            request_bytes=len(json.dumps(payload).encode()),
            request_sha256=report.digest(
                {"endpoint": report.DECISIONS, "headers": HEADERS, "payload": payload}
            ),
            prompt_sha256=report.digest(prompt),
            protocol_sha256=report.sha(protocol_path),
            manifest_sha256=protocol["manifest_sha256"],
            resolved_model=REVISION,
            provider="TypeSafe",
        )
    ledger = tmp_path / "data/jev14-v1/attempts.jsonl"
    write_rows(ledger, records)
    return tmp_path, ledger, records


def test_build_report_checks_frozen_evidence_and_writes_immutable_snapshot(frozen):
    root, _, _ = frozen
    summary, cases = report.build_report(root)
    assert summary["primary_context"]["max_input_tokens"] == 500
    output = root / "report"
    report.write_report(root, output)
    assert (output / "cases.csv").is_file()
    prose = (output / "README.md").read_text()
    assert "not human ground truth" in prose
    assert "Unknown does not mean free" in prose
    assert "512-pixel source" in prose
    assert "forwarding by OpenRouter was not independently confirmed" in prose
    assert "0.19%" in prose
    assert cases[0]["disposition"] == "valid"
    with pytest.raises(FileExistsError):
        report.write_report(root, output)


@pytest.mark.parametrize(
    "mutation", ["order", "view", "protocol", "request", "response", "provider", "usage", "cost"]
)
def test_corrupt_raw_record_is_rejected(frozen, mutation):
    root, ledger, rows = frozen
    if mutation == "order":
        rows.reverse()
    elif mutation == "view":
        rows[0]["delivered_sha256"] = "f" * 64
    elif mutation == "protocol":
        rows[0]["protocol_sha256"] = "f" * 64
    elif mutation == "request":
        rows[0]["request_sha256"] = "f" * 64
    elif mutation == "response":
        rows[0]["normalized"]["technical_quality"] = 1.0
    elif mutation == "provider":
        rows[0]["provider"] = "Another provider"
    elif mutation == "usage":
        rows[0]["response"] = copy.deepcopy(rows[0]["response"])
        rows[0]["response"]["usage"]["input_tokens"] = 999
    else:
        rows[0]["cost_usd"] = 10.0
    write_rows(ledger, rows)
    with pytest.raises(ValueError):
        report.build_report(root)


def test_unknown_cost_failure_still_has_a_report(frozen):
    root, ledger, rows = frozen
    row = rows[0]
    row.update(
        status=429,
        normalized=None,
        validation_error="HTTP failure",
        usage={},
        cost_usd=None,
        effective_cost_usd=None,
        resolved_model=None,
        provider=None,
        response={"error": {"message": "Rate limited"}},
    )
    write_rows(ledger, [row])
    summary, cases = report.build_report(root)
    assert summary["status"] == "partial"
    assert summary["unattempted"] == 3
    assert summary["operations"]["unknown_cost_count"] == 1
    assert summary["operations"]["effective_provider_inference_cost_usd"] is None
    assert cases[0]["http_status"] == 429


def test_changed_pinned_source_rejected(frozen):
    root, _, _ = frozen
    (root / "prompts/final/ai.json").write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        report.build_report(root)


def test_parser_review_applies_to_every_row_and_preserves_raw_records(frozen):
    root, _, rows = frozen
    rows[0].update(
        normalized=None, validation_error="ValueError: Probability distribution does not sum to one"
    )
    before = copy.deepcopy(rows)
    prompts = {task: json.loads((root / f"prompts/final/{task}.json").read_text()) for task in report.TASKS}
    called = []

    def parser(response, questions):
        called.append(response)
        return normalize(response, questions)

    reviewed, audit = report.review_records(rows, prompts, REVISION, parser)
    assert len(called) == len(rows)
    assert audit == {
        "applied_to_responses": 4,
        "original_strict_parser_rejections": 1,
        "recovered_strict_parser_rejections": 1,
        "revised_valid": 4,
    }
    assert (
        reviewed[0]["strict_validation_error"] == "ValueError: Probability distribution does not sum to one"
    )
    assert reviewed[0]["normalized"]["technical_quality"] == 3.5
    assert "validation_error" not in reviewed[0]
    assert rows == before


def test_parser_review_cannot_recover_wrong_revision_or_http_failure(frozen):
    root, _, rows = frozen
    rows[0].update(status=502, normalized=None, validation_error="HTTP failure")
    rows[1].update(resolved_model="wrong", normalized=None, validation_error="Wrong model")
    prompts = {task: json.loads((root / f"prompts/final/{task}.json").read_text()) for task in report.TASKS}
    reviewed, audit = report.review_records(rows, prompts, REVISION, normalize)
    assert all(row["normalized"] is None for row in reviewed[:2])
    assert audit["recovered_strict_parser_rejections"] == 0
    assert audit["revised_valid"] == 2


def test_recorded_parser_review_must_match_uniform_reparse(frozen):
    root, _, rows = frozen
    rows[0]["reviewed_normalized"] = {"invented": 1}
    prompts = {task: json.loads((root / f"prompts/final/{task}.json").read_text()) for task in report.TASKS}
    with pytest.raises(ValueError, match="uniform response parsing"):
        report.review_records(rows, prompts, REVISION, normalize)


@pytest.fixture
def amended(frozen):
    root, ledger, rows = frozen
    track = root / "tracks/jev14"
    for name in ("rounding.py", "resume.py"):
        (track / name).write_bytes((report.ROOT / "tracks/jev14" / name).read_bytes())
    rows[0]["response"]["answers"]["technical_quality"]["probabilities"] = {
        "0": 0,
        "1": 0.10,
        "2": 0.40,
        "3": 0.40,
        "4": 0.09,
    }
    rows[0].update(
        normalized=None, validation_error="ValueError: Probability distribution does not sum to one"
    )
    write_rows(ledger, rows[:2])
    amendment = {
        "protocol_sha256": report.sha(track / "protocol.json"),
        "prefix_rows": 2,
        "prefix_sha256": report.sha(ledger),
        "parser_sha256": report.sha(track / "rounding.py"),
        "resume_sha256": report.sha(track / "resume.py"),
        "maximum_attempts": 4,
    }
    write_json(track / "parser-amendment.json", amendment)
    for row in rows[2:]:
        row.update(
            reviewed_normalized=copy.deepcopy(row["normalized"]),
            amendment_sha256=report.sha(track / "parser-amendment.json"),
        )
    write_rows(ledger, rows)
    return frozen


def test_rounding_amendment_reparses_every_response_without_changing_values(amended):
    root, ledger, _ = amended
    before = ledger.read_bytes()
    summary, cases = report.build_report(root)
    review = summary["parser_review"]
    assert summary["status"] == "complete"
    assert review["original_strict_parser_rejections"] == 1
    assert review["recovered_strict_parser_rejections"] == 1
    assert review["applied_to_responses"] == 4
    assert review["revised_valid"] == 4
    assert cases[0]["technical_quality"] == 3.5
    assert cases[0]["parser_review_applied"] is True
    assert cases[0]["strict_parser_valid"] is False
    assert all(row["parser_review_applied"] for row in cases)
    assert "no values were renormalised" in report.render_markdown(summary, cases)
    assert ledger.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    ["prefix", "parser", "resume", "amendment", "missing_review", "reviewed_value", "reviewed_error"],
)
def test_invalid_parser_amendment_evidence_rejected(amended, mutation):
    root, ledger, rows = amended
    track = root / "tracks/jev14"
    if mutation == "prefix":
        rows[0]["seconds"] = 10
    elif mutation in {"parser", "resume"}:
        (track / ("rounding.py" if mutation == "parser" else "resume.py")).write_text("changed")
    elif mutation == "amendment":
        rows[-1]["amendment_sha256"] = "f" * 64
    elif mutation == "missing_review":
        rows[-1].pop("reviewed_normalized")
    elif mutation == "reviewed_value":
        rows[-1]["reviewed_normalized"]["ai_probability"] = 0.0
    else:
        rows[-1]["reviewed_validation_error"] = "unrecorded failure"
    write_rows(ledger, rows)
    with pytest.raises(ValueError):
        report.build_report(root)


def test_review_does_not_relax_other_schema_errors(frozen):
    root, _, rows = frozen
    rows[0].update(normalized=None, validation_error="ValueError: Unknown choice")
    prompts = {task: json.loads((root / f"prompts/final/{task}.json").read_text()) for task in report.TASKS}
    reviewed, audit = report.review_records(rows, prompts, REVISION, normalize)
    assert reviewed[0]["normalized"] is None
    assert "outside probability rounding" in reviewed[0]["validation_error"]
    assert audit["recovered_strict_parser_rejections"] == 0
