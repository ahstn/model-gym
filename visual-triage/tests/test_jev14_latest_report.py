"""Test the human-rated quality and provenance-labelled AI comparisons."""

import copy
import json
import math
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import DECISIONS, digest, normalize_decisions
from tracks.jev14 import latest_report as report
from tracks.jev14.client import HEADERS, MODEL, REVISION, image_info, make_payload, normalize
from tracks.jev14.rounding import normalize as rounded_normalize


@pytest.fixture
def cohort():
    cases, rows, samples, baseline = [], [], {}, {}
    for task in report.TRACKS:
        samples[task], baseline[task] = [], []
        for index in range(2):
            identifier, sha = f"{task}-{index}", str(index) * 64
            sample = {
                "id": identifier,
                "image": "unused.jpg",
                "group_id": identifier,
                "split": "test",
                "delivered_sha256": sha,
                "labels": {"quality_mos_1to5": [1.5, 3.5][index], "ai_generated": bool(index)},
            }
            samples[task].append(sample)
            case = {
                "case_id": identifier,
                "sample": identifier,
                "phase": "primary",
                "task": task,
                "delivered_sha256": sha,
                "parent_delivered_sha256": sha,
                "image_width": 1024 if task == "quality" else 512,
                "image_height": 768 if task == "quality" else 512,
                "image_bytes": 1000,
            }
            cases.append(case)
            prediction = (
                {"technical_quality": [1.8, 3.7][index], "aesthetics": 3.0}
                if task == "quality"
                else {"ai_probability": [0.1, 0.7][index], "medium": "photo", "uncertainty": "medium"}
            )
            row = {
                **case,
                "arm": "jev14",
                "status": 200,
                "normalized": prediction,
                "seconds": 1.0,
                "cost_usd": 0.02,
                "usage": {"cost": 0.02, "input_tokens": 500},
                "prompt_sha256": task,
            }
            rows.append(row)
            baseline[task].append({**row, "arm": "decider", "cost_usd": 0.01, "usage": {"cost": 0.01}})
    return cases, rows, samples, baseline


def test_human_quality_and_ai_origin_are_separate_objectives(cohort):
    summary, rows = report.assemble_report(*cohort, 262144, bootstrap_replicates=0)
    quality = summary["panels"]["quality"]["jev14"]["metrics"]
    assert quality["mae_valid"] == pytest.approx(0.25)
    assert quality["bad_detection"]["valid_only"]["recall"] == 1
    assert quality["aesthetics"]["scored_against_gold"] is False
    ai = summary["panels"]["ai"]["jev14"]["metrics"]
    assert ai["valid_only"]["accuracy"] == 1
    assert ai["brier_valid"] == pytest.approx(0.05)
    assert rows[0]["gold_field"] == "quality_mos_1to5"
    assert rows[2]["gold_field"] == "ai_generated"
    assert summary["operations"]["effective_provider_inference_cost_usd"] == pytest.approx(0.08)
    assert summary["panels"]["quality"]["paired_reward_delta"]["selected_minus_baseline"] == 0


def test_missing_and_invalid_do_not_become_real_or_high_quality(cohort):
    cohort[1].pop()
    cohort[1][0].update(normalized=None, status=502, validation_error="HTTP failure", usage={}, cost_usd=None)
    summary, rows = report.assemble_report(*cohort, 262144, bootstrap_replicates=0)
    assert summary["status"] == "partial"
    assert summary["unattempted"] == 1
    assert summary["operations"]["unknown_cost_count"] == 1
    assert rows[0]["technical_quality"] is None
    assert rows[-1]["ai_probability"] is None
    assert summary["panels"]["ai"]["jev14"]["metrics"]["accuracy_all_expected_invalid_as_wrong"] == 0.5
    assert summary["panels"]["quality"]["jev14"]["metrics"]["mae_invalid_or_unattempted_penalty4"] == 2.1
    assert summary["panels"]["quality"]["paired_valid_decider"]["expected"] == 1


def test_metadata_gold_mapping_preserves_test_groups():
    doc = {
        "samples": [
            {"id": "train", "split": "train", "gold": True},
            {"id": "test", "split": "test", "gold": False, "group_id": "device-1"},
        ]
    }
    selected = report.test_samples("ai", doc)
    assert selected == [
        {
            "id": "test",
            "split": "test",
            "gold": False,
            "group_id": "device-1",
            "labels": {"ai_generated": False},
        }
    ]
    assert "labels" not in doc["samples"][1]
    doc["samples"][1]["gold"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        report.test_samples("ai", doc)


@pytest.mark.parametrize("mutation", ["case", "row", "hash", "foreign"])
def test_changed_or_duplicate_cases_are_rejected(cohort, mutation):
    if mutation == "case":
        cohort[0].append(copy.deepcopy(cohort[0][0]))
    elif mutation == "row":
        cohort[1].append(copy.deepcopy(cohort[1][0]))
    elif mutation == "hash":
        cohort[1][0]["delivered_sha256"] = "f" * 64
    else:
        cohort[1][0]["case_id"] = "another"
    with pytest.raises(ValueError):
        report.assemble_report(*cohort, 262144, bootstrap_replicates=0)


def test_context_fraction_is_measured_from_usage(cohort):
    cohort[1][0]["usage"]["input_tokens"] = 2000
    summary, _ = report.assemble_report(*cohort, 10000, bootstrap_replicates=0)
    context = summary["panels"]["quality"]["context"]
    assert context["maximum_input_tokens"] == 2000
    assert context["maximum_context_fraction"] == 0.2
    assert context["median_input_tokens"] == 1250


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def prompt_for(task):
    questions = (
        {
            key: {"type": "score", "criteria": [str(i) for i in range(5)], "instructions": "Rate it"}
            for key in ("technical_quality", "aesthetics")
        }
        if task == "quality"
        else {
            "ai_probability": {"type": "noul", "instructions": "AI origin"},
            "medium": {
                "type": "choice",
                "criteria": {"photo": "Photo", "graphic": "Graphic"},
                "instructions": "Medium",
            },
            "uncertainty": {
                "type": "choice",
                "criteria": {"low": "Low", "medium": "Medium"},
                "instructions": "Uncertainty",
            },
        }
    )
    return {"id": task, "task": task, "context": "Inspect pixels", "questions": questions}


def response_for(prediction, prompt, *, model=REVISION, cost=0.02):
    answers = {}
    for key, question in prompt["questions"].items():
        kind, value = question["type"], prediction[key]
        answer = {"type": kind, kind: value - 1 if kind == "score" else value}
        if kind == "score":
            score = value - 1
            lower, upper = math.floor(score), math.ceil(score)
            probabilities = dict.fromkeys(map(str, range(5)), 0.0)
            probabilities[str(lower)] = 1 - (score - lower)
            if upper != lower:
                probabilities[str(upper)] = score - lower
            answer["probabilities"] = probabilities
        elif kind == "choice":
            answer["probabilities"] = {label: float(label == value) for label in question["criteria"]}
        answers[key] = answer
    return {
        "answers": answers,
        "model": model,
        "provider": "TypeSafe" if model == REVISION else "Perplexity",
        "usage": {"cost": cost, "input_tokens": 500},
    }


@pytest.fixture
def frozen(tmp_path, cohort):
    cases, rows, samples, baselines = cohort
    prompts, datasets, baseline_paths, source_paths = {}, {}, {}, []
    for case in cases:
        path = tmp_path / "data" / (case["sample"] + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (320, 320)).save(path)
        sample = next(sample for sample in samples[case["task"]] if sample["id"] == case["sample"])
        sample.update(image=str(path.relative_to(tmp_path)), delivered_sha256=report.sha(path))
        case.update(
            image=sample["image"],
            delivered_sha256=sample["delivered_sha256"],
            parent_delivered_sha256=sample["delivered_sha256"],
            benchmark=f"latest-{case['task']}",
            group_id=sample["group_id"],
            **image_info(path),
        )
        if case["task"] == "ai":
            sample["gold"] = sample["labels"]["ai_generated"]
        case["gold"] = sample["labels"]["quality_mos_1to5" if case["task"] == "quality" else "ai_generated"]
    for task in report.TRACKS:
        prompt = prompt_for(task)
        prompts[task], datasets[task], baseline_paths[task] = (
            f"prompts/{task}.json",
            f"datasets/{task}.json",
            f"baselines/{task}.jsonl",
        )
        write_json(tmp_path / prompts[task], prompt)
        write_json(tmp_path / datasets[task], {"samples": samples[task]})
        for row in baselines[task]:
            sample = next(sample for sample in samples[task] if sample["id"] == row["sample"])
            response = response_for(row["normalized"], prompt, model="decider-revision", cost=0.01)
            row.update(
                delivered_sha256=sample["delivered_sha256"],
                prompt_sha256=digest(prompt),
                response=response,
                normalized=normalize_decisions(response, prompt["questions"]),
                metadata_arm="image_only",
                context_sha256=digest(None),
            )
        write_rows(tmp_path / baseline_paths[task], baselines[task])
        source_paths += [prompts[task], datasets[task], baseline_paths[task]]
    manifest = tmp_path / "data/jev14-latest-v1/manifest.json"
    write_json(manifest, {"cases": cases})
    protocol_path = tmp_path / "tracks/jev14/latest/protocol.json"
    protocol = {
        "model": MODEL,
        "expected_revision": REVISION,
        "request_headers": HEADERS,
        "context_tokens": 262144,
        "source_inputs": {path: {"sha256": report.sha(tmp_path / path)} for path in source_paths},
        "execution_sha256": {},
        "prompts": prompts,
        "datasets": datasets,
        "baselines": baseline_paths,
        "counts": {"quality": 2, "ai": 2},
        "maximum_attempts": 4,
        "manifest_sha256": report.sha(manifest),
    }
    write_json(protocol_path, protocol)
    for index, (case, row) in enumerate(zip(cases, rows), 1):
        prompt = prompt_for(case["task"])
        response = response_for(row["normalized"], prompt)
        payload = make_payload(tmp_path, case, prompt)
        row.update(
            case,
            attempt=index,
            response=response,
            normalized=rounded_normalize(response, prompt["questions"]),
            strict_normalized=normalize(response, prompt["questions"]),
            strict_validation_error=None,
            prompt_sha256=digest(prompt),
            protocol_sha256=report.sha(protocol_path),
            manifest_sha256=protocol["manifest_sha256"],
            endpoint=DECISIONS,
            request_headers=HEADERS,
            request_bytes=len(json.dumps(payload).encode()),
            request_sha256=digest({"endpoint": DECISIONS, "headers": HEADERS, "payload": payload}),
            resolved_model=REVISION,
            provider="TypeSafe",
        )
    ledger = tmp_path / "data/jev14-latest-v1/attempts.jsonl"
    write_rows(ledger, rows)
    return tmp_path, ledger, rows


def test_latest_frozen_report_and_immutable_outputs(frozen):
    root, _, _ = frozen
    summary, cases = report.build_report(root, bootstrap_replicates=0)
    assert summary["status"] == "complete"
    assert summary["strict_parser_rejections"] == 0
    assert len(cases) == 4
    output = root / "output"
    report.write_report(root, output)
    prose = (output / "README.md").read_text()
    assert "human ratings" in prose
    assert "no EXIF or C2PA text" in prose
    assert "No Jev prompt tuning" in prose
    assert "different images" in prose
    with pytest.raises(FileExistsError):
        report.write_report(root, output)


@pytest.mark.parametrize("mutation", ["order", "image", "prompt", "normalized", "strict", "cost"])
def test_latest_changed_attempt_rejected(frozen, mutation):
    root, ledger, rows = frozen
    if mutation == "order":
        rows.reverse()
    elif mutation == "image":
        rows[0]["delivered_sha256"] = "f" * 64
    elif mutation == "prompt":
        rows[0]["prompt_sha256"] = "f" * 64
    elif mutation == "normalized":
        rows[0]["normalized"]["technical_quality"] = 5.0
    elif mutation == "strict":
        rows[0]["strict_normalized"]["technical_quality"] = 5.0
    else:
        rows[0]["cost_usd"] = 10.0
    write_rows(ledger, rows)
    with pytest.raises(ValueError):
        report.build_report(root, bootstrap_replicates=0)


def test_latest_failure_and_unknown_cost_remain_reportable(frozen):
    root, ledger, rows = frozen
    rows[0].update(
        status=400,
        normalized=None,
        strict_normalized=None,
        strict_validation_error="ValueError: HTTP failure",
        validation_error="ValueError: HTTP, model or non-rounding validation failed",
        usage={},
        cost_usd=None,
        response={"error": {"message": "bad image"}},
        resolved_model=None,
        provider=None,
    )
    write_rows(ledger, rows[:1])
    summary, cases = report.build_report(root, bootstrap_replicates=0)
    assert summary["status"] == "partial"
    assert summary["operations"]["unknown_cost_count"] == 1
    assert cases[0]["disposition"] == "invalid"


def test_latest_baseline_prompt_mismatch_rejected(frozen):
    root, _, _ = frozen
    source = root / "baselines/quality.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    prompt = prompt_for("quality")
    samples = report.test_samples("quality", json.loads((root / "datasets/quality.json").read_text()))
    rows[0]["prompt_sha256"] = "wrong"
    with pytest.raises(ValueError, match="prompt or task"):
        report.validate_baseline("quality", samples, rows, prompt)


def test_parser_csv_flags_distinguish_native_acceptance_from_rounding_recovery(frozen):
    root, ledger, rows = frozen
    prompt = prompt_for("quality")
    row = rows[0]
    row["response"]["answers"]["technical_quality"]["probabilities"] = {
        "0": 0.21,
        "1": 0.78,
        "2": 0,
        "3": 0,
        "4": 0,
    }
    row.update(
        strict_normalized=None,
        strict_validation_error="ValueError: Probability distribution does not sum to one",
        normalized=rounded_normalize(row["response"], prompt["questions"]),
    )
    write_rows(ledger, rows)
    summary, cases = report.build_report(root, bootstrap_replicates=0)
    assert sum(case["strict_parser_valid"] for case in cases) == 3
    assert sum(case["parser_review_applied"] for case in cases) == 1
    assert summary["strict_parser_rejections"] == 1
    assert cases[0]["technical_quality"] == 1.8


def test_existing_gates_include_endpoints_and_keep_failures_separate():
    rows = [
        {"valid": True, "gold": False, "prediction": {"ai_probability": 0.2}},
        {"valid": True, "gold": True, "prediction": {"ai_probability": 0.8}},
        {"valid": True, "gold": False, "prediction": {"ai_probability": 0.5}},
        {"valid": False, "gold": True, "prediction": None},
    ]
    assert report.fixed_gates(rows) == {
        "accept": {"real": 1, "ai": 0},
        "reject": {"real": 0, "ai": 1},
        "review": {"real": 1, "ai": 0},
        "invalid": {"real": 0, "ai": 1},
    }
