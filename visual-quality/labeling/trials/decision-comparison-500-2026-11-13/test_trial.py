"""Offline tests for selection, score comparison and immutable request reuse."""

import copy
import hashlib
import json
import threading

import prepare
import pytest
import report
import run


def test_selection_is_diverse_unique_and_independent_of_reference_values():
    rows = [
        {
            "case_id": f"R{i:04}",
            "content_cluster": i % 64,
            "assessments": {
                m: {"technical_score": 1 + (i % 5 if m == "luna" else 0), "aesthetic_score": 3}
                for m in ("luna", "deepseek", "liquid", "jev")
            },
        }
        for i in range(1000)
    ]
    selected = prepare.select(rows)
    assert len(selected) == len({r["case_id"] for r in selected}) == 500
    assert len({r["content_cluster"] for r in selected}) == 64
    assert sum(r["selection_arm"] == "high_disagreement" for r in selected) == 375
    for row in rows:
        row["assessments"]["reference"] = {"technical_score": 5, "aesthetic_score": 1}
    assert [r["case_id"] for r in prepare.select(list(reversed(rows)))] == [r["case_id"] for r in selected]


def test_missing_scores_do_not_enter_agreement_or_latency_as_zero():
    cases = [
        {
            "assessments": {
                "reference": {"technical_score": 2, "aesthetic_score": 3},
                "gemini": {"technical_score": None, "aesthetic_score": None, "latency_seconds": None},
            }
        }
    ]
    result = report.metrics(cases, "gemini")
    assert result["numeric_pairs"] == 0
    assert result["technical"]["mae"] is None
    assert result["latency_median_seconds"] is None


def test_new_requests_exclude_labels_stats_and_reuse_exact_payload(tmp_path, monkeypatch):
    runner = run.Runner.__new__(run.Runner)
    runner.model, runner.root, runner.plan_sha = "gemini", tmp_path, "frozen-plan"
    runner.plan = {
        "max_attempts": 3,
        "common": "Rate independently",
        "questions": {"test": {"type": "noul", "instructions": "Photo?"}},
    }
    runner.route = {"requested_model": "google/gemini-decision-1"}
    runner.lock, runner.done, runner.accounted = threading.Lock(), 0, 0
    image = tmp_path / "view.png"
    image.write_bytes(b"frozen-image-bytes")
    case = {
        "case_id": "R1",
        "trial_image_path": str(image),
        "view_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "image_stats": {"sharpness_peak": 100},
        "assessments": {"reference": {"technical_score": 1}},
    }
    sent = []

    def request(case, body, request_sha, number):
        sent.append(json.loads(body))
        return {
            "parsed": {
                "status": "valid",
                "technical_score": 2.5,
                "aesthetic_score": 3.5,
                "price_usd": "0.001",
            },
            "latency_seconds": 0.75,
            "http_status": 200,
        }

    monkeypatch.setattr(runner, "once", request)
    runner.case(case)
    runner.case(copy.deepcopy(case))
    assert len(sent) == 1
    assert set(sent[0]) == {"model", "state", "questions"}
    assert "sharpness_peak" not in json.dumps(sent) and "reference" not in json.dumps(sent)
    image.write_bytes(b"changed")
    with pytest.raises(AssertionError):
        runner.case(case)


def test_unknown_new_attempt_cost_is_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "DATA", tmp_path)
    request = tmp_path / "results/gemini/attempts/R1/01/request.json"
    request.parent.mkdir(parents=True)
    request.write_text("{}")
    result = report.costs([], "gemini")
    assert result["known_usd"] == "0" and result["unpriced_attempts"] == 1
    assert result["new_requests"] == 1
