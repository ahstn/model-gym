"""Reject stale, incomplete or relabelled visual adjudications."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import api_contract
import protected_adjudication as adjudication
import pytest


@pytest.fixture
def review_fixture(tmp_path):
    image, receipt, timing = (tmp_path / name for name in ("view.png", "result.json", "timing.json"))
    for path in (image, receipt, timing):
        path.write_text(path.name)
    scores = {
        arm: {
            "case_id": "R0001",
            "model": "luna",
            "arm": arm,
            "technical_score": score,
            "aesthetic_score": 3.0,
            "attempt_id": arm + "/01",
            "raw_response_path": str(receipt),
            "raw_response_sha256": api_contract.file_sha(receipt),
        }
        for arm, score in (("baseline", 3.0), ("own_stats", 2.5))
    }
    reference = {
        "case_id": "R0001",
        "eligibility": "eligible",
        "review_timing_path": str(timing),
        "review_timing_sha256": api_contract.file_sha(timing),
        "technical_quality": {"score": 3.0, "status": "ok"},
        "aesthetics": {"score": 3.0, "status": "ok"},
    }
    queue = {
        "case_id": "R0001",
        "model": "luna",
        "arm": "own_stats",
        "image_path": str(image),
        "view_sha256": api_contract.file_sha(image),
        "reasons": ["prospectively_selected_style_case"],
    }
    context = {
        "bindings": {"inputs": {"references": {"sha256": "ref"}, "freeze": {"sha256": "freeze"}}},
        "references": {"R0001": reference},
        "scores": {("R0001", "luna", arm): row for arm, row in scores.items()},
        "queue": {adjudication.key(queue): queue},
    }
    review_input = {"queue": [{**queue, "scores": scores}]}
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(review_input))
    input_sha = api_contract.file_sha(input_path)
    record = {
        **adjudication.pair_bindings(queue, context),
        "input_file_sha256": input_sha,
        "verdict": "regression",
        "visible_evidence": "Intentional background blur surrounds a sharp subject.",
        "primary_reference_changed": False,
        "model_scores_changed": False,
        "tasks": {
            task: {
                "verdict": "regression" if task == "technical_quality" else "unchanged",
                "evidence": "Visible image evidence.",
                "baseline_score": 3.0,
                "treatment_score": 2.5 if task == "technical_quality" else 3.0,
                "frozen_reference_score": 3.0,
            }
            for task in ("technical_quality", "aesthetics")
        },
    }
    review = {
        "input": {"path": str(input_path), "sha256": input_sha},
        "reference_freeze": {"sha256": "freeze", "references_sha256": "ref"},
        "images": {
            "R0001": {
                "view_sha256": queue["view_sha256"],
                "opened_exact_image": True,
                "primary_reference_changed": False,
            }
        },
        "comparisons": {adjudication.key(queue): record},
    }
    return review, review_input, context


def test_exact_review_is_reusable(review_fixture):
    assert len(adjudication.validate_review(*review_fixture, luna=True)) == 1


@pytest.mark.parametrize(
    "mutation", ["missing_comparison", "score", "narrative_score", "freeze", "not_opened", "labels_changed"]
)
def test_stale_or_incomplete_review_is_rejected(review_fixture, mutation):
    review, review_input, context = copy.deepcopy(review_fixture)
    record = next(iter(review["comparisons"].values()))
    if mutation == "missing_comparison":
        review["comparisons"].clear()
    elif mutation == "score":
        context["scores"][("R0001", "luna", "own_stats")]["technical_score"] = 2.0
    elif mutation == "narrative_score":
        record["tasks"]["technical_quality"]["baseline_score"] = 4.0
    elif mutation == "freeze":
        review["reference_freeze"]["sha256"] = "different"
    elif mutation == "not_opened":
        review["images"]["R0001"]["opened_exact_image"] = False
    elif mutation == "labels_changed":
        record["primary_reference_changed"] = True
    with pytest.raises(ValueError):
        adjudication.validate_review(review, review_input, context, luna=True)


@pytest.mark.parametrize("field", ["image_path", "raw_response_path"])
def test_changed_bytes_are_rejected(review_fixture, field):
    review, review_input, context = review_fixture
    source = (
        next(iter(context["queue"].values()))
        if field == "image_path"
        else next(iter(context["scores"].values()))
    )
    Path(source[field]).write_text("changed bytes")
    with pytest.raises(ValueError):
        adjudication.validate_review(review, review_input, context, luna=True)
