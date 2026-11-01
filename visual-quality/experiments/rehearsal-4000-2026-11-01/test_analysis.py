"""Synthetic checks for reference agreement, missingness and honest accounting."""

from __future__ import annotations

import copy
import csv
from decimal import Decimal

import analysis_metrics as m
import analyze as a
import pytest

FROZEN = "2026-11-01T10:00:00+00:00"
START = "2026-11-01T11:00:00+00:00"


def cases(count=2):
    return [
        {
            "case_id": f"R{i:04}",
            "view_sha256": f"hash-{i}",
            "family_id": f"family-{i}",
            "benchmark_partition": "representative" if i == 0 else "rehearsal",
        }
        for i in range(count)
    ]


def reference(case, technical=3.0, aesthetic=4.0):
    return {
        "case_id": case["case_id"],
        "view_sha256": case["view_sha256"],
        "eligibility": "eligible",
        "latency_seconds": None,
        "price_usd": 0.0,
        "review_elapsed_seconds": 12.25,
        **{
            task: {
                "task": task,
                "status": "ok",
                "score": score,
                "evidence": "Visible reference evidence.",
                "limitations": [],
            }
            for task, score in zip(a.api_contract.TASKS, (technical, aesthetic), strict=True)
        },
    }


def attempt(case, model="luna", arm="baseline", technical=3.2, aesthetic=4.2, price="0.00123", sequence=1):
    lite = model == "qrealign_lite"
    return {
        "case_id": case["case_id"],
        "view_sha256": case["view_sha256"],
        "model": model,
        "arm": arm,
        "attempt_id": f"{case['case_id']}/{arm}/{sequence}",
        "sequence": sequence,
        "request_started_at": START,
        "response_received_at": "2026-11-01T11:00:01+00:00",
        "latency_seconds": 1.23456,
        "http_status": 200,
        "price_usd": "0.00" if lite else price,
        "price_basis": "user_assigned_zero; infrastructure separate" if lite else "receipt",
        "status": "valid",
        "technical_score": technical,
        "aesthetic_score": aesthetic,
        "technical_score_native": technical,
        "aesthetic_score_native": aesthetic,
        "task_valid": {"quality" if lite else "technical_quality": True, "aesthetics": True},
    }


def test_complete_expected_coverage_and_missing_values_are_not_scores():
    corpus = cases()
    rows = a.build_long(corpus, [], [reference(corpus[0])], frozen_at=FROZEN)
    assert len(rows) == 19  # 14 model slots + one reference + four derived means.
    assert len({(r["case_id"], r["model"], r["arm"]) for r in rows}) == 19
    missing = next(r for r in rows if r["model"] == "luna")
    assert missing["status"] == "missing"
    assert missing["technical_score"] is missing["price_usd"] is missing["latency_seconds"] is None
    lite = next(r for r in rows if r["model"] == "qrealign_lite")
    assert lite["price_usd"] == "0.00" and lite["attempt_count"] == 0
    ref = next(r for r in rows if r["model"] == "reference")
    assert ref["latency_seconds"] is None and ref["review_elapsed_seconds"] == 12.25


def test_full_size_contract_has_27500_rows_without_inventing_attempts():
    corpus = cases(4000)
    for i, case in enumerate(corpus[:500]):
        case["benchmark_partition"] = (
            "development" if i < 100 else "representative" if i < 400 else "challenge"
        )
    rows = a.build_long(corpus, [], [reference(case) for case in corpus[:500]], frozen_at=FROZEN)
    assert len(rows) == 27500
    assert sum(r["model"] == "luna_deepseek_mean" for r in rows) == 5000
    assert not any(r["attempt_count"] for r in rows)


def test_partial_task_survives_and_mean_uses_only_two_available_valid_components():
    corpus = cases()
    luna = attempt(corpus[0], technical=3.123456789, aesthetic=None)
    luna["status"] = "invalid"
    luna["task_valid"]["aesthetics"] = False
    deepseek = attempt(corpus[0], "deepseek", technical=3.876543211, aesthetic=4.1)
    rows = a.build_long(corpus, [luna, deepseek], [reference(corpus[0])], frozen_at=FROZEN)
    mean = next(
        r
        for r in rows
        if r["model"] == "luna_deepseek_mean" and r["case_id"] == "R0000" and r["arm"] == "baseline"
    )
    assert mean["technical_score"] == 3.5 and mean["aesthetic_score"] is None
    assert mean["status"] == "partial"
    assert mean["price_usd"] is None and mean["latency_seconds"] is None
    reported_luna = next(
        r for r in rows if r["model"] == "luna" and r["arm"] == "baseline" and r["case_id"] == "R0000"
    )
    assert reported_luna["technical_score"] == 3.123456789
    assert reported_luna["technical_score_display"] == "3.12"


def test_receipts_count_each_attempt_once_and_keep_unknown_retry_cost():
    corpus = cases()
    rejected = attempt(corpus[0], "deepseek", price=None)
    rejected.update(
        status="rejected_retryable",
        technical_score=None,
        aesthetic_score=None,
        task_valid={"technical_quality": False, "aesthetics": False},
    )
    scored = attempt(corpus[0], "deepseek", sequence=2)
    lite = attempt(corpus[0], "qrealign_lite")
    attempts = [rejected, scored, lite]
    rows = a.build_long(corpus, attempts, [reference(corpus[0])], frozen_at=FROZEN)
    totals = {row["model"]: row for row in a.operational(rows, attempts)["model_totals"]}
    assert totals["deepseek"]["attempts"] == 2
    assert Decimal(totals["deepseek"]["known_price_usd"]) == Decimal("0.00123")
    assert totals["deepseek"]["total_price_usd"] is None and totals["deepseek"]["unpriced_attempts"] == 1
    assert totals["qrealign_lite"]["total_price_usd"] == "0.00"
    assert "luna_deepseek_mean" not in totals
    row = next(
        r for r in rows if r["model"] == "deepseek" and r["case_id"] == "R0000" and r["arm"] == "baseline"
    )
    assert row["total_attempt_price_usd"] is None and row["attempt_count"] == 2
    assert row["total_attempt_latency_seconds"] == pytest.approx(2 * 1.23456)


def test_duplicates_changed_images_and_prefreeze_calls_are_rejected():
    corpus = cases()
    row = attempt(corpus[0])
    with pytest.raises(ValueError, match="Duplicate attempt"):
        a.build_long(corpus, [row, row], [], frozen_at=FROZEN)
    bad = {**row, "view_sha256": "other"}
    with pytest.raises(ValueError, match="changed view"):
        a.build_long(corpus, [bad], [], frozen_at=FROZEN)
    bad = {**row, "request_started_at": "2026-11-01T09:00:00+00:00"}
    with pytest.raises(ValueError, match="preceded"):
        a.build_long(corpus, [bad], [], frozen_at=FROZEN)
    second = {**row, "attempt_id": "second"}
    with pytest.raises(ValueError, match="Multiple completed"):
        a.build_long(corpus, [row, second], [], frozen_at=FROZEN)


def test_lite_zero_override_is_explicit_and_no_latency_is_fabricated():
    corpus = cases()
    row = attempt(corpus[0], "qrealign_lite")
    row["price_basis"] = "provider cost"
    with pytest.raises(ValueError, match="override"):
        a.build_long(corpus, [row], [], frozen_at=FROZEN)
    row.update(
        price_basis="user_assigned_zero",
        latency_seconds=None,
        status="in_flight",
        technical_score=None,
        aesthetic_score=None,
        task_valid={},
    )
    output = a.build_long(corpus, [row], [], frozen_at=FROZEN)
    lite = next(r for r in output if r["case_id"] == row["case_id"] and r["model"] == "qrealign_lite")
    assert lite["latency_seconds"] is None and lite["technical_score"] is None


def test_accuracy_has_task_specific_and_common_task_denominators():
    corpus = cases()
    row = attempt(corpus[0], aesthetic=None)
    row["task_valid"]["aesthetics"] = False
    rows = a.build_long(corpus, [row], [reference(corpus[0])], frozen_at=FROZEN)
    score = next(
        r
        for r in m.accuracy(rows)["panels"]
        if r["panel"] == "representative" and r["model"] == "luna" and r["arm"] == "baseline"
    )
    assert score["technical_score"]["n"] == 1
    assert score["technical_score"]["mae"] == pytest.approx(0.2)
    assert score["aesthetic_score"]["n"] == 0
    assert score["combined_common_tasks"]["n"] == 0
    assert score["all_five_baselines_common"]["n"] == 0


def test_spearman_uses_average_ties_and_handles_constant_values():
    assert m.ranks([1, 1, 2, 3]).tolist() == [0.5, 0.5, 2, 3]
    assert m.spearman([1, 1, 2, 3], [3, 3, 2, 1]) == pytest.approx(-1)
    assert m.spearman([1, 1], [1, 2]) is None
    assert m.spearman([1], [1]) is None


def test_paired_family_bootstrap_does_not_treat_related_images_as_independent():
    result = m.bootstrap_families([0] * 100 + [1], ["big-family"] * 100 + ["small-family"], repeats=10000)
    assert result["n_images"] == 101 and result["n_families"] == 2
    assert result["estimate"] == pytest.approx(1 / 101)
    assert result["ci95"] == [0, 1]
    assert result == m.bootstrap_families(
        [0] * 100 + [1], ["big-family"] * 100 + ["small-family"], repeats=10000
    )


def test_primary_pair_does_not_drop_images_when_unrelated_lite_is_missing():
    corpus = cases()
    attempts = [
        attempt(corpus[0], model, arm, technical=score, aesthetic=4)
        for model in ("luna", "deepseek")
        for arm, score in [("baseline", 4), ("own_stats", 3), ("swapped_stats", 5)]
    ]
    rows = a.build_long(corpus, attempts, [reference(corpus[0])], frozen_at=FROZEN)
    primary = next(row for row in m.treatment_effects(rows, repeats=100) if row["primary"])
    assert primary["technical_score"]["n_images"] == 1
    assert primary["technical_score"]["estimate"] == -1
    assert primary["technical_practical_and_interval_target_met"] is None
    assert primary["aesthetic_one_sided95_noninferiority_met"] is None


def test_ladders_keep_ties_strict_inversions_and_all_severities():
    corpus = cases(4)
    for i, case in enumerate(corpus):
        case.update(
            family_id="one-family",
            benchmark_partition="rehearsal",
            stratum="controlled",
            role="source_parent" if i == 0 else "variant",
            severity=None if i == 0 else i,
            operator="gaussian_blur",
        )
    rows = a.build_long(
        corpus,
        [attempt(case, technical=score) for case, score in zip(corpus, [4, 4, 4.2, 3], strict=True)],
        [],
        frozen_at=None,
    )
    ladder = next(r for r in m.ladders(corpus, rows)["families"] if r["model"] == "luna")
    assert ladder["adjacent_ties"] == 1 and ladder["strict_inversions"] == 1
    assert not ladder["nonincreasing"] and ladder["parent_minus_severity3"] == 1
    assert ladder["complete"]


def test_qc_flags_only_review_and_random_fill_is_label_independent():
    corpus = cases(20)
    for case in corpus:
        case["benchmark_partition"] = "representative"
    refs = [reference(case, technical=3) for case in corpus]
    attempts = [attempt(case, technical=4 if i == 0 else 3) for i, case in enumerate(corpus)]
    rows = a.build_long(corpus, attempts, refs, frozen_at=FROZEN)
    original = copy.deepcopy(rows)
    stats = {
        case["case_id"]: {
            "metrics": {"sharpness_peak": 1 if i == 0 else 10, "noise_sigma": 1},
            "percentiles": {"sharpness_peak": 5 if i == 0 else 80, "noise_sigma": 50},
            "gate": None,
        }
        for i, case in enumerate(corpus)
    }
    report = m.qc_review(rows, stats, {"sharpness_peak_p25": 2, "noise_sigma_p90": 100})
    qc = next(
        r
        for r in report["rows"]
        if r["panel"] == "representative" and r["model"] == "luna" and r["arm"] == "baseline"
    )
    assert qc["review_budget"] == 2
    assert qc["fixed_budget"]["precision"] == 0.5 and qc["fixed_budget"]["recall"] == 1
    assert qc["reserved_random_audit_slots"] == qc["unflagged_audits_in_budget"] == 1
    assert qc["random_review_expected_large_errors"] == pytest.approx(0.1)
    assert rows == original


def test_csv_preserves_precision_and_leaves_unknown_cells_empty(tmp_path):
    corpus = cases()
    record = attempt(corpus[0], technical=3.123456789, price="0.00000012345")
    rows = a.build_long(corpus, [record], [reference(corpus[0])], frozen_at=FROZEN)
    a.write_exports(tmp_path, rows, {"operations": a.operational(rows, [record])})
    with (tmp_path / "scores-long.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    luna = next(
        r for r in exported if r["model"] == "luna" and r["case_id"] == "R0000" and r["arm"] == "baseline"
    )
    assert luna["technical_score"] == "3.123456789" and luna["technical_score_display"] == "3.12"
    assert Decimal(luna["price_usd"]) == Decimal("0.00000012345")
    mean = next(r for r in exported if r["model"] == "luna_deepseek_mean")
    assert mean["price_usd"] == mean["latency_seconds"] == ""


def test_lite_raw_journal_reparse_detects_edited_normalized_label(tmp_path, monkeypatch):
    import base64
    import json

    corpus = cases(1)
    corpus[0]["benchmark_partition"] = "rehearsal"
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture-image")
    corpus[0].update(image_path=str(image), view_sha256=a.api_contract.file_sha(image))
    manifest = tmp_path / "manifest.json"
    a.api_runner.write(manifest, {"cases": corpus})
    directory = tmp_path / "results/lite/rehearsal"
    identity = {"model": a.lite_runner.MODEL, "revision": a.lite_runner.REVISION}
    config = {
        "manifest_sha256": a.api_contract.file_sha(manifest),
        "client_sha256": a.api_contract.file_sha(a.lite_runner.__file__),
        "case_ids": [corpus[0]["case_id"]],
        "model": a.lite_runner.MODEL,
        "revision": a.lite_runner.REVISION,
        "server_identity_sha256": a.lite_runner.digest_bytes(a.lite_runner.canonical(identity)),
    }
    run_id = a.lite_runner.digest_bytes(a.lite_runner.canonical(config))
    a.api_runner.write(
        directory / "run.json", {"configuration": config, "run_id": run_id, "created_at": START}
    )
    a.api_runner.write(directory / "server-identity.json", identity)
    task = {"native_0_1": 0.5, "score_1_5": 3.0, "probabilities": [0, 0, 1, 0, 0], "input_tokens": 100}
    body = {"quality": task, "aesthetics": task}
    raw = directory / "raw/fixture.json"
    a.api_runner.write(raw, {"status": 200, "body": json.dumps(body), "headers": {}})
    payload = a.lite_runner.canonical({"image": base64.b64encode(image.read_bytes()).decode("ascii")})
    row = {
        **attempt(corpus[0], "qrealign_lite"),
        **a.lite_runner.normalize_response(body),
        "run_id": run_id,
        "raw_response_path": str(raw),
        "request_sha256": a.lite_runner.digest_bytes(payload),
    }
    path = directory / "attempts" / (corpus[0]["case_id"] + ".json")
    a.api_runner.write(path, row)
    monkeypatch.setattr(a, "DATA", tmp_path)
    records, _ = a.load_attempts(tmp_path, manifest, corpus)
    assert len(records) == 1 and records[0]["technical_score"] == 3
    row["technical_score"] = 4
    a.api_runner.write(path, row)
    with pytest.raises(ValueError, match="independently reparsed"):
        a.load_attempts(tmp_path, manifest, corpus)


def test_small_successful_subset_cannot_pass_coverage_gate():
    corpus = cases(300)
    for case in corpus:
        case["benchmark_partition"] = "representative"
    attempts = [
        attempt(case, model, arm, technical=score, aesthetic=4)
        for case in corpus[:2]
        for model in ("luna", "deepseek")
        for arm, score in [("baseline", 4), ("own_stats", 3), ("swapped_stats", 5)]
    ]
    rows = a.build_long(corpus, attempts, [reference(case) for case in corpus], frozen_at=FROZEN)
    primary = next(row for row in m.treatment_effects(rows, repeats=100) if row["primary"])
    assert primary["technical_practical_and_interval_target_met"]
    assert primary["aesthetic_one_sided95_noninferiority_met"]
    assert primary["technical_score"]["expected_panel_images"] == 300
    assert primary["technical_score"]["numeric_reference_cases"] == 300
    assert primary["technical_score"]["paired_cases"] == 2
    assert primary["technical_score"]["paired_coverage"] == pytest.approx(2 / 300)
    assert not primary["comparison_decision_eligible"] and not primary["comparison_decision_ready"]
    assert not primary["advancement_ready"]


def test_extra_treatment_missingness_blocks_gate_even_with_95_percent_pairs():
    corpus = cases(20)
    for case in corpus:
        case["benchmark_partition"] = "representative"
    attempts = [
        attempt(case, model, "baseline", technical=4, aesthetic=4)
        for case in corpus
        for model in ("luna", "deepseek")
    ]
    attempts += [
        attempt(case, model, "own_stats", technical=3, aesthetic=4)
        for case in corpus[:19]
        for model in ("luna", "deepseek")
    ]
    rows = a.build_long(corpus, attempts, [reference(case) for case in corpus], frozen_at=FROZEN)
    primary = next(row for row in m.treatment_effects(rows, repeats=100) if row["primary"])
    assert primary["technical_score"]["paired_coverage"] == 0.95
    assert primary["technical_score"]["baseline_missing_or_invalid_scores"] == 0
    assert primary["technical_score"]["treatment_missing_or_invalid_scores"] == 1
    assert not primary["comparison_decision_eligible"]


def test_qc_reserves_random_audits_when_flags_exceed_budget():
    corpus = cases(300)
    for case in corpus:
        case["benchmark_partition"] = "representative"
    rows = a.build_long(
        corpus, [attempt(case) for case in corpus], [reference(case) for case in corpus], frozen_at=FROZEN
    )
    stats = {
        case["case_id"]: {
            "metrics": {"sharpness_peak": i + 1, "noise_sigma": 1},
            "percentiles": {"sharpness_peak": 5 if i < 75 else 80, "noise_sigma": 50},
            "gate": None,
        }
        for i, case in enumerate(corpus)
    }
    result = m.qc_review(rows, stats, {})
    qc = next(
        row
        for row in result["rows"]
        if row["panel"] == "representative" and row["model"] == "luna" and row["arm"] == "baseline"
    )
    assert qc["flagged"]["reviewed"] == 75
    assert qc["review_budget"] == 30 and len(qc["selected_case_ids"]) == 30
    assert qc["reserved_random_audit_slots"] == qc["unflagged_audits_in_budget"] == 6
    assert len(set(qc["selected_case_ids"])) == 30
    assert all(int(identity[1:]) >= 75 for identity in qc["random_unflagged_audit_case_ids"])


@pytest.mark.parametrize(
    "baseline,own_luna,expected", [(3.3, 3.2, True), (3.1, 3.0, True), (3.3, 3.20000002, False)]
)
def test_inclusive_practical_margin_survives_binary_subtraction(baseline, own_luna, expected):
    corpus = cases(300)
    for case in corpus:
        case["benchmark_partition"] = "representative"
    attempts = [
        attempt(case, model, arm, technical=score, aesthetic=4)
        for case in corpus
        for model in ("luna", "deepseek")
        for arm, score in [("baseline", baseline), ("own_stats", own_luna if model == "luna" else baseline)]
    ]
    rows = a.build_long(corpus, attempts, [reference(case) for case in corpus], frozen_at=FROZEN)
    primary = next(row for row in m.treatment_effects(rows, repeats=100) if row["primary"])
    assert primary["comparison_decision_eligible"]
    assert primary["technical_score"]["ci95"][1] < 0
    assert primary["technical_practical_and_interval_target_met"] is expected


@pytest.mark.parametrize("delta,expected", [("0.03", True), ("0.03000001", False)])
def test_inclusive_aesthetic_margin_retains_tight_boundary(delta, expected):
    corpus = cases(300)
    for case in corpus:
        case["benchmark_partition"] = "representative"
    own = float(Decimal("4.3") + Decimal(delta))
    attempts = [
        attempt(case, model, arm, technical=technical, aesthetic=aesthetic)
        for case in corpus
        for model in ("luna", "deepseek")
        for arm, technical, aesthetic in [("baseline", 3.3, 4.3), ("own_stats", 3.2, own)]
    ]
    rows = a.build_long(corpus, attempts, [reference(case) for case in corpus], frozen_at=FROZEN)
    primary = next(row for row in m.treatment_effects(rows, repeats=100) if row["primary"])
    assert primary["aesthetic_one_sided95_noninferiority_met"] is expected


def test_zero_upper_ci_still_fails_strict_improvement_requirement():
    corpus = cases(2)
    for case in corpus:
        case["benchmark_partition"] = "representative"
    attempts = [
        attempt(case, model, arm, technical=score, aesthetic=4)
        for i, case in enumerate(corpus)
        for model in ("luna", "deepseek")
        for arm, score in [("baseline", 4), ("own_stats", 3.75 if i == 0 and model == "luna" else 4)]
    ]
    rows = a.build_long(corpus, attempts, [reference(case) for case in corpus], frozen_at=FROZEN)
    primary = next(row for row in m.treatment_effects(rows, repeats=1000) if row["primary"])
    assert primary["technical_score"]["estimate"] == -0.0625
    assert primary["technical_score"]["ci95"][1] == 0
    assert primary["technical_practical_and_interval_target_met"] is False


def test_qc_large_error_includes_exact_nominal_boundary_but_not_nearby_smaller_error():
    corpus = cases(2)
    for case in corpus:
        case["benchmark_partition"] = "representative"
    attempts = [
        attempt(case, model, technical=1 if model == "luna" else 3.1 - i * 0.00000002)
        for i, case in enumerate(corpus)
        for model in ("luna", "deepseek")
    ]
    rows = a.build_long(
        corpus, attempts, [reference(case, technical=1.3) for case in corpus], frozen_at=FROZEN
    )
    stats = {
        case["case_id"]: {
            "metrics": {"sharpness_peak": 1, "noise_sigma": 1},
            "percentiles": {"sharpness_peak": 5, "noise_sigma": 50},
            "gate": None,
        }
        for case in corpus
    }
    output = m.qc_review(rows, stats, {})
    mean = next(
        row
        for row in output["rows"]
        if row["panel"] == "representative"
        and row["model"] == "luna_deepseek_mean"
        and row["arm"] == "baseline"
    )
    assert mean["valid_reference_cases_with_stats"] == 2
    assert mean["large_errors"] == 1


@pytest.mark.parametrize(
    "artifact", ["attempts/baseline--R0000/01/request.json", "attempts.jsonl", "empty_job_folder"]
)
def test_missing_api_plan_cannot_hide_orphaned_attempts(tmp_path, monkeypatch, artifact):
    monkeypatch.setattr(a, "DATA", tmp_path)
    phase = tmp_path / "results/api/deepseek/rehearsal"
    if artifact == "empty_job_folder":
        (phase / "attempts/baseline--R0000/01").mkdir(parents=True)
    else:
        path = phase / artifact
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
    with pytest.raises(ValueError, match="Orphaned API journals"):
        a.load_attempts(tmp_path, tmp_path / "manifest.json", cases())


@pytest.mark.parametrize(
    "empty_path", [None, "results/api/luna/benchmark", "results/api/luna/benchmark/attempts"]
)
def test_genuinely_unstarted_phase_remains_reportable(tmp_path, monkeypatch, empty_path):
    monkeypatch.setattr(a, "DATA", tmp_path)
    if empty_path:
        (tmp_path / empty_path).mkdir(parents=True)
    records, inputs = a.load_attempts(tmp_path, tmp_path / "manifest.json", cases())
    assert records == [] and inputs == {}


def test_common_baseline_metrics_preserve_valid_sibling_task():
    corpus = cases(3)
    for case in corpus:
        case["benchmark_partition"] = "representative"
    references = [reference(case, technical=i + 2, aesthetic=i + 2) for i, case in enumerate(corpus)]
    attempts = [
        attempt(case, model, technical=i + 2.2, aesthetic=i + 2.1)
        for i, case in enumerate(corpus)
        for model in m.BASE_MODELS
    ]
    broken = next(
        row for row in attempts if row["case_id"] == corpus[-1]["case_id"] and row["model"] == "liquid"
    )
    broken.update(status="invalid", aesthetic_score=None)
    broken["task_valid"]["aesthetics"] = False
    rows = a.build_long(corpus, attempts, references, frozen_at=FROZEN)
    score = next(
        row
        for row in m.accuracy(rows)["panels"]
        if row["panel"] == "representative" and row["model"] == "luna" and row["arm"] == "baseline"
    )
    technical = score["technical_score_all_five_baselines_common"]
    aesthetic = score["aesthetic_score_all_five_baselines_common"]
    assert technical["n"] == 3 and aesthetic["n"] == 2
    assert technical["mae"] == pytest.approx(0.2) and technical["bias"] == pytest.approx(0.2)
    assert aesthetic["mae"] == pytest.approx(0.1) and aesthetic["bias"] == pytest.approx(0.1)
    assert technical["spearman"] == pytest.approx(1)
    assert aesthetic["spearman"] == pytest.approx(1)
    assert score["all_five_baselines_common"]["n"] == 2


def test_decimal_chat_means_preserve_equal_ties_without_rounding_native_values():
    corpus = cases(2)
    for case in corpus:
        case["benchmark_partition"] = "representative"
    # Binary averages differ despite both pairs having an exact decimal mean of 1.2.
    assert (1.0 + 1.4) / 2 != (1.1 + 1.3) / 2
    attempts = [
        attempt(case, model, technical=score, aesthetic=aesthetic)
        for case, pair in zip(corpus, [(1.0, 1.4), (1.1, 1.3)], strict=True)
        for model, score, aesthetic in zip(
            ("luna", "deepseek"), pair, (4.123456789, 4.876543209), strict=True
        )
    ]
    rows = a.build_long(corpus, attempts, [reference(case) for case in corpus], frozen_at=FROZEN)
    means = [row for row in rows if row["model"] == "luna_deepseek_mean" and row["arm"] == "baseline"]
    assert [row["technical_score"] for row in means] == [1.2, 1.2]
    assert [row["technical_score_native"] for row in means] == [1.2, 1.2]
    assert all(row["aesthetic_score"] == row["aesthetic_score_native"] == 4.499999999 for row in means)
    assert m.ranks([row["technical_score"] for row in means]).tolist() == [0.5, 0.5]
    # Exact tie semantics remain unchanged for other models' adjacent native values.
    assert m.ranks([1.2, 1.2, 1.2000000000000002]).tolist() == [0.5, 0.5, 2]


@pytest.mark.parametrize("panel", ["representative", "challenge"])
def test_exploratory_baseline_comparison_direction_and_sibling_coverage(panel):
    corpus = cases(3)
    for case in corpus:
        case["benchmark_partition"] = panel
    scores = {
        "luna": (3.4, 4.4),
        "deepseek": (3.6, 4.6),
        "jev": (3.25, 4.1),
        "liquid": (4.0, 4.8),
        "qrealign_lite": (3.75, 4.25),
    }
    attempts = [
        attempt(case, model, technical=technical, aesthetic=aesthetic)
        for case in corpus
        for model, (technical, aesthetic) in scores.items()
    ]
    invalid = next(
        row for row in attempts if row["case_id"] == corpus[-1]["case_id"] and row["model"] == "jev"
    )
    invalid.update(status="invalid", aesthetic_score=None)
    invalid["task_valid"]["aesthetics"] = False
    rows = a.build_long(corpus, attempts, [reference(case) for case in corpus], frozen_at=FROZEN)
    comparisons = m.baseline_comparisons(rows, repeats=100)
    assert len(comparisons) == 10
    jev = next(row for row in comparisons if row["panel"] == panel and row["model"] == "jev")
    assert jev["technical_score"]["n_images"] == 3
    assert jev["aesthetic_score"]["n_images"] == jev["combined_error"]["n_images"] == 2
    assert jev["technical_score"]["estimate"] == pytest.approx(-0.25)
    assert jev["aesthetic_score"]["estimate"] == pytest.approx(-0.4)
    assert jev["combined_error"]["estimate"] == pytest.approx(-0.325)
    assert jev["technical_score"]["ci95"] == pytest.approx([-0.25, -0.25])
    liquid = next(row for row in comparisons if row["panel"] == panel and row["model"] == "liquid")
    assert liquid["technical_score"]["estimate"] == pytest.approx(0.5)
    assert all(row["exploratory"] and row["multiplicity_adjustment"] == "none" for row in comparisons)
    assert jev["technical_score"]["seed"] == m.SEED
    assert comparisons == m.baseline_comparisons(rows, repeats=100)
    import report

    rendered = report.baseline_comparison_section({"baseline_comparisons": comparisons})
    assert "uncorrected for multiple comparisons" in rendered
    assert "-0.250 [-0.250, -0.250] (3)" in rendered


@pytest.mark.parametrize(("http_status", "status", "transport_error", "expected"), [
    (200, "valid", None, "http_request_through_complete_response"),
    (503, "failed", None, "http_request_through_complete_response"),
    (None, "unknown", "ReadTimeout", "incomplete_or_unmeasured_http_attempt"),
    (200, "unknown", None, "incomplete_or_unmeasured_http_attempt"),
    (200, "failed", "ConnectionError", "incomplete_or_unmeasured_http_attempt"),
])
def test_latency_kind_requires_completed_http_response_but_keeps_failure_elapsed(http_status, status, transport_error, expected):
    case = cases(1)[0]
    row = attempt(case)
    row.update(http_status=http_status, status=status, transport_error=transport_error)
    normalized = a.normalize_attempt(case, row)
    assert normalized["latency_kind"] == expected
    assert normalized["latency_seconds"] == row["latency_seconds"]
    assert normalized["response_received_at"] == row["response_received_at"]
