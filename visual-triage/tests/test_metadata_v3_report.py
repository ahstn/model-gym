"""Offline checks for costs, dataset shortcuts, and paired metadata evidence."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import digest
from tracks.metadata_v3 import cost_recovery, report
from tracks.metadata_v3.features import ARMS, assemble_context, extract_features


def comparison_row(identifier, gold, reward, group=None, **overrides):
    return {
        "id": identifier,
        "group_id": group or identifier,
        "gold": gold,
        "split": "test",
        "delivered_sha256": f"pixels-{identifier}",
        "reward": reward,
        **overrides,
    }


def sample(identifier, gold, split="dev", camera_present=False):
    return {
        "id": identifier,
        "group_id": identifier,
        "gold": gold,
        "split": split,
        "image": f"data/{identifier}.jpg",
        "delivered_sha256": f"pixels-{identifier}",
        "features": {
            "camera": {
                "status": "ok",
                "exif_availability": "present" if camera_present else "absent",
                "iso": 100 if camera_present else None,
                "exposure_seconds": None,
                "f_number": None,
                "focal_length_mm": None,
            }
        },
    }


def test_expenses_include_failed_attempts_and_known_zero_costs():
    summary = report.expense_summary(
        [
            {"status": 200, "record_type": "task", "effective_cost_usd": 0.2},
            {"status": 429, "record_type": "task", "effective_cost_usd": 0.0},
            {"status": 500, "record_type": "reflection", "effective_cost_usd": 0.3},
        ]
    )
    assert summary["attempts"] == 3
    assert summary["effective_cost_usd"] == pytest.approx(0.5)
    assert summary["known_effective_cost_usd"] == pytest.approx(0.5)
    assert summary["unknown_cost_count"] == 0


@pytest.mark.parametrize("invalid", [None, True, -0.01, float("nan"), float("inf"), "0.001"])
def test_unknown_or_invalid_cost_is_not_reported_as_zero(invalid):
    summary = report.expense_summary(
        [
            {"effective_cost_usd": 0.2},
            {"effective_cost_usd": invalid},
            {"transport_error": "ConnectionError"},
        ]
    )
    assert summary["attempts"] == 3
    assert summary["effective_cost_usd"] is None
    assert summary["known_effective_cost_usd"] == pytest.approx(0.2)
    assert summary["unknown_cost_count"] == 2


def test_no_attempts_has_zero_known_spend():
    summary = report.expense_summary([])
    assert summary["attempts"] == 0
    assert summary["effective_cost_usd"] == 0
    assert summary["known_effective_cost_usd"] == 0
    assert summary["unknown_cost_count"] == 0


@pytest.mark.parametrize(
    "changed",
    [
        {"delivered_sha256": "different-pixels"},
        {"group_id": "different-parent"},
        {"gold": True},
        {"split": "dev"},
    ],
)
def test_paired_comparison_rejects_changed_pixels_or_pair_identity(changed):
    left = [comparison_row("one", False, 0.4)]
    right = [{**left[0], "reward": 0.6, **changed}]
    with pytest.raises(ValueError):
        report.paired_class_balanced_delta(left, right, replicates=50)


def test_paired_comparison_rejects_missing_and_repeated_ids():
    left = [comparison_row("one", False, 0.4), comparison_row("two", True, 0.5)]
    with pytest.raises(ValueError):
        report.paired_class_balanced_delta(left, left[:1], replicates=50)
    with pytest.raises(ValueError):
        report.paired_class_balanced_delta(left, [left[0], left[0], left[1]], replicates=50)


def test_paired_delta_balances_classes_then_source_groups():
    # Three variants share a real parent. Each parent and each class must have
    # equal weight, regardless of variant count or the class sample counts.
    left = [
        *[comparison_row(f"real-a-{i}", False, 0.0, "real-a") for i in range(3)],
        comparison_row("real-b", False, 0.0),
        comparison_row("generated-a", True, 0.0),
    ]
    right = [{**row, "reward": 0.0 if row["group_id"] == "real-a" else 1.0} for row in left]
    result = report.paired_class_balanced_delta(left, right, replicates=1000)
    assert result["delta"] == pytest.approx(0.75)
    assert result["delta"] != pytest.approx(0.4)  # Incorrect image-weighted mean.
    assert result["n_images"] == 5
    assert result["n_groups"] == 3
    assert result["interval"] == pytest.approx([0.5, 1.0])
    assert report.paired_class_balanced_delta(left, right, replicates=1000) == result
    assert report.paired_class_balanced_delta(left, right, replicates=0)["interval"] is None


def test_paired_comparison_rejects_groups_with_mixed_gold_classes():
    rows = [comparison_row("real", False, 0.4, "shared"), comparison_row("generated", True, 0.5, "shared")]
    with pytest.raises(ValueError):
        report.paired_class_balanced_delta(rows, rows, replicates=50)


def test_single_class_paired_control_requires_explicit_opt_in():
    native = [comparison_row("one", False, 0.99), comparison_row("two", False, 0.75)]
    control = [{**native[0], "reward": 0.19}, {**native[1], "reward": 0.55}]
    assert report.paired_class_balanced_delta(native, control, replicates=100)["delta"] is None
    result = report.paired_class_balanced_delta(native, control, replicates=100, require_both_classes=False)
    assert result["delta"] == pytest.approx(-0.5)
    assert result["interval"] == pytest.approx([-0.8, -0.2])
    assert result["single_class_subset"] is True


def test_camera_availability_flags_perfect_development_label_leakage():
    samples = [
        sample("dev-real-a", 0, camera_present=True),
        sample("dev-real-b", 0, camera_present=True),
        sample("dev-ai-a", 1),
        sample("dev-ai-b", 1),
        # Test examples intentionally contradict the dev-only diagnostic rule.
        sample("test-real", 0, split="test"),
        sample("test-ai", 1, split="test", camera_present=True),
    ]
    result = report.coverage_summary(samples)
    development = result["by_split"]["dev"]
    assert development["real"]["count"] == 2
    assert development["real"]["camera_present"] == 2
    assert development["generated"]["count"] == 2
    assert development["generated"]["camera_present"] == 0
    assert result["availability_only_dev"]["accuracy"] == 1
    assert result["availability_only_dev"]["perfect_label_separation"] is True


def test_camera_presence_does_not_imply_usable_numeric_values():
    present = sample("present", 0, camera_present=True)
    present["features"]["camera"]["iso"] = None
    absent_real = sample("absent-real", 0)
    result = report.coverage_summary([present, absent_real, sample("ai", 1)])
    real = result["by_split"]["dev"]["real"]
    assert real["camera_present"] == 1
    assert real["camera_numeric_present"] == 0
    assert result["availability_only_dev"]["accuracy"] == pytest.approx(2 / 3)
    assert result["availability_only_dev"]["perfect_label_separation"] is False


def test_generated_exif_container_does_not_hide_numeric_camera_shortcut():
    real = sample("camera-photo", 0, camera_present=True)
    generated = sample("generated-with-exif", 1, camera_present=True)
    generated["features"]["camera"]["iso"] = None
    # Held-out examples contradict the diagnostic and must not affect its fit.
    held_out = sample("test-generated-with-camera", 1, split="test", camera_present=True)
    result = report.coverage_summary([real, generated, held_out])
    exif_rule = result["availability_only_dev"]
    numeric_rule = result["numeric_camera_only_dev"]
    assert exif_rule["n"] == numeric_rule["n"] == 2
    assert exif_rule["accuracy"] == 0.5
    assert exif_rule["perfect_label_separation"] is False
    assert numeric_rule["accuracy"] == 1.0
    assert numeric_rule["perfect_label_separation"] is True


@pytest.mark.parametrize(
    "samples", [[], [sample("real-only", 0, camera_present=True)], [sample("ai-only", 1)]]
)
def test_empty_or_one_class_development_is_not_perfect_label_separation(samples):
    result = report.coverage_summary(samples)
    for rule in ("availability_only_dev", "numeric_camera_only_dev"):
        assert result[rule]["perfect_label_separation"] is False
        if not samples:
            assert result[rule]["accuracy"] is None


def test_native_metrics_keep_failed_and_unattempted_cases_in_denominator():
    samples = [
        sample("valid", 0, split="test"),
        sample("failed", 1, split="test"),
        sample("missing", 1, split="test"),
    ]
    records = [
        {
            "sample": "valid",
            "delivered_sha256": "pixels-valid",
            "status": 200,
            "normalized": {"ai_probability": 0.2, "medium": "photo", "uncertainty": "high"},
            "effective_cost_usd": 0.001,
            "seconds": 0.1,
        },
        {
            "sample": "failed",
            "delivered_sha256": "pixels-failed",
            "status": 503,
            "normalized": None,
            "effective_cost_usd": 0.002,
            "seconds": 0.2,
        },
    ]
    result = report.summarise_cases(samples, records)
    summary = result["summary"]
    metrics = summary["metrics"]
    assert summary["expected"] == 3
    assert summary["unattempted"] == 1
    assert metrics["n_valid"] == 1
    assert metrics["accuracy_all_expected_invalid_as_wrong"] == pytest.approx(1 / 3)
    assert metrics["balanced_accuracy_all_expected"] == pytest.approx(0.5)
    assert metrics["brier_invalid_or_unattempted_penalty1"] == pytest.approx(0.68)
    assert metrics["class_counts_expected"] == {"real": 1, "generated": 2}
    cases = {row["id"]: row for row in result["per_image"]}
    assert cases["failed"]["attempted"] is True
    assert cases["failed"]["valid"] is False
    assert cases["missing"]["attempted"] is False
    assert cases["failed"]["reward"] == cases["missing"]["reward"] == 0


@pytest.fixture
def frozen_study(tmp_path):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
        return path

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    image = tmp_path / "view.png"
    Image.new("RGB", (16, 16), (80, 100, 120)).save(image)
    features = extract_features(image, image)
    samples = [
        {
            **sample(identifier, gold, split="test"),
            "features": copy.deepcopy(features),
            "features_sha256": digest(features),
            "preprocess_seconds": 0.3 if gold else 0.2,
        }
        for identifier, gold in (("real", 0), ("generated", 1))
    ]
    control = copy.deepcopy(samples[0])
    control.update(
        id="spoofed-real",
        base_sample_id="real",
        condition="spoofed_camera",
        arms=["camera"],
        control_kind="context_only",
        expected_c2pa_status=None,
        c2pa_validation_pass=None,
    )
    control["features"]["camera"].update(exif_availability="present", iso=100)
    control["features_sha256"] = digest(control["features"])
    dataset = save(tmp_path / "dataset.json", {"samples": samples})
    controls = save(tmp_path / "controls.json", {"cases": [control]})
    protocol = save(
        tmp_path / "protocol.json", {"dataset_sha256": sha(dataset), "controls_sha256": sha(controls)}
    )
    prompt = {"id": "fixture", "context": "Assess image origin from supplied evidence."}
    prompt_path = save(tmp_path / "baseline.json", prompt)
    runs = tmp_path / "runs"

    def record(member, arm, valid=True, probability=0.1):
        return {
            "record_type": "task",
            "sample": member["id"],
            "metadata_arm": arm,
            "task": "ai",
            "arm": "decider",
            "split": "test",
            "timestamp": "2026-10-12T12:01:00+00:00",
            "protocol_sha256": sha(protocol),
            "prompt_sha256": digest(prompt),
            "features_sha256": member["features_sha256"],
            "delivered_sha256": member["delivered_sha256"],
            "context_sha256": digest(assemble_context(member["features"], arm)),
            "status": 200 if valid else 503,
            "seconds": 0.1,
            "effective_cost_usd": 0.001,
            "cost_usd": 0.001,
            "usage": {"cost": 0.001},
            "normalized": {"ai_probability": probability, "medium": "photo", "uncertainty": "low"}
            if valid
            else None,
        }

    for arm in ARMS:
        save(
            runs / arm / "selection.json",
            {
                "metadata_arm": arm,
                "protocol_sha256": sha(protocol),
                "selection_split": "dev",
                "selected_at": "2026-10-12T12:00:00+00:00",
                "test_used": False,
                "baseline_prompt_path": str(prompt_path),
                "selected_prompt_path": str(prompt_path),
                "selected_prompt_sha256": digest(prompt),
            },
        )
        ledger = runs / arm / "final" / "baseline.jsonl"
        ledger.parent.mkdir(parents=True)
        ledger.write_text(
            "".join(json.dumps(record(member, arm, valid=not member["gold"])) + "\n" for member in samples)
        )
    return {
        "arguments": {
            "dataset_path": dataset,
            "protocol_path": protocol,
            "runs_root": runs,
            "project_root": tmp_path,
            "bootstrap_replicates": 20,
        },
        "controls_path": controls,
        "control_ledger": runs / "camera" / "controls" / "spoofed_camera.jsonl",
        "control_record": record(control, "camera", probability=0.9),
    }


def test_report_keeps_controls_separate_and_counts_reused_baseline_once(frozen_study):
    before = report.build_report(**frozen_study["arguments"])
    path = frozen_study["control_ledger"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(frozen_study["control_record"]) + "\n")
    result = report.build_report(**frozen_study["arguments"], controls_path=frozen_study["controls_path"])
    assert result["native"] == before["native"]
    assert result["comparisons"] == before["comparisons"]
    assert before["expenses"]["attempts"] == 14
    assert result["expenses"]["attempts"] == 15
    assert result["expenses"]["effective_cost_usd"] == pytest.approx(0.015)
    assert result["expenses"]["by_phase"]["controls"]["attempts"] == 1
    assert len(result["cases"]) == 29
    for arm in ARMS:
        native = result["native"][arm]
        assert native["baseline"]["metrics"]["n_expected"] == 2
        assert native["baseline"]["metrics"]["n_valid"] == 1
        assert native["selected"]["reused_baseline"] is True
        assert native["selected"]["metrics"]["accuracy_all_expected_invalid_as_wrong"] == 0.5
        empty = native["baseline"]["metadata_slices"]["numeric_camera_present:present"]["metrics"]
        assert empty["class_counts_expected"] == {"real": 0, "generated": 0}
        assert empty["n_expected"] == 0
        assert empty["accuracy_all_expected_invalid_as_wrong"] is None
    assert (
        result["controls"]["spoofed_camera"]["camera"]["metrics"]["accuracy_all_expected_invalid_as_wrong"]
        == 0
    )
    shift = result["controls"]["spoofed_camera"]["camera"]["same_pixels_vs_native_selected"]
    assert shift["paired_valid"] == shift["decision_flips"] == 1
    assert shift["mean_probability_change"] == pytest.approx(0.8)
    assert result["controls"]["spoofed_camera"]["camera"]["integrity_fixtures"] == {
        "asserted": 0,
        "passed": 0,
        "not_origin_accuracy": True,
    }
    assert result["freeze_timing"]["all_selections_preceded_evaluation"] is True
    paired = result["controls"]["spoofed_camera"]["camera"]["paired_native_subset"]
    assert paired["n_unpaired_intended"] == 0
    assert paired["native_metrics"]["n_expected"] == 1
    assert paired["native_metrics"]["brier_invalid_or_unattempted_penalty1"] == pytest.approx(0.01)
    assert paired["control_metrics"]["brier_invalid_or_unattempted_penalty1"] == pytest.approx(0.81)
    assert paired["native_metrics"]["class_counts_expected"] == {"real": 1, "generated": 0}
    assert paired["reward_change"]["single_class_subset"] is True
    assert paired["reward_change"]["delta"] == pytest.approx(-0.8)
    assert paired["brier_change"] == pytest.approx(0.8)
    assert paired["brier_change_interval"] == pytest.approx([0.8, 0.8])


def test_builder_preprocessing_field_counts_native_once_and_not_inherited_control_time(frozen_study):
    result = report.build_report(**frozen_study["arguments"], controls_path=frozen_study["controls_path"])
    native = result["preprocessing"]["native_unique_cases"]
    assert native["cases"] == 2
    assert native["seconds"] == native["known_seconds"] == pytest.approx(0.5)
    assert native["unknown_count"] == 0
    control = result["preprocessing"]["control_unique_cases"]
    assert control["cases"] == 1
    assert control["seconds"] is None
    assert control["known_seconds"] == 0
    assert control["unknown_count"] == 1
    for case in result["cases"]:
        if case["condition"] == "native":
            assert case["preprocessing_seconds"] == (0.2 if case["case"] == "real" else 0.3)
        else:
            assert case["preprocessing_seconds"] is None
    assert result["controls"]["spoofed_camera"]["camera"]["preprocessing"]["unknown_count"] == 1


def test_preprocessing_supports_legacy_field_without_overriding_native_zero_or_null():
    members = [
        {"preprocess_seconds": 0.2},
        {"preprocessing_seconds": 0.3},
        {"preprocess_seconds": 0.0, "preprocessing_seconds": 999},
    ]
    total = report._preprocessing(members)
    assert total["seconds"] == total["known_seconds"] == pytest.approx(0.5)
    assert total["unknown_count"] == 0
    total = report._preprocessing([*members, {"preprocess_seconds": None, "preprocessing_seconds": 999}])
    assert total["seconds"] is None
    assert total["known_seconds"] == pytest.approx(0.5)
    assert total["unknown_count"] == 1


@pytest.mark.parametrize("inherited", [{"preprocess_seconds": 0.2}, {"preprocessing_seconds": 0.3}])
def test_control_preprocessing_is_unmeasured_for_both_inherited_field_names(inherited):
    result = report._preprocessing([{**inherited, "base_sample_id": "native-case"}])
    assert result["seconds"] is None
    assert result["known_seconds"] == 0
    assert result["unknown_count"] == 1


@pytest.mark.parametrize("failed_attempt", [False, True])
def test_matched_control_retains_missing_or_invalid_results(frozen_study, failed_attempt):
    if failed_attempt:
        path = frozen_study["control_ledger"]
        path.parent.mkdir(parents=True)
        row = {**frozen_study["control_record"], "status": 503, "normalized": None}
        path.write_text(json.dumps(row) + "\n")
    result = report.build_report(**frozen_study["arguments"], controls_path=frozen_study["controls_path"])
    control = result["controls"]["spoofed_camera"]["camera"]
    paired = control["paired_native_subset"]
    assert paired["n_unpaired_intended"] == 0
    assert paired["control_metrics"]["n_expected"] == 1
    assert paired["control_metrics"]["n_valid"] == 0
    assert paired["control_metrics"]["brier_invalid_or_unattempted_penalty1"] == 1
    assert paired["native_metrics"]["brier_invalid_or_unattempted_penalty1"] == pytest.approx(0.01)
    assert paired["reward_change"]["delta"] == pytest.approx(-0.99)
    assert paired["brier_change"] == pytest.approx(0.99)
    assert paired["brier_change_interval"] == pytest.approx([0.99, 0.99])
    assert control["unattempted"] == (0 if failed_attempt else 1)


def test_matched_control_retains_failed_native_result(frozen_study):
    native_path = frozen_study["arguments"]["runs_root"] / "camera" / "final" / "baseline.jsonl"
    rows = [json.loads(line) for line in native_path.read_text().splitlines()]
    rows[0].update(status=503, normalized=None)
    native_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    control_path = frozen_study["control_ledger"]
    control_path.parent.mkdir(parents=True)
    control_path.write_text(json.dumps(frozen_study["control_record"]) + "\n")
    result = report.build_report(**frozen_study["arguments"], controls_path=frozen_study["controls_path"])
    paired = result["controls"]["spoofed_camera"]["camera"]["paired_native_subset"]
    assert paired["native_metrics"]["n_expected"] == 1
    assert paired["native_metrics"]["n_valid"] == 0
    assert paired["native_metrics"]["brier_invalid_or_unattempted_penalty1"] == 1
    assert paired["control_metrics"]["brier_invalid_or_unattempted_penalty1"] == pytest.approx(0.81)
    assert paired["reward_change"]["delta"] == pytest.approx(0.19)
    assert paired["brier_change"] == pytest.approx(-0.19)
    assert paired["brier_change_interval"] == pytest.approx([-0.19, -0.19])


@pytest.mark.parametrize("unknown_exif", [False, True])
def test_evaluated_metadata_and_source_slices_include_class_counts(tmp_path, unknown_exif):
    image = tmp_path / "view.png"
    Image.new("RGB", (16, 16), (80, 100, 120)).save(image)
    features = extract_features(image, image)
    members = []
    settings = [
        ("real", 0, "present", True, None),
        ("generated-exif", 1, "present", False, True),
        ("generated-plain", 1, "absent", False, False),
    ]
    if unknown_exif:
        settings.append(("real-unknown", 0, "unknown", False, None))
    for identifier, gold, exif, has_numeric, claim in settings:
        value = copy.deepcopy(features)
        value["camera"].update(exif_availability=exif, iso=100 if has_numeric else None)
        if claim is not None:
            value["provenance"].update(
                status="valid",
                manifest_present=True,
                signature_valid=True,
                hard_binding_valid=True,
                ai_generation_claim=claim,
            )
        members.append(
            {
                **sample(identifier, gold, split="test"),
                "features": value,
                "features_sha256": digest(value),
                "source": {
                    "dataset": "fixture",
                    "camera_group": None if gold else "camera-A",
                    "generator": "gen-A" if gold else None,
                },
            }
        )
    prompt = {"context": "Assess supplied evidence."}
    ledger = tmp_path / "results.jsonl"
    records = [
        {
            "sample": member["id"],
            "metadata_arm": "all_context",
            "task": "ai",
            "arm": "decider",
            "protocol_sha256": "protocol",
            "prompt_sha256": digest(prompt),
            "features_sha256": member["features_sha256"],
            "delivered_sha256": member["delivered_sha256"],
            "context_sha256": digest(assemble_context(member["features"], "all_context")),
            "status": 200,
            "effective_cost_usd": 0.001,
            "seconds": 0.1,
            "normalized": {
                "ai_probability": 0.9 if member["gold"] else 0.1,
                "medium": "photo",
                "uncertainty": "low",
            },
        }
        for member in members
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in records))
    result, _ = report._evaluate(members, ledger, prompt, "all_context", "protocol", 20)
    summary = result["summary"]
    expected = {
        "exif_present:present": {"real": 1, "generated": 1},
        "exif_present:absent": {"real": 0, "generated": 1},
        "numeric_camera_present:present": {"real": 1, "generated": 0},
        "numeric_camera_present:absent": {"real": int(unknown_exif), "generated": 2},
        "c2pa_ai_claim:present": {"real": 0, "generated": 1},
        "c2pa_ai_claim:absent": {"real": 1 + unknown_exif, "generated": 1},
    }
    if unknown_exif:
        expected["exif_present:unknown"] = {"real": 1, "generated": 0}
    assert set(summary["metadata_slices"]) == set(expected)
    for name, counts in expected.items():
        metrics = summary["metadata_slices"][name]["metrics"]
        assert metrics["class_counts_expected"] == counts
        assert metrics["n_expected"] == metrics["n_valid"] == sum(counts.values())
    for name, counts in {
        "dataset:fixture": {"real": 1 + unknown_exif, "generated": 2},
        "camera_group:camera-A": {"real": 1 + unknown_exif, "generated": 0},
        "generator:gen-A": {"real": 0, "generated": 2},
    }.items():
        assert summary["source_slices"][name]["metrics"]["class_counts_expected"] == counts


@pytest.mark.parametrize(
    "status,validator_pass,passed",
    [("invalid", True, 1), ("invalid", False, 0), ("invalid", None, 0), ("valid", True, 0)],
)
def test_byte_control_integrity_requires_expected_status_and_validator(
    monkeypatch, tmp_path, status, validator_pass, passed
):
    case = {
        "id": "fixture",
        "condition": "c2pa_invalid_signature",
        "arms": ["provenance"],
        "features": {"provenance": {"status": status}},
        "expected_c2pa_status": "invalid",
        "control_kind": "native_byte_intervention",
        "c2pa_validation_pass": validator_pass,
    }
    monkeypatch.setattr(
        report, "_evaluate", lambda *args: ({"summary": {}, "per_image": [{"valid": False}]}, {})
    )
    monkeypatch.setattr(report, "_compact", lambda *args: {})
    summaries, _ = report._control_report(
        [case],
        [],
        {"provenance": {"selected": []}},
        {"provenance": {"selected_prompt": {}}},
        tmp_path,
        "sha",
        0,
    )
    assert summaries["c2pa_invalid_signature"]["provenance"]["integrity_fixtures"] == {
        "asserted": 1,
        "passed": passed,
        "not_origin_accuracy": True,
    }


def test_selected_prompt_exports_preserve_body_hash_and_raw_receipts(frozen_study):
    arguments = frozen_study["arguments"]
    result = report.build_report(**arguments)
    receipts = {arm: (arguments["runs_root"] / arm / "selection.json").read_bytes() for arm in ARMS}
    output_root = arguments["project_root"] / "tracked"
    selected = report.export_selected_prompts(result["selection"], output_root, arguments["project_root"])
    assert set(selected) == set(ARMS)
    for arm, selection in selected.items():
        path = arguments["project_root"] / selection["selected_export_path"]
        prompt = json.loads(path.read_text())
        assert path == output_root / "prompts" / f"{arm}.json"
        assert prompt["id"] == selection["selected_prompt_id"] == "fixture"
        assert digest(prompt) == selection["selected_prompt_sha256"]
        assert selection["selected_prompt_path"] == selection["selected_export_path"]
        assert not any(Path(value).is_absolute() for key, value in selection.items() if key.endswith("_path"))
        assert (arguments["runs_root"] / arm / "selection.json").read_bytes() == receipts[arm]


@pytest.fixture
def recovery_projection(frozen_study, monkeypatch):
    """Report projection only; recovery tests own strict receipt validation."""
    arguments = frozen_study["arguments"]
    runs = arguments["runs_root"]
    protocol_sha = hashlib.sha256(arguments["protocol_path"].read_bytes()).hexdigest()
    failure = {
        "record_type": "task",
        "task": "ai",
        "arm": "decider",
        "metadata_arm": "camera",
        "sample": "failed-training-case",
        "split": "train",
        "request_sha256": "a" * 64,
        "transport_error": "ConnectionError",
        "status": None,
        "normalized": None,
        "effective_cost_usd": None,
        "cost_usd": None,
        "protocol_sha256": protocol_sha,
    }
    ledger = runs / "camera" / "candidates" / "failed" / "train.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps(failure) + "\n")
    amendment = {
        "original_protocol_sha256": protocol_sha,
        "frozen_at": "2026-10-12T11:59:00+00:00",
        "review_reason": "Keep unresolved transport costs separate and reserve guard funds.",
        "allowance_usd": 0.05,
        "actual_cost_remains_unknown": True,
        "no_retry": True,
        "pricing_assumption": {"proven_billing_ceiling": False},
    }
    amendment_path = arguments["protocol_path"].with_name(cost_recovery.AMENDMENT_NAME)
    amendment_path.write_text(json.dumps(amendment) + "\n")
    amendment_sha = hashlib.sha256(amendment_path.read_bytes()).hexdigest()
    receipt = {
        "amendment_sha256": amendment_sha,
        "ledger_path": str(ledger.relative_to(runs)),
        "sample": failure["sample"],
        "request_sha256": failure["request_sha256"],
        "record_sha256": digest(failure),
        "metadata_arm": "camera",
        "allowance_usd": 0.05,
        "actual_cost_usd": None,
        "no_retry": True,
    }
    receipt_path = runs / "recovery-receipts" / "approved.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(json.dumps(receipt) + "\n")
    validated = {
        "amendment": amendment,
        "costs": {
            "effective_cost_usd": None,
            "known_reported_usd": 0.014,
            "unknown_cost_count": 1,
            "reserved_unverified_usd": 0.05,
            "budget_accounted_usd": 0.064,
            "amendment_sha256": amendment_sha,
            "arms": {
                arm: {
                    "known_reported_usd": 0.002,
                    "reserved_unverified_usd": 0.05 if arm == "camera" else 0.0,
                    "budget_accounted_usd": 0.052 if arm == "camera" else 0.002,
                }
                for arm in ARMS
            },
        },
    }
    calls = []

    def allowance_summary(runs_root, path):
        calls.append((Path(runs_root), Path(path)))
        return copy.deepcopy(validated)

    monkeypatch.setattr(report, "allowance_summary", allowance_summary)
    return {
        **frozen_study,
        "amendment_path": amendment_path,
        "receipt_path": receipt_path,
        "validated": validated,
        "calls": calls,
        "raw": {path: path.read_bytes() for path in runs.rglob("*.json*")},
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_recovery_allowance_is_separate_from_unknown_actual_cost(recovery_projection, explicit):
    fixture = recovery_projection
    extra = {"amendment_path": fixture["amendment_path"]} if explicit else {}
    result = report.build_report(**fixture["arguments"], **extra)
    expenses = result["expenses"]
    assert expenses["attempts"] == 15
    assert expenses["effective_cost_usd"] is None
    assert expenses["unknown_cost_count"] == 1
    assert expenses["known_effective_cost_usd"] == pytest.approx(0.014)
    assert expenses["reserved_unverified_usd"] == pytest.approx(0.05)
    assert expenses["budget_accounted_usd"] == pytest.approx(0.064)
    camera = expenses["by_arm"]["camera"]
    assert camera["attempts"] == 3
    assert camera["effective_cost_usd"] is None
    assert camera["unknown_cost_count"] == 1
    assert camera["reserved_unverified_usd"] == pytest.approx(0.05)
    assert camera["budget_accounted_usd"] == pytest.approx(0.052)
    for arm in set(ARMS) - {"camera"}:
        assert expenses["by_arm"][arm]["reserved_unverified_usd"] == 0
        assert expenses["by_arm"][arm]["budget_accounted_usd"] == pytest.approx(0.002)
    recovery = expenses["recovery"]
    assert recovery["actual_cost_remains_unknown"] is recovery["no_retry"] is True
    assert recovery["allowance_usd"] == pytest.approx(0.05)
    assert recovery["pricing_assumption"]["proven_billing_ceiling"] is False
    assert len(recovery["receipt_inventory"]) == 1
    receipt = recovery["receipt_inventory"][0]
    assert receipt["sha256"] == hashlib.sha256(fixture["receipt_path"].read_bytes()).hexdigest()
    assert receipt["sample"] == "failed-training-case"
    assert receipt["request_sha256"] == "a" * 64
    assert receipt["actual_cost_usd"] is None
    assert recovery["amendment_sha256"] == hashlib.sha256(fixture["amendment_path"].read_bytes()).hexdigest()
    assert fixture["calls"] == [(fixture["arguments"]["runs_root"], fixture["amendment_path"])]
    assert all(path.read_bytes() == value for path, value in fixture["raw"].items())


@pytest.mark.parametrize("mismatch", ["protocol", "unknown_count", "known_subtotal"])
def test_recovery_projection_rejects_inventory_mismatch(recovery_projection, mismatch):
    validated = recovery_projection["validated"]
    if mismatch == "protocol":
        validated["amendment"]["original_protocol_sha256"] = "different-protocol"
    elif mismatch == "unknown_count":
        validated["costs"]["unknown_cost_count"] = 2
    else:
        validated["costs"]["known_reported_usd"] = 0.015
    with pytest.raises(ValueError):
        report.build_report(**recovery_projection["arguments"])


@pytest.mark.parametrize("extra_receipt", [False, True])
def test_recovery_report_rejects_missing_or_extra_receipt(recovery_projection, extra_receipt):
    path = recovery_projection["receipt_path"]
    if extra_receipt:
        path.with_name("duplicate.json").write_bytes(path.read_bytes())
    else:
        path.unlink()
    with pytest.raises(ValueError, match="receipt inventory"):
        report.build_report(**recovery_projection["arguments"])


@pytest.fixture
def reallocation_projection(recovery_projection, monkeypatch):
    fixture = recovery_projection
    caps = {arm: 1.10 if arm == "image_only" else 0.65 for arm in ARMS}
    reallocation = {
        "first_amendment_sha256": hashlib.sha256(fixture["amendment_path"].read_bytes()).hexdigest(),
        "frozen_at": "2026-10-12T12:00:00+00:00",
        "review_reason": "Reserve unused global headroom for the image-only arm.",
        "global_budget_usd": 5.0,
        "original_run_receipt_arm_usd": 0.65,
        "guard_budget_by_arm": caps,
        "reallocated_unused_global_headroom_usd": 0.45,
        "no_retry": True,
        "new_unknown_batch_action": "Record approved allowances and stop; check network before manual resume",
    }
    path = fixture["amendment_path"].with_name("cost-recovery-reallocation.json")
    path.write_text(json.dumps(reallocation) + "\n")
    validated = copy.deepcopy(fixture["validated"])
    validated["reallocation"] = reallocation
    validated["costs"]["guard_budget_by_arm"] = caps
    for arm in ARMS:
        validated["costs"]["arms"][arm]["guard_budget_usd"] = caps[arm]
    calls = []

    def reallocation_summary(runs_root, reallocation_path):
        calls.append((Path(runs_root), Path(reallocation_path)))
        return copy.deepcopy(validated)

    monkeypatch.setattr(report, "reallocation_summary", reallocation_summary)
    return {
        **fixture,
        "reallocation_path": path,
        "reallocation_validated": validated,
        "reallocation_calls": calls,
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_reallocation_reports_new_guard_without_resolving_unknown_cost(reallocation_projection, explicit):
    fixture = reallocation_projection
    extra = {"reallocation_path": fixture["reallocation_path"]} if explicit else {}
    result = report.build_report(**fixture["arguments"], **extra)
    expenses = result["expenses"]
    caps = {arm: 1.10 if arm == "image_only" else 0.65 for arm in ARMS}
    assert expenses["guard_budget_by_arm"] == caps
    for arm, cap in caps.items():
        assert expenses["by_arm"][arm]["guard_budget_usd"] == cap
    assert expenses["effective_cost_usd"] is None
    assert expenses["unknown_cost_count"] == 1
    assert expenses["known_effective_cost_usd"] == pytest.approx(0.014)
    assert expenses["reserved_unverified_usd"] == pytest.approx(0.05)
    assert expenses["budget_accounted_usd"] == pytest.approx(0.064)
    assert expenses["attempts"] == 15
    second = expenses["recovery"]["reallocation"]
    assert second["sha256"] == hashlib.sha256(fixture["reallocation_path"].read_bytes()).hexdigest()
    assert second["global_budget_usd"] == 5.0
    assert second["original_run_receipt_arm_usd"] == 0.65
    assert second["guard_budget_by_arm"] == caps
    assert second["no_retry"] is True
    assert fixture["calls"] == []
    assert fixture["reallocation_calls"] == [
        (fixture["arguments"]["runs_root"], fixture["reallocation_path"])
    ]
    assert all(path.read_bytes() == value for path, value in fixture["raw"].items())
    rendered = report.markdown_report(result)
    sentence = next(line for line in rendered.splitlines() if "second operational amendment" in line)
    assert "image_only" in sentence and "$1.10" in sentence and "$0.45" in sentence
    assert "global cap stays $5.00" in sentence
    assert "Missing costs remain unknown" in rendered


def test_reallocation_rejects_wrong_first_amendment_binding(reallocation_projection):
    fixture = reallocation_projection
    fixture["reallocation_validated"]["reallocation"]["first_amendment_sha256"] = "wrong-amendment"
    with pytest.raises(ValueError, match="amendment"):
        report.build_report(**fixture["arguments"])


@pytest.fixture
def timeout_projection(reallocation_projection, monkeypatch):
    fixture = reallocation_projection
    extension = {
        "reallocation_sha256": hashlib.sha256(fixture["reallocation_path"].read_bytes()).hexdigest(),
        "frozen_at": "2026-10-12T12:00:30+00:00",
        "review_reason": "Keep unresolved read timeout charges unknown and reserve guard funds.",
        "eligible_transport_errors": ["ConnectionError", "ReadTimeout"],
        "added_transport_error": "ReadTimeout",
        "allowance_usd": 0.05,
        "actual_cost_remains_unknown": True,
        "no_retry": True,
        "new_unknown_batch_action": "Record allowances and stop; verify network before manual resume",
    }
    path = fixture["amendment_path"].with_name("cost-recovery-timeouts.json")
    path.write_text(json.dumps(extension) + "\n")
    validated = copy.deepcopy(fixture["reallocation_validated"])
    validated["timeout_extension"] = extension
    calls = []

    def timeout_summary(runs_root, extension_path):
        calls.append((Path(runs_root), Path(extension_path)))
        return copy.deepcopy(validated)

    monkeypatch.setattr(report, "timeout_summary", timeout_summary)
    return {**fixture, "timeout_extension_path": path, "timeout_validated": validated, "timeout_calls": calls}


@pytest.mark.parametrize("explicit", [False, True])
def test_timeout_extension_preserves_actual_unknown_costs_and_existing_caps(timeout_projection, explicit):
    fixture = timeout_projection
    extra = {"timeout_extension_path": fixture["timeout_extension_path"]} if explicit else {}
    result = report.build_report(**fixture["arguments"], **extra)
    expenses = result["expenses"]
    assert expenses["effective_cost_usd"] is None
    assert expenses["unknown_cost_count"] == 1
    assert expenses["known_effective_cost_usd"] == pytest.approx(0.014)
    assert expenses["reserved_unverified_usd"] == pytest.approx(0.05)
    assert expenses["budget_accounted_usd"] == pytest.approx(0.064)
    assert expenses["attempts"] == 15
    caps = {arm: 1.10 if arm == "image_only" else 0.65 for arm in ARMS}
    assert expenses["guard_budget_by_arm"] == caps
    for arm, cap in caps.items():
        assert expenses["by_arm"][arm]["guard_budget_usd"] == cap
    recovery = expenses["recovery"]
    third = recovery["timeout_extension"]
    assert third["sha256"] == hashlib.sha256(fixture["timeout_extension_path"].read_bytes()).hexdigest()
    assert third["eligible_transport_errors"] == ["ConnectionError", "ReadTimeout"]
    assert third["added_transport_error"] == "ReadTimeout"
    assert third["actual_cost_remains_unknown"] is third["no_retry"] is True
    assert third["allowance_usd"] == 0.05
    assert (
        recovery["reallocation"]["sha256"]
        == fixture["timeout_validated"]["timeout_extension"]["reallocation_sha256"]
    )
    assert (
        recovery["amendment_sha256"] == fixture["timeout_validated"]["reallocation"]["first_amendment_sha256"]
    )
    assert fixture["calls"] == fixture["reallocation_calls"] == []
    assert fixture["timeout_calls"] == [
        (fixture["arguments"]["runs_root"], fixture["timeout_extension_path"])
    ]
    assert all(path.read_bytes() == value for path, value in fixture["raw"].items())


@pytest.mark.parametrize("wrong_link", ["first", "second"])
def test_timeout_extension_rejects_changed_amendment_chain(timeout_projection, wrong_link):
    fixture = timeout_projection
    if wrong_link == "second":
        fixture["timeout_validated"]["timeout_extension"]["reallocation_sha256"] = "wrong-second"
    else:
        fixture["timeout_validated"]["reallocation"]["first_amendment_sha256"] = "wrong-first"
    with pytest.raises(ValueError):
        report.build_report(**fixture["arguments"])


@pytest.fixture
def shared_pool_projection(timeout_projection, monkeypatch):
    fixture = timeout_projection
    fourth = {
        "timeout_extension_sha256": hashlib.sha256(
            fixture["timeout_extension_path"].read_bytes()
        ).hexdigest(),
        "frozen_at": "2026-10-12T12:00:45+00:00",
        "review_reason": "Reserve unresolved allowances only within the existing shared global cap.",
        "global_budget_usd": 5.0,
        "known_spend_budget_per_arm_usd": 0.65,
        "unknown_allowance_usd": 0.05,
        "unknown_allowance_scope": "shared_global_only",
        "global_guard_scope": "Known costs plus unresolved allowances plus all inflight reservations",
        "arm_guard_scope": "Known costs plus original prospective known-cost headroom; excludes unresolved allowances",
        "actual_cost_remains_unknown": True,
        "no_retry": True,
        "transport_policy_unchanged": True,
        "circuit_stop_unchanged": True,
    }
    path = fixture["amendment_path"].with_name("cost-recovery-shared-pool.json")
    path.write_text(json.dumps(fourth) + "\n")
    validated = copy.deepcopy(fixture["timeout_validated"])
    validated["shared_pool_amendment"] = fourth
    costs = validated["costs"]
    costs.pop("guard_budget_by_arm")
    costs.update(
        global_budget_usd=5.0,
        reserved_inflight_usd=0.03,
        budget_guard_scope="Known reported costs per arm; unresolved allowances and all inflight reservations use the shared global cap",
        shared_unknown_allowance_pool_usd=0.05,
        known_spend_budget_per_arm_usd=0.65,
        known_spend_budget_by_arm={arm: 0.65 for arm in ARMS},
    )
    for arm in ARMS:
        costs["arms"][arm].pop("guard_budget_usd")
        costs["arms"][arm].update(known_spend_budget_usd=0.65, guard_scope="known_reported_usd only")
    diagnostic = fixture["arguments"]["project_root"] / "diagnostics" / "transport.jsonl"
    diagnostic.parent.mkdir()
    diagnostic.write_text(
        json.dumps(
            {
                "record_type": "task",
                "metadata_arm": "camera",
                "effective_cost_usd": 99.0,
                "transport_error": "ReadTimeout",
            }
        )
        + "\n"
    )
    calls = []

    def shared_pool_summary(runs_root, shared_pool_path):
        calls.append((Path(runs_root), Path(shared_pool_path)))
        return copy.deepcopy(validated)

    monkeypatch.setattr(report, "shared_pool_summary", shared_pool_summary)
    return {
        **fixture,
        "shared_pool_path": path,
        "shared_pool_validated": validated,
        "shared_pool_calls": calls,
        "diagnostic_path": diagnostic,
        "diagnostic_bytes": diagnostic.read_bytes(),
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_shared_pool_keeps_current_known_spend_caps_distinct_from_prior_guards(
    shared_pool_projection, explicit
):
    fixture = shared_pool_projection
    extra = {"shared_pool_path": fixture["shared_pool_path"]} if explicit else {}
    result = report.build_report(**fixture["arguments"], **extra)
    expenses = result["expenses"]
    assert expenses["attempts"] == 15
    assert expenses["effective_cost_usd"] is None
    assert expenses["unknown_cost_count"] == 1
    assert expenses["known_effective_cost_usd"] == pytest.approx(0.014)
    assert expenses["reserved_unverified_usd"] == expenses["shared_unknown_allowance_pool_usd"] == 0.05
    assert expenses["budget_accounted_usd"] == pytest.approx(0.064)
    assert expenses["global_budget_usd"] == 5.0
    assert expenses["reserved_inflight_usd"] == 0.03
    assert expenses["global_guard_accounted_usd"] == pytest.approx(0.094)
    assert expenses["known_spend_budget_by_arm"] == {arm: 0.65 for arm in ARMS}
    assert "guard_budget_by_arm" not in expenses
    for arm in ARMS:
        assert expenses["by_arm"][arm]["known_spend_budget_usd"] == 0.65
        assert "guard_budget_usd" not in expenses["by_arm"][arm]
    recovery = expenses["recovery"]
    assert recovery["reallocation"]["guard_budget_by_arm"]["image_only"] == 1.10
    fourth = recovery["shared_pool_amendment"]
    assert fourth["sha256"] == hashlib.sha256(fixture["shared_pool_path"].read_bytes()).hexdigest()
    assert fourth["unknown_allowance_scope"] == "shared_global_only"
    assert fourth["known_spend_budget_per_arm_usd"] == 0.65
    assert fourth["actual_cost_remains_unknown"] is fourth["no_retry"] is True
    assert fourth["transport_policy_unchanged"] is fourth["circuit_stop_unchanged"] is True
    assert fixture["calls"] == fixture["reallocation_calls"] == fixture["timeout_calls"] == []
    assert fixture["shared_pool_calls"] == [(fixture["arguments"]["runs_root"], fixture["shared_pool_path"])]
    assert fixture["diagnostic_path"].read_bytes() == fixture["diagnostic_bytes"]
    assert all(path.read_bytes() == value for path, value in fixture["raw"].items())
    rendered = report.markdown_report(result)
    assert "second operational amendment" not in rendered.lower()
    assert "third operational amendment" not in rendered.lower()
    assert "$5.00" in rendered and "$0.65" in rendered
    assert "All unknown allowances share" in rendered
    assert "$0.094000" in rendered and "$0.030000" in rendered


def test_shared_pool_rejects_wrong_timeout_extension_link(shared_pool_projection):
    fixture = shared_pool_projection
    fixture["shared_pool_validated"]["shared_pool_amendment"]["timeout_extension_sha256"] = "wrong-timeout"
    with pytest.raises(ValueError):
        report.build_report(**fixture["arguments"])


@pytest.fixture
def http_projection(shared_pool_projection, monkeypatch):
    fixture = shared_pool_projection
    fifth = {
        "shared_pool_sha256": hashlib.sha256(fixture["shared_pool_path"].read_bytes()).hexdigest(),
        "frozen_at": "2026-10-12T12:00:50+00:00",
        "review_reason": "Retain unresolved transient HTTP charges without retrying failed requests.",
        "added_http_statuses": [500, 502, 503, 504],
        "error_body_required": True,
        "response_contract": "Only an error object with matching integer code and nonempty message; no ID, usage or known cost",
        "allowance_usd": 0.05,
        "actual_cost_remains_unknown": True,
        "no_retry": True,
        "circuit_stop_unchanged": True,
    }
    path = fixture["amendment_path"].with_name("cost-recovery-http-errors.json")
    path.write_text(json.dumps(fifth) + "\n")
    validated = copy.deepcopy(fixture["shared_pool_validated"])
    validated["http_extension"] = fifth
    calls = []

    def http_summary(runs_root, http_extension_path):
        calls.append((Path(runs_root), Path(http_extension_path)))
        return copy.deepcopy(validated)

    monkeypatch.setattr(report, "http_summary", http_summary)
    return {**fixture, "http_extension_path": path, "http_validated": validated, "http_calls": calls}


@pytest.mark.parametrize("explicit", [False, True])
def test_http_extension_has_priority_without_changing_costs_or_limits(http_projection, explicit):
    fixture = http_projection
    extra = {"http_extension_path": fixture["http_extension_path"]} if explicit else {}
    result = report.build_report(**fixture["arguments"], **extra)
    expenses = result["expenses"]
    assert expenses["attempts"] == 15
    assert expenses["effective_cost_usd"] is None
    assert expenses["unknown_cost_count"] == 1
    assert expenses["known_effective_cost_usd"] == pytest.approx(0.014)
    assert expenses["reserved_unverified_usd"] == expenses["shared_unknown_allowance_pool_usd"] == 0.05
    assert expenses["budget_accounted_usd"] == pytest.approx(0.064)
    assert expenses["global_guard_accounted_usd"] == pytest.approx(0.094)
    assert expenses["global_budget_usd"] == 5.0
    assert expenses["known_spend_budget_by_arm"] == {arm: 0.65 for arm in ARMS}
    fifth = expenses["recovery"]["http_extension"]
    assert fifth["sha256"] == hashlib.sha256(fixture["http_extension_path"].read_bytes()).hexdigest()
    assert fifth["added_http_statuses"] == [500, 502, 503, 504]
    assert fifth["error_body_required"] is True
    assert (
        fifth["actual_cost_remains_unknown"] is fifth["no_retry"] is fifth["circuit_stop_unchanged"] is True
    )
    assert fifth["allowance_usd"] == 0.05
    assert fifth["response_contract"] == fixture["http_validated"]["http_extension"]["response_contract"]
    assert (
        expenses["recovery"]["shared_pool_amendment"]["sha256"]
        == fixture["http_validated"]["http_extension"]["shared_pool_sha256"]
    )
    assert (
        fixture["calls"]
        == fixture["reallocation_calls"]
        == fixture["timeout_calls"]
        == fixture["shared_pool_calls"]
        == []
    )
    assert fixture["http_calls"] == [(fixture["arguments"]["runs_root"], fixture["http_extension_path"])]
    assert all(path.read_bytes() == value for path, value in fixture["raw"].items())


def test_http_extension_rejects_wrong_shared_pool_link(http_projection):
    fixture = http_projection
    fixture["http_validated"]["http_extension"]["shared_pool_sha256"] = "wrong-shared-pool"
    with pytest.raises(ValueError):
        report.build_report(**fixture["arguments"])


def test_unknown_native_charge_keeps_failed_latency_and_intended_denominator(frozen_study):
    path = frozen_study["arguments"]["runs_root"] / "camera" / "final" / "baseline.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1].update(effective_cost_usd=None, cost_usd=None, usage={}, seconds=7.25)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    raw = path.read_bytes()
    result = report.build_report(**frozen_study["arguments"])
    for variant in ("baseline", "selected"):
        native = result["native"]["camera"][variant]
        assert native["expenses"]["effective_cost_usd"] is None
        assert native["expenses"]["known_effective_cost_usd"] == pytest.approx(0.001)
        assert native["expenses"]["unknown_cost_count"] == 1
        assert native["operations"]["latency_count"] == 2
        assert native["operations"]["latency_p50_seconds"] == pytest.approx(3.675)
        assert native["operations"]["latency_p95_seconds"] == 7.25
        assert native["metrics"]["n_expected"] == 2
        assert native["metrics"]["n_valid"] == 1
        assert native["metrics"]["accuracy_all_expected_invalid_as_wrong"] == 0.5
        assert native["metrics"]["brier_invalid_or_unattempted_penalty1"] == pytest.approx(0.505)
    assert result["expenses"]["attempts"] == 14
    assert result["expenses"]["effective_cost_usd"] is None
    assert result["expenses"]["unknown_cost_count"] == 1
    assert path.read_bytes() == raw
    rendered = report.markdown_report(result)
    section = rendered.split("## Common prompt: fixed feature effect", 1)[1].split("\n## ", 1)[0]
    row = next(line for line in section.splitlines() if line.startswith("| camera |"))
    assert "0.001000 known; 1 unknown" in row
    assert "3.675" in row and "7.250" in row


def test_markdown_separates_native_tables_paired_gains_and_cost_allowances(recovery_projection):
    result = report.build_report(**recovery_projection["arguments"])
    rendered = report.markdown_report(result)
    for heading in ("## Common prompt: fixed feature effect", "## Independently selected GEPA pipelines"):
        section = rendered.split(heading, 1)[1].split("\n## ", 1)[0]
        table_start = section.index("| Arm |")
        table = section[table_start:].split("\n\n", 1)[0].splitlines()
        assert all(len(line.strip().strip("|").split("|")) == 8 for line in table)
        assert "gain" not in table[0].lower()
    paired = rendered.split("## Paired reward changes", 1)[1].split("\n## ", 1)[0]
    paired_table = [line for line in paired.splitlines() if line.startswith("|")]
    assert len(paired_table) == 9
    assert all(len(line.strip().strip("|").split("|")) == 4 for line in paired_table)
    cost_text = rendered[rendered.lower().index("actual total cost") :].lower()
    assert "unknown" in cost_text
    assert "unverified" in cost_text and "allowance" in cost_text
    assert "0.050000" in cost_text and "0.064000" in cost_text
    assert "charged $0.050000" not in cost_text and "charge: $0.050000" not in cost_text


def test_markdown_uses_percentages_without_mutating_fraction_metrics(frozen_study):
    result = report.build_report(**frozen_study["arguments"])
    # This fixture has only test samples. Supply two diagnostic rates to check
    # presentation without adding development labels to its frozen dataset.
    result["coverage"]["availability_only_dev"]["accuracy"] = 0.5
    result["coverage"]["numeric_camera_only_dev"]["accuracy"] = 0.75
    before = copy.deepcopy(result)
    rendered = report.markdown_report(result)
    assert result == before
    for heading in ("## Common prompt: fixed feature effect", "## Independently selected GEPA pipelines"):
        section = rendered.split(heading, 1)[1].split("\n## ", 1)[0]
        header = next(line for line in section.splitlines() if line.startswith("| Arm |"))
        assert "Accuracy (%)" in header
        assert "AI recall (%)" in header
        assert "Real FPR (%)" in header
        assert "Brier (0–1)" in header
        row = next(line for line in section.splitlines() if line.startswith("| camera"))
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 8
        assert cells[1:5] == ["50.0%", "0.0%", "0.0%", "0.5050"]
    lead = rendered.split("\n## ", 1)[0]
    assert "50.0%" in lead and "75.0%" in lead
    raw = result["native"]["camera"]["baseline"]["metrics"]
    assert raw["accuracy_all_expected_invalid_as_wrong"] == 0.5
    assert raw["recall_all_expected"] == 0.0
    assert raw["valid_only"]["false_positive_rate"] == 0.0
    assert raw["brier_invalid_or_unattempted_penalty1"] == pytest.approx(0.505)
    for link in (
        "README.md#frozen-experiment",
        "../../research/metadata-2026-10-12/paper-and-design.md",
        "../../research/metadata-2026-10-12/real-sources.md",
        "../../research/metadata-2026-10-12/ai-sources.md",
        "../../research/metadata-2026-10-12/c2pa-sources.md",
    ):
        assert f"]({link})" in rendered


def test_selected_prompt_export_rechecks_frozen_hash(frozen_study):
    arguments = frozen_study["arguments"]
    result = report.build_report(**arguments)
    Path(result["selection"]["camera"]["selected_prompt_path"]).write_text('{"id":"changed"}')
    output_root = arguments["project_root"] / "tracked"
    with pytest.raises(ValueError, match="Selected prompt changed before export"):
        report.export_selected_prompts(result["selection"], output_root, arguments["project_root"])
    assert not (output_root / "prompts").exists()


def test_report_rejects_calls_before_all_arm_selections(frozen_study):
    path = frozen_study["arguments"]["runs_root"] / "texture" / "selection.json"
    selection = json.loads(path.read_text())
    selection["selected_at"] = "2026-10-12T12:02:00+00:00"
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="preceded the last arm selection"):
        report.build_report(**frozen_study["arguments"])


def test_c2pa_absence_or_a_valid_claim_never_becomes_guaranteed_gold(frozen_study):
    samples = json.loads(frozen_study["arguments"]["dataset_path"].read_text())["samples"]
    claimed = samples[0]
    claimed["features"]["provenance"].update(
        status="valid",
        manifest_present=True,
        signature_valid=True,
        hard_binding_valid=True,
        trust_status="trusted",
        ai_generation_claim=True,
    )
    summary = report.provenance_summary(samples)
    assert summary["explicit_ai_claims"] == 1
    assert summary["signature_and_binding_valid_ai_claims"] == 1
    assert summary["validated_ai_claims_agreeing_with_source"] == 0
    assert "does not prove truth" in summary["interpretation"]
    assert summary["trust_status"] == {"trusted": 1, "unknown": 1}
    claimed["features"]["provenance"]["trust_status"] = "untrusted"
    untrusted = report.provenance_summary(samples)
    assert untrusted["trust_status"].get("trusted", 0) == 0
    assert untrusted["trust_status"]["untrusted"] == 1
    assert untrusted["signature_and_binding_valid_ai_claims"] == 1
