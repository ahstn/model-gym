import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_analyze as analyze


def record(identifier="one", task="quality", normalized=None, **extra):
    if normalized is None:
        normalized = {"technical_quality": 3.0, "aesthetics": 3.0}
    return {
        "benchmark": "test",
        "task": task,
        "arm": "decider",
        "prompt_id": "r1",
        "requested_reasoning": None,
        "provider": "provider",
        "sample": identifier,
        "normalized": normalized,
        "cost_usd": 0.01,
        "seconds": 1.0,
        "ledger": "test.jsonl",
        **extra,
    }


def ai_record(identifier, probability, **extra):
    return record(
        identifier,
        task="ai",
        normalized={"ai_probability": probability, "medium": "photo", "uncertainty": 0.2},
        **extra,
    )


def test_rank_correlations_and_auc_handle_ties_and_missing_classes():
    assert analyze.ranks([30, 10, 10, 20]) == [4, 1.5, 1.5, 3]
    assert analyze.correlations([(1, 3), (2, 2), (3, 1)])["spearman"] == pytest.approx(-1)
    assert analyze.pearson([1, 1], [2, 3]) is None
    assert analyze.pearson([], []) is None
    assert analyze.auc([0.1, 0.9], [0, 1]) == 1
    assert analyze.auc([0.9, 0.1], [0, 1]) == 0
    assert analyze.auc([0.5, 0.5], [0, 1]) == 0.5
    assert analyze.auc([0.3, 0.8], [1, 1]) is None


def test_wilson_intervals_show_uncertainty_at_zero_and_perfect_rates():
    assert analyze.wilson_interval(0, 0) is None
    assert analyze.wilson_interval(0, 30) == pytest.approx([0, 0.11351339317396876])
    assert analyze.wilson_interval(30, 30) == pytest.approx([0.8864866068260312, 1])
    assert analyze.wilson_interval(30, 60) == pytest.approx([0.377350242415557, 0.622649757584443])
    with pytest.raises(ValueError, match="Wilson"):
        analyze.wilson_interval(31, 30)


def test_parent_groups_and_invalid_penalty_have_equal_group_weight():
    records = [record(name) for name in ("a", "b", "c", "d")]
    records[2]["normalized"] = None
    references = {name: {"technical_quality": 3.0, "aesthetics": 3.0} for name in ("a", "b", "c")}
    references["d"] = {"technical_quality": 5.0, "aesthetics": 5.0}
    samples = {
        name: {"id": name, "source": {"parent_id": "parent" if name != "d" else "other"}}
        for name in references
    }
    group = analyze.summarize(records, references, samples)["groups"][0]
    metrics = group["quality"]["assistant_reference"]
    assert group["valid"] == 3
    assert metrics["technical_quality"]["mae_valid"] == pytest.approx(2 / 3)
    assert metrics["technical_quality"]["mae_invalid_penalty_4"] == 1.5
    assert metrics["mean_axis_group_balanced_mae_invalid_penalty_4"] == pytest.approx(5 / 3)
    assert metrics["technical_quality"]["reference_groups"] == 2


def test_ai_metrics_keep_invalid_rows_and_unknown_labels_separate():
    records = [
        ai_record(str(index), probability) for index, probability in enumerate((0.9, 0.4, 0.5, 0.1, 0.7, 0.9))
    ]
    records[4]["normalized"] = None
    labels = [True, True, False, False, False, None]
    samples = {
        str(index): {"id": str(index), "labels": {"ai_generated": label}}
        for index, label in enumerate(labels)
    }
    group = analyze.summarize(records, samples=samples)["groups"][0]
    metrics = group["ai"]
    assert metrics["labeled"] == 5
    assert metrics["invalid_labeled"] == 1
    assert metrics["unlabeled"] == 1
    assert metrics["confusion_valid_only"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert metrics["balanced_accuracy_valid_only"] == 0.5
    assert metrics["accuracy_all_labeled_invalid_as_wrong"] == 0.4
    assert metrics["brier_valid_only"] == pytest.approx(0.1575)
    assert metrics["review_count_valid"] == 2
    assert metrics["review_fraction_valid"] == 0.4
    assert metrics["automatic_coverage_all_attempts"] == 0.5
    assert metrics["ai_recall_wilson95_valid_only"] == pytest.approx(analyze.wilson_interval(1, 2))
    assert metrics["false_positive_rate_wilson95_valid_only"] == pytest.approx(analyze.wilson_interval(1, 2))
    assert metrics["accuracy_wilson95_valid_only"] == pytest.approx(analyze.wilson_interval(2, 4))


def test_review_boundaries_are_not_in_the_open_review_interval():
    records = [ai_record("low", 0.2), ai_record("high", 0.8)]
    metrics = analyze.summarize(records)["groups"][0]["ai"]
    assert metrics["review_count_valid"] == 0
    assert metrics["automatic_coverage_all_attempts"] == 1
    assert metrics["balanced_accuracy_valid_only"] is None
    assert metrics["auroc_valid_only"] is None


def test_live_choice_uncertainty_and_no_valid_flag_are_valid():
    row = ai_record("choice", 0.27)
    row["normalized"]["uncertainty"] = "high"
    assert "valid" not in row["normalized"]
    assert analyze.valid_prediction(row)
    assert analyze.summarize([row])["total"]["valid"] == 1
    row["normalized"]["uncertainty"] = "unrecognized"
    assert not analyze.valid_prediction(row)


def test_byok_uses_upstream_not_zero_charge_or_an_assumed_additive_fee():
    row = record(
        cost_usd=0.002,
        effective_cost_usd=0.022,
        usage={"cost": 0.002, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.02}},
    )
    total = analyze.summarize([row])["total"]
    assert total["openrouter_charged_cost_usd"] == 0.002
    assert total["effective_provider_inference_cost_usd"] == 0.02
    assert total["cost_per_attempt_usd"] == 0.02
    assert total["cost_basis_counts"] == {"byok_upstream_inference": 1}


def test_missing_byok_upstream_cost_is_unknown_not_free():
    row = record(cost_usd=0, usage={"cost": 0, "is_byok": True})
    total = analyze.summarize([row])["total"]
    assert total["reported_cost_usd"] == 0
    assert total["cost_complete"] is False
    assert total["effective_provider_inference_cost_usd"] is None
    assert total["cost_per_attempt_usd"] is None
    assert total["unknown_cost_count"] == 1


def test_cost_per_valid_includes_failures_and_p95_uses_nearest_rank():
    records = [record(str(index), seconds=index + 1) for index in range(20)]
    records[-1]["normalized"] = None
    total = analyze.summarize(records)["total"]
    assert total["latency_p50_seconds"] == 10.5
    assert total["latency_p95_seconds"] == 19
    assert total["cost_per_valid_result_usd"] == pytest.approx(0.2 / 19)


def test_image_hash_mismatch_and_duplicate_attempts_are_errors():
    row = record(delivered_sha256="actual")
    with pytest.raises(ValueError, match="hash"):
        analyze.summarize([row], {"one": {"delivered_sha256": "other"}})
    with pytest.raises(ValueError, match="Duplicate"):
        analyze.summarize([row, copy.deepcopy(row)])
    with pytest.raises(ValueError, match="absent"):
        analyze.summarize([row], samples={"different": {"id": "different"}})


def test_effort_provider_and_partition_form_separate_groups():
    records = [record(str(index)) for index in range(4)]
    records[1]["requested_reasoning"] = {"effort": "low"}
    records[2]["provider"] = "other"
    references = {"3": {"development_partition": "check"}}
    assert len(analyze.summarize(records, references)["groups"]) == 4


def test_agiqa_shared_prompt_and_legacy_ai_truth_do_not_infer_real_labels():
    base = {
        "component": "agiqa",
        "source_labels": {"prompt": "statue   of a man"},
        "known_ai_generated": True,
    }
    one, two = {"id": "one", "source": base}, {"id": "two", "source": {**base, "generator": "other"}}
    assert analyze.sample_group(one) == analyze.sample_group(two)
    assert analyze.ai_label(one) is True
    assert analyze.ai_label({"source": {"content_kind": "photo"}}) is None
    assert analyze.ai_label({"labels": {"ai_generated": "false"}}) is None


def test_human_raw_targets_require_explicit_construct_and_scale_mapping():
    sample = {"labels": {"mos_quality": 90}}
    assert analyze.human_target(sample, "technical_quality") == {"raw": None, "normalized": None}
    sample["labels"]["technical_quality"] = {"value": 0.5, "scale": [0.01, 1], "normalized_1_to_5": 2.98}
    assert analyze.human_target(sample, "technical_quality") == {"raw": 0.5, "normalized": 2.98}
    del sample["labels"]["technical_quality"]["normalized_1_to_5"]
    assert analyze.human_target(sample, "technical_quality") == {"raw": 0.5, "normalized": None}
    assert analyze.human_target(sample, "aesthetics") == {"raw": None, "normalized": None}


def test_expected_manifest_counts_are_partitioned_and_unknown_cost_serializes():
    refs = {
        name: {"development_partition": partition}
        for name, partition in (("a", "tune"), ("b", "tune"), ("c", "check"))
    }
    samples = {name: {"id": name} for name in refs}
    report = analyze.summarize([record("a", cost_usd=None)], refs, samples)
    group = report["groups"][0]
    assert group["manifest_expected_count"] == 2
    assert group["manifest_unattempted_ids"] == ["b"]
    assert report["manifest_unattempted_ids"] == ["b", "c"]
    json.dumps(report, allow_nan=False)


def test_cli_writes_reproducible_input_hashes(tmp_path):
    ledger = tmp_path / "run.jsonl"
    manifest = tmp_path / "manifest.json"
    refs = tmp_path / "refs.json"
    output = tmp_path / "report.json"
    ledger.write_text(json.dumps(record()) + "\n")
    manifest.write_text(json.dumps({"benchmark": "test", "samples": [{"id": "one"}]}))
    refs.write_text(json.dumps({"samples": [{"id": "one", "technical_quality": 4, "aesthetics": 2}]}))
    subprocess.run(
        [
            sys.executable,
            str(Path(analyze.__file__)),
            "--ledgers",
            str(ledger),
            "--references",
            str(refs),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    result = json.loads(output.read_text())
    assert result["total"]["attempted"] == 1
    assert len(result["inputs"]) == 3
    assert all(len(item["sha256"]) == 64 for item in result["inputs"])
    assert result["groups"][0]["quality"]["assistant_reference"]["mean_axis_mae_valid"] == 1
