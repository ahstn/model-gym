"""Focused checks for fixed paired metrics, clustering, and missing observations."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "prompt_calibration_report", Path(__file__).with_name("report.py")
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def cases():
    result = []
    for family in range(6):
        for severity in range(4):
            result.append(
                {
                    "role": "source_parent" if severity == 0 else "variant",
                    "family_id": f"controlled-{family}",
                    "severity": severity or None,
                    "operator": ("gaussian_blur", "gaussian_noise", "jpeg_compression")[family // 2],
                }
            )
    for role, count in (("vizwiz", 14), ("standalone_megalith", 12)):
        result.extend(
            {"role": role, "family_id": f"{role}-{number}", "severity": None, "operator": None}
            for number in range(count)
        )
    for number, case in enumerate(result, 1):
        case.update(case_id=f"B{number:02d}", image_id=f"image-{number}", view_id=f"view-{number}")
    return result


def compared(old=3.5, new=3.0):
    selected = cases()
    references = {
        case["case_id"]: {"case_id": case["case_id"], "score": 3, "interval": [3, 3]} for case in selected
    }
    runs = {
        arm: {case["view_id"]: {"technical_quality": {"status": "ok", "score": value}} for case in selected}
        for arm, value in (("v1", old), ("v2", new))
    }
    return selected, references, runs


def test_interval_math_and_secondary_metrics_keep_nulls_and_boundaries():
    assert report.distance(None, [1, 2]) is None
    assert report.distance(2, None) is None
    assert report.distance(2, [2, 4]) == 0
    assert report.distance(1, [2, 4]) == 1
    assert report.distance(5, [2, 4]) == 1
    assert report.critical(4, [1, 2]) is True
    assert report.critical(2, [4, 5]) is True
    assert report.critical(3.9, [1, 2]) is False
    selected, references, runs = compared(old=4, new=3)
    runs["v1"].pop("view-1")
    rows = report.case_rows("b", selected, references, {}, runs)
    metrics = report.arm_metrics(rows, "v1")
    assert metrics["valid_reference_comparisons"] == 49
    assert metrics["mae_from_assistant_point"] == 1
    assert metrics["large_interval_error_count"] == 49
    assert metrics["interval_hit_rate"] == 0
    assert rows[0]["v1_score"] is None
    assert ",," in report.csv_text(rows)
    with pytest.raises(ValueError, match="Invalid assistant reference"):
        report.validate_reference({"score": 3, "interval": [1, 2]})
    with pytest.raises(ValueError, match="Invalid assistant reference"):
        report.validate_reference({"score": float("nan"), "interval": [1, 5]})
    with pytest.raises(ValueError, match="Duplicate or unexpected"):
        report.indexed([{"case_id": "B01"}, {"case_id": "B01"}], "case_id", {"B01"}, "test")


def test_bootstrap_samples_whole_families_and_keeps_image_weights():
    selected = cases()
    report.validate_cases(selected, "b")
    deltas = {
        case["case_id"]: (-1 if int(case["family_id"].rsplit("-", 1)[1]) < 3 else 1)
        if case["role"] in {"variant", "source_parent"}
        else 0
        for case in selected
    }
    interval = report.bootstrap_interval(selected, deltas)
    # Six equal binary families produce a binomial 95% range of 1..5 positive draws.
    assert interval == pytest.approx([-0.32, 0.32])
    assert report.bootstrap_interval(list(reversed(selected)), deltas) == interval
    constant = {
        case["case_id"]: 1 if case["role"] in {"variant", "source_parent"} else 0 for case in selected
    }
    assert report.bootstrap_interval(selected, constant) == pytest.approx([0.48, 0.48])
    naive = (
        np.random.default_rng(report.BOOTSTRAP_SEED)
        .choice(list(deltas.values()), size=(10_000, 50))
        .mean(axis=1)
    )
    assert np.quantile(naive, 0.975) < interval[1]
    with pytest.raises(ValueError, match="50 paired"):
        report.bootstrap_interval(selected[:-1], deltas)
    selected[24]["family_id"] = selected[0]["family_id"]
    with pytest.raises(ValueError, match="different strata"):
        report.validate_cases(selected, "b")


def test_fixed_gates_reject_regression_and_missing_pairs_without_zero_fill():
    selected, references, runs = compared(old=3.15, new=3)
    rows = report.case_rows("b", selected, references, {}, runs)
    assert report.paired_result(selected, rows)["decision"] == "improved_against_assistant_references"
    for case in selected:
        runs["v1"][case["view_id"]]["technical_quality"]["score"] = 3.149999
    boundary = report.paired_result(selected, report.case_rows("b", selected, references, {}, runs))
    assert boundary["gates"]["mean_improvement_at_least_0_15"] is False
    selected, references, runs = compared(old=3.15, new=3)
    # Overall improvement stays positive, but one source stratum worsens beyond its bound.
    for case in selected:
        if case["role"] == "standalone_megalith":
            runs["v2"][case["view_id"]]["technical_quality"]["score"] = 3.3
    result = report.paired_result(selected, report.case_rows("b", selected, references, {}, runs))
    assert result["gates"]["no_stratum_worsens_more_than_0_10"] is False
    selected, references, runs = compared(old=3.5, new=3)
    for case in selected:
        if case["role"] == "standalone_megalith":
            runs["v2"][case["view_id"]]["technical_quality"]["score"] = 3.6
    result = report.paired_result(selected, report.case_rows("b", selected, references, {}, runs))
    assert result["gates"]["no_stratum_worsens_more_than_0_10"] is True
    for case in selected:
        if case["role"] == "standalone_megalith":
            runs["v2"][case["view_id"]]["technical_quality"]["score"] = 3.600001
    result = report.paired_result(selected, report.case_rows("b", selected, references, {}, runs))
    assert result["gates"]["no_stratum_worsens_more_than_0_10"] is False
    selected, references, runs = compared()
    references["B01"].update(score=1.5, interval=[1, 2])
    runs["v2"]["view-1"]["technical_quality"]["score"] = 4
    result = report.paired_result(selected, report.case_rows("b", selected, references, {}, runs))
    assert result["new_critical_case_ids"] == ["B01"]
    assert result["gates"]["no_new_critical_contradictions"] is False
    runs["v2"]["view-1"]["technical_quality"] = {"status": "unscorable", "score": None}
    result = report.paired_result(selected, report.case_rows("b", selected, references, {}, runs))
    assert result["decision"] == "inconclusive"
    assert result["valid_pairs"] == 49
    assert result["paired_95_percent_bootstrap_interval"] is None
    assert result["mean_improvement"] == 0.5
    assert result["missing_pair_case_ids"] == ["B01"]
    selected, references, runs = compared(old=3, new=3)
    result = report.paired_result(selected, report.case_rows("b", selected, references, {}, runs))
    assert result["gates"]["lower_95_percent_bound_above_zero"] is False
    assert result["decision"] == "inconclusive"
    rows = report.case_rows("b", selected, references, {}, runs)
    with pytest.raises(ValueError, match="Duplicate or unexpected"):
        report.paired_result(selected, rows[:-1] + [rows[0]])


def test_bootstrap_zero_bound_cannot_pass_from_float_residue():
    selected, references, runs = compared(old=1.8, new=1)
    for case in selected:
        references[case["case_id"]].update(score=1, interval=[1, 1])
        if case["role"] in {"source_parent", "variant"}:
            positive = int(case["family_id"].rsplit("-", 1)[1]) < 3
            runs["v1"][case["view_id"]]["technical_quality"]["score"] = 2.3 if positive else 1
            runs["v2"][case["view_id"]]["technical_quality"]["score"] = 1 if positive else 2.3
    result = report.paired_result(selected, report.case_rows("b", selected, references, {}, runs))
    assert result["paired_95_percent_bootstrap_interval"][0] == 0
    assert result["gates"]["lower_95_percent_bound_above_zero"] is False
    assert result["decision"] == "inconclusive"


@pytest.fixture
def isolated_study(tmp_path, monkeypatch):
    project = tmp_path / "visual-quality"
    here = project / "experiments/prompt-calibration-2026-10-18"
    data = project / "data/prompt-calibration-2026-10-18"
    for key, value in (("REPO", tmp_path), ("PROJECT", project), ("HERE", here), ("DATA", data)):
        monkeypatch.setattr(report, key, value)
    for directory in (
        here / "references",
        here / "audits",
        data / "a/v1",
        project / "prompts/visual-rating-v2",
    ):
        directory.mkdir(parents=True)


def test_reference_freeze_rejects_changes_and_pins_source_files(isolated_study):
    merged = report.HERE / "references/a.jsonl"
    source = report.HERE / "references/a-review-1.json"
    freeze_path = report.HERE / "references/a-freeze.json"
    content = json.dumps({"case_id": "A01", "score": 3, "interval": [2, 4]}) + "\n"
    merged.write_text(content)
    source.write_text(content)
    with pytest.raises(ValueError, match="require their frozen hash record"):
        report.load_references("a", {"A01"})
    freeze = {
        "stage": "a",
        "human_gold": False,
        "merged_sha256": report.sha(merged),
        "source_files_sha256": {str(source.relative_to(report.PROJECT)): report.sha(source)},
    }
    freeze_path.write_text(json.dumps(freeze))
    references, verification = report.load_references("a", {"A01"})
    assert set(references) == {"A01"}
    assert verification["status"] == "verified"
    manifest = {
        "stages": {
            stage: {"views_dir": f"views/{stage}", "dataset_dir": f"datasets/{stage}"} for stage in ("a", "b")
        }
    }
    assert {merged, freeze_path, source} <= report.input_paths(manifest)
    merged.write_text(content.replace('"score": 3', '"score": 4'))
    with pytest.raises(ValueError, match="merged references differ"):
        report.load_references("a", {"A01"})
    merged.write_text(content)
    source.write_text(content + " ")
    with pytest.raises(ValueError, match="source files differ"):
        report.load_references("a", {"A01"})
    source.unlink()
    freeze["source_files_sha256"][str(source.relative_to(report.PROJECT))] = None
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="source files differ"):
        report.load_references("a", {"A01"})


def test_prompt_revision_binds_stage_a_audit_bytes(isolated_study, monkeypatch):
    from visual_quality import labelling

    monkeypatch.setattr(
        labelling,
        "_prompt_bundle",
        lambda path: {"version": path.name, "sha256": "1" * 64 if path.name.endswith("v1") else "2" * 64},
    )
    inputs = {
        "source_references_sha256": report.HERE / "references/a.jsonl",
        "source_observations_sha256": report.DATA / "a/v1/observations.jsonl",
        "source_audits_sha256": report.HERE / "audits/a.jsonl",
    }
    for path in inputs.values():
        path.write_text("{}\n")
    directory = report.PROJECT / "prompts/visual-rating-v2"
    files = {}
    for name in ("system.txt", "technical-quality.txt", "response.schema.json", "prompt-version.json"):
        path = directory / name
        path.write_text("synthetic prompt file")
        files[str(path.relative_to(report.REPO))] = report.sha(path)
    lock = {
        "schema_version": "prompt-revision-v1",
        "source_stage": "a",
        "stage_b_not_inspected": True,
        "prompt_version": "visual-rating-v2",
        "prompt_sha256": "2" * 64,
        "files": files,
        **{key: report.sha(path) for key, path in inputs.items()},
    }
    (report.HERE / "prompt-revision.json").write_text(json.dumps(lock))
    runs = {"a/v1": {}, "b/v1": {}, "b/v2": {}}
    assert report.validate_prompts(runs)["status"] == "verified"
    inputs["source_audits_sha256"].write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="stage-A inputs"):
        report.validate_prompts(runs)
    inputs["source_audits_sha256"].unlink()
    lock.pop("source_audits_sha256")
    (report.HERE / "prompt-revision.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="stage-A inputs"):
        report.validate_prompts(runs)
