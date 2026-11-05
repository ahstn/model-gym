import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import digest
from tracks import metrics
from tracks import report as reporting
from tracks import run as runtime


@pytest.fixture
def frozen_track(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    directory = tmp_path / "tracks/quality"
    directory.mkdir(parents=True)
    samples = []
    for split in ("train", "dev", "test"):
        for index, gold in enumerate((1.5, 4.5)):
            identifier = f"{split}-{index}"
            samples.append(
                {
                    "id": identifier,
                    "image": f"data/{identifier}.png",
                    "delivered_sha256": hashlib.sha256(identifier.encode()).hexdigest(),
                    "group_id": split,
                    "split": split,
                    "source": {"dataset": "test"},
                    "labels": {"quality_mos_1to5": gold},
                }
            )
    dataset = {"track": "quality", "samples": samples}
    (directory / "dataset.json").write_text(json.dumps(dataset))
    prompt = {
        "id": "baseline",
        "task": "quality",
        "context": "Rate technical defects independently of aesthetics.",
        "questions": {
            key: {
                "type": "score",
                "instructions": "Rate visible image.",
                "criteria": ["bad", "poor", "fair", "good", "excellent"],
            }
            for key in ("technical_quality", "aesthetics")
        },
    }
    baseline = directory / "baseline.json"
    baseline.write_text(json.dumps(prompt))
    runtime.create_protocol("quality", baseline)
    return directory, dataset, prompt


def fake_records(dataset, prompt, split, offset=0):
    return [
        {
            "benchmark": "quality",
            "sample": sample["id"],
            "task": "quality",
            "arm": "decider",
            "status": 200,
            "normalized": {
                "technical_quality": sample["labels"]["quality_mos_1to5"] + offset,
                "aesthetics": 3,
            },
            "manifest_sha256": digest(dataset),
            "prompt_sha256": digest(prompt),
            "delivered_sha256": sample["delivered_sha256"],
            "cost_usd": 0.01,
            "seconds": 0.2,
        }
        for sample in dataset["samples"]
        if sample["split"] == split
    ]


def test_group_and_exact_view_leakage_rejected(frozen_track):
    _, dataset, _ = frozen_track
    bad = copy.deepcopy(dataset)
    bad["samples"][2]["group_id"] = bad["samples"][0]["group_id"]
    with pytest.raises(ValueError, match="crosses"):
        runtime.validate_dataset("quality", bad)
    bad = copy.deepcopy(dataset)
    bad["samples"][2]["delivered_sha256"] = bad["samples"][0]["delivered_sha256"]
    with pytest.raises(ValueError, match="crosses"):
        runtime.validate_dataset("quality", bad)


def test_minibatch_requires_whole_groups(frozen_track):
    _, dataset, _ = frozen_track
    with pytest.raises(ValueError, match="whole"):
        runtime.select_split(dataset, "train", ["train-0"])
    with pytest.raises(ValueError, match="requested split"):
        runtime.select_split(dataset, "train", ["test-0", "test-1"])


def test_dataset_hash_and_context_only_mutation(frozen_track):
    directory, dataset, prompt = frozen_track
    _, protocol, _ = runtime.read_protocol("quality")
    candidate = directory / "candidate.json"
    prompt["context"] = "A changed context is allowed."
    candidate.write_text(json.dumps(prompt))
    runtime.validate_candidate(candidate, protocol)
    prompt["questions"]["aesthetics"]["instructions"] = "Changed question"
    candidate.write_text(json.dumps(prompt))
    with pytest.raises(ValueError, match="Only prompt context"):
        runtime.validate_candidate(candidate, protocol)
    (directory / "dataset.json").write_text(json.dumps(dataset) + "\n")
    with pytest.raises(ValueError, match="Frozen dataset"):
        runtime.read_protocol("quality")


def test_missing_selection_and_excess_budget_prevent_calls(frozen_track, monkeypatch):
    directory, _, _ = frozen_track
    monkeypatch.setattr(runtime, "run_benchmark", lambda args: pytest.fail("Must not call API"))
    with pytest.raises(FileNotFoundError):
        runtime.eval_prompt("quality", "test", directory / "baseline.json", directory / "test.jsonl")
    with pytest.raises(ValueError):
        runtime.eval_prompt(
            "quality", "train", directory / "baseline.json", directory / "train.jsonl", budget_usd=2.01
        )


def test_eval_uses_existing_runner_and_returns_trace_feedback(frozen_track, monkeypatch):
    directory, dataset, prompt = frozen_track
    calls = []

    def fake_run(args):
        calls.append(args)
        args.output.write_text(
            "".join(json.dumps(record) + "\n" for record in fake_records(dataset, prompt, "train"))
        )

    monkeypatch.setattr(runtime, "run_benchmark", fake_run)
    result = runtime.eval_prompt("quality", "train", directory / "baseline.json", directory / "train.jsonl")
    assert result["summary"]["objective"]["score"] == 1
    assert len(result["ledger_records"]) == 2
    assert result["per_image"][0]["feedback"]["target"] == {"quality_mos_1to5": 1.5}
    assert calls[0].models == ["decider"] and calls[0].max_tokens == 4096
    assert calls[0].budget_usd == 2 and calls[0].max_calls == 360


def test_selection_requires_dev_gain_and_pins_evidence(frozen_track):
    directory, dataset, prompt = frozen_track
    baseline_ledger = directory / "baseline-dev.jsonl"
    candidate_ledger = directory / "candidate-dev.jsonl"
    baseline_ledger.write_text(
        "".join(json.dumps(record) + "\n" for record in fake_records(dataset, prompt, "dev", 0.1))
    )
    candidate = copy.deepcopy(prompt)
    candidate["id"], candidate["context"] = "candidate", "A more precise context."
    candidate_path = directory / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    candidate_ledger.write_text(
        "".join(json.dumps(record) + "\n" for record in fake_records(dataset, candidate, "dev", 0.09))
    )
    selected = runtime.freeze_selection(
        directory / "protocol.json", candidate_path, baseline_ledger, candidate_ledger
    )
    assert selected["selected"] == "baseline"
    assert selected["candidate_gain"] < 0.005
    protocol_path, protocol, _ = runtime.read_protocol("quality")
    with pytest.raises(ValueError, match="permits only"):
        runtime.validate_test_selection(
            directory / "prompt-selection.json", protocol_path, protocol, candidate_path
        )
    baseline_ledger.write_text(baseline_ledger.read_text() + "\n")
    with pytest.raises(ValueError, match="evidence changed"):
        runtime.validate_test_selection(
            directory / "prompt-selection.json", protocol_path, protocol, directory / "baseline.json"
        )


def test_metrics_nonfinite_invalids_do_not_disappear(frozen_track):
    _, dataset, prompt = frozen_track
    samples = runtime.select_split(dataset, "dev")
    records = fake_records(dataset, prompt, "dev")
    records[0]["normalized"]["technical_quality"] = float("nan")
    result = metrics.summarize("quality", samples, records)
    assert result["summary"]["metrics"]["mae_valid"] == 0
    assert result["summary"]["metrics"]["mae_invalid_or_unattempted_penalty4"] == 2
    assert result["summary"]["objective"]["score"] == 0.5
    assert result["summary"]["operations"]["reported_cost_usd"] == 0.02
    assert result["summary"]["operations"]["latency_count"] == 2


def test_quality_threshold_is_strict_and_no_human_aesthetics(frozen_track):
    _, dataset, prompt = frozen_track
    samples = runtime.select_split(dataset, "dev")
    samples[0]["labels"]["quality_mos_1to5"] = 2.6
    records = fake_records(dataset, prompt, "dev")
    result = metrics.summarize("quality", samples, records)
    binary = result["summary"]["metrics"]["bad_detection"]["valid_only"]
    assert binary["confusion"] == {"tp": 0, "tn": 2, "fp": 0, "fn": 0}
    assert result["summary"]["metrics"]["aesthetics"]["scored_against_gold"] is False


def test_ai_tied_auc_review_and_pair_metrics():
    samples, records = [], []
    for i, label in enumerate((False, True)):
        samples.append(
            {
                "id": str(i),
                "group_id": "pair",
                "split": "test",
                "image": "data/image.jpg",
                "delivered_sha256": str(i),
                "labels": {"ai_generated": label},
            }
        )
        records.append(
            {
                "sample": str(i),
                "task": "ai",
                "arm": "luna",
                "prompt_sha256": "same",
                "delivered_sha256": str(i),
                "status": 200,
                "normalized": {"ai_probability": 0.5, "medium": "photo", "uncertainty": "high"},
                "seconds": i + 1,
                "cost_usd": 0,
                "usage": {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.001}},
            }
        )
    result = metrics.summarize("ai_detection", samples, records)
    scores = result["summary"]["metrics"]
    assert scores["auroc_valid"] == 0.5 and scores["brier_valid"] == 0.25
    assert scores["valid_only"]["confusion"] == {"tp": 1, "tn": 0, "fp": 1, "fn": 0}
    assert scores["automatic_coverage_all_expected"] == 0
    assert scores["pairs"]["expected"] == 1 and scores["pairs"]["both_correct"] == 0
    assert result["summary"]["operations"]["effective_provider_inference_cost_usd"] == 0.002


def test_metrics_reject_hash_mismatch_and_mixed_prompts(frozen_track):
    _, dataset, prompt = frozen_track
    samples = runtime.select_split(dataset, "dev")
    records = fake_records(dataset, prompt, "dev")
    records[0]["delivered_sha256"] = "changed"
    with pytest.raises(ValueError, match="delivered hash"):
        metrics.summarize("quality", samples, records)
    records = fake_records(dataset, prompt, "dev")
    records[1]["prompt_sha256"] = "other"
    with pytest.raises(ValueError, match="one model"):
        metrics.summarize("quality", samples, records)


def setup_final_artifacts(directory, dataset, prompt, task_cost=0.01, reflection_cost=0.005):
    optimisation = directory / "optimisation"
    candidate = optimisation / "candidates/base"
    candidate.mkdir(parents=True)
    baseline_dev = candidate / "dev.jsonl"
    records = fake_records(dataset, prompt, "dev")
    for record in records:
        record["effective_cost_usd"] = record["cost_usd"] = task_cost
    baseline_dev.write_text("".join(json.dumps(record) + "\n" for record in records))
    (optimisation / "reflection.jsonl").write_text(json.dumps({"effective_cost_usd": reflection_cost}) + "\n")
    (optimisation / "summary.json").write_text(
        json.dumps(
            {
                "costs": {
                    "task_calls": 2,
                    "reflection_calls": 1,
                    "effective_cost_usd": 2 * task_cost + reflection_cost,
                }
            }
        )
    )
    runtime.freeze_selection(
        directory / "protocol.json", directory / "baseline.json", baseline_dev, baseline_dev
    )


def test_final_driver_deduplicates_baseline_and_subtracts_optimisation(frozen_track, monkeypatch):
    directory, dataset, prompt = frozen_track
    setup_final_artifacts(directory, dataset, prompt, task_cost=0.2, reflection_cost=0.1)
    calls = []

    def fake_eval(track, split, prompt_path, output, **kwargs):
        calls.append(kwargs)
        output.parent.mkdir(parents=True)
        records = fake_records(dataset, prompt, "test")
        for record in records:
            record["effective_cost_usd"] = 0.01
        output.write_text("".join(json.dumps(record) + "\n" for record in records))

    monkeypatch.setattr(runtime, "eval_prompt", fake_eval)
    result = runtime.run_final("quality")
    assert result["configurations"] == ["baseline"]
    assert len(calls) == 1 and calls[0]["budget_usd"] == 1.5
    assert result["spending"]["total_usd"] == 0.52
    runtime.run_final("quality")
    assert len(calls) == 1


def test_track_total_guard_prevents_final_calls(frozen_track, monkeypatch):
    directory, dataset, prompt = frozen_track
    setup_final_artifacts(directory, dataset, prompt, task_cost=0.9, reflection_cost=0.15)
    monkeypatch.setattr(runtime, "eval_prompt", lambda *args, **kwargs: pytest.fail("Must not call API"))
    with pytest.raises(ValueError, match="Combined optimisation"):
        runtime.run_final("quality")


def test_track_spending_stops_on_unknown_or_changed_cost(frozen_track):
    directory, dataset, prompt = frozen_track
    setup_final_artifacts(directory, dataset, prompt)
    reflection = directory / "optimisation/reflection.jsonl"
    reflection.write_text(json.dumps({"effective_cost_usd": None}) + "\n")
    with pytest.raises(ValueError, match="spending is unknown"):
        runtime.track_spending("quality")
    reflection.write_text(json.dumps({"effective_cost_usd": 0.1}) + "\n")
    with pytest.raises(ValueError, match="summary and recorded"):
        runtime.track_spending("quality")


def test_zero_errors_have_nonzero_wilson_upper_bound():
    scores = metrics.confusion([False, False, True, True], [False, False, True, True])
    assert scores["false_positive_rate"] == 0
    assert scores["false_positive_rate_wilson95"][1] > 0
    assert scores["recall_wilson95"][0] < 1
    assert "accuracy_wilson95" not in scores
    assert "accuracy_wilson95_naive_independent_images" in scores


def test_paired_group_delta_uses_same_group_samples():
    baseline = [
        {
            "id": str(i),
            "group_id": str(i // 2),
            "delivered_sha256": str(i),
            "gold": bool(i % 2),
            "split": "test",
            "reward": 0.5,
        }
        for i in range(4)
    ]
    selected = [{**row, "reward": 0.7} for row in baseline]
    result = metrics.paired_reward_delta(baseline, selected, 100)
    assert result["selected_minus_baseline"] == pytest.approx(0.2)
    assert result["paired_group_bootstrap95"] == pytest.approx([0.2, 0.2])
    unchanged = metrics.paired_reward_delta(baseline, baseline, 100)
    assert unchanged["paired_group_bootstrap95"] == [0, 0]
    selected[0]["delivered_sha256"] = "changed"
    with pytest.raises(ValueError, match="inputs or groups"):
        metrics.paired_reward_delta(baseline, selected, 100)


def test_report_keeps_small_costs_visible_and_avoids_fake_gain_interval(frozen_track, monkeypatch):
    directory, dataset, prompt = frozen_track
    monkeypatch.setattr(reporting, "ROOT", directory.parents[1])
    setup_final_artifacts(directory, dataset, prompt)
    final = directory / "final"
    final.mkdir()
    records = fake_records(dataset, prompt, "test")
    for record in records:
        record["effective_cost_usd"] = record["cost_usd"] = 0.00002078
    (final / "baseline.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    result = reporting.build_report("quality", 20)
    markdown = (directory / "report/README.md").read_text()
    assert "Spearman" in markdown
    assert "no independent prompt gain was measured" in markdown
    assert "Selected minus baseline reward" not in markdown
    assert "0.00002078" in (directory / "report/per-image.md").read_text()
    assert result["paired_comparison"]["paired_group_bootstrap95"] is None
    assert result["paired_comparison"]["independent_candidate_evaluated"] is False


def test_accuracy_intervals_resample_pairs_without_splitting_members():
    rows = []
    for group in (0, 1):
        for label in (False, True):
            probability = float(label if group == 0 else not label)
            rows.append(
                {
                    "id": f"{group}-{label}",
                    "group_id": str(group),
                    "gold": label,
                    "valid": True,
                    "prediction": {"ai_probability": probability},
                    "reward": float(group == 0),
                    "error": float(group != 0),
                    "correct": group == 0,
                }
            )
    result = metrics.bootstrap_intervals("ai_detection", rows, 100)
    assert result["intervals"]["accuracy_valid"]["percentile95"] == [0, 1]
    assert result["intervals"]["balanced_accuracy_valid"]["percentile95"] == [0, 1]
    assert "false_positive_rate" not in result["intervals"]


def test_versioned_quality_changes_descriptions_but_preserves_scale(frozen_track):
    directory, dataset, _ = frozen_track
    dataset["track"] = "quality_v2"
    path = directory / "v2-dataset.json"
    path.write_text(json.dumps(dataset))
    prompt = json.loads((Path(__file__).resolve().parents[1] / "prompts/final/quality.json").read_text())
    baseline = directory / "v2-baseline.json"
    baseline.write_text(json.dumps(prompt))
    protocol_path = directory / "v2-protocol.json"
    protocol = runtime.create_protocol("quality_v2", baseline, dataset_path=path, output_path=protocol_path)
    assert protocol["model_alias"] == "decider" and protocol["track_budget_usd"] == 5
    assert protocol["max_proposals"] == 20
    runtime.read_protocol("quality_v2", protocol_path)
    prompt["questions"]["technical_quality"]["instructions"] = "Inspect visible technical damage."
    prompt["questions"]["technical_quality"]["criteria"][0] = "1 bad: major visible damage"
    candidate = directory / "v2-candidate.json"
    candidate.write_text(json.dumps(prompt))
    runtime.validate_candidate(candidate, protocol)
    original_aesthetics = prompt["questions"]["aesthetics"]["instructions"]
    prompt["questions"]["aesthetics"]["instructions"] = "An unscored aesthetic change"
    candidate.write_text(json.dumps(prompt))
    with pytest.raises(ValueError, match="aesthetics"):
        runtime.validate_candidate(candidate, protocol)
    prompt["questions"]["aesthetics"]["instructions"] = original_aesthetics
    prompt["questions"]["technical_quality"]["criteria"][0] = "5 excellent: inverted scale"
    candidate.write_text(json.dumps(prompt))
    with pytest.raises(ValueError, match="ascending"):
        runtime.validate_candidate(candidate, protocol)


def test_versioned_metrics_use_task_semantics(frozen_track):
    _, dataset, prompt = frozen_track
    samples = runtime.select_split(dataset, "dev")
    records = fake_records(dataset, prompt, "dev")
    v2 = metrics.summarize("quality_v2", samples, records)
    v1 = metrics.summarize("quality", samples, records)
    assert v2["summary"]["metrics"] == v1["summary"]["metrics"]
    assert v2["per_image"] == v1["per_image"]


def test_versioned_ai_uses_native_decider_probability_request(tmp_path, monkeypatch):
    import bench_runner
    from bench_protocol import DECISIONS, MODELS

    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(
        bench_runner, "image_part", lambda sample: {"type": "image_url", "image_url": {"url": "fake"}}
    )
    prompt = json.loads((Path(__file__).resolve().parents[1] / "prompts/final/ai.json").read_text())
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(prompt))
    samples = [
        {
            "id": split,
            "group_id": split,
            "split": split,
            "image": f"{split}.png",
            "delivered_sha256": hashlib.sha256(split.encode()).hexdigest(),
            "labels": {"ai_generated": True},
        }
        for split in ("train", "dev", "test", "external")
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"track": "ai_detection_v2", "samples": samples}))
    protocol_path = tmp_path / "protocol.json"
    runtime.create_protocol("ai_detection_v2", baseline, dataset_path=dataset_path, output_path=protocol_path)
    called = []

    def response(endpoint, payload):
        called.append((endpoint, payload))
        answers = {"ai_probability": {"type": "noul", "noul": 0.8}}
        for key in ("medium", "uncertainty"):
            labels = prompt["questions"][key]["criteria"]
            first = next(iter(labels))
            answers[key] = {
                "type": "choice",
                "choice": first,
                "probabilities": {label: float(label == first) for label in labels},
            }
        return {"status": 200, "seconds": 0.1, "response": {"usage": {"cost": 0.001}, "answers": answers}}

    monkeypatch.setattr(bench_runner, "post", response)
    result = runtime.eval_prompt(
        "ai_detection_v2", "train", baseline, tmp_path / "train.jsonl", protocol_path=protocol_path
    )
    assert called[0][0] == DECISIONS
    assert called[0][1]["model"] == MODELS["decider"]
    assert called[0][1]["questions"]["ai_probability"]["type"] == "noul"
    assert result["per_image"][0]["prediction"]["ai_probability"] == 0.8
    with pytest.raises(ValueError, match="train/dev"):
        from tracks.optimise import split_groups

        split_groups(json.loads(dataset_path.read_text()), "external")


def test_versioned_external_final_is_scored_without_selection_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    directory = tmp_path / "tracks/quality_v2"
    directory.mkdir(parents=True)
    prompt = json.loads((Path(__file__).resolve().parents[1] / "prompts/final/quality.json").read_text())
    (directory / "baseline.json").write_text(json.dumps(prompt))
    samples = [
        {
            "id": f"{split}-{index}",
            "group_id": split,
            "split": split,
            "image": f"data/{split}-{index}.png",
            "delivered_sha256": hashlib.sha256(f"{split}-{index}".encode()).hexdigest(),
            "labels": {"quality_mos_1to5": gold},
        }
        for split in ("train", "dev", "test", "external")
        for index, gold in enumerate((1.5, 4.5))
    ]
    dataset = {"track": "quality_v2", "samples": samples}
    (directory / "dataset.json").write_text(json.dumps(dataset))
    runtime.create_protocol("quality_v2", directory / "baseline.json")
    setup_final_artifacts(directory, dataset, prompt)
    calls = []

    def evaluate(track, split, prompt_path, output, **kwargs):
        calls.append(split)
        output.parent.mkdir(parents=True, exist_ok=True)
        records = fake_records(dataset, prompt, split)
        for record in records:
            record["effective_cost_usd"] = record["cost_usd"]
        output.write_text("".join(json.dumps(record) + "\n" for record in records))

    monkeypatch.setattr(runtime, "eval_prompt", evaluate)
    runtime.run_final("quality_v2")
    assert calls == ["test", "external"]
    runtime.run_final("quality_v2")
    assert calls == ["test", "external"]
    result = reporting.build_report("quality_v2", 0)
    assert result["external_validation"]["selection_used"] is False
    assert result["external_validation"]["evaluations"]["baseline"]["summary"]["expected"] == 2
    metadata = directory / "final-metadata"
    metadata.mkdir()
    metadata_ledger = metadata / "test.jsonl"
    metadata_ledger.write_text(json.dumps({"effective_cost_usd": 0.3}) + "\n")
    spending = runtime.track_spending("quality_v2")
    assert spending["metadata_calls"] == 1
    assert spending["metadata_usd"] == 0.3
    assert spending["final_usd"] == 0.04
    assert spending["total_usd"] == pytest.approx(0.365)
    metadata_ledger.write_text(json.dumps({"effective_cost_usd": None}) + "\n")
    with pytest.raises(ValueError, match="spending is unknown"):
        runtime.track_spending("quality_v2")
    metadata_ledger.write_text(json.dumps({"effective_cost_usd": 4.96}) + "\n")
    (directory / "final/baseline.jsonl").unlink()
    monkeypatch.setattr(
        runtime, "eval_prompt", lambda *args, **kwargs: pytest.fail("Metadata spend must stop primary calls")
    )
    with pytest.raises(ValueError, match="Combined optimisation"):
        runtime.run_final("quality_v2")
    external_dev = directory / "bad-dev.jsonl"
    external_dev.write_text(
        "".join(json.dumps(record) + "\n" for record in fake_records(dataset, prompt, "external"))
    )
    with pytest.raises(ValueError, match="cannot be used for prompt selection"):
        runtime.freeze_selection(
            directory / "protocol.json",
            directory / "baseline.json",
            external_dev,
            external_dev,
            output_path=directory / "bad-selection.json",
        )


def test_v1_spending_shape_ignores_new_metadata_ledgers(frozen_track):
    directory, dataset, prompt = frozen_track
    setup_final_artifacts(directory, dataset, prompt)
    metadata = directory / "final-metadata"
    metadata.mkdir()
    (metadata / "test.jsonl").write_text("Unknown future format")
    assert runtime.track_spending("quality") == {
        "optimisation_calls": 2,
        "reflection_calls": 1,
        "final_calls": 0,
        "optimisation_usd": 0.025,
        "final_usd": 0,
        "total_usd": 0.025,
        "track_budget_usd": 2.0,
    }


def test_v1_total_preserves_historical_grouped_sum(frozen_track):
    directory, dataset, prompt = frozen_track
    setup_final_artifacts(directory, dataset, prompt)
    final = directory / "final"
    final.mkdir()
    (final / "baseline.jsonl").write_text(
        "".join(json.dumps({"effective_cost_usd": cost}) + "\n" for cost in (0.1, 0.2))
    )
    # Exact equality matters because the old JSON reports must rebuild byte for byte.
    assert runtime.track_spending("quality")["total_usd"] == 0.32500000000000007


@pytest.mark.parametrize("entrypoint", ["read_protocol", "track_spending", "run_final"])
def test_historical_track_preflight_rejects_missing_archive_before_inputs(tmp_path, monkeypatch, entrypoint):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    calls = []

    def require(root, scope=None):
        calls.append((root, scope))
        raise ValueError("Archived evidence is missing; restore it before continuing")

    monkeypatch.setattr(runtime, "require_restored", require)
    monkeypatch.setattr(runtime, "run_benchmark", lambda *args: pytest.fail("Must not call API"))
    with pytest.raises(ValueError, match="Archived evidence"):
        getattr(runtime, entrypoint)("quality_v2")
    assert calls == [(tmp_path, "tracks/quality_v2/")]
