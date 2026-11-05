import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracks.optimise import (
    BudgetStop,
    RunBudget,
    TextProposer,
    TrackAdapter,
    optimise,
    read_ledger,
    select_candidate,
    split_groups,
)


@pytest.fixture
def template():
    return json.loads((Path(__file__).resolve().parents[1] / "prompts/final/ai.json").read_text())


def test_test_split_and_cross_split_groups_rejected():
    dataset = {
        "samples": [
            {"id": "a", "group_id": "pair-a", "split": "train"},
            {"id": "b", "group_id": "pair-a", "split": "test"},
        ]
    }
    with pytest.raises(ValueError, match="train/dev"):
        split_groups(dataset, "test")
    with pytest.raises(ValueError, match="crosses"):
        split_groups(dataset, "train")


def test_small_development_gain_retains_baseline():
    result = SimpleNamespace(best_idx=1, val_aggregate_scores=[0.9, 0.9049])
    assert select_candidate(result) == 0
    result.val_aggregate_scores[1] = 0.906
    assert select_candidate(result) == 1


def test_reflection_charged_and_unknown_cost_stops(tmp_path, template):
    budget = RunBudget(tmp_path, budget_usd=0.11)
    called = []

    def response(endpoint, payload):
        called.append(payload)
        return {
            "status": 200,
            "seconds": 0.2,
            "response": {
                "usage": {"cost": 0.02},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"context": "A sufficiently long revised image context."})
                        }
                    }
                ],
            },
        }

    proposer = TextProposer("ai_detection", template, budget, post_fn=response)
    with pytest.raises(BudgetStop, match="spend"):
        proposer(
            {"context": template["context"]}, {"context": [{"Feedback": "Prediction error."}]}, ["context"]
        )
    assert len(called) == 1
    assert budget.totals() == {"task_calls": 0, "reflection_calls": 1, "effective_cost_usd": 0.02}
    row = read_ledger(tmp_path / "reflection.jsonl")[0]
    row["effective_cost_usd"] = None
    (tmp_path / "reflection.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(BudgetStop, match="unknown cost"):
        budget.check()


def test_only_context_can_mutate_and_only_train_can_reflect(tmp_path, template):
    adapter = TrackAdapter(
        "ai_detection",
        template,
        tmp_path / "protocol.json",
        tmp_path,
        RunBudget(tmp_path),
        lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="context string"):
        adapter.save_candidate({"context": template["context"], "questions": {}})
    batch = SimpleNamespace(trajectories=[{"split": "dev", "feedback": []}])
    with pytest.raises(ValueError, match="train feedback"):
        adapter.make_reflective_dataset({}, batch, ["context"])


def test_repeated_complete_group_requests_once_and_keeps_batch_results(tmp_path, template):
    dataset = {"samples": [{"id": side, "group_id": "pair", "split": "train"} for side in ("real", "ai")]}
    requests = []

    def evaluate(track, split, prompt_path, output_path, **kwargs):
        ids = kwargs["sample_ids"]
        requests.append(ids)
        assert ids == ["real", "ai"]
        old = read_ledger(output_path)
        if not old:
            output_path.write_text(
                "".join(json.dumps({"sample": sample, "effective_cost_usd": 0.001}) + "\n" for sample in ids)
            )
        return {"per_image": [{"id": sample, "reward": 0.5, "feedback": "Brier error 0.5"} for sample in ids]}

    budget = RunBudget(tmp_path, max_calls=2, budget_usd=1)
    adapter = TrackAdapter("ai_detection", template, tmp_path / "protocol.json", tmp_path, budget, evaluate)
    group = split_groups(dataset, "train")[0]
    batch = [group, group]
    for _ in range(2):
        result = adapter.evaluate(batch, {"context": template["context"]}, True)
        assert result.scores == [0.5, 0.5]
        assert len(result.outputs) == len(result.trajectories) == 2
        assert result.outputs[0] == result.outputs[1]
        assert result.trajectories[0] == result.trajectories[1]
    assert requests == [["real", "ai"], ["real", "ai"]]
    assert budget.totals()["task_calls"] == 2


def test_real_gepa_loop_groups_cache_budget_and_hides_test(tmp_path, template):
    pytest.importorskip("gepa")
    samples = [
        {"id": f"{split}-{group}-{side}", "group_id": f"{split}-{group}", "split": split}
        for split in ("train", "dev", "test")
        for group in range(4)
        for side in ("real", "ai")
    ]
    dataset = {"samples": samples}
    requested, reflected = [], []
    budget = RunBudget(tmp_path, max_calls=100, budget_usd=1)

    def evaluate(track, split, prompt_path, output_path, **kwargs):
        assert split != "test"
        ids = kwargs["sample_ids"]
        requested.extend(ids)
        assert len(ids) % 2 == 0
        prompt = json.loads(prompt_path.read_text())
        assert prompt["questions"] == template["questions"]
        reward = 0.9 if "REVISED" in prompt["context"] else 0.5
        old = read_ledger(output_path)
        done = {row["sample"] for row in old}
        with output_path.open("a") as handle:
            for sample in ids:
                if sample not in done:
                    handle.write(json.dumps({"sample": sample, "effective_cost_usd": 0.001}) + "\n")
        return {
            "per_image": [
                {"id": sample, "reward": reward, "feedback": f"Brier error {1 - reward}"} for sample in ids
            ]
        }

    def reflection(endpoint, payload):
        reflected.append(payload)
        context = (
            "REVISED: Judge the visible image with calibrated uncertainty and preserve the task meanings."
        )
        return {
            "status": 200,
            "seconds": 0.2,
            "response": {
                "usage": {"cost": 0.001},
                "choices": [{"message": {"content": json.dumps({"context": context})}}],
            },
        }

    adapter = TrackAdapter("ai_detection", template, tmp_path / "protocol.json", tmp_path, budget, evaluate)
    proposer = TextProposer("ai_detection", template, budget, max_proposals=2, post_fn=reflection)
    result = optimise(adapter, proposer, split_groups(dataset, "train"), split_groups(dataset, "dev"))
    assert result["selected_dev_reward"] == pytest.approx(0.9)
    assert result["selected_index"] == 1
    assert result["costs"]["reflection_calls"] == 2
    assert result["costs"]["task_calls"] == 32
    assert result["costs"]["effective_cost_usd"] == pytest.approx(0.034)
    assert all(not sample.startswith("test-") for sample in requested)
    assert "test-" not in json.dumps(reflected)
    assert "train-" not in json.dumps(reflected)
    assert "dev-" not in json.dumps(reflected)
    assert result["test_used"] is False


@pytest.mark.parametrize("usage", [None, [], "malformed"])
def test_malformed_usage_retains_attempt_and_stops(tmp_path, template, usage):
    def response(endpoint, payload):
        return {"status": 200, "seconds": 0.2, "response": {"usage": usage}}

    budget = RunBudget(tmp_path)
    proposer = TextProposer("ai_detection", template, budget, post_fn=response)
    with pytest.raises(BudgetStop, match="unknown cost"):
        proposer({"context": template["context"]}, {"context": []}, ["context"])
    assert len(read_ledger(tmp_path / "reflection.jsonl")) == 1


def test_original_prompt_bytes_preserved(tmp_path, template):
    raw = json.dumps(template, separators=(",", ":")).encode()
    adapter = TrackAdapter(
        "ai_detection",
        template,
        tmp_path / "protocol.json",
        tmp_path,
        RunBudget(tmp_path),
        lambda *args, **kwargs: None,
        raw,
    )
    path = adapter.save_candidate({"context": template["context"]})
    assert path.read_bytes() == raw


def test_adapter_uses_real_runner_without_repeat_requests(tmp_path, monkeypatch):
    pytest.importorskip("gepa")
    import bench_runner
    from tracks import run

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(
        bench_runner, "image_part", lambda sample: {"type": "image_url", "image_url": {"url": "fake"}}
    )
    prompt = json.loads((Path(__file__).resolve().parents[1] / "prompts/final/quality.json").read_text())
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(prompt) + "\n")
    samples = [
        {
            "id": f"{split}-{i}",
            "split": split,
            "group_id": split,
            "image": f"{split}-{i}.jpg",
            "delivered_sha256": str(i) * 64,
            "labels": {"quality_mos_1to5": 2.0},
        }
        for split in ("train", "dev", "test")
        for i in range(2)
    ]
    # Distinct hashes across splits are required by the real freeze guard.
    for i, sample in enumerate(samples):
        sample["delivered_sha256"] = f"{i:064x}"
    dataset = {"track": "quality", "samples": samples}
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset) + "\n")
    protocol_path = tmp_path / "protocol.json"
    run.create_protocol("quality", baseline, dataset_path=dataset_path, output_path=protocol_path)
    called = []

    def response(endpoint, payload):
        called.append(payload)
        return {
            "status": 200,
            "seconds": 0.2,
            "response": {
                "usage": {"cost": 0.001},
                "answers": {
                    field: {
                        "type": "score",
                        "score": 1.0,
                        "probabilities": {str(i): float(i == 1) for i in range(5)},
                    }
                    for field in prompt["questions"]
                },
            },
        }

    monkeypatch.setattr(bench_runner, "post", response)
    output = tmp_path / "optimisation"
    adapter = TrackAdapter(
        "quality", prompt, protocol_path, output, RunBudget(output), run.eval_prompt, baseline.read_bytes()
    )
    groups = split_groups(dataset, "train")
    first = adapter.evaluate(groups, {"context": prompt["context"]}, True)
    second = adapter.evaluate(groups, {"context": prompt["context"]}, True)
    assert len(called) == 2
    assert first.scores == second.scores == [1.0]
    assert adapter.budget.totals()["task_calls"] == 2


def test_resume_replays_recorded_proposal_without_another_call(tmp_path, template):
    calls = []

    def response(endpoint, payload):
        calls.append(payload)
        return {
            "status": 200,
            "response": {
                "usage": {"cost": 0.001},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"context": "Use calibrated uncertainty for image origin assessment."}
                            )
                        }
                    }
                ],
            },
        }

    budget = RunBudget(tmp_path)
    arguments = ({"context": template["context"]}, {"context": []}, ["context"])
    first = TextProposer("ai_detection_v2", template, budget, max_proposals=20, post_fn=response)
    answer = first(*arguments)
    resumed = TextProposer("ai_detection_v2", template, budget, max_proposals=20, post_fn=response)
    assert resumed(*arguments) == answer
    assert len(calls) == 1
    changed = TextProposer("ai_detection_v2", template, budget, max_proposals=20, post_fn=response)
    with pytest.raises(ValueError, match="Resume proposal input changed"):
        changed(arguments[0], {"context": [{"Feedback": "Different input"}]}, arguments[2])
    assert len(calls) == 1


def test_v2_twenty_proposals_use_train_pixels_and_resume_without_paid_calls(tmp_path, template):
    pytest.importorskip("gepa")
    samples = [
        {"id": f"{split}-{group}-{side}", "group_id": f"{split}-{group}", "split": split}
        for split in ("train", "dev", "test", "external")
        for group in range(8)
        for side in ("real", "ai")
    ]
    dataset = {"samples": samples}
    requested, reflected = [], []
    budget = RunBudget(tmp_path, max_calls=4000, budget_usd=5)

    def evaluate(track, split, prompt_path, output_path, **kwargs):
        assert split in {"train", "dev"}
        ids = kwargs["sample_ids"]
        requested.extend(ids)
        old = read_ledger(output_path)
        done = {row["sample"] for row in old}
        with output_path.open("a") as handle:
            for identifier in ids:
                if identifier not in done:
                    handle.write(json.dumps({"sample": identifier, "effective_cost_usd": 0.001}) + "\n")
        return {
            "per_image": [
                {
                    "id": identifier,
                    "image": f"data/{identifier}.jpg",
                    "split": split,
                    "delivered_sha256": "a" * 64,
                    "reward": 0.5,
                    "feedback": {"target": True, "prediction": 0.5},
                }
                for identifier in ids
            ]
        }

    def reflection(endpoint, payload):
        reflected.append(payload)
        return {
            "status": 200,
            "response": {
                "usage": {"cost": 0.001},
                "choices": [
                    {
                        "message": {
                            "content": "null"
                            if len(reflected) % 5 == 0
                            else json.dumps(
                                {"context": "Judge calibrated image origin and preserve uncertainty."}
                            )
                        }
                    }
                ],
            },
        }

    def make_adapter():
        return TrackAdapter(
            "ai_detection_v2", template, tmp_path / "protocol.json", tmp_path, budget, evaluate
        )

    def make_proposer():
        return TextProposer(
            "ai_detection_v2",
            template,
            budget,
            max_proposals=20,
            post_fn=reflection,
            image_fn=lambda image: {"type": "image_url", "image_url": {"url": "fake-training-pixels"}},
        )

    first = optimise(
        make_adapter(), make_proposer(), split_groups(dataset, "train"), split_groups(dataset, "dev")
    )
    assert len(reflected) == 20 and first["costs"]["reflection_calls"] == 20
    assert first["proposals"] == {"attempted": 20, "accepted": 16, "rejected": 4}
    assert all(not identifier.startswith(("test-", "external-")) for identifier in requested)
    assert all(len(payload["messages"][1]["content"]) == 9 for payload in reflected)
    assert "dev-" not in json.dumps(reflected) and "external-" not in json.dumps(reflected)
    attempts = read_ledger(tmp_path / "reflection.jsonl")
    assert all(len(row["reflection_images"]) == 4 for row in attempts)
    assert sum(row["proposal_outcome"] == "rejected" for row in attempts) == 4
    second = optimise(
        make_adapter(), make_proposer(), split_groups(dataset, "train"), split_groups(dataset, "dev")
    )
    assert len(reflected) == 20
    assert first["costs"] == second["costs"]


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        "null",
        "[]",
        "{}",
        '{"context":null}',
        '{"context":"short"}',
        '{"context":"A valid length but extra schema field is present.","extra":"x"}',
    ],
)
def test_known_cost_malformed_context_is_retained_rejected_and_replayed(tmp_path, template, content):
    called = []

    def response(endpoint, payload):
        called.append(payload)
        return {
            "status": 200,
            "response": {"usage": {"cost": 0.001}, "choices": [{"message": {"content": content}}]},
        }

    budget = RunBudget(tmp_path)
    arguments = ({"context": template["context"]}, {"context": []}, ["context"])
    proposer = TextProposer("ai_detection_v2", template, budget, max_proposals=20, post_fn=response)
    assert proposer(*arguments) == arguments[0]
    row = read_ledger(tmp_path / "reflection.jsonl")[0]
    assert row["proposal_outcome"] == "rejected" and row["effective_cost_usd"] == 0.001
    assert row["response"]["choices"][0]["message"]["content"] == content
    resumed = TextProposer("ai_detection_v2", template, budget, max_proposals=20, post_fn=response)
    assert resumed(*arguments) == arguments[0]
    assert len(called) == 1 and budget.totals()["reflection_calls"] == 1
    assert not (tmp_path / "candidates").exists()


@pytest.mark.parametrize("mutation", ["non_json", "null", "list", "schema", "aesthetics", "scale", "length"])
def test_quality_rubric_rejection_does_not_write_candidate_and_resume_reuses_attempt(tmp_path, mutation):
    import copy

    template = json.loads((Path(__file__).resolve().parents[1] / "prompts/final/quality.json").read_text())
    rubric = copy.deepcopy(template["questions"])
    if mutation == "schema":
        rubric["technical_quality"]["type"] = "noul"
    elif mutation == "aesthetics":
        rubric["aesthetics"]["instructions"] = "An unscored aesthetic change"
    elif mutation == "scale":
        rubric["technical_quality"]["criteria"][0] = "5 excellent: reverse the scale"
    text = {"non_json": "not JSON", "null": "null", "list": "[]", "length": "x" * 12001}.get(
        mutation, json.dumps(rubric)
    )
    calls = []

    def response(endpoint, payload):
        calls.append(payload)
        return {
            "status": 200,
            "response": {
                "usage": {"cost": 0.001},
                "choices": [{"message": {"content": json.dumps({"rubric": text})}}],
            },
        }

    budget = RunBudget(tmp_path)
    adapter = TrackAdapter(
        "quality_v2", template, tmp_path / "protocol.json", tmp_path, budget, lambda *a, **k: None
    )
    candidate = adapter.seed_candidate()
    proposer = TextProposer("quality_v2", template, budget, max_proposals=20, post_fn=response)
    assert proposer(candidate, {"rubric": []}, ["rubric"]) == {"rubric": candidate["rubric"]}
    assert read_ledger(tmp_path / "reflection.jsonl")[0]["proposal_outcome"] == "rejected"
    resumed = TextProposer("quality_v2", template, budget, max_proposals=20, post_fn=response)
    assert resumed(candidate, {"rubric": []}, ["rubric"]) == {"rubric": candidate["rubric"]}
    assert len(calls) == 1
    with pytest.raises((ValueError, TypeError, KeyError, AttributeError)):
        adapter.save_candidate({**candidate, "rubric": text})
    assert not (tmp_path / "candidates").exists()


def test_optimisation_cannot_restart_from_missing_archived_attempts(tmp_path, monkeypatch, template):
    from tracks import optimise as runtime

    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    output = tmp_path / "tracks/ai_detection_v2/optimisation"
    scopes = []

    def require(root, scope=None):
        scopes.append((root, scope))
        raise ValueError("Archived evidence is missing; restore it before continuing")

    monkeypatch.setattr(runtime, "require_restored", require)
    budget = RunBudget(output)
    proposer = TextProposer(
        "ai_detection_v2", template, budget, post_fn=lambda *args: pytest.fail("Must not call API")
    )
    with pytest.raises(ValueError, match="Archived evidence"):
        proposer({"context": template["context"]}, {"context": []}, ["context"])
    assert scopes == [(tmp_path, "tracks/ai_detection_v2/optimisation/")]
    assert not output.exists()
