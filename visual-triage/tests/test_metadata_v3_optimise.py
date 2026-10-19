"""True GEPA attempts, exact replay and train-only reflection without API calls."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_protocol import digest
from tracks.metadata_v3 import optimise, runtime


@pytest.fixture
def study(tmp_path, monkeypatch):
    prompt = json.loads((Path(__file__).resolve().parents[1] / "prompts/final/ai.json").read_text())
    samples = [
        {
            "id": f"{split}-{group}-{side}",
            "group_id": f"{split}-{group}",
            "split": split,
            "image": "fake",
            "delivered_sha256": "view",
            "gold": side,
            "features": {},
            "features_sha256": digest({}),
        }
        for split in ("train", "dev", "test")
        for group in range(5)
        for side in (0, 1)
    ]
    dataset = {"samples": samples}
    monkeypatch.setattr(optimise, "context_for", lambda sample, arm: {"arm": arm})
    monkeypatch.setattr(optimise, "feature_text", lambda sample, arm: "Frozen feature context")
    monkeypatch.setattr(
        optimise,
        "image_part",
        lambda sample: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,fixture"}},
    )
    budget = runtime.StudyBudget(tmp_path / "runs", max_calls=4000)
    requests = []

    def evaluator(members, candidate, arm, output, budget, protocol_sha256, split, **kwargs):
        assert split != "test"
        assert len({s["id"] for s in members}) == len(members)
        old = runtime.read_ledger(output)
        done = {r["sample"] for r in old}
        for sample in members:
            if sample["id"] not in done:
                budget.check(arm, 1)
                requests.append(sample["id"])
                runtime.append_record(
                    output,
                    {
                        "record_type": "task",
                        "metadata_arm": arm,
                        "sample": sample["id"],
                        "effective_cost_usd": 0.00001,
                    },
                )
        reward = 0.6 if "REVISED" in candidate["context"] else 0.5
        return {"per_image": [{**s, "reward": reward, "feedback": {"loss": 1 - reward}} for s in members]}

    return dataset, prompt, budget, evaluator, requests


def test_true_gepa_twenty_proposals_and_independent_selection(study):
    dataset, prompt, budget, evaluator, requests = study
    paid = []

    def send(endpoint, payload):
        paid.append(payload)
        assert payload["reasoning"] == {"effort": "low"}
        text = json.dumps(payload)
        assert "test-" not in text and "dev-" not in text
        assert text.count("data:image/jpeg;base64") <= 4
        return {
            "status": 200,
            "seconds": 0.1,
            "response": {
                "usage": {"cost": 0.00001},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "context": f"REVISED {len(paid)}: Assess image origin from the supplied evidence with calibrated uncertainty."
                                }
                            )
                        }
                    }
                ],
            },
        }

    summary = optimise.optimise_arm(
        dataset,
        prompt,
        "camera",
        budget.root / "camera",
        budget,
        "protocol",
        evaluate_fn=evaluator,
        post_fn=send,
    )
    assert len(paid) == summary["proposals"]["attempted"] == 20
    assert summary["selected_dev_reward"] == pytest.approx(0.6)
    assert summary["test_used"] is False
    assert all(not identifier.startswith("test-") for identifier in requests)
    assert (budget.root / "camera/selection.json").exists()
    with pytest.raises(ValueError, match="fresh"):
        optimise.optimise_arm(dataset, prompt, "camera", budget.root / "camera", budget, "protocol")


def test_malformed_proposal_replays_without_call_and_changed_input_fails(study):
    dataset, prompt, budget, evaluator, _requests = study
    adapter = optimise.Adapter(dataset, prompt, "tone", budget.root / "tone", budget, "protocol", evaluator)
    count = []

    def send(endpoint, payload):
        count.append(payload)
        return {
            "status": 200,
            "seconds": 0.1,
            "response": {"usage": {"cost": 0.001}, "choices": [{"message": {"content": "null"}}]},
        }

    proposer = optimise.Proposer(adapter, post_fn=send)
    arguments = (adapter.seed_candidate(), {"context": []}, ["context"])
    assert proposer(*arguments) == adapter.seed_candidate()
    row = runtime.read_ledger(proposer.path)[0]
    assert (
        row["proposal_outcome"] == "rejected"
        and row["response"]["choices"][0]["message"]["content"] == "null"
    )
    resumed = optimise.Proposer(adapter, post_fn=send)
    assert resumed(*arguments) == adapter.seed_candidate() and len(count) == 1
    altered = optimise.Proposer(adapter, post_fn=send)
    with pytest.raises(ValueError, match="input changed"):
        altered({"context": prompt["context"] + " Changed"}, {"context": []}, ["context"])
    assert len(count) == 1


def test_repeated_group_outputs_and_no_test_reflection(study):
    dataset, prompt, budget, evaluator, requests = study
    adapter = optimise.Adapter(
        dataset, prompt, "camera", budget.root / "camera", budget, "protocol", evaluator
    )
    group = optimise.split_groups(dataset, "train")[0]
    result = adapter.evaluate([group, group], adapter.seed_candidate(), True)
    assert len(result.scores) == 2 and len(requests) == 2
    reflection = adapter.make_reflective_dataset(adapter.seed_candidate(), result, ["context"])
    assert len(reflection["context"]) == 2
    assert all(example["context"] == {"arm": "camera"} for example in reflection["context"])
    result.trajectories[0]["split"] = "dev"
    with pytest.raises(ValueError, match="training"):
        adapter.make_reflective_dataset(adapter.seed_candidate(), result, ["context"])


def test_unknown_reflection_retains_attempt_and_resume_stops(study):
    dataset, prompt, budget, evaluator, _requests = study
    adapter = optimise.Adapter(dataset, prompt, "tone", budget.root / "tone", budget, "protocol", evaluator)
    count = []

    def send(endpoint, payload):
        count.append(payload)
        return {"transport_error": "ConnectionError", "seconds": 0.01}

    args = (adapter.seed_candidate(), {"context": []}, ["context"])
    for proposer in (optimise.Proposer(adapter, post_fn=send), optimise.Proposer(adapter, post_fn=send)):
        with pytest.raises(runtime.BudgetStop, match="Unknown"):
            proposer(*args)
    assert len(count) == 1


def test_whole_gepa_resume_replays_proposals_and_completes_twenty(study):
    dataset, prompt, budget, evaluator, _requests = study
    paid = []

    def send(endpoint, payload):
        paid.append(payload)
        return {
            "status": 200,
            "seconds": 0.1,
            "response": {
                "usage": {"cost": 0.00001},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "context": f"REVISED {len(paid)}: Assess supplied evidence for image origin with calibrated uncertainty."
                                }
                            )
                        }
                    }
                ],
            },
        }

    fail_once = [True]

    def interrupted(*args, **kwargs):
        if len(paid) == 7 and fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("Offline interruption before transport")
        return evaluator(*args, **kwargs)

    with pytest.raises(RuntimeError, match="Offline interruption"):
        optimise.optimise_arm(
            dataset,
            prompt,
            "tone",
            budget.root / "tone",
            budget,
            "protocol",
            evaluate_fn=interrupted,
            post_fn=send,
        )
    assert len(paid) == 7
    result = optimise.optimise_arm(
        dataset,
        prompt,
        "tone",
        budget.root / "tone",
        budget,
        "protocol",
        evaluate_fn=interrupted,
        post_fn=send,
        resume=True,
    )
    assert len(paid) == result["proposals"]["attempted"] == 20
    assert budget.pending() == {}


def test_eight_image_group_reflects_four_worst_individual_members(study):
    dataset, prompt, budget, evaluator, _requests = study
    adapter = optimise.Adapter(
        dataset, prompt, "camera", budget.root / "camera", budget, "protocol", evaluator
    )
    identifiers = [sample["id"] for sample in dataset["samples"] if sample["split"] == "train"][:8]
    group = {"group_id": "eight-members", "split": "train", "sample_ids": identifiers}
    result = adapter.evaluate([group], adapter.seed_candidate(), True)
    for index, row in enumerate(result.trajectories[0]["members"]):
        row["reward"] = index / 10
    examples = adapter.make_reflective_dataset(adapter.seed_candidate(), result, ["context"])["context"]
    assert len(examples) == 4
    assert [example["image"]["id"] for example in examples] == identifiers[:4]
    assert all(example["context"] == {"arm": "camera"} for example in examples)
    assert all(example["image"]["split"] == "train" for example in examples)
