"""The latest-panel run uses only frozen test cases and unchanged image requests."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_jev14_run import fake_post, prompt, save_json

from bench_protocol import digest
from tracks.jev14 import latest
from tracks.jev14 import run as study


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path
    track = root / "tracks/jev14"
    data = root / "data/jev14-latest-v1"
    track.mkdir(parents=True)
    data.mkdir(parents=True)
    for module in (latest, study):
        monkeypatch.setattr(module, "ROOT", root)
        monkeypatch.setattr(module, "TRACK", track)
    monkeypatch.setattr(latest, "DATA", data)
    monkeypatch.setattr(latest, "PROTOCOL", track / "latest/protocol.json")
    monkeypatch.setattr(latest, "MANIFEST", data / "manifest.json")
    monkeypatch.setattr(latest, "LEDGER", data / "attempts.jsonl")
    monkeypatch.setattr(latest, "__file__", str(track / "latest.py"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    for path in (
        track / "latest.py",
        track / "client.py",
        track / "rounding.py",
        track / "run.py",
        root / "bench_protocol.py",
        root / "bench_runner.py",
    ):
        path.write_text("# frozen fixture\n")
    image = root / "view.jpg"
    Image.new("RGB", (512, 512), "blue").save(image, "JPEG")
    prompts = {"quality": root / "tracks/quality_v2/prompts/selected.json", "ai": root / "selected-ai.json"}
    for task in ("quality", "ai"):
        save_json(prompts[task], prompt(task))
        samples = [
            {
                "id": f"{task}-{i:03d}",
                "image": image.name,
                "delivered_sha256": study.sha(image),
                "split": "test",
                "labels": {"quality_mos_1to5": 2.5},
                "gold": True,
                "group_id": f"group-{task}-{i}",
                "source": {"generator": "private-source"},
            }
            for i in range(100 if task == "quality" else 160)
        ]
        ignored = [
            {"id": f"excluded-{split}", "image": "missing-training-image.jpg", "split": split}
            for split in ("train", "dev")
        ]
        save_json(root / latest.DATASETS[task], {"samples": samples + ignored})
        baseline = [
            {
                "sample": s["id"],
                "delivered_sha256": s["delivered_sha256"],
                "prompt_sha256": digest(prompt(task)),
            }
            for s in samples
        ]
        path = root / latest.BASELINES[task]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in baseline))
    save_json(
        root / "data/metadata-v3/runs/image_only/selection.json",
        {"selected_prompt_path": str(prompts["ai"]), "selected_prompt_sha256": digest(prompt("ai"))},
    )
    for path in (
        "tracks/quality_v2/protocol.json",
        "tracks/metadata_v3/protocol.json",
        "tracks/quality_v2/selection.json",
        "data/jev14-v1/research/openrouter-decisions-models.json",
    ):
        save_json(root / path, {})
    prior = root / "data/jev14-v1/attempts.jsonl"
    prior.write_text(
        json.dumps({"status": 200, "effective_cost_usd": 0.006})
        + "\n"
        + json.dumps({"status": 400, "effective_cost_usd": None})
        + "\n"
    )
    return root


def test_freeze_selects_exact_test_cohorts_and_request_excludes_source_and_gold(prepared):
    assert latest.freeze()["cases"] == 260
    protocol, cases, prompts, rows = latest.load_study()
    assert len([c for c in cases if c["task"] == "quality"]) == 100
    assert len([c for c in cases if c["task"] == "ai"]) == 160
    assert not rows and all(not c["sample"].startswith("excluded") for c in cases)
    for case in cases[:2]:
        payload = latest.make_payload(prepared, case, prompts[case["task"]])
        assert set(payload) == {"model", "state", "questions"}
        assert len(payload["state"]) == 2
        text = json.dumps(payload)
        assert "private-source" not in text and "group_id" not in text and '"gold"' not in text
    assert protocol["prior_known_cost_usd"] == 0.006
    assert protocol["prior_unknown_cost_reserve_usd"] == 0.012


@pytest.mark.parametrize("field", ["sample", "delivered_sha256", "prompt_sha256"])
def test_baseline_identity_image_or_prompt_mismatch_blocks_freeze(prepared, field):
    path = prepared / latest.BASELINES["quality"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = "changed"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="differ"):
        latest.freeze()
    assert not latest.PROTOCOL.exists()


def test_frozen_latest_source_pin_blocks_mutation(prepared):
    latest.freeze()
    with (prepared / latest.DATASETS["quality"]).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="frozen latest-trial input"):
        latest.load_study()


def test_review_accepts_only_rounding_validation_error_and_keeps_original_score(prepared):
    task_prompt = prompt("quality")
    row = fake_post({}, task_prompt)
    row["response"]["answers"]["aesthetics"].update(
        score=2.58, probabilities={"0": 0, "1": 0.06, "2": 0.34, "3": 0.54, "4": 0.05}
    )
    row.update(normalized=None, validation_error="ValueError: Probability distribution does not sum to one")
    reviewed = latest.review(row, task_prompt)
    assert reviewed["normalized"]["aesthetics"] == 3.58
    assert reviewed["response"]["answers"]["aesthetics"]["score"] == 2.58
    row["validation_error"] = "ValueError: Non-finite or out-of-range value"
    reviewed = latest.review(row, task_prompt)
    assert reviewed["normalized"] is None and reviewed["validation_error"]


def test_prior_unknown_reserve_guards_budget_but_is_not_reported_as_actual_cost(prepared, monkeypatch):
    latest.freeze()
    protocol = json.loads(latest.PROTOCOL.read_text())
    protocol["minimum_interval_seconds"] = 0
    protocol["budget_usd"] = 0.025
    save_json(latest.PROTOCOL, protocol)
    calls = []

    def post(payload, task_prompt):
        calls.append(payload)
        return fake_post(payload, task_prompt)

    monkeypatch.setattr(latest, "post", post)
    with pytest.raises(ValueError, match="budget guard"):
        latest.run(1)
    assert calls == []
    protocol["budget_usd"] = 2
    save_json(latest.PROTOCOL, protocol)
    result = latest.run(1)
    assert len(calls) == 1
    assert result["reported_cost_usd"] == 0.000042


def test_priced_failed_call_stops_and_cannot_be_skipped(prepared, monkeypatch):
    latest.freeze()
    calls = []

    def post(payload, task_prompt):
        calls.append(payload)
        row = fake_post(payload, task_prompt)
        row["status"] = 502
        return row

    monkeypatch.setattr(latest, "post", post)
    assert latest.run(2) == {"status": "stopped", "attempts": 1}
    saved = json.loads(latest.LEDGER.read_text())
    assert saved["effective_cost_usd"] == 0.000042
    assert saved["normalized"] is None
    with pytest.raises(ValueError, match="failed prior call"):
        latest.run(1)
    assert len(calls) == 1
