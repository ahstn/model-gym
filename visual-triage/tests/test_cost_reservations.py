"""One reviewed reservation supports resume without rewriting unknown actual cost."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_runner
from cost_reservations import freeze, guard_cost, reservation
from tracks import metadata_ablation, report, run
from tracks.optimise import BudgetStop, RunBudget, TrackAdapter, split_groups


@pytest.fixture
def failed_track(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(metadata_ablation, "ROOT", tmp_path)
    fake_image = {"type": "image_url", "image_url": {"url": "fake-training-image"}}
    monkeypatch.setattr(bench_runner, "image_part", lambda sample: fake_image)
    monkeypatch.setattr(metadata_ablation, "image_part", lambda sample: fake_image)
    directory = tmp_path / "tracks/quality_v2"
    directory.mkdir(parents=True)
    prompt = json.loads((Path(__file__).resolve().parents[1] / "prompts/final/quality.json").read_text())
    baseline = directory / "baseline.json"
    baseline.write_text(json.dumps(prompt))
    samples = []
    for index, split in enumerate(("train", "dev", "test")):
        image = tmp_path / f"{split}.png"
        Image.new("RGB", (4, 5), (index * 30, 10, 20)).save(image)
        samples.append(
            {
                "id": split,
                "group_id": split,
                "split": split,
                "image": image.name,
                "delivered_sha256": run.file_sha(image),
                "labels": {"quality_mos_1to5": 2.0},
            }
        )
    dataset = {"track": "quality_v2", "benchmark": "reservation-test", "samples": samples}
    (directory / "dataset.json").write_text(json.dumps(dataset))
    protocol_path = directory / "protocol.json"
    run.create_protocol("quality_v2", baseline)
    output = directory / "optimisation"
    budget = RunBudget(output, budget_usd=4.5)
    adapter = TrackAdapter(
        "quality_v2", prompt, protocol_path, output, budget, run.eval_prompt, baseline.read_bytes()
    )
    calls = []

    def fail(endpoint, payload):
        calls.append(payload)
        return {"transport_error": "ConnectionError", "seconds": 0.15}

    monkeypatch.setattr(bench_runner, "post", fail)
    with pytest.raises(ValueError, match="Unknown request cost"):
        adapter.evaluate(split_groups(dataset, "train"), adapter.seed_candidate(), True)
    candidate = adapter.save_candidate(adapter.seed_candidate())
    ledger = candidate.parent / "train.jsonl"
    raw = ledger.read_bytes()
    row = json.loads(raw)
    (output / "run.json").write_text(json.dumps({"track": "quality_v2", "max_proposals": 20}))
    evidence = directory / "evidence.json"
    evidence.write_text(json.dumps({"catalogue_price": "fixture", "actual_cost": None}))
    return directory, dataset, prompt, adapter, ledger, row, raw, evidence, calls


def reserve(fixture):
    _directory, _dataset, _prompt, _adapter, ledger, row, _raw, evidence, _calls = fixture
    return freeze(
        ledger,
        row["sample"],
        row["request_sha256"],
        evidence,
        "Reviewed fixture failure; reserve only, actual unknown",
    )


def test_exact_failed_attempt_resumes_without_retry_and_keeps_actual_unknown(failed_track, capsys):
    _directory, dataset, _prompt, adapter, ledger, row, raw, _evidence, calls = failed_track
    with pytest.raises(BudgetStop, match="unknown cost"):
        adapter.budget.check()
    reserve(failed_track)
    result = adapter.evaluate(split_groups(dataset, "train"), adapter.seed_candidate(), True)
    assert result.scores == [0.0]
    assert len(calls) == 1 and ledger.read_bytes() == raw
    output = capsys.readouterr().out
    assert "budget-accounted cost: $0.100000" in output
    assert "known reported cost including BYOK: $0.000000" in output
    assert "reserved unverified cost: $0.100000; actual total cost: unknown" in output
    assert "effective reported cost including BYOK: $0.100000" not in output
    assert row["effective_cost_usd"] is None and row["cost_usd"] is None
    assert adapter.budget.totals() == {
        "task_calls": 1,
        "reflection_calls": 0,
        "effective_cost_usd": None,
        "known_reported_usd": 0,
        "reserved_unverified_usd": 0.1,
        "budget_accounted_usd": 0.1,
        "unknown_cost_count": 1,
    }
    other = {**row, "sample": "unreviewed"}
    with pytest.raises(ValueError, match="Unknown cost"):
        guard_cost(other, ledger, None)


@pytest.mark.parametrize("change", ["ledger", "evidence", "protocol", "run", "sample", "request", "amount"])
def test_reservation_hashes_and_scope_fail_closed(failed_track, change):
    directory, _dataset, _prompt, _adapter, ledger, row, raw, evidence, calls = failed_track
    plan_path = reserve(failed_track)
    if change == "ledger":
        ledger.write_bytes(raw + b" ")
        ledger.write_bytes(b" " + ledger.read_bytes())
    elif change in {"evidence", "protocol", "run"}:
        path = {
            "evidence": evidence,
            "protocol": directory / "protocol.json",
            "run": directory / "optimisation/run.json",
        }[change]
        path.write_bytes(path.read_bytes() + b" ")
    else:
        plan = json.loads(plan_path.read_text())
        field = {"sample": "sample", "request": "request_sha256", "amount": "reservation_usd"}[change]
        plan["reservations"][0][field] = 0.01 if change == "amount" else "changed"
        plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        reservation(row, ledger)
    assert len(calls) == 1


def test_prefix_can_append_known_attempt_but_sidecar_cannot_be_replaced(failed_track):
    _directory, _dataset, _prompt, _adapter, ledger, row, raw, _evidence, _calls = failed_track
    reserve(failed_track)
    ledger.write_bytes(raw + json.dumps({"sample": "later", "effective_cost_usd": 0.01}).encode() + b"\n")
    assert reservation(row, ledger) == 0.1
    with pytest.raises(FileExistsError):
        reserve(failed_track)


def test_reserved_cost_counts_in_final_metadata_and_report_guards(failed_track, monkeypatch):
    directory, dataset, prompt, adapter, ledger, _row, raw, _evidence, calls = failed_track
    reserve(failed_track)
    metadata_ablation.freeze("quality_v2")

    def success(endpoint, payload):
        calls.append(payload)
        return {
            "status": 200,
            "seconds": 0.1,
            "response": {
                "usage": {"cost": 0.001},
                "answers": {
                    key: {
                        "type": "score",
                        "score": 1,
                        "probabilities": {str(i): float(i == 1) for i in range(5)},
                    }
                    for key in prompt["questions"]
                },
            },
        }

    monkeypatch.setattr(bench_runner, "post", success)
    adapter.evaluate(split_groups(dataset, "dev"), adapter.seed_candidate())
    candidate = adapter.save_candidate(adapter.seed_candidate())
    (directory / "optimisation/summary.json").write_text(json.dumps({"costs": adapter.budget.totals()}))
    run.freeze_selection(
        directory / "protocol.json", candidate, candidate.parent / "dev.jsonl", candidate.parent / "dev.jsonl"
    )
    spending = run.track_spending("quality_v2")
    assert spending["total_usd"] is None
    assert spending["optimisation_usd"] is None
    assert spending["budget_accounted_usd"] == pytest.approx(0.101)
    result = run.run_final("quality_v2")
    assert result["spending"]["budget_accounted_usd"] == pytest.approx(0.102)
    report.build_report("quality_v2", 0)
    text = (directory / "report/README.md").read_text()
    assert "Actual total cost remains unknown" in text and "Unreconciled budget reservation: $0.10" in text
    monkeypatch.setattr(metadata_ablation, "run_benchmark", bench_runner.run)
    metadata_ablation.run("quality_v2")
    assert run.track_spending("quality_v2")["budget_accounted_usd"] == pytest.approx(0.103)
    assert ledger.read_bytes() == raw


def test_reservation_does_not_allow_a_new_unknown_or_v1_attempt(failed_track):
    directory, _dataset, _prompt, adapter, ledger, row, raw, evidence, _calls = failed_track
    reserve(failed_track)
    with ledger.open("a") as output:
        output.write(json.dumps({**row, "sample": "new", "request_sha256": "new"}) + "\n")
    with pytest.raises(BudgetStop, match="unknown cost"):
        adapter.budget.check()
    v1 = directory.parent / "quality"
    v1.mkdir()
    old = v1 / "train.jsonl"
    old.write_bytes(raw)
    assert reservation(row, old) is None
    with pytest.raises(ValueError, match="v2 ledger"):
        freeze(old, row["sample"], row["request_sha256"], evidence, "Cannot override v1")


def test_reserved_summary_accepts_reordered_ledger_sums(failed_track):
    directory, _dataset, _prompt, adapter, _ledger, row, _raw, _evidence, _calls = failed_track
    reserve(failed_track)
    amounts = [0.01] + [0.000001] * 40
    for index, amount in enumerate(amounts):
        path = directory / "optimisation/candidates" / f"cost-{index:02d}" / "train.jsonl"
        path.parent.mkdir()
        path.write_text(json.dumps({**row, "effective_cost_usd": amount, "usage": {"cost": amount}}) + "\n")
    costs = adapter.budget.totals()
    forward = reverse = 0.0
    for amount in amounts:
        forward += amount
    for amount in reversed(amounts):
        reverse += amount
    assert forward != reverse and abs(forward - reverse) < 1e-9
    costs["known_reported_usd"] = reverse
    costs["budget_accounted_usd"] = reverse + 0.1
    summary = directory / "optimisation/summary.json"
    summary.write_text(json.dumps({"costs": costs}))
    assert run.track_spending("quality_v2")["known_reported_usd"] == pytest.approx(forward)

    for field, invalid in (
        ("known_reported_usd", reverse + 1e-8),
        ("reserved_unverified_usd", 0.1 + 1e-8),
        ("budget_accounted_usd", reverse + 0.1 + 1e-8),
        ("known_reported_usd", float("nan")),
        ("unknown_cost_count", 2),
        ("unknown_cost_count", 1.0),
    ):
        summary.write_text(json.dumps({"costs": {**costs, field: invalid}}))
        with pytest.raises((ValueError, TypeError)):
            run.track_spending("quality_v2")
