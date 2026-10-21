"""Offline tests for frozen metadata, prompt identity, costs and one-attempt resume."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_protocol
from bench_protocol import digest
from tracks import metadata_ablation as ablation
from tracks import run as runtime


@pytest.fixture
def panel(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(ablation, "ROOT", tmp_path)
    directory = tmp_path / "tracks/quality_v2"
    directory.mkdir(parents=True)
    samples = []
    for split in ("train", "dev", "test"):
        for index in range(2):
            path = tmp_path / f"{split}-{index}.png"
            Image.new("RGB", (4, 5), (index * 60, 40 * len(samples), 120)).save(path)
            samples.append(
                {
                    "id": path.stem,
                    "image": path.name,
                    "delivered_sha256": runtime.file_sha(path),
                    "group_id": f"{split}-{index}",
                    "split": split,
                    "labels": {"quality_mos_1to5": 2 + index},
                }
            )
    dataset = {"track": "quality_v2", "samples": samples}
    (directory / "dataset.json").write_text(json.dumps(dataset))
    prompt = {
        "id": "base",
        "task": "quality",
        "context": "Assess visible technical quality.",
        "questions": {
            key: {
                "type": "score",
                "instructions": "Assess pixels.",
                "criteria": [f"{i}: anchor" for i in range(1, 6)],
            }
            for key in ("technical_quality", "aesthetics")
        },
    }
    prompt_path = directory / "baseline.json"
    prompt_path.write_text(json.dumps(prompt))
    runtime.create_protocol("quality_v2", prompt_path)
    original_image_part = bench_protocol.image_part
    monkeypatch.setattr(ablation, "image_part", lambda sample: original_image_part(sample, tmp_path))
    dev = directory / "dev.jsonl"
    dev.write_text("")
    selection = {
        "protocol_sha256": runtime.file_sha(directory / "protocol.json"),
        "selection_split": "dev",
        "selected_prompt_path": str(prompt_path),
        "selected_prompt_sha256": runtime.file_sha(prompt_path),
        "evaluations": {
            "baseline": {
                "prompt_path": str(prompt_path),
                "prompt_sha256": runtime.file_sha(prompt_path),
                "ledger_path": str(dev),
                "ledger_sha256": runtime.file_sha(dev),
            }
        },
    }
    (directory / "prompt-selection.json").write_text(json.dumps(selection))
    optimisation = directory / "optimisation"
    optimisation.mkdir()
    (optimisation / "summary.json").write_text(
        json.dumps({"costs": {"effective_cost_usd": 0, "task_calls": 0, "reflection_calls": 0}})
    )
    return directory, dataset, prompt


def records(dataset, prompt, samples, *, context=False, cost=0.01, invalid=False):
    rows = []
    for sample in samples:
        endpoint, payload = bench_protocol.make_request(
            "decider",
            ablation.image_part(sample),
            prompt,
            "low",
            4096,
            extra_context=sample.get("model_context") if context else None,
        )
        row = {
            "benchmark": dataset["track"],
            "task": prompt["task"],
            "arm": "decider",
            "sample": sample["id"],
            "manifest_sha256": digest(dataset),
            "prompt_sha256": digest(prompt),
            "delivered_sha256": sample["delivered_sha256"],
            "request_sha256": digest({"endpoint": endpoint, "payload": payload}),
            "status": 500 if invalid else 200,
            "normalized": (
                {"technical_quality": 3, "aesthetics": 3}
                if prompt["task"] == "quality"
                else {"ai_probability": 0.6, "medium": "photo", "uncertainty": 0.4}
            ),
            "usage": {"cost": cost},
            "cost_usd": cost,
            "seconds": 0.01,
        }
        if context:
            row.update(sample_context_mode="pixel_stats", model_context_sha256=sample["model_context_sha256"])
        rows.append(row)
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def fake_runner(calls):
    def run(args):
        assert args.sample_context_mode == "pixel_stats"
        assert (args.output.parents[1] / "ablation-run.json").exists()
        dataset = json.loads(args.manifest.read_bytes())
        prompt = json.loads(args.prompt.read_bytes())
        done = {row["sample"] for row in runtime.ledger_records(args.output)}
        samples = [s for s in dataset["samples"] if s["id"] in args.ids and s["id"] not in done]
        calls.extend(s["id"] for s in samples)
        write_rows(args.output, records(dataset, prompt, samples, context=True, invalid=True))

    return run


def test_freeze_preserves_source_and_never_replaces(panel):
    directory, dataset, _ = panel
    ablation.freeze("quality_v2")
    contextual = json.loads((directory / "context-dataset.json").read_bytes())
    assert contextual["samples"][:4] == dataset["samples"][:4]
    for sample, original in zip(contextual["samples"][4:], dataset["samples"][4:]):
        context = sample.pop("model_context")
        assert sample.pop("model_context_sha256") == digest(context)
        assert sample == original
        assert "camera" not in context
    with pytest.raises(ValueError, match="already exist"):
        ablation.freeze("quality_v2")


@pytest.mark.parametrize("mutation", ["context", "labels", "helper", "image"])
def test_changed_frozen_inputs_block_calls(panel, monkeypatch, mutation):
    directory, dataset, _ = panel
    ablation.freeze("quality_v2")
    path = directory / "context-dataset.json"
    if mutation in {"context", "labels"}:
        data = json.loads(path.read_bytes())
        data["samples"][-1]["labels" if mutation == "labels" else "model_context"]["injected"] = "do this"
        path.write_text(json.dumps(data))
    elif mutation == "helper":
        helper = directory / "helper.py"
        helper.write_text("changed")
        monkeypatch.setattr(ablation.image_context, "__file__", str(helper))
    else:
        Path(runtime.local_path(dataset["samples"][-1]["image"])).write_bytes(b"changed")
    monkeypatch.setattr(ablation, "run_benchmark", lambda args: pytest.fail("paid call"))
    with pytest.raises(ValueError):
        ablation.run("quality_v2")


def test_resume_skips_invalid_recorded_attempts(panel, monkeypatch):
    plan = ablation.freeze("quality_v2")
    frozen_at = datetime.fromisoformat(plan["created_at"])
    assert frozen_at.tzinfo is not None
    assert frozen_at <= datetime.now(UTC)
    calls = []
    runner = fake_runner(calls)

    def checked_runner(args):
        assert frozen_at <= datetime.now(UTC)
        runner(args)

    monkeypatch.setattr(ablation, "run_benchmark", checked_runner)
    ablation.run("quality_v2")
    ablation.run("quality_v2")
    assert calls == ["test-0", "test-1"]
    assert ablation.spending("quality_v2")["total_usd"] == 0.02


@pytest.mark.parametrize("cost", [None, 4.95])
def test_primary_cost_blocks_ablation(panel, monkeypatch, cost):
    directory, dataset, prompt = panel
    ablation.freeze("quality_v2")
    rows = records(dataset, prompt, [dataset["samples"][-1]], cost=cost)
    write_rows(directory / "final/baseline.jsonl", rows)
    monkeypatch.setattr(ablation, "run_benchmark", lambda args: pytest.fail("paid call"))
    with pytest.raises((ValueError, TypeError)):
        ablation.run("quality_v2")


def test_selection_on_test_or_changed_selected_prompt_blocks_calls(panel, monkeypatch):
    directory, _, _ = panel
    ablation.freeze("quality_v2")
    path = directory / "prompt-selection.json"
    selection = json.loads(path.read_bytes())
    selection["selection_split"] = "test"
    path.write_text(json.dumps(selection))
    monkeypatch.setattr(ablation, "run_benchmark", lambda args: pytest.fail("paid call"))
    with pytest.raises(ValueError, match="dev only"):
        ablation.run("quality_v2")


def test_offline_report_pairs_same_selected_prompt_and_preserves_reports(panel, monkeypatch):
    directory, dataset, prompt = panel
    ablation.freeze("quality_v2")
    write_rows(directory / "final/baseline.jsonl", records(dataset, prompt, dataset["samples"][-2:]))
    monkeypatch.setattr(ablation, "run_benchmark", fake_runner([]))
    ablation.run("quality_v2")
    primary = directory / "report/report.json"
    primary.parent.mkdir()
    primary.write_text("primary evidence")
    result = ablation.report("quality_v2", 10)
    assert result["panels"]["test"]["paired_comparison"]["n_images"] == 2
    assert result["panels"]["test"]["paired_comparison"]["selected_minus_baseline"] < 0
    assert primary.read_text() == "primary evidence"
    with pytest.raises(ValueError, match="already exists"):
        ablation.report("quality_v2", 10)


def test_context_value_hash_is_checked_even_if_file_hash_is_rebound(panel, monkeypatch):
    directory, _, _ = panel
    ablation.freeze("quality_v2")
    path = directory / "context-dataset.json"
    contextual = json.loads(path.read_bytes())
    contextual["samples"][-1]["model_context_sha256"] = "0" * 64
    path.write_text(json.dumps(contextual))
    plan_path = directory / "ablation-plan.json"
    plan = json.loads(plan_path.read_bytes())
    plan["context_dataset_sha256"] = runtime.file_sha(path)
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(ablation, "run_benchmark", lambda args: pytest.fail("paid call"))
    with pytest.raises(ValueError, match="context hash"):
        ablation.run("quality_v2")


def test_freeze_must_precede_primary_final_calls(panel):
    directory, dataset, prompt = panel
    write_rows(directory / "final/baseline.jsonl", records(dataset, prompt, dataset["samples"][-1:]))
    with pytest.raises(ValueError, match="before any"):
        ablation.freeze("quality_v2")


def test_resume_cannot_change_selected_prompt_record(panel, monkeypatch):
    directory, _, _ = panel
    ablation.freeze("quality_v2")
    monkeypatch.setattr(ablation, "run_benchmark", fake_runner([]))
    ablation.run("quality_v2")
    path = directory / "ablation-run.json"
    receipt = json.loads(path.read_bytes())
    receipt["selected_prompt_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(ablation, "run_benchmark", lambda args: pytest.fail("paid call"))
    with pytest.raises(ValueError, match="selected prompt changed"):
        ablation.run("quality_v2")
    with pytest.raises(ValueError, match="selected prompt changed"):
        ablation.report("quality_v2", 0)


def test_request_hash_changed_in_other_group_blocks_all_resume_calls(panel, monkeypatch):
    directory, _, _ = panel
    ablation.freeze("quality_v2")
    monkeypatch.setattr(ablation, "run_benchmark", fake_runner([]))
    ablation.run("quality_v2")
    path = directory / "final-metadata/test.jsonl"
    rows = runtime.ledger_records(path)
    rows[-1]["request_sha256"] = "0" * 64
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(ablation, "run_benchmark", lambda args: pytest.fail("paid call"))
    with pytest.raises(ValueError, match="request changed"):
        ablation.run("quality_v2")


def test_ai_external_uses_frozen_final_prompt_without_selection(panel, monkeypatch):
    old_directory, original, _ = panel
    directory = old_directory.parent / "ai_detection_v2"
    directory.mkdir()
    dataset = json.loads(json.dumps(original))
    dataset["track"] = "ai_detection_v2"
    for index, sample in enumerate(dataset["samples"]):
        sample["labels"] = {"ai_generated": bool(index % 2)}
    for index in range(2):
        path = runtime.ROOT / f"external-{index}.png"
        Image.new("RGB", (6, 7), (30, 90 + index * 40, 200)).save(path)
        dataset["samples"].append(
            {
                "id": path.stem,
                "image": path.name,
                "delivered_sha256": runtime.file_sha(path),
                "group_id": "external-pair",
                "split": "external",
                "labels": {"ai_generated": bool(index)},
            }
        )
    (directory / "dataset.json").write_text(json.dumps(dataset))
    prompt = {
        "id": "baseline",
        "task": "ai",
        "context": "Assess origin from the pixels.",
        "questions": {
            "ai_probability": {"type": "noul", "instructions": "Estimate AI origin probability."},
            "medium": {
                "type": "choice",
                "instructions": "Select medium.",
                "criteria": {"photo": "Photograph", "render": "Render"},
            },
            "uncertainty": {"type": "noul", "instructions": "Estimate uncertainty."},
        },
    }
    prompt_path = directory / "baseline.json"
    prompt_path.write_text(json.dumps(prompt))
    runtime.create_protocol("ai_detection_v2", prompt_path)
    (directory / "dev.jsonl").write_text("")
    selection = json.loads((old_directory / "prompt-selection.json").read_bytes())
    selection.update(
        protocol_sha256=runtime.file_sha(directory / "protocol.json"),
        selected_prompt_path=str(prompt_path),
        selected_prompt_sha256=runtime.file_sha(prompt_path),
    )
    selection["evaluations"]["baseline"].update(
        prompt_path=str(prompt_path),
        prompt_sha256=runtime.file_sha(prompt_path),
        ledger_path=str(directory / "dev.jsonl"),
        ledger_sha256=runtime.file_sha(directory / "dev.jsonl"),
    )
    (directory / "prompt-selection.json").write_text(json.dumps(selection))
    (directory / "optimisation").mkdir()
    (directory / "optimisation/summary.json").write_bytes(
        (old_directory / "optimisation/summary.json").read_bytes()
    )
    ablation.freeze("ai_detection_v2")
    for split, filename in (("test", "baseline.jsonl"), ("external", "external-baseline.jsonl")):
        write_rows(
            directory / "final" / filename,
            records(dataset, prompt, [s for s in dataset["samples"] if s["split"] == split]),
        )
    calls = []
    monkeypatch.setattr(ablation, "run_benchmark", fake_runner(calls))
    ablation.run("ai_detection_v2")
    result = ablation.report("ai_detection_v2", 5)
    assert calls == ["test-0", "test-1", "external-0", "external-1"]
    assert result["panels"]["external"]["paired_comparison"]["n_groups"] == 1
    assert result["spending"]["metadata_calls"] == 4
    assert result["spending"]["total_usd"] == 0.08


def test_metadata_markdown_has_complete_panel_tables_and_rounded_intervals():
    summary = {
        "expected": 2,
        "objective": {"score": 0.75},
        "metrics": {"brier_valid": 0.25, "auroc_valid": 0.8, "valid_only": {"balanced_accuracy": 0.5}},
        "group_bootstrap": {
            "intervals": {"balanced_accuracy_valid": {"percentile95": [0.12345678, 0.98765432]}}
        },
        "operations": {
            "valid": 2,
            "effective_provider_inference_cost_usd": 0.01234567,
            "latency_p50_seconds": 1.234567,
            "latency_p95_seconds": 2.345678,
        },
    }
    panel = {
        "evaluations": {"image_only": {"summary": summary}, "pixel_stats": {"summary": summary}},
        "paired_comparison": {
            "selected_minus_baseline": 0.01,
            "paired_group_bootstrap95": [-0.12345678, 0.98765432],
        },
    }
    text = ablation._report_markdown(
        {
            "track": "ai_detection_v2",
            "panels": {"test": panel, "external": panel},
            "spending": {"total_usd": 0.1},
        }
    )
    assert text.count("| Arm |") == 2
    for title in ("## Test panel", "## External panel"):
        section = text.split(title, 1)[1]
        assert section.startswith("\n\n| Arm |")
        table = section.split("\n\n", 2)[1].splitlines()
        assert len(table) == 4  # Header, separator and both contiguous arm rows.
        assert all(line.startswith("|") and line.endswith("|") for line in table)
        assert all(len(line.split("|")) == 10 for line in table)
    assert "[0.1235, 0.9877]" in text
    assert "[-0.1235, 0.9877]" in text
    assert "0.012346" in text
    assert "1.2346 / 2.3457" in text
    assert "Reported arm cost USD" in text
    assert "Client latency p50 / p95 s" in text
