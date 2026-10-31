"""Offline tests for paid-call admission, independent routes, and immutable replay."""

import base64
import copy
import importlib.util
import json
import threading
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jr1sAAAAASUVORK5CYII="
)
KEY = "offline-secret-do-not-persist"


def envelope(model, scores=("3.90", "4.10"), amount="0.001", provider="Anthropic"):
    return {
        "model": model,
        "provider": provider,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            task: {
                                "task": task,
                                "status": "unscorable" if score is None else "ok",
                                "score": score,
                                "evidence": "Visible detail supports this rating.",
                                "limitations": [],
                            }
                            for task, score in zip(("technical_quality", "aesthetics"), scores, strict=True)
                        }
                    )
                },
            }
        ],
        "usage": {"cost": amount, "runinfra": {"cost_microcents": int(Decimal(amount) * 100_000_000)}},
    }


@pytest.fixture
def study(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ablation_runner_tests", HERE / "runner.py")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    monkeypatch.setattr(r, "DATA", tmp_path / "data")
    config = json.loads((HERE.parent / "dual-score-trial-2026-10-27/routes.json").read_text())
    config["haiku"] = copy.deepcopy(config["luna"])
    config["haiku"].update(
        requested_model="anthropic/claude-haiku-5.5",
        accepted_models=["anthropic/claude-haiku-5.5", "anthropic/claude-haiku-5.5-20261015"],
        accepted_providers=["Anthropic"],
    )
    config["haiku"]["settings"].update(
        model="anthropic/claude-haiku-5.5", reasoning={"effort": "low", "exclude": True}
    )
    config["haiku"]["settings"]["provider"]["only"] = ["anthropic"]
    source = tmp_path / "method"
    source.mkdir()
    (source / "preflight-v1").mkdir()
    (source / "preflight-v1/archive.json").write_text(
        json.dumps({"routes": {route: {"admission_debit_usd": "0.01"} for route in config}})
    )
    (source / "routes.json").write_text(json.dumps(config))
    for name in ("runner.py", "wire.py", "test_runner.py", "test_wire.py", "arms.json"):
        (source / name).write_bytes((HERE / name).read_bytes())
    for name in (
        "protocol.md",
        "report.py",
        "test_report.py",
        "freeze_references.py",
        "review-instructions.md",
        "collage-selection.json",
    ):
        (source / name).write_text("Frozen test input\n")
    monkeypatch.setattr(r, "HERE", source)
    for key_name in ("RUN_INFRA_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(key_name, KEY)
    image = tmp_path / "image.png"
    image.write_bytes(PNG)
    pin = source / "protocol.md"
    responses, calls = [], []
    guard = threading.Lock()

    def response(body=None, status=200):
        return SimpleNamespace(
            status_code=status,
            headers={},
            text=json.dumps(body or envelope(config["deepseek"]["settings"]["model"])),
        )

    def post(url, **kwargs):
        with guard:
            calls.append((url, kwargs))
            assert responses, "Unexpected provider call"
            result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(r.requests, "post", post)

    def make_run(count=6, phase="images", route="deepseek", concurrency=4):
        route_config = r.route_entry(route)
        run = {
            "route": route,
            "route_config": route_config,
            "endpoint": route_config["endpoint"],
            "phase": phase,
            "prompts": {
                arm["id"]: r.prompt_bundle(r.PROJECT / "prompts/visual-rating-v1", arm)
                for arm in r.load_arms(source / "arms.json")
            },
            "settings": route_config["settings"],
            "previous_cost_usd": "0.002",
            "policy": {"max_concurrency": concurrency},
            "budget": {"max_cost_usd": "5.00", "reserve_per_request_usd": "0.20"},
            "inputs": {str(pin): r.file_sha256(pin)},
            "cases": [],
        }
        for i in range(count):
            arm = r.ARM_IDS[i % 6] if phase == "images" else "baseline"
            case_id = f"C{i // 6:03}" if phase == "images" else f"probe-{i + 1}"
            case = {
                "case_id": case_id,
                "arm_id": arm,
                "attempt_id": arm + "--" + case_id,
                "image_id": "image-1",
                "view_id": "view-1",
                "image_path": str(image),
                "source_sha256": "2" * 64,
                "view_sha256": r.file_sha256(image),
                "photo_domain_eligible": True,
            }
            if phase == "probes":
                case["expected_scores"] = {
                    "technical_quality": r.PROBE_SCORES[i],
                    "aesthetics": r.PROBE_SCORES[-1 - i],
                }
            body = r.payload(run, case)
            case.update(request_sha256=r.digest(body), request_bytes=len(r.encoded(body)))
            run["cases"].append(case)
        run["run_id"] = r.digest(run)
        r.durable_json(r.directory(run) / "run.json", run)
        return run

    return SimpleNamespace(
        r=r,
        image=image,
        pin=pin,
        source=source,
        calls=calls,
        responses=responses,
        response=response,
        make_run=make_run,
    )


def test_success_is_journalled_before_send_and_resume_is_noop(study, monkeypatch):
    r, run = study.r, study.make_run()
    real_post = r.requests.post

    def check_journal(url, **kwargs):
        folders = list((r.directory(run) / "attempts").iterdir())
        assert all((folder / "request.json").exists() for folder in folders)
        return real_post(url, **kwargs)

    monkeypatch.setattr(r.requests, "post", check_journal)
    study.responses.extend(study.response() for _ in run["cases"])
    report = r.execute(r.read_run("deepseek"))
    assert report["status"] == "complete"
    assert report["attempted"] == 6
    rows = r.read_jsonl(r.directory(run) / "observations.jsonl")
    assert {row["arm_id"] for row in rows} == set(r.ARM_IDS)
    assert {row["case_id"] for row in rows} == {"C000"}
    assert len({row["provenance"]["prompt_sha256"] for row in rows}) == 6
    assert all(not any(row["task_masks"].values()) for row in rows)
    assert r.verify(run) == r.execute(run) == report
    assert len(study.calls) == 6
    for _, kwargs in study.calls:
        sent = json.loads(kwargs["data"])
        assert base64.b64decode(sent["messages"][1]["content"][1]["image_url"]["url"].split(",", 1)[1]) == PNG
        assert kwargs["allow_redirects"] is False


def test_invalid_task_is_kept_separate_and_not_repaired(study):
    r, run = study.r, study.make_run(count=1)
    study.responses.append(study.response(envelope(run["settings"]["model"], (3.987, "4.10"))))
    report = r.execute(run)
    assert report["status"] == "complete"
    assert report["task_valid_observations"] == {"technical_quality": 0, "aesthetics": 1}
    row = r.read_jsonl(r.directory(run) / "observations.jsonl")[0]
    assert row["technical_quality"] == {"status": "invalid", "score": None}
    assert row["aesthetics"]["score"] == 4.1


@pytest.mark.parametrize("failure", ["transport", "model", "provider", "cost", "http", "length"])
def test_failure_drains_admitted_batch_and_never_retries(study, failure):
    r, run = study.r, study.make_run(route="haiku", concurrency=2)
    body = envelope(run["settings"]["model"])
    status = 200
    if failure == "model":
        body["model"] = "another-model"
    elif failure == "provider":
        body["provider"] = "Another provider"
    elif failure == "cost":
        body["usage"] = {"cost": 0, "is_byok": True}
    elif failure == "http":
        status = 429
    elif failure == "length":
        body["choices"][0]["finish_reason"] = "length"
    study.responses.extend(
        [
            r.requests.Timeout(KEY) if failure == "transport" else study.response(body, status),
            study.response(envelope(run["settings"]["model"])),
        ]
    )
    report = r.execute(run)
    assert report["status"] == "blocked"
    assert report["attempted"] == len(study.calls) == 2
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    assert len(study.calls) == 2
    assert r.verify(run) == report
    for path in (r.directory(run) / "attempts").rglob("*.json"):
        assert KEY not in path.read_text()


def test_interrupted_attempt_cannot_be_resubmitted(study):
    r, run = study.r, study.make_run()
    case = run["cases"][0]
    journal = {k: case[k] for k in ("case_id", "arm_id", "attempt_id", "request_sha256", "request_bytes")}
    r.durable_json(
        r.directory(run) / "attempts" / case["attempt_id"] / "request.json",
        {**journal, "run_id": run["run_id"], "created_at": r.now()},
    )
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    assert study.calls == []
    assert r.verify(run)["attempts"][0]["reason"] == "interrupted_request"


def test_budget_shrinks_batch_and_includes_prior_probe_cost(study):
    r, run = study.r, study.make_run(count=3)
    run["budget"]["reserve_per_request_usd"] = "2.0"
    run["previous_cost_usd"] = "2.99"
    run["run_id"] = r.digest({k: v for k, v in run.items() if k != "run_id"})
    study.responses.append(study.response(envelope(run["settings"]["model"], amount="0.02")))
    with pytest.raises(ValueError, match="admission budget"):
        r.execute(run)
    assert len(study.calls) == 1
    assert r.verify(run)["total_known_cost_usd"] == "3.01"


def test_edited_method_image_and_exports_fail_verification(study):
    r, run = study.r, study.make_run(count=1)
    study.responses.append(study.response())
    r.execute(run)
    study.pin.write_text("changed")
    for operation in (r.execute, r.verify, r.reconstruct):
        with pytest.raises(ValueError, match="Pinned input changed"):
            operation(run)
    study.pin.write_text("Frozen test input\n")
    study.image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen image bytes changed"):
        r.verify(run)
    study.image.write_bytes(PNG)
    r.write_jsonl(r.directory(run) / "observations.jsonl", [])
    with pytest.raises(ValueError, match="preserved replies"):
        r.verify(run)


def test_five_probes_cover_extremes_null_and_value_mismatch(study):
    r, run = study.r, study.make_run(count=5, phase="probes")
    study.responses.extend(
        study.response(envelope(run["settings"]["model"], tuple(c["expected_scores"].values())))
        for c in run["cases"]
    )
    assert r.execute(run)["valid_observations"] == 5
    other = study.make_run(count=5, phase="probes", route="haiku")
    study.responses.append(study.response(envelope(other["settings"]["model"])))
    report = r.execute(other)
    assert report["status"] == "blocked"
    assert report["attempted"] == 1
    assert report["attempts"][0]["reason"] == "conformance_value_mismatch"


def test_order_has_each_case_arm_once_and_varies_by_route(study):
    r = study.r
    cases = [{"case_id": f"N{i:03}"} for i in range(50)]
    ordered = r.interleave(cases, "deepseek")
    assert ordered == r.interleave(cases, "deepseek")
    assert ordered != r.interleave(cases, "luna")
    assert len({c["attempt_id"] for c in ordered}) == 300
    for index in range(0, 300, 6):
        block = ordered[index : index + 6]
        assert len({c["case_id"] for c in block}) == 1
        assert {c["arm_id"] for c in block} == set(r.ARM_IDS)


@pytest.mark.parametrize("route", ["deepseek", "gemini", "luna", "haiku"])
def test_probe_freeze_pins_methods_routes_arms_and_rejects_rewrite(study, route):
    r = study.r
    run = r.freeze(route, "probes")
    assert r.read_run(route, "probes") == run
    assert run["budget"]["max_cost_usd"] == "4.99"
    assert run["budget"]["prior_preflight_admission_debit_usd"] == "0.01"
    assert run["policy"]["max_concurrency"] == 4
    assert {Path(p).name for p in run["inputs"]} >= {
        "runner.py",
        "wire.py",
        "report.py",
        "test_report.py",
        "protocol.md",
        "arms.json",
        "routes.json",
        "freeze_references.py",
    }
    with pytest.raises(ValueError, match="already frozen"):
        r.freeze(route, "probes")


def test_byok_and_ambiguous_billing_are_explicit(study):
    r = study.r
    body = {"usage": {"cost": "0.02", "is_byok": True, "cost_details": {"upstream_inference_cost": "0.50"}}}
    result = r.billing(body, "openrouter")
    assert result["cost_usd"] == "0.52"
    assert result["gateway_cost_usd"] == "0.02"
    del body["usage"]["is_byok"]
    assert "unconfirmed" in r.billing(body, "openrouter")["cost_source"]
    assert r.billing(body, "openrouter")["cost_usd"] == "0.52"
    body["usage"]["is_byok"] = True
    body["usage"]["cost_details"] = {}
    assert r.billing(body, "openrouter")["cost_usd"] is None


def test_haiku_alias_and_low_reasoning_are_preserved(study):
    r, run = study.r, study.make_run(count=1, route="haiku")
    study.responses.append(study.response(envelope(run["route_config"]["accepted_models"][-1])))
    assert r.execute(run)["status"] == "complete"
    sent = json.loads(study.calls[0][1]["data"])
    assert sent["reasoning"] == {"effort": "low", "exclude": True}
    assert sent["provider"]["only"] == ["anthropic"]
    assert "temperature" not in sent


def test_at_most_four_calls_are_in_flight(study, monkeypatch):
    r, run = study.r, study.make_run(count=12)
    barrier, guard = threading.Barrier(4, timeout=10), threading.Lock()
    active, maximum = 0, 0

    def post(_url, **_kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        barrier.wait()
        with guard:
            active -= 1
        return study.response()

    monkeypatch.setattr(r.requests, "post", post)
    assert r.execute(run)["status"] == "complete"
    assert maximum == 4


@pytest.fixture
def frozen_references(study):
    r = study.r
    cases = [
        {
            "case_id": f"N{i:03}",
            "image_path": str(study.image),
            "view_sha256": r.file_sha256(study.image),
            "image_id": f"image-{i}",
            "view_id": f"view-{i}",
            "source_sha256": "2" * 64,
            "photo_domain_eligible": True,
        }
        for i in range(50)
    ]
    rows = [
        {
            "case_id": c["case_id"],
            "view_sha256": c["view_sha256"],
            **{
                task: {"score": 3.9, "interval": [3.6, 4.2], "evidence": "Important detail remains clear."}
                for task in r.TASKS
            },
        }
        for c in cases
    ]
    manifest, references = study.source / "manifest.json", study.source / "references.jsonl"
    r.atomic_json(manifest, {"cases": cases, "inputs": {}})
    r.write_jsonl(references, rows)
    reviewer = study.source / "reviewer.jsonl"
    r.write_jsonl(reviewer, rows)
    frozen = {
        "count": 50,
        "human_gold": False,
        "reference_kind": "frozen_assistant_judgments",
        "references_sha256": r.file_sha256(references),
        "manifest_sha256": r.file_sha256(manifest),
        "reviewer_files": {str(reviewer): r.file_sha256(reviewer)},
        "review_instructions_sha256": r.file_sha256(study.source / "review-instructions.md"),
        "blinding": {
            "provider_image_requests_before_freeze": 0,
            "reviewers_received_provider_outputs": False,
        },
    }
    r.atomic_json(references.with_name("references-freeze.json"), frozen)
    return manifest, references, frozen


def test_image_freeze_requires_complete_conformance_and_all_blind_references(study, frozen_references):
    r = study.r
    manifest, references, frozen = frozen_references
    assert len(r.validate_references(manifest, references)) == 50
    probes = r.freeze("deepseek", "probes")
    r.save_report(probes)
    with pytest.raises(ValueError, match="five conformance probes"):
        r.freeze("deepseek", "images", manifest, references)
    study.responses.extend(
        study.response(envelope(probes["settings"]["model"], tuple(c["expected_scores"].values())))
        for c in probes["cases"]
    )
    assert r.execute(probes)["status"] == "complete"
    run = r.freeze("deepseek", "images", manifest, references)
    assert len(run["cases"]) == 300
    assert set(run["prompts"]) == set(r.ARM_IDS)
    assert run["previous_cost_usd"] == "0.005"
    assert r.read_run("deepseek") == run
    frozen["blinding"]["provider_image_requests_before_freeze"] = 1
    r.atomic_json(references.with_name("references-freeze.json"), frozen)
    with pytest.raises(ValueError, match="blinding"):
        r.validate_references(manifest, references)
    with pytest.raises(ValueError, match="Pinned input changed"):
        r.read_run("deepseek")


def test_reference_view_mismatch_cannot_pass_even_with_updated_file_hash(study, frozen_references):
    r = study.r
    manifest, references, frozen = frozen_references
    rows = r.read_jsonl(references)
    rows[0]["view_sha256"] = "f" * 64
    r.write_jsonl(references, rows)
    frozen["references_sha256"] = r.file_sha256(references)
    r.atomic_json(references.with_name("references-freeze.json"), frozen)
    with pytest.raises(ValueError, match="delivered view differ"):
        r.validate_references(manifest, references)
    assert study.calls == []


def test_each_batch_checks_pins_without_reencoding_every_image(study, monkeypatch):
    r, run = study.r, study.make_run(count=12)
    real_payload, count = r.payload, 0

    def counted_payload(*args):
        nonlocal count
        count += 1
        return real_payload(*args)

    monkeypatch.setattr(r, "payload", counted_payload)
    study.responses.extend(study.response() for _ in run["cases"])
    assert r.execute(run)["status"] == "complete"
    assert count == 24  # full validation once, then one verification per admitted send
    r.verify(run)
    assert count == 36  # independent read-only verification always checks every payload


def test_cli_failure_in_one_route_does_not_skip_other_routes(study, monkeypatch, capsys):
    r = study.r
    seen = []
    monkeypatch.setattr(r.sys, "argv", ["runner.py", "verify", "--route", "all"])
    monkeypatch.setattr(r, "read_run", lambda route, phase: {"route": route, "phase": phase})

    def verify_route(run):
        seen.append(run["route"])
        if run["route"] == "deepseek":
            raise ValueError("One failed route")
        return {**run, "status": "complete"}

    monkeypatch.setattr(r, "verify", verify_route)
    with pytest.raises(SystemExit) as stopped:
        r.main()
    assert stopped.value.code == 1
    assert set(seen) == set(r.ROUTES)
    output = capsys.readouterr().out
    assert output.count('"status": "complete"') == 3
    assert output.count('"status": "error"') == 1
    assert study.calls == []
