"""Offline validation, provenance, budget, concurrency, and recovery checks."""

import base64
import importlib.util
import json
import threading
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jr1sAAAAASUVORK5CYII="
)
KEY = "offline-secret-never-persist"


def envelope(model, score="3.90", amount="0.001", **changes):
    rating = {
        "task": "technical_quality",
        "status": "unscorable" if score is None else "ok",
        "score": score,
        "evidence": "Fine detail remains clear.",
        "limitations": [],
        **changes,
    }
    return {
        "model": model,
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(rating)}}],
        "usage": {"cost": amount, "runinfra": {"cost_microcents": int(Decimal(amount) * 100_000_000)}},
    }


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    spec = importlib.util.spec_from_file_location(
        "trial200_runner_tests", Path(__file__).with_name("runner.py")
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "DATA", tmp_path / "data")
    monkeypatch.setenv("RUN_INFRA_API_KEY", KEY)
    image = tmp_path / "view.png"
    image.write_bytes(PNG)
    pin = tmp_path / "pin.txt"
    pin.write_text("immutable")
    calls, responses = [], []
    queue_lock = threading.Lock()

    def post(url, **kwargs):
        with queue_lock:
            calls.append((url, kwargs))
            assert responses, "Unexpected provider request"
            result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(runner.requests, "post", post)

    def response(body=None, status=200):
        return SimpleNamespace(
            status_code=status,
            headers={},
            text=json.dumps(body if body is not None else envelope(runner.SLUG)),
        )

    def make_run(count=5, phase="images", concurrency=4):
        run = {
            "phase": phase,
            "prompt": {
                "version": "test-v1",
                "sha256": "1" * 64,
                "system": "Rate.",
                "user": "Inspect.",
                "response_format": {"type": "json_object"},
            },
            "settings": {"model": runner.SLUG},
            "previous_cost_usd": "0.002",
            "policy": {"max_concurrency": concurrency},
            "budget": {"max_cost_usd": "1.00", "reserve_per_request_usd": "0.1579008"},
            "inputs": {str(pin): runner.file_sha256(pin)},
            "cases": [],
        }
        for i in range(count):
            case = {
                "case_id": f"C{i:03}",
                "image_id": f"image-{i}",
                "view_id": f"view-{i}",
                "image_path": str(image),
                "source_sha256": "2" * 64,
                "view_sha256": runner.file_sha256(image),
                "photo_domain_eligible": [True, False, None][i % 3],
            }
            if phase == "probes":
                case["expected_score"] = runner.PROBE_SCORES[i]
            body = runner.payload(run, case)
            case.update(request_sha256=runner.digest(body), request_bytes=len(runner.encoded(body)))
            run["cases"].append(case)
        run["run_id"] = runner.digest(run)
        runner.directory(run).mkdir(parents=True)
        runner.atomic_json(runner.directory(run) / "run.json", run)
        return run

    return SimpleNamespace(
        runner=runner,
        image=image,
        pin=pin,
        calls=calls,
        responses=responses,
        response=response,
        make_run=make_run,
    )


def test_success_preserves_wire_provenance_and_reuses_complete_run(study):
    r, run = study.runner, study.make_run()
    study.responses.extend(study.response() for _ in run["cases"])
    report = r.execute(r.read_run())
    assert report["status"] == "complete"
    assert report["attempted"] == report["scored"] == 5
    assert Decimal(report["total_known_cost_usd"]) == Decimal("0.007")
    rows = r.read_jsonl(r.directory(run) / "observations.jsonl")
    assert [row["photo_domain_eligible"] for row in rows] == [True, False, None, True, False]
    for row in rows:
        assert row["wire_score"] == "3.90"
        assert row["technical_quality"]["score"] == 3.9
        assert row["adapter_version"] == "string-score-to-decimal-v1"
        assert row["aesthetics"] == {"status": "not_requested", "score": None}
        assert row["review_status"] == "pending_human_review"
        assert all(value is False for value in row["task_masks"].values())
    for _, kwargs in study.calls:
        body = json.loads(kwargs["data"])
        image_url = body["messages"][1]["content"][1]["image_url"]["url"]
        assert base64.b64decode(image_url.split(",", 1)[1]) == PNG
        assert kwargs["allow_redirects"] is False
    assert r.verify(run) == report
    assert r.execute(run) == report
    assert len(study.calls) == 5


def test_invalid_priced_rating_continues_without_retry_or_observation(study):
    r, run = study.runner, study.make_run(concurrency=1)
    study.responses.extend(
        [
            study.response(envelope(r.SLUG, 3.8999999999999999)),
            study.response(envelope(r.SLUG, evidence="word " * 31)),
            *[study.response() for _ in range(3)],
        ]
    )
    report = r.execute(run)
    assert report["status"] == "complete"
    assert report["invalid_observations"] == 2
    assert report["valid_observations"] == 3
    assert report["attempts"][0]["wire_score_valid"] is False
    assert report["attempts"][1]["wire_score_valid"] is True
    assert "30 words" in report["attempts"][1]["reason"]
    assert len(study.calls) == 5
    assert r.verify(run) == report


@pytest.mark.parametrize(
    "failure", ["http", "model", "incomplete", "null_message", "unknown_cost", "transport"]
)
def test_uncertain_or_envelope_failure_drains_batch_then_blocks_resume(study, failure):
    r, run = study.runner, study.make_run(count=6)
    body, status = envelope(r.SLUG), 200
    if failure == "http":
        status = 429
        body["error"] = {"message": "Unavailable"}
    elif failure == "model":
        body["model"] = "different-model"
    elif failure == "incomplete":
        body["choices"][0]["finish_reason"] = "length"
    elif failure == "null_message":
        body["choices"][0]["message"] = None
    elif failure == "unknown_cost":
        body["usage"]["runinfra"]["cost_microcents"] = 1
    study.responses.append(
        r.requests.Timeout(KEY) if failure == "transport" else study.response(body, status)
    )
    study.responses.extend(study.response() for _ in range(3))
    report = r.execute(run)
    assert report["status"] == "blocked"
    assert report["attempted"] == len(study.calls) == 4
    assert all(
        (r.directory(run) / "attempts" / case["case_id"] / "result.json").exists()
        for case in run["cases"][:4]
    )
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    assert len(study.calls) == 4
    assert r.verify(run) == report
    for path in (r.directory(run) / "attempts").rglob("*.json"):
        assert KEY not in path.read_text()


def test_inflight_budget_includes_probes_and_full_batch(study):
    r, run = study.runner, study.make_run(count=4)
    run["previous_cost_usd"] = "0.40"
    with pytest.raises(ValueError, match="admission budget"):
        r.execute(run)
    assert study.calls == []
    assert not (r.directory(run) / "attempts").exists()


def test_interrupted_attempt_never_retries(study):
    r, run = study.runner, study.make_run()
    case = run["cases"][0]
    folder = r.directory(run) / "attempts" / case["case_id"]
    folder.mkdir(parents=True)
    request = {k: case[k] for k in ("case_id", "request_sha256", "request_bytes")}
    request.update(run_id=run["run_id"], created_at=r.now())
    r.atomic_json(folder / "request.json", request)
    with pytest.raises(ValueError, match="failed or uncertain"):
        r.execute(run)
    assert study.calls == []
    assert r.verify(run)["attempts"][0]["reason"] == "interrupted_request"


def test_raw_response_rebuild_detects_modified_export(study):
    r, run = study.runner, study.make_run(count=1)
    study.responses.append(study.response())
    r.execute(run)
    r.write_jsonl(r.directory(run) / "observations.jsonl", [])
    with pytest.raises(ValueError, match="preserved replies"):
        r.verify(run)


def test_read_run_detects_view_and_method_changes(study):
    r = study.runner
    study.make_run(count=1)
    study.pin.write_text("changed")
    with pytest.raises(ValueError, match="Pinned input changed"):
        r.read_run()
    study.pin.write_text("immutable")
    study.image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen image bytes changed"):
        r.read_run()


def test_conformance_all_five_and_null_pass_but_wrong_value_blocks(study):
    r, run = study.runner, study.make_run(phase="probes")
    study.responses.extend(study.response(envelope(r.SLUG, score)) for score in r.PROBE_SCORES)
    report = r.execute(run)
    assert report["status"] == "complete"
    assert report["valid_observations"] == 5
    assert report["scored"] == 4
    assert len(study.calls) == 5


def test_failed_conformance_stops_before_remaining_probes(study):
    r, run = study.runner, study.make_run(phase="probes")
    study.responses.append(study.response())
    report = r.execute(run)
    assert report["status"] == "blocked"
    assert report["attempts"][0]["reason"] == "conformance_value_mismatch"
    assert len(study.calls) == 1


def test_missing_references_block_before_any_provider_call(study, tmp_path):
    r = study.runner
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": []}))
    with pytest.raises(FileNotFoundError):
        r.validate_references(manifest, tmp_path / "missing-references.jsonl")
    assert study.calls == []


@pytest.fixture
def frozen_references(study, tmp_path):
    r = study.runner
    cases = [{"case_id": f"R{i:03}", "view_sha256": "a" * 64} for i in range(200)]
    rows = [
        {
            "case_id": row["case_id"],
            "view_sha256": row["view_sha256"],
            "score": 3.9,
            "interval": [3.6, 4.2],
            "evidence": "Fine detail is visible.",
        }
        for row in cases
    ]
    manifest, references = tmp_path / "manifest.json", tmp_path / "references.jsonl"
    r.atomic_json(manifest, {"cases": cases, "inputs": {}})
    r.write_jsonl(references, rows)
    frozen = {
        "count": 200,
        "human_gold": False,
        "reference_kind": "frozen_assistant_judgments",
        "references_sha256": r.file_sha256(references),
        "manifest_sha256": r.file_sha256(manifest),
        "reviewer_files": {str(study.pin): r.file_sha256(study.pin)},
        "review_instructions_sha256": r.file_sha256(r.HERE / "review-instructions.md"),
        "blinding": {
            "provider_image_requests_before_freeze": 0,
            "reviewers_received_provider_outputs": False,
        },
    }
    r.atomic_json(tmp_path / "references-freeze.json", frozen)
    return manifest, references, frozen


def test_reference_freeze_checks_all_case_views_and_source_hashes(study, frozen_references):
    r = study.runner
    manifest, references, frozen = frozen_references
    assert len(r.validate_references(manifest, references)) == 200
    study.pin.write_text("changed review")
    with pytest.raises(ValueError, match="reviewer file changed"):
        r.validate_references(manifest, references)
    study.pin.write_text("immutable")
    rows = r.read_jsonl(references)
    rows[0]["view_sha256"] = "b" * 64
    r.write_jsonl(references, rows)
    frozen["references_sha256"] = r.file_sha256(references)
    r.atomic_json(references.with_name("references-freeze.json"), frozen)
    with pytest.raises(ValueError, match="delivered view differ"):
        r.validate_references(manifest, references)


def test_main_freeze_requires_complete_conformance(study, frozen_references, monkeypatch):
    r = study.runner
    manifest, references, _ = frozen_references
    monkeypatch.setattr(r, "catalog_entry", lambda _: {})
    monkeypatch.setattr(r, "read_run", lambda _: {})
    monkeypatch.setattr(r, "verify", lambda _: {"status": "blocked", "valid_observations": 4})
    with pytest.raises(ValueError, match="five conformance probes"):
        r.freeze(Path("not-read"), "images", manifest, references)
    assert study.calls == []
    assert not (r.DATA / "images/run.json").exists()


def test_main_run_has_at_most_four_calls_in_flight(study, monkeypatch):
    r, run = study.runner, study.make_run(count=8)
    barrier, guard = threading.Barrier(4, timeout=5), threading.Lock()
    active, maximum, total = 0, 0, 0

    def post(_url, **_kwargs):
        nonlocal active, maximum, total
        with guard:
            active += 1
            total += 1
            maximum = max(maximum, active)
        barrier.wait()
        with guard:
            active -= 1
        return study.response()

    monkeypatch.setattr(r.requests, "post", post)
    report = r.execute(run)
    assert report["status"] == "complete"
    assert maximum == 4
    assert total == 8
