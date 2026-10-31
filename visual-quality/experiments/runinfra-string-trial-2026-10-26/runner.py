"""Frozen RunInfra string-score trial with journalled attempts and bounded spend."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import requests
from wire import ADAPTER_VERSION, parse_wire, prompt_bundle, score_wire_valid

from visual_quality.labels import LabelValidationError
from visual_quality.storage import atomic_json, file_sha256, read_jsonl, write_jsonl

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data/runinfra-string-trial-2026-10-26"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
SLUG = "deepseek-v4-1-flash-1017"
ENDPOINT = "https://api.runinfra.ai/v1/chat/completions"
PROBE_SCORES = ("1.00", "3.90", "4.10", "5.00", None)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def encoded(body: dict) -> bytes:
    return json.dumps(body, allow_nan=False).encode()


def directory(run: dict) -> Path:
    return DATA / run["phase"]


def payload(run: dict, case: dict) -> dict:
    if run["phase"] == "probes":
        expected = {
            "task": "technical_quality",
            "status": "unscorable" if case["expected_score"] is None else "ok",
            "score": case["expected_score"],
            "evidence": "Literal transport conformance check.",
            "limitations": [],
        }
        messages = [
            {
                "role": "system",
                "content": "Copy the supplied JSON object exactly. This is a text-only format test.",
            },
            {"role": "user", "content": json.dumps(expected)},
        ]
    else:
        raw = Path(case["image_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != case["view_sha256"]:
            raise ValueError("Frozen image bytes changed")
        messages = [
            {"role": "system", "content": run["prompt"]["system"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": run["prompt"]["user"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(raw).decode(),
                            "detail": "high",
                        },
                    },
                ],
            },
        ]
    return {**run["settings"], "messages": messages, "response_format": run["prompt"]["response_format"]}


def catalog_entry(path: Path) -> dict:
    catalog = json.loads(path.read_text())
    if not isinstance(catalog, list) or len(catalog) != 1:
        raise ValueError("Expected one exact catalog entry")
    model = catalog[0]
    if (
        model["id"] != SLUG
        or model["availability"] != "available"
        or not {"text", "image"}.issubset(model["input_modalities"])
        or not model["json_mode"]
        or "none" not in model["reasoning_efforts"]
    ):
        raise ValueError("Required model capabilities are unavailable")
    if model["pricing"] != {"input": 0.15, "output": 0.6}:
        raise ValueError("Published prices changed; review the budget")
    return model


def validate_references(manifest_path: Path, references_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    references = read_jsonl(references_path)
    freeze_path = references_path.with_name("references-freeze.json")
    frozen = json.loads(freeze_path.read_text())
    if (
        len(cases) != 200
        or len(references) != 200
        or frozen["count"] != 200
        or frozen["human_gold"] is not False
        or frozen["reference_kind"] != "frozen_assistant_judgments"
        or frozen["references_sha256"] != file_sha256(references_path)
        or frozen["manifest_sha256"] != file_sha256(manifest_path)
    ):
        raise ValueError("All 200 blinded references must be frozen before image requests")
    blinding = frozen["blinding"]
    if (
        blinding["provider_image_requests_before_freeze"] != 0
        or blinding["reviewers_received_provider_outputs"] is not False
    ):
        raise ValueError("Reference blinding is not established")
    if frozen["review_instructions_sha256"] != file_sha256(HERE / "review-instructions.md"):
        raise ValueError("Frozen review instructions changed")
    if not isinstance(frozen["reviewer_files"], dict) or not frozen["reviewer_files"]:
        raise ValueError("Frozen reviewer source files are required")
    by_id = {row["case_id"]: row for row in references}
    if len(by_id) != 200 or set(by_id) != {case["case_id"] for case in cases}:
        raise ValueError("Reference identities differ from cohort")
    for case in cases:
        row = by_id[case["case_id"]]
        if row["view_sha256"] != case["view_sha256"]:
            raise ValueError("Reference and delivered view differ")
        if not isinstance(row["evidence"], str) or not row["evidence"].strip():
            raise ValueError("A reference requires visible evidence")
        score, interval = row["score"], row["interval"]
        if score is None:
            if interval is not None or not row.get("unscorable_reason"):
                raise ValueError("Unscorable reference requires an explicit reason")
        elif (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isinstance(interval, list)
            or len(interval) != 2
            or any(isinstance(n, bool) or not isinstance(n, (int, float)) for n in interval)
            or not 1 <= interval[0] <= score <= interval[1] <= 5
            or Decimal(str(score)) % Decimal("0.1")
        ):
            raise ValueError("Reference score or interval is invalid")
    for name, expected in {**manifest.get("inputs", {}), **frozen["reviewer_files"]}.items():
        if file_sha256(Path(name)) != expected:
            raise ValueError("Frozen source or reviewer file changed")
    return cases


def freeze(
    catalog_path: Path, phase: str, manifest_path: Path | None = None, references_path: Path | None = None
) -> dict:
    target = DATA / phase
    if (target / "run.json").exists():
        raise ValueError("Run already frozen")
    model = catalog_entry(catalog_path)
    files = [HERE / name for name in ("runner.py", "wire.py", "protocol.md", "format-decision.md")]
    files += [PROJECT / "src/visual_quality/labels.py", PROJECT / "src/visual_quality/labelling.py"]
    files += list((PROJECT / "prompts/visual-rating-v1").glob("*"))
    previous_cost = "0"
    if phase == "images":
        if manifest_path is None or references_path is None:
            raise ValueError("Manifest and frozen references are required")
        cases = validate_references(manifest_path, references_path)
        probe_run = read_run("probes")
        probes = verify(probe_run)
        if probes["status"] != "complete" or probes["valid_observations"] != 5:
            raise ValueError("All five conformance probes must pass")
        previous_cost = probes["known_cost_usd"]
        files += [manifest_path, references_path, references_path.with_name("references-freeze.json")]
        files += [
            HERE / name
            for name in ("review-instructions.md", "report.py", "test_report.py", "audit-selection.json")
        ]
        frozen_references = json.loads(references_path.with_name("references-freeze.json").read_text())
        files += [Path(name) for name in frozen_references["reviewer_files"]]
        files += [p for p in (DATA / "probes").rglob("*") if p.is_file() and p.name != ".lock"]
    else:
        cases = [
            {"case_id": f"probe-{i + 1}", "expected_score": score} for i, score in enumerate(PROBE_SCORES)
        ]
    run = {
        "schema_version": "runinfra-string-trial-run-v1",
        "phase": phase,
        "created_at": now(),
        "endpoint": ENDPOINT,
        "requested_model": MODEL,
        "model_snapshot": model,
        "prompt": prompt_bundle(PROJECT / "prompts/visual-rating-v1"),
        "settings": {
            "model": SLUG,
            "max_tokens": 1024,
            "temperature": 0,
            "reasoning_effort": "none",
            "stream": False,
        },
        "policy": {
            "automatic_retries": False,
            "fallback": False,
            "max_concurrency": 4,
            "reference_kind": "frozen_assistant_judgments",
            "human_gold": False,
        },
        "budget": {
            "max_cost_usd": "1.00",
            "reserve_per_request_usd": str(
                Decimal(str(model["pricing"]["input"])) * model["context_window"] / 1_000_000
                + Decimal(str(model["pricing"]["output"])) * 1024 / 1_000_000
            ),
        },
        "previous_cost_usd": previous_cost,
        "inputs": {str(p.resolve()): file_sha256(p) for p in files},
        "cases": cases,
        "training_ready": False,
    }
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case["case_id"]):
            raise ValueError("Case identifier is unsafe")
        body = payload(run, case)
        case.update(request_sha256=digest(body), request_bytes=len(encoded(body)))
        if case["request_bytes"] > model["max_request_bytes"]:
            raise ValueError(f"Request exceeds provider limit: {case['case_id']}")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case identifier")
    run["run_id"] = digest(run)
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / "run.json", run)
    return run


def read_run(phase: str = "images") -> dict:
    run = json.loads((DATA / phase / "run.json").read_text())
    if run["phase"] != phase or digest({k: v for k, v in run.items() if k != "run_id"}) != run["run_id"]:
        raise ValueError("Frozen run metadata changed")
    for name, expected in run["inputs"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Pinned input changed: {name}")
    for case in run["cases"]:
        body = payload(run, case)
        if digest(body) != case["request_sha256"] or len(encoded(body)) != case["request_bytes"]:
            raise ValueError("Request differs from frozen inputs")
    return run


def cost(body: dict) -> Decimal | None:
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        return None
    try:
        value = usage.get("cost")
        if isinstance(value, bool) or value is None:
            return None
        amount = Decimal(str(value))
        micro = usage.get("runinfra", {}).get("cost_microcents")
        if (
            not amount.is_finite()
            or amount < 0
            or type(micro) is not int
            or micro < 0
            or Decimal(micro) / 100_000_000 != amount
        ):
            return None
        return amount
    except (ValueError, TypeError, ArithmeticError, AttributeError):
        return None


def observation(run: dict, case: dict, request: dict, body: dict, rating: dict, score: str | None) -> dict:
    record = {
        "schema_version": "runinfra-string-score-observation-v1",
        "label_kind": "teacher_scalar",
        "case_id": case["case_id"],
        "wire_score": score,
        "adapter_version": ADAPTER_VERSION,
        "technical_quality": rating,
        "aesthetics": {"status": "not_requested", "score": None},
        "review_status": "pending_human_review",
        "task_masks": {"technical_quality": False, "aesthetics": False, "photo_domain": False},
        "provenance": {
            "provider": "runinfra",
            "model_requested": run["settings"]["model"],
            "model_returned": body["model"],
            "run_id": run["run_id"],
            "attempt_id": case["case_id"],
            "prompt_version": run["prompt"]["version"],
            "prompt_sha256": run["prompt"]["sha256"],
            "request_sha256": request["request_sha256"],
            "created_at": request["created_at"],
            "usage": body.get("usage", {}),
        },
    }
    if run["phase"] == "images":
        record.update(
            {
                key: case[key]
                for key in ("image_id", "view_id", "source_sha256", "view_sha256", "photo_domain_eligible")
            }
        )
        record["delivered_sha256"] = case["view_sha256"]
    else:
        record["label_kind"] = "conformance_probe"
    return record


def interpret(run: dict, case: dict, request: dict, result: dict) -> tuple[dict, dict | None]:
    details = {
        "case_id": case["case_id"],
        "status": "failed",
        "cost_usd": None,
        "reason": None,
        "wire_score_valid": None,
    }
    if result["transport_error"]:
        details["reason"] = "transport_outcome_unknown"
        return details, None
    try:
        body = json.loads(result["body"])
        if not isinstance(body, dict):
            raise TypeError("Non-object envelope")
        amount = cost(body)
        details["cost_usd"] = str(amount) if amount is not None else None
        if result["http_status"] != 200 or body.get("error"):
            details["reason"] = "http_or_provider_error"
            return details, None
        if body.get("model") not in {MODEL, SLUG}:
            raise ValueError("Unexpected returned model")
        choices = body["choices"]
        if not isinstance(choices, list) or len(choices) != 1 or choices[0]["finish_reason"] != "stop":
            raise ValueError("Incomplete response")
        message = choices[0]["message"]
        if not isinstance(message, dict) or message.get("refusal"):
            raise ValueError("Invalid message or refusal")
        content = message["content"]
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        details["reason"] = f"invalid_envelope: {type(error).__name__}"
        return details, None
    details["wire_score_valid"] = score_wire_valid(content)
    try:
        rating, score = parse_wire(content)
    except LabelValidationError as error:
        details.update(status="invalid", reason=f"invalid_rating: {error}")
        return details, None
    if run["phase"] == "probes" and score != case["expected_score"]:
        details.update(status="invalid", reason="conformance_value_mismatch")
        return details, None
    details.update(status="valid", reason=None if amount is not None else "cost_unavailable")
    return details, observation(run, case, request, body, rating, score)


def reconstruct(run: dict) -> tuple[dict, list[dict]]:
    target = directory(run)
    observations, attempts = [], []
    attempt_root = target / "attempts"
    expected_ids = {case["case_id"] for case in run["cases"]}
    if attempt_root.exists() and not {p.name for p in attempt_root.iterdir()}.issubset(expected_ids):
        raise ValueError("Unexpected attempt directory")
    for case in run["cases"]:
        folder = attempt_root / case["case_id"]
        if not folder.exists():
            continue
        if not (folder / "request.json").exists():
            raise ValueError("Incomplete attempt directory; inspect manually")
        request = json.loads((folder / "request.json").read_text())
        expected = {key: case[key] for key in ("case_id", "request_sha256", "request_bytes")}
        expected["run_id"] = run["run_id"]
        if any(request.get(key) != value for key, value in expected.items()):
            raise ValueError("Attempt request metadata differs from frozen run")
        if not (folder / "result.json").exists():
            attempts.append(
                {
                    "case_id": case["case_id"],
                    "status": "unknown",
                    "cost_usd": None,
                    "reason": "interrupted_request",
                    "wire_score_valid": None,
                }
            )
            continue
        details, record = interpret(run, case, request, json.loads((folder / "result.json").read_text()))
        attempts.append(details)
        if record is not None:
            observations.append(record)
    stopped = any(a["status"] in {"failed", "unknown"} or a["cost_usd"] is None for a in attempts)
    if run["phase"] == "probes":
        stopped = stopped or any(a["status"] != "valid" for a in attempts)
    status = "blocked" if stopped else "complete" if len(attempts) == len(run["cases"]) else "partial"
    spend = sum((Decimal(a["cost_usd"]) for a in attempts if a["cost_usd"] is not None), Decimal(0))
    return {
        "schema_version": "runinfra-string-trial-report-v1",
        "run_id": run["run_id"],
        "phase": run["phase"],
        "status": status,
        "planned": len(run["cases"]),
        "attempted": len(attempts),
        "valid_observations": len(observations),
        "invalid_observations": sum(a["status"] == "invalid" for a in attempts),
        "scored": sum(r["technical_quality"]["score"] is not None for r in observations),
        "known_cost_usd": str(spend),
        "total_known_cost_usd": str(spend + Decimal(run["previous_cost_usd"])),
        "unpriced_attempts": sum(a["cost_usd"] is None for a in attempts),
        "attempts": attempts,
        "training_ready": False,
    }, observations


def save_report(run: dict) -> dict:
    report, observations = reconstruct(run)
    write_jsonl(directory(run) / "observations.jsonl", observations)
    atomic_json(directory(run) / "report.json", report)
    return report


def send(run: dict, case: dict, key: str) -> None:
    """One POST for an already journalled case; this function never retries."""
    body = payload(run, case)
    if digest(body) != case["request_sha256"]:
        raise ValueError("Request changed after freezing")
    try:
        response = requests.post(
            ENDPOINT,
            data=encoded(body),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            timeout=(15, 180),
            allow_redirects=False,
        )
        result = {
            "http_status": response.status_code,
            "body": response.text.replace(key, "[REDACTED]"),
            "headers": {
                k.lower(): v.replace(key, "[REDACTED]")
                for k, v in response.headers.items()
                if k.lower().startswith(("x-runinfra-", "x-request-id", "retry-after"))
            },
            "transport_error": None,
            "received_at": now(),
        }
    except requests.RequestException as error:
        result = {
            "http_status": None,
            "body": None,
            "headers": {},
            "transport_error": type(error).__name__,
            "received_at": now(),
        }
    atomic_json(directory(run) / "attempts" / case["case_id"] / "result.json", result)


def execute(run: dict) -> dict:
    key = os.environ.get("RUN_INFRA_API_KEY")
    if not key:
        raise ValueError("RUN_INFRA_API_KEY is unavailable")
    with (directory(run) / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = save_report(run)
        if report["status"] == "blocked":
            raise ValueError("Existing failed or uncertain attempts prevent further calls")
        pending = [
            case for case in run["cases"] if not (directory(run) / "attempts" / case["case_id"]).exists()
        ]
        concurrency = 1 if run["phase"] == "probes" else run["policy"]["max_concurrency"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            while pending:
                batch = pending[:concurrency]
                reserve = Decimal(run["budget"]["reserve_per_request_usd"]) * len(batch)
                if Decimal(report["total_known_cost_usd"]) + reserve > Decimal(run["budget"]["max_cost_usd"]):
                    raise ValueError("Next batch exceeds the admission budget")
                # Journal the whole admitted batch before any request. A crash fails closed on resume.
                for case in batch:
                    folder = directory(run) / "attempts" / case["case_id"]
                    folder.mkdir(parents=True)
                    request = {k: case[k] for k in ("case_id", "request_sha256", "request_bytes")}
                    request.update(run_id=run["run_id"], created_at=now())
                    atomic_json(folder / "request.json", request)
                futures = [pool.submit(send, run, case, key) for case in batch]
                # Drain every admitted call before considering any further submissions.
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                report = save_report(run)
                print(
                    json.dumps(
                        {
                            k: report[k]
                            for k in (
                                "phase",
                                "attempted",
                                "scored",
                                "invalid_observations",
                                "total_known_cost_usd",
                                "status",
                            )
                        }
                    ),
                    flush=True,
                )
                if report["status"] == "blocked":
                    break
                pending = pending[len(batch) :]
        return report


def verify(run: dict) -> dict:
    report, observations = reconstruct(run)
    if (
        json.loads((directory(run) / "report.json").read_text()) != report
        or read_jsonl(directory(run) / "observations.jsonl") != observations
    ):
        raise ValueError("Saved exports differ from preserved replies")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["freeze-probes", "probe", "verify-probes", "freeze", "run", "verify"]
    )
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--references", type=Path, default=HERE / "references.jsonl")
    args = parser.parse_args()
    phase = "probes" if args.command in {"freeze-probes", "probe", "verify-probes"} else "images"
    if args.command.startswith("freeze"):
        if args.catalog is None:
            parser.error("--catalog is required for freezing")
        run = freeze(args.catalog, phase, args.manifest, args.references)
        report = save_report(run)
    else:
        run = read_run(phase)
        report = execute(run) if args.command in {"run", "probe"} else verify(run)
    print(json.dumps({k: v for k, v in report.items() if k != "attempts"}, indent=2))


if __name__ == "__main__":
    main()
