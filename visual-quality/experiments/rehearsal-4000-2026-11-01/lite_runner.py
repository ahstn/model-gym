"""Resumable, measured Q-ReAlign Lite requests to the existing authenticated Runpod scorer.

The server and scoring method are unchanged from the 72-image trial. Bounded
client concurrency overlaps proxy/upload overhead; GPU inference remains serial.
Every HTTP attempt is durable, including failures. Ambiguous attempts are never
automatically resent. Lifecycle operations belong to the Runpod supervisor.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

import requests

MODEL = "q-future/Q-ReAlign-Lite-4B"
REVISION = "06add46a6bc4a7dfcf0a59a2fb1ed15a6e3d30b7"
TASKS = {"technical_score": "quality", "aesthetic_score": "aesthetics"}
WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.0)
CLIENT_VERSION = "rehearsal-lite-v2"
DEFAULT_URL = "https://j8qvat8tmzyhgr-8888.proxy.runpod.net"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir(path: Path) -> None:
    """Persist each new directory and its parent entry before a request can start."""
    if not path.is_dir():
        durable_mkdir(path.parent)
        path.mkdir(exist_ok=True)
    fsync_directory(path)
    fsync_directory(path.parent)


def write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    durable_mkdir(path.parent)
    if exclusive:
        with path.open("x") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(path.parent)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric value")
    return float(value)


def validate_task(task: dict) -> tuple[float, float]:
    native = finite_number(task["native_0_1"])
    score = finite_number(task["score_1_5"])
    probabilities = [finite_number(value) for value in task["probabilities"]]
    if len(probabilities) != 5 or any(not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Invalid five-level probabilities")
    if abs(sum(probabilities) - 1) > 1e-5:
        raise ValueError("Level probabilities do not sum to one")
    if not 0 <= native <= 1 or not 1 <= score <= 5:
        raise ValueError("Score is outside the native or normalized range")
    expected = sum(probability * weight for probability, weight in zip(probabilities, WEIGHTS))
    if abs(native - expected) > 1e-5 or abs(score - (1 + 4 * native)) > 1e-5:
        raise ValueError("Score is inconsistent with the level distribution or scale mapping")
    tokens = task["input_tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError("Invalid input token count")
    return score, native


def normalize_response(response: dict) -> dict:
    result = {"task_valid": {}, "validation_errors": {}, "native_model_0_1": {}}
    for output_name, source_name in TASKS.items():
        try:
            score, native = validate_task(response[source_name])
        except (KeyError, TypeError, ValueError) as error:
            result[output_name] = None
            result[output_name + "_native"] = None
            result["task_valid"][source_name] = False
            result["validation_errors"][source_name] = f"{type(error).__name__}: {error}"
        else:
            result[output_name] = score
            result[output_name + "_native"] = score  # No policy adjustment is applied.
            result["native_model_0_1"][source_name] = native
            result["task_valid"][source_name] = True
    valid_count = sum(result["task_valid"].values())
    result["status"] = {0: "invalid", 1: "partial", 2: "valid"}[valid_count]
    return result


def is_benchmark(case: dict) -> bool:
    return case.get("benchmark_partition") not in (None, "", "rehearsal", "nonbenchmark")


def select_cases(cases: list[dict], partition: str, case_ids: set[str] | None = None) -> list[dict]:
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case identifiers")
    if partition != "all":
        cases = [case for case in cases if is_benchmark(case) == (partition == "benchmark")]
    if case_ids is not None:
        available = {case["case_id"] for case in cases}
        if not case_ids <= available:
            raise ValueError("Requested case IDs are absent from the selected partition")
        cases = [case for case in cases if case["case_id"] in case_ids]
    return sorted(cases, key=lambda case: digest_bytes(("lite-order-v1:" + case["case_id"]).encode()))


def reference_admission(
    cases: list[dict], freeze_path: Path | None, manifest_path: Path | None = None
) -> dict:
    if not any(is_benchmark(case) for case in cases):
        return {"required": False}
    if freeze_path is None:
        raise ValueError("Benchmark requests require a frozen independent reference receipt")
    if manifest_path is None:
        raise ValueError("Benchmark admission requires the full corpus manifest")
    manifest_bytes = manifest_path.read_bytes()
    full_cases = json.loads(manifest_bytes)["cases"]
    by_id = {case["case_id"]: case for case in full_cases}
    benchmark = {identity: case for identity, case in by_id.items() if is_benchmark(case)}
    families = {case["family_id"] for case in benchmark.values()}
    partitions = {
        part: sum(case["benchmark_partition"] == part for case in benchmark.values())
        for part in ("development", "representative", "challenge")
    }
    if (
        len(full_cases) != 4000
        or len(by_id) != 4000
        or len({case["view_sha256"] for case in full_cases}) != 4000
        or len(benchmark) != 500
        or len(families) != 500
        or partitions != {"development": 100, "representative": 300, "challenge": 100}
        or any(case["family_id"] in families for case in full_cases if not is_benchmark(case))
    ):
        raise ValueError("Full corpus must contain exactly 500 distinct benchmark families")
    for case in cases:
        frozen = by_id.get(case["case_id"])
        if frozen is None or any(
            case[key] != frozen[key] for key in ("view_sha256", "family_id", "benchmark_partition")
        ):
            raise ValueError("Selected case differs from the full corpus manifest")
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes)
    manifest_sha256 = digest_bytes(manifest_bytes)
    if freeze.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Reference freeze belongs to a different corpus manifest")
    frozen_at = datetime.fromisoformat(freeze["frozen_at"])
    if frozen_at.tzinfo is None or frozen_at > datetime.now(UTC):
        raise ValueError("Invalid reference freeze timestamp")
    reference_path = Path(freeze["references_path"])
    if not reference_path.is_absolute():
        reference_path = freeze_path.parent / reference_path
    reference_bytes = reference_path.read_bytes()
    if digest_bytes(reference_bytes) != freeze["references_sha256"]:
        raise ValueError("Independent reference file changed since freeze")
    rows = [json.loads(line) for line in reference_bytes.splitlines() if line]
    recorded = [row["case_id"] for row in rows]
    if (
        len(recorded) != 500
        or set(recorded) != set(benchmark)
        or len(freeze["case_ids"]) != 500
        or set(freeze["case_ids"]) != set(benchmark)
    ):
        raise ValueError("All 500 matching benchmark references must be frozen before any subset request")
    if any(row["view_sha256"] != benchmark[row["case_id"]]["view_sha256"] for row in rows):
        raise ValueError("Frozen reference image differs from the full corpus manifest")
    return {
        "required": True,
        "freeze_sha256": digest_bytes(freeze_bytes),
        "references_sha256": freeze["references_sha256"],
        "frozen_at": freeze["frozen_at"],
        "manifest_sha256": manifest_sha256,
        "benchmark_cases": len(benchmark),
        "benchmark_families": len(families),
    }


def prepare_attempt(case: dict, run_id: str, directory: Path) -> tuple[dict, bytes]:
    image = Path(case["image_path"]).read_bytes()
    image_digest = digest_bytes(image)
    if image_digest != case["view_sha256"]:
        raise ValueError(f"Delivered image hash changed for {case['case_id']}")
    payload = canonical({"image": base64.b64encode(image).decode("ascii")})
    if len(payload) > 10_000_000:
        raise ValueError("Payload exceeds the unchanged server request limit")
    attempt_id = digest_bytes(canonical([run_id, case["case_id"], image_digest]))
    record = {
        "attempt_id": attempt_id,
        "run_id": run_id,
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "benchmark_partition": case.get("benchmark_partition"),
        "model": "qrealign_lite",
        "checkpoint": MODEL,
        "revision": REVISION,
        "arm": "baseline",
        "view_sha256": image_digest,
        "request_sha256": digest_bytes(payload),
        "request_bytes": len(payload),
        "price_usd": "0.00",
        "price_basis": "user_assigned_zero; infrastructure accounted separately",
        "scope_policy": "native outputs retained; no eligibility or image-stats floor",
        "raw_response_path": str(directory / "raw" / (attempt_id + ".json")),
        "status": "reserved",
    }
    write_json(directory / "attempts" / (case["case_id"] + ".json"), record, exclusive=True)
    return record, payload


def send_attempt(record: dict, payload: bytes, directory: Path, url: str, token: str) -> dict:
    request = requests.Request(
        "POST",
        url + "/score",
        data=payload,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    ).prepare()
    attempt_path = directory / "attempts" / (record["case_id"] + ".json")
    record.update(status="in_flight", request_started_at=utc_now())
    write_json(attempt_path, record)
    record["request_started_at"] = utc_now()
    started = time.perf_counter()
    try:
        # No adapter retries. Preparing/encoding the image is outside request latency.
        with requests.Session() as session:
            response = session.send(request, timeout=(30, 300))
        elapsed = time.perf_counter() - started
        record.update(
            latency_seconds=elapsed, response_received_at=utc_now(), http_status=response.status_code
        )
        raw = {
            "status": response.status_code,
            "body": response.text,
            "headers": {
                key: value
                for key, value in response.headers.items()
                if key.lower() in {"content-type", "content-length", "date", "x-request-id"}
            },
        }
        write_json(Path(record["raw_response_path"]), raw, exclusive=True)
        if response.status_code != 200:
            record.update(status="http_error", error=f"HTTP {response.status_code}")
        else:
            parsed = response.json()
            record.update(normalize_response(parsed))
            record["server_processing_seconds"] = parsed.get("seconds")
            record["peak_vram_bytes"] = parsed.get("peak_vram")
    except (requests.RequestException, OSError, ValueError, TypeError, KeyError) as error:
        record.update(
            latency_seconds=time.perf_counter() - started,
            response_received_at=record.get("response_received_at"),
            failure_observed_at=utc_now(),
            status="transport_or_parse_error",
            # Exception text can contain a request URL, but never Authorization headers or the payload.
            error=f"{type(error).__name__}: {str(error)[:300]}",
        )
    write_json(attempt_path, record)
    return record


def resume_record(path: Path, case: dict, run_id: str) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["run_id"] != run_id or record["view_sha256"] != case["view_sha256"]:
        raise ValueError("Existing request belongs to different frozen inputs")
    if record["status"] not in ("valid", "partial", "invalid"):
        raise ValueError(f"Unresolved attempt for {case['case_id']}; inspect before any new request")
    if not Path(record["raw_response_path"]).is_file():
        raise ValueError("Completed attempt is missing its raw response")
    return record


def execute(cases: list[dict], *, directory: Path, config: dict, token: str, max_seconds: float) -> dict:
    run_id = digest_bytes(canonical(config))
    durable_mkdir(directory)
    run_file = directory / "run.json"
    if run_file.exists():
        if json.loads(run_file.read_text())["configuration"] != config:
            raise ValueError("Runner configuration changed; use a distinct run directory")
    else:
        write_json(
            run_file, {"run_id": run_id, "configuration": config, "created_at": utc_now()}, exclusive=True
        )
    completed, pending = [], []
    for case in cases:
        previous = resume_record(directory / "attempts" / (case["case_id"] + ".json"), case, run_id)
        (completed if previous else pending).append(previous or case)
    started = time.monotonic()
    failed = False
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
        futures = {}
        position = 0
        while futures or (position < len(pending) and not failed):
            while not failed and position < len(pending) and len(futures) < config["concurrency"]:
                if time.monotonic() - started >= max_seconds:
                    failed = True
                    break
                case = pending[position]
                position += 1
                record, payload = prepare_attempt(case, run_id, directory)
                future = pool.submit(send_attempt, record, payload, directory, config["url"], token)
                futures[future] = case["case_id"]
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                record = future.result()
                completed.append(record)
                failed |= record["status"] not in ("valid", "partial", "invalid")
                print(
                    json.dumps(
                        {
                            "completed": len(completed),
                            "total": len(cases),
                            "case_id": record["case_id"],
                            "status": record["status"],
                            "latency_seconds": round(record["latency_seconds"], 3),
                        }
                    ),
                    flush=True,
                )
    summary = {
        "run_id": run_id,
        "selected_cases": len(cases),
        "completed_cases": len(completed),
        "valid_cases": sum(record["status"] == "valid" for record in completed),
        "status": "complete" if len(completed) == len(cases) and not failed else "needs_review",
        "updated_at": utc_now(),
        "invocation_seconds": time.monotonic() - started,
        "reported_price_usd": "0.00",
        "price_basis": "user_assigned_zero",
    }
    write_json(directory / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--token-file", type=Path, default=Path("/tmp/qrealign-lite-token"))
    parser.add_argument("--partition", choices=("all", "rehearsal", "benchmark"), default="all")
    parser.add_argument("--case-ids", help="Comma-separated subset, for a separate smoke run")
    parser.add_argument(
        "--references-freeze", type=Path, help="JSON receipt required before any benchmark request"
    )
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-seconds", type=float, default=8 * 3600)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    cases = select_cases(
        manifest["cases"], args.partition, set(args.case_ids.split(",")) if args.case_ids else None
    )
    if not cases:
        parser.error("No selected cases")
    admission = reference_admission(cases, args.references_freeze, args.manifest)
    if admission["required"] and admission["manifest_sha256"] != digest_bytes(manifest_bytes):
        raise ValueError("Corpus manifest changed during benchmark admission")
    token = args.token_file.read_text().strip()
    health = requests.get(
        args.url + "/health", headers={"Authorization": "Bearer " + token}, timeout=(30, 60)
    )
    health.raise_for_status()
    identity = health.json()
    if identity.get("model") != MODEL or identity.get("revision") != REVISION:
        raise ValueError("Server identity does not match the pinned checkpoint")
    config = {
        "version": CLIENT_VERSION,
        "model": MODEL,
        "revision": REVISION,
        "manifest_sha256": digest_bytes(manifest_bytes),
        "client_sha256": digest_bytes(Path(__file__).read_bytes()),
        "case_ids": [case["case_id"] for case in cases],
        "url": args.url,
        "concurrency": args.concurrency,
        "partition": args.partition,
        "reference_admission": admission,
        "server_identity_sha256": digest_bytes(canonical(identity)),
        "latency_definition": (
            "Immediately before HTTP send through complete HTTP response; "
            "includes upload/network/proxy/queue/server"
        ),
        "policy": "No automatic retries; unchanged serial server; no eligibility floor",
    }
    durable_mkdir(args.output)
    identity_path = args.output / "server-identity.json"
    if not identity_path.exists():
        write_json(identity_path, identity, exclusive=True)
    summary = execute(cases, directory=args.output, config=config, token=token, max_seconds=args.max_seconds)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
