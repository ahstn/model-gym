"""Repeat frozen Jev image panels without modifying the historical worktree."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/Users/ahstn/.codex/worktrees/d187/model-gym/visual-triage")
HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/jev14-2026-10-18"
PROTOCOL = HERE / "protocol.json"
MANIFEST = HERE / "manifest.json"
LEDGER = DATA / "attempts.jsonl"
sys.path.insert(0, str(ROOT))

from bench_protocol import DECISIONS, digest
from bench_runner import effective_cost
from tracks.jev14.client import HEADERS, MODEL, REVISION, make_payload, post
from tracks.jev14.latest import review


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def freeze():
    if PROTOCOL.exists() or MANIFEST.exists():
        raise ValueError("Run is already frozen")
    catalog = Path("/private/tmp/model-gym-decisions-catalog-2026-10-18.json")
    model = next(row for row in read(catalog)["data"] if row["id"] == MODEL)
    if model["canonical_slug"] != REVISION or "image" not in model["architecture"]["input_modalities"]:
        raise ValueError("Catalog revision or modality changed")
    sources, cases = set(), []
    for panel, protocol_path, manifest_path in (
        ("shared", ROOT / "tracks/jev14/protocol.json", ROOT / "data/jev14-v1/manifest.json"),
        ("latest", ROOT / "tracks/jev14/latest/protocol.json", ROOT / "data/jev14-latest-v1/manifest.json"),
    ):
        original = read(protocol_path)
        if sha(manifest_path) != original["manifest_sha256"]:
            raise ValueError("Historical manifest changed")
        sources.update((protocol_path, manifest_path))
        for path, pin in original["source_inputs"].items():
            source = ROOT / path
            if sha(source) != pin["sha256"]:
                raise ValueError(f"Historical source changed: {path}")
            sources.add(source)
        for path, pin in original["execution_sha256"].items():
            if sha(ROOT / path) != pin:
                raise ValueError(f"Historical execution code changed: {path}")
        prompts = original.get("prompts", {task: f"prompts/final/{task}.json" for task in ("quality", "ai")})
        for case in read(manifest_path)["cases"]:
            if case["phase"] != "primary":
                continue
            case = {
                **case,
                "panel": panel,
                "row_id": panel + "/" + case["case_id"],
                "prompt_path": prompts[case["task"]],
            }
            make_payload(ROOT, case, read(ROOT / case["prompt_path"]))
            cases.append(case)
    expected = {
        ("shared", "quality"): 58,
        ("shared", "ai"): 60,
        ("latest", "quality"): 100,
        ("latest", "ai"): 160,
    }
    if Counter((case["panel"], case["task"]) for case in cases) != expected:
        raise ValueError("Unexpected panel allocation")
    for panel, directory in (
        ("shared", "shared-results-2026-10-15"),
        ("latest", "latest-results-2026-10-15"),
    ):
        sources.add(ROOT / "tracks/jev14" / directory / "summary.json")
    # Pin all local harness modules, including transitive reporting imports.
    sources.update(ROOT.glob("*.py"))
    sources.update((ROOT / "tracks").rglob("*.py"))
    sources.update(ROOT / case["prompt_path"] for case in cases)
    DATA.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(catalog, DATA / "catalog.json")
    write_new(MANIFEST, {"cases": cases})
    write_new(
        PROTOCOL,
        {
            "frozen_at": datetime.now(UTC).isoformat(),
            "model": MODEL,
            "expected_revision": REVISION,
            "baseline_root": str(ROOT),
            "endpoint": DECISIONS,
            "request_headers": HEADERS,
            "maximum_attempts": len(cases),
            "budget_usd": 0.10,
            "reserve_per_request_usd": 0.012,
            "minimum_interval_seconds": 1,
            "manifest_sha256": sha(MANIFEST),
            "runner_sha256": sha(Path(__file__)),
            "catalog_sha256": sha(DATA / "catalog.json"),
            "catalog_model": model,
            "source_inputs": {str(path.relative_to(ROOT)): sha(path) for path in sorted(sources)},
            "comparison": "Same frozen image bytes and prompts. Historical Decider and Luna responses; separate panels.",
            "retry_policy": "No retries. Stop on HTTP, transport, validation or unknown-cost failure.",
            "parsing": "Existing bounded two-decimal rounding; preserve native values.",
            "order": "Original primary case order, shared panel followed by latest panel.",
            "exclusions": "Two incompatible shared panoramas and six development size probes are not rerun.",
        },
    )
    return {"status": "frozen", "planned": len(cases), "budget_usd": 0.10}


def load_study():
    protocol = read(PROTOCOL)
    if sha(MANIFEST) != protocol["manifest_sha256"] or sha(Path(__file__)) != protocol["runner_sha256"]:
        raise ValueError("Frozen runner or manifest changed")
    if sha(DATA / "catalog.json") != protocol["catalog_sha256"]:
        raise ValueError("Catalog snapshot changed")
    for path, pin in protocol["source_inputs"].items():
        if sha(ROOT / path) != pin:
            raise ValueError(f"Frozen source changed: {path}")
    cases = read(MANIFEST)["cases"]
    if len(cases) != protocol["maximum_attempts"] or len({case["row_id"] for case in cases}) != len(cases):
        raise ValueError("Invalid allocation")
    for case in cases:
        if sha(ROOT / case["image"]) != case["delivered_sha256"]:
            raise ValueError("Frozen image changed")
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    if len(rows) > len(cases):
        raise ValueError("Attempt limit exceeded")
    for index, (case, row) in enumerate(zip(cases, rows), 1):
        prompt = read(ROOT / case["prompt_path"])
        payload = make_payload(ROOT, case, prompt)
        if (
            row["row_id"] != case["row_id"]
            or row["attempt"] != index
            or row["protocol_sha256"] != sha(PROTOCOL)
            or row["manifest_sha256"] != protocol["manifest_sha256"]
            or row["prompt_sha256"] != digest(prompt)
            or row["request_sha256"]
            != digest({"endpoint": DECISIONS, "headers": HEADERS, "payload": payload})
        ):
            raise ValueError("Saved request differs from frozen inputs")
        strict = {**row, "normalized": row.get("strict_normalized")}
        strict.pop("validation_error", None)
        if row.get("strict_validation_error"):
            strict["validation_error"] = row["strict_validation_error"]
        parsed = review(strict, prompt)
        if parsed["normalized"] != row["normalized"] or parsed.get("validation_error") != row.get(
            "validation_error"
        ):
            raise ValueError("Saved normalized response changed")
        response = row.get("response", {})
        if row["usage"] != (response.get("usage") or {}) or row.get("resolved_model") != response.get(
            "model"
        ):
            raise ValueError("Saved usage or model changed")
        cost, basis = effective_cost(row["usage"])
        if row["effective_cost_usd"] != cost or row["cost_basis"] != basis:
            raise ValueError("Saved cost changed")
    return protocol, cases, rows


def run(max_new):
    with (DATA / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, cases, rows = load_study()
        pending = DATA / "pending.json"
        if pending.exists():
            raise ValueError("Unreconciled request intent; inspect before further calls")
        if any(
            row.get("validation_error")
            or row.get("transport_error")
            or row.get("cost_validation_error")
            or row["effective_cost_usd"] is None
            for row in rows
        ):
            raise ValueError("Prior failed or unpriced request; no automatic retry")
        if not 1 <= max_new <= len(cases):
            raise ValueError("Invalid request count")
        spent = sum(row["effective_cost_usd"] for row in rows)
        for case in cases[len(rows) : len(rows) + max_new]:
            if spent + protocol["reserve_per_request_usd"] > protocol["budget_usd"]:
                raise ValueError("Local budget guard reached")
            if rows:
                previous = datetime.fromisoformat(rows[-1]["timestamp"]).timestamp()
                time.sleep(max(0, protocol["minimum_interval_seconds"] - (time.time() - previous)))
            prompt = read(ROOT / case["prompt_path"])
            payload = make_payload(ROOT, case, prompt)
            write_new(pending, {"row_id": case["row_id"], "timestamp": datetime.now(UTC).isoformat()})
            row = review(post(payload, prompt), prompt)
            row.update(
                {key: value for key, value in case.items() if key not in {"image", "eligible", "gold"}}
            )
            row.update(
                arm="jev14",
                attempt=len(rows) + 1,
                prompt_id=prompt["id"],
                prompt_sha256=digest(prompt),
                protocol_sha256=sha(PROTOCOL),
                manifest_sha256=protocol["manifest_sha256"],
            )
            with LEDGER.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            pending.unlink()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "attempt": len(rows),
                        "row_id": case["row_id"],
                        "status": row.get("status"),
                        "valid": row["normalized"] is not None,
                        "cost": row["effective_cost_usd"],
                        "seconds": round(row["seconds"], 3),
                    }
                ),
                flush=True,
            )
            if (
                row.get("validation_error")
                or row.get("transport_error")
                or row.get("cost_validation_error")
                or row["effective_cost_usd"] is None
            ):
                return {"status": "stopped", "attempts": len(rows)}
            spent += row["effective_cost_usd"]
        return {
            "status": "complete" if len(rows) == len(cases) else "partial",
            "attempts": len(rows),
            "cost_usd": spent,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "run"))
    parser.add_argument("--max-new", type=int, default=378)
    args = parser.parse_args()
    result = (
        freeze()
        if args.command == "freeze"
        else run(args.max_new)
        if args.command == "run"
        else {"verified_attempts": len(load_study()[2])}
    )
    print(json.dumps(result, indent=2))
