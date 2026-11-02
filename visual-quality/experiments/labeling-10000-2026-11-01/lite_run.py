"""Use the unchanged Q-ReAlign Lite scorer for new views; reuse pinned old observations."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "rehearsal-4000-2026-11-01"
SPEC = importlib.util.spec_from_file_location("prior_lite_runner", PRIOR / "lite_runner.py")
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


def reuse_rows(cases: list[dict]) -> list[dict]:
    """Reuse only the same pinned model's valid baseline results for unchanged bytes."""
    old_cases = {c["case_id"]: c for c in json.loads((PRIOR / "manifest.json").read_text())["cases"]}
    scores_sha256 = client.digest_bytes((PRIOR / "scores-long.jsonl").read_bytes())
    records = {}
    with (PRIOR / "scores-long.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["model"] == "qrealign_lite" and row["arm"] == "baseline":
                records[row["case_id"]] = row
    reused = []
    for case in cases:
        original = records.get(case["case_id"])
        if original is None:
            continue
        if original["view_sha256"] != case["view_sha256"]:
            raise ValueError("Historical case ID reused for different bytes")
        if old_cases[case["case_id"]]["family_id"] != case["family_id"]:
            raise ValueError("Historical family changed")
        raw_path = Path(original["raw_response_path"])
        raw = json.loads(raw_path.read_text())
        parsed = json.loads(raw["body"])
        checked = client.normalize_response(parsed)
        if checked["status"] != "valid":
            raise ValueError("Historical Lite response failed revalidation")
        for field in ("technical_score", "aesthetic_score"):
            if abs(original[field] - checked[field]) > 1e-10:
                raise ValueError("Historical Lite score changed")
        reused.append(
            {
                **original,
                "reuse": {
                    "source": str(PRIOR / "scores-long.jsonl"),
                    "source_sha256": scores_sha256,
                    "source_attempt_id": original.get("attempt_id"),
                    "checkpoint": client.MODEL,
                    "revision": client.REVISION,
                    "new_request_sent": False,
                    "current_run_incremental_price_usd": "0.00",
                },
                "price_usd": "0.00",
                "price_basis": "user_assigned_zero",
            }
        )
    return reused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--partition", choices=("rehearsal", "benchmark"), default="rehearsal")
    parser.add_argument("--references-freeze", type=Path)
    parser.add_argument(
        "--case-ids", type=Path, help="Explicit frozen subset for a continuation or reference packet"
    )
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--url")
    parser.add_argument("--token-file", type=Path, default=Path("/tmp/qrealign-lite-token"))
    parser.add_argument("--max-seconds", type=float, default=8 * 3600)
    args = parser.parse_args()
    raw_manifest = args.manifest.read_bytes()
    cases = json.loads(raw_manifest)["cases"]
    if args.reuse_only:
        rows = reuse_rows(cases)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        print(json.dumps({"reused": len(rows), "output": str(args.output)}))
        return
    if not args.url:
        parser.error("--url is required for new inference")
    old_ids = {c["case_id"] for c in json.loads((PRIOR / "manifest.json").read_text())["cases"]}
    selected = client.select_cases([c for c in cases if c["case_id"] not in old_ids], args.partition)
    if args.case_ids:
        admitted = json.loads(args.case_ids.read_text())
        available = {c["case_id"] for c in selected}
        if not isinstance(admitted, list) or len(set(admitted)) != len(admitted) or set(admitted) - available:
            raise ValueError("Continuation IDs must be unique members of the selected partition")
        selected = [c for c in selected if c["case_id"] in set(admitted)]
    if not selected:
        raise ValueError("No new images selected")
    pins = {}
    if args.partition == "benchmark":
        from api_runner import reference_pins

        pins = reference_pins(selected, args.references_freeze)
    token = args.token_file.read_text().strip()
    response = requests.get(
        args.url + "/health", headers={"Authorization": "Bearer " + token}, timeout=(30, 60)
    )
    response.raise_for_status()
    identity = response.json()
    expected_path = HERE.parent / "qrealign-lite-72-2026-10-31/server-identity.json"
    if identity != json.loads(expected_path.read_text()):
        raise ValueError("Live scorer identity differs from the prior validated model")
    config = {
        "version": "labeling-10000-lite-v1",
        "model": client.MODEL,
        "revision": client.REVISION,
        "manifest_sha256": client.digest_bytes(raw_manifest),
        "case_ids": [c["case_id"] for c in selected],
        "url": args.url,
        "concurrency": args.concurrency,
        "partition": args.partition,
        "reference_pins": pins,
        "client_sha256": client.digest_bytes((PRIOR / "lite_runner.py").read_bytes()),
        "adapter_sha256": client.digest_bytes(Path(__file__).read_bytes()),
        "server_identity_sha256": client.digest_bytes(client.canonical(identity)),
        "latency_kind": "http_request_through_complete_response",
        "reported_price_usd": "0.00",
        "policy": "Unchanged serial scorer; no eligibility floor or automatic retries",
    }
    client.write_json(args.output / "server-identity.json", identity)
    summary = client.execute(
        selected, directory=args.output, config=config, token=token, max_seconds=args.max_seconds
    )
    print(json.dumps(summary))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
