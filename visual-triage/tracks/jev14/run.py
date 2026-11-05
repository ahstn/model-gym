"""Freeze and run the shared Jev 1.14 image benchmark without changing prior evidence."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_protocol import DECISIONS, digest, image_part, number, validate_prompt
from bench_runner import effective_cost
from tracks.jev14.client import (
    CONTEXT_TOKENS,
    HEADERS,
    MODEL,
    REVISION,
    image_info,
    make_payload,
    normalize,
    post,
)

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "tracks/jev14"
DATA = ROOT / "data/jev14-v1"
PROTOCOL = TRACK / "protocol.json"
MANIFEST = DATA / "manifest.json"
LEDGER = DATA / "attempts.jsonl"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")


def case_record(sample, task, phase, path=None, suffix=""):
    path = path or ROOT / sample["image"]
    info = image_info(path)
    return {
        "case_id": f"{phase}/{task}/{sample['id']}{suffix}",
        "sample": sample["id"],
        "task": task,
        "phase": phase,
        "benchmark": task,
        "image": str(path.relative_to(ROOT)),
        "delivered_sha256": sha(path),
        "parent_delivered_sha256": sample["delivered_sha256"],
        **info,
    }


def prepare_cases():
    primary, compatibility, probes, excluded = [], [], [], []
    image_dir = DATA / "views"
    image_dir.mkdir(parents=True, exist_ok=True)
    for task in ("quality", "ai"):
        samples = json.loads((ROOT / f"evidence/benchmarks/{task}.json").read_text())["samples"]
        for sample in samples:
            image_part(sample, ROOT)  # Validate the original hash before any transformation.
            case = case_record(sample, task, "primary")
            if case["eligible"]:
                primary.append(case)
                continue
            if task != "quality" or sample["id"] not in {"quality-000", "quality-019"}:
                raise ValueError("Unexpected incompatible benchmark image")
            excluded.append({**case, "reason": "Frozen panorama height is below 256 pixels"})
            with Image.open(ROOT / sample["image"]) as source:
                rgb = source.convert("RGB")
                padded = Image.new("RGB", (rgb.width, 256), (128, 128, 128))
                top = (256 - rgb.height) // 2
                padded.paste(rgb, (0, top))
                if padded.crop((0, top, rgb.width, top + rgb.height)).tobytes() != rgb.tobytes():
                    raise ValueError("Padding changed source pixels")
                path = image_dir / f"{sample['id']}-padded.png"
                if path.exists():
                    raise ValueError("Prepared view already exists")
                padded.save(path)
            compatibility.append(case_record(sample, task, "compatibility", path))
    development = json.loads((ROOT / "evidence/samples.json").read_text())["samples"]
    for sample in development:
        if sample["id"] not in {"sample-024", "sample-025"}:
            continue
        image_part(sample, ROOT)
        for side in (320, 512, 1024):
            path = image_dir / f"{sample['id']}-{side}.png"
            if path.exists():
                raise ValueError("Prepared probe already exists")
            with Image.open(ROOT / sample["image"]) as source:
                if source.size != (512, 512):
                    raise ValueError("Size probe source changed")
                source.convert("RGB").resize((side, side), Image.Resampling.LANCZOS).save(path)
            probes.append(case_record(sample, "quality", "size_probe", path, f"/{side}"))
    primary.sort(key=lambda case: (int(case["sample"].split("-")[-1]), case["task"]))
    cases = probes + primary + compatibility
    if Counter(case["phase"] for case in cases) != {"primary": 118, "compatibility": 2, "size_probe": 6}:
        raise ValueError("Unexpected case allocation")
    if any(not case["eligible"] for case in cases):
        raise ValueError("A prepared image is outside Jev's limits")
    return {"cases": cases, "primary_exclusions": excluded}


def freeze():
    if PROTOCOL.exists() or MANIFEST.exists():
        raise ValueError("Study already frozen; do not overwrite it")
    sources = [ROOT / "evidence/benchmarks/freeze.json", ROOT / "evidence/samples.json"]
    for task in ("quality", "ai"):
        prompt_path = ROOT / f"prompts/final/{task}.json"
        validate_prompt(json.loads(prompt_path.read_text()), task)
        sources += [
            prompt_path,
            ROOT / f"evidence/benchmarks/{task}.json",
            ROOT / f"evidence/assessments/{task}-60.json",
            ROOT / f"evidence/final/{task}.jsonl",
        ]
    sources += sorted((DATA / "research").glob("*.json"))
    source_inputs = {
        str(path.relative_to(ROOT)): {"sha256": sha(path), "bytes": path.stat().st_size} for path in sources
    }
    prior = json.loads((ROOT / "evidence/benchmarks/freeze.json").read_text())
    for task in ("quality", "ai"):
        if sha(ROOT / f"prompts/final/{task}.json") != prior["prompt_hashes"][task]:
            raise ValueError("Original frozen prompt changed")
    manifest = prepare_cases()
    write_new(MANIFEST, manifest)
    execution = [Path(__file__), TRACK / "client.py", ROOT / "bench_protocol.py", ROOT / "bench_runner.py"]
    protocol = {
        "version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "expected_revision": REVISION,
        "request_headers": HEADERS,
        "context_tokens": CONTEXT_TOKENS,
        "manifest_sha256": sha(MANIFEST),
        "source_inputs": source_inputs,
        "execution_sha256": {str(path.relative_to(ROOT)): sha(path) for path in execution},
        "primary_counts": {"quality": 58, "ai": 60},
        "compatibility_cases": 2,
        "size_probe_cases": 6,
        "maximum_attempts": 126,
        "budget_usd": 2,
        "reserve_per_request_usd": 0.012,
        "minimum_interval_seconds": 1,
        "retry_policy": "No retries; stop on failure or unknown cost",
        "prompt_policy": "Unchanged final prompts; no Jev-specific optimisation or threshold fitting",
        "ai_threshold": 0.5,
        "quality_scale": [1, 5],
        "views": "Primary bytes unchanged. Two panoramas: separate neutral PNG padding to height256.",
        "probe": "Two development AGIQA images at320,512,1024square; upscaling adds no source detail.",
        "cost": "Use reported usage.cost; no charge inferred from missing cost. Catalogue price is context.",
        "comparison": "Historical model calls; same primary pixels/prompts, different run dates.",
    }
    write_new(PROTOCOL, protocol)
    return {"protocol_sha256": sha(PROTOCOL), "cases": len(manifest["cases"])}


def load_study():
    protocol = json.loads(PROTOCOL.read_text())
    if sha(MANIFEST) != protocol["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    for relative, pin in protocol["source_inputs"].items():
        if sha(ROOT / relative) != pin["sha256"]:
            raise ValueError(f"Frozen source changed: {relative}")
    for relative, expected in protocol["execution_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen execution code changed: {relative}")
    cases = json.loads(MANIFEST.read_text())["cases"]
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line] if LEDGER.exists() else []
    if len(rows) > len(cases):
        raise ValueError("Too many attempts")
    for index, (case, row) in enumerate(zip(cases, rows), 1):
        if (
            row["case_id"] != case["case_id"]
            or row["delivered_sha256"] != case["delivered_sha256"]
            or row["protocol_sha256"] != sha(PROTOCOL)
            or type(row.get("attempt")) is not int
            or row["attempt"] != index
            or row.get("manifest_sha256") != protocol["manifest_sha256"]
        ):
            raise ValueError("Ledger differs from frozen request order")
        if (
            row.get("validation_error")
            or row.get("transport_error")
            or row.get("cost_validation_error")
            or row.get("status") != 200
            or not isinstance(row.get("normalized"), dict)
            or row.get("resolved_model") != REVISION
            or row.get("effective_cost_usd") is None
        ):
            raise ValueError("Previous failed or unpriced call needs review; no automatic resume")
        number(row["effective_cost_usd"], 0, float("inf"))
        prompt = json.loads((ROOT / f"prompts/final/{case['task']}.json").read_text())
        payload = make_payload(ROOT, case, prompt)
        expected_request = digest({"endpoint": DECISIONS, "headers": HEADERS, "payload": payload})
        if (
            row.get("prompt_sha256") != digest(prompt)
            or row.get("request_sha256") != expected_request
            or row.get("request_headers") != HEADERS
            or row.get("requested_model") != MODEL
            or row.get("endpoint") != DECISIONS
        ):
            raise ValueError("Saved request differs from the frozen inputs")
        response = row.get("response", {})
        if (
            response.get("model") != REVISION
            or row.get("usage") != response.get("usage")
            or row["normalized"] != normalize(response, prompt["questions"])
        ):
            raise ValueError("Saved result differs from its API response")
        cost, basis = effective_cost(row["usage"])
        if row["effective_cost_usd"] != cost or row.get("cost_basis") != basis:
            raise ValueError("Saved cost differs from reported usage")
    return protocol, cases, rows


def run(max_new):
    protocol, cases, rows = load_study()
    if not 1 <= max_new <= protocol["maximum_attempts"]:
        raise ValueError("Invalid request count")
    prompts = {
        task: json.loads((ROOT / f"prompts/final/{task}.json").read_text()) for task in ("quality", "ai")
    }
    for case in cases:
        if sha(ROOT / case["image"]) != case["delivered_sha256"]:
            raise ValueError("Frozen image changed")
    spent = sum(row["effective_cost_usd"] for row in rows)
    for case in cases[len(rows) : len(rows) + max_new]:
        if spent + protocol["reserve_per_request_usd"] > protocol["budget_usd"]:
            raise ValueError("Request would exceed the local budget guard")
        if rows:
            previous = datetime.fromisoformat(rows[-1]["timestamp"]).timestamp()
            time.sleep(max(0, protocol["minimum_interval_seconds"] - (time.time() - previous)))
        prompt = prompts[case["task"]]
        payload = make_payload(ROOT, case, prompt)
        row = post(payload, prompt)
        row.update({key: value for key, value in case.items() if key not in {"image", "eligible"}})
        row.update(
            arm="jev14",
            attempt=len(rows) + 1,
            prompt_id=prompt["id"],
            prompt_sha256=digest(prompt),
            manifest_sha256=protocol["manifest_sha256"],
            protocol_sha256=sha(PROTOCOL),
        )
        with LEDGER.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(row)
        print(
            json.dumps(
                {
                    "attempt": len(rows),
                    "case": case["case_id"],
                    "status": row.get("status"),
                    "valid": row["normalized"] is not None,
                    "cost": row["effective_cost_usd"],
                    "seconds": row["seconds"],
                    "input_tokens": row["usage"].get("input_tokens"),
                }
            ),
            flush=True,
        )
        if row.get("validation_error") or row["effective_cost_usd"] is None:
            return {"status": "stopped", "attempts": len(rows)}
        spent += row["effective_cost_usd"]
    return {
        "status": "complete" if len(rows) == len(cases) else "partial",
        "attempts": len(rows),
        "reported_cost_usd": spent,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--max-new", type=int, default=126)
    args = parser.parse_args()
    print(json.dumps(freeze() if args.command == "freeze" else run(args.max_new), indent=2))


if __name__ == "__main__":
    main()
