"""Freeze and run a native-file provenance check on the existing held-out cohort."""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tracks.openai_provenance.client import ENDPOINT, check_image

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "tracks/openai_provenance"
DATA = ROOT / "data/openai-provenance-v1"
SOURCE = ROOT / "tracks/metadata_v3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze():
    if (TRACK / "protocol.json").exists() or (DATA / "manifest.json").exists():
        raise ValueError("Study inputs already exist; preserve the frozen study")
    source_protocol = json.loads((SOURCE / "protocol.json").read_text())
    if sha(SOURCE / "dataset.json") != source_protocol["dataset_sha256"]:
        raise ValueError("Original dataset no longer matches its frozen protocol")
    samples = json.loads((SOURCE / "dataset.json").read_text())["samples"]
    groups = {name: [] for name in ("dalle3", "real", "dalle2", "firefly", "midjourney-v5")}
    for sample in samples:
        if sample["split"] != "test":
            continue
        group = sample["source"].get("generator") or "real"
        native = ROOT / sample["native_image"]
        if sha(native) != sample["native_sha256"]:
            raise ValueError("Native file no longer matches the frozen source")
        groups[group].append(
            {
                "case": sample["id"],
                "native_image": sample["native_image"],
                "native_sha256": sample["native_sha256"],
                "delivered_sha256": sample["delivered_sha256"],
                "gold_ai": bool(sample["gold"]),
                "source": group,
                "openai_origin": group in {"dalle2", "dalle3"},
                "source_group": sample["group_id"],
                "local_c2pa": sample["features"]["provenance"],
            }
        )
    for values in groups.values():
        values.sort(key=lambda row: row["case"])
    if {k: len(v) for k, v in groups.items()} != {
        "dalle3": 20,
        "real": 80,
        "dalle2": 20,
        "firefly": 20,
        "midjourney-v5": 20,
    }:
        raise ValueError("Expected the complete original 160-image test cohort")
    cases = []
    for index in range(20):
        for offset, generator in enumerate(("dalle3", "dalle2", "firefly", "midjourney-v5")):
            cases.extend((groups[generator][index], groups["real"][index * 4 + offset]))
    manifest = {"version": 1, "cases": cases}
    write_new(DATA / "manifest.json", manifest)
    protocol = {
        "version": 1,
        "study": "openai-native-provenance-v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "endpoint": ENDPOINT,
        "source_protocol_sha256": sha(SOURCE / "protocol.json"),
        "source_dataset_sha256": sha(SOURCE / "dataset.json"),
        "manifest_sha256": sha(DATA / "manifest.json"),
        "guide_sha256": sha(DATA / "research/guide.md"),
        "execution_sha256": {
            str(p.relative_to(ROOT)): sha(p) for p in (TRACK / "client.py", TRACK / "study.py")
        },
        "cases": len(cases),
        "source_counts": dict(Counter(c["source"] for c in cases)),
        "view": "Original native bytes; generic multipart filename; no prompt or class labels",
        "comparison": "Same identities as Decider; Decider used 512px JPEG crops and optional text",
        "order": "Twenty blocks of dalle3, real, dalle2, real, firefly, real, midjourney-v5, real",
        "minimum_interval_seconds": 6.2,
        "maximum_attempts": 160,
        "retries": "None; stop on any failed request and retain its record",
        "interpretation": "not_detected is no supported OpenAI signal, never a real-image label",
        "cost": "No price or per-request charge in the fetched developer guide; retain null costs",
        "limitations": [
            "No current GPT-image positive controls",
            "Older DALL-E 2/3 exports",
            "Third-party generation lies outside the API attribution target",
            "Native input differs from Decider's delivered view",
        ],
    }
    write_new(TRACK / "protocol.json", protocol)
    return {"frozen_cases": len(cases), "protocol_sha256": sha(TRACK / "protocol.json")}


def load_study():
    protocol = json.loads((TRACK / "protocol.json").read_text())
    if sha(DATA / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    for name, expected in protocol["execution_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen execution code changed")
    cases = json.loads((DATA / "manifest.json").read_text())["cases"]
    rows = []
    ledger = DATA / "attempts.jsonl"
    if ledger.exists():
        rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    identifiers = [row["case"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate submitted cases are forbidden")
    expected = {case["case"]: case for case in cases}
    for row in rows:
        case = expected.get(row["case"])
        if not case or row["input_sha256"] != case["native_sha256"]:
            raise ValueError("Ledger contains an unexpected case or image hash")
        if row["protocol_sha256"] != sha(TRACK / "protocol.json"):
            raise ValueError("Ledger references another protocol")
    return protocol, cases, rows


def run(max_new):
    protocol, cases, rows = load_study()
    if max_new < 1 or max_new > protocol["maximum_attempts"]:
        raise ValueError("max-new must be between one and the frozen attempt limit")
    if any(not row["success"] for row in rows):
        raise ValueError("A previous failure stopped this study; inspect it before any further calls")
    done = {row["case"] for row in rows}
    pending = [case for case in cases if case["case"] not in done][:max_new]
    for index, case in enumerate(pending):
        if rows:
            last = datetime.fromisoformat(rows[-1]["started_at"]).timestamp()
            time.sleep(max(0.0, protocol["minimum_interval_seconds"] - (time.time() - last)))
        row = check_image(ROOT / case["native_image"], case["native_sha256"])
        row.update(
            case=case["case"],
            source=case["source"],
            protocol_sha256=sha(TRACK / "protocol.json"),
            attempt=len(rows) + 1,
        )
        with (DATA / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        rows.append(row)
        print(
            json.dumps(
                {
                    "completed": len(rows),
                    "planned": len(cases),
                    "case": case["case"],
                    "success": row["success"],
                    "http_status": row["http_status"],
                    "seconds": row["elapsed_seconds"],
                    "signals": {k: v["outcome"] for k, v in row["signals"].items()},
                    "error_kind": (row["error"] or {}).get("kind"),
                }
            ),
            flush=True,
        )
        if not row["success"]:
            return {"status": "stopped", "attempts": len(rows), "unattempted": len(cases) - len(rows)}
    return {"status": "complete" if len(rows) == len(cases) else "partial", "attempts": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--max-new", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(freeze() if args.command == "freeze" else run(args.max_new), indent=2))


if __name__ == "__main__":
    main()
