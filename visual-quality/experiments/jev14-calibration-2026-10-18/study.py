"""Freeze the new cohort, then select a rule before validation inference."""

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data/jev14-calibration-2026-10-18"
BASELINE = Path("/Users/ahstn/.codex/worktrees/d187/model-gym/visual-triage")
PROMPT_SOURCE = BASELINE / (
    "data/metadata-v3/runs/image_only/candidates/"
    "fc15792f2b64d250730af9cfaca097ff21ad203d75991d5e915883af1715174f/prompt.json"
)
sys.path.insert(0, str(BASELINE))

from bench_protocol import DECISIONS, digest
from tracks.jev14.client import HEADERS, MODEL, REVISION, make_payload


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def freeze():
    if (HERE / "protocol.json").exists():
        raise ValueError("Protocol is already frozen")
    if any((DATA / f"{split}.jsonl").exists() for split in ("calibration", "validation")):
        raise ValueError("Inference has already started")
    cases = read(HERE / "manifest.json")["cases"]
    if any(type(case["expected_ai"]) is not bool for case in cases):
        raise ValueError("Origin provenance must use Boolean labels")
    counts = Counter((case["split"], case["expected_ai"]) for case in cases)
    expected = {
        ("calibration", False): 300,
        ("calibration", True): 300,
        ("validation", False): 500,
        ("validation", True): 500,
    }
    if counts != expected or len({case["row_id"] for case in cases}) != 1600:
        raise ValueError("Unexpected case allocation")
    if any(case["source_split"] != "validation" for case in cases):
        raise ValueError("Study requires the single-copy official validation pool")
    families = {
        split: {c["family_id"] for c in cases if c["split"] == split}
        for split in ("calibration", "validation")
    }
    if families["calibration"] & families["validation"]:
        raise ValueError("Caption family crosses splits")
    if len({c["delivered_sha256"] for c in cases}) != 1600:
        raise ValueError("Duplicate delivered image")
    for split, n in (("calibration", 60), ("validation", 100)):
        allocation = Counter(c["generator"] for c in cases if c["split"] == split and c["expected_ai"])
        if len(allocation) != 5 or set(allocation.values()) != {n}:
            raise ValueError("Generated image allocation is not balanced")
    catalog_path = DATA / "research/catalog.json"
    model = next(row for row in read(catalog_path)["data"] if row["id"] == MODEL)
    if model["canonical_slug"] != REVISION or "image" not in model["architecture"]["input_modalities"]:
        raise ValueError("Current model revision or modality differs")
    prompt_path = HERE / "prompt.json"
    if not prompt_path.exists():
        shutil.copyfile(PROMPT_SOURCE, prompt_path)
    if sha(prompt_path) != sha(PROMPT_SOURCE):
        raise ValueError("Prompt differs from the previous image-only benchmark")
    prompt = read(prompt_path)
    audits = []
    for case in cases:
        if not case["source_url"].startswith("https://") or not case.get("source_sha256"):
            raise ValueError("Missing public image provenance")
        payload = make_payload(DATA, case, prompt)
        audits.append(
            {
                "row_id": case["row_id"],
                "source_url": case["source_url"],
                "source_asset_url": case["source_asset_url"],
                "source_row_index": case["source_row_index"],
                "native_row_group": case["native_row_group"],
                "native_group_row": case["native_group_row"],
                "source_sha256": case["source_sha256"],
                "delivered_sha256": case["delivered_sha256"],
                "request_sha256": digest({"endpoint": DECISIONS, "headers": HEADERS, "payload": payload}),
            }
        )
    write_new(
        HERE / "payload-audit.json",
        {
            "destination": DECISIONS,
            "model": MODEL,
            "images": len(audits),
            "outbound_fields": "Model ID, fixed public research prompt, typed questions and downloaded public image bytes only.",
            "excluded_fields": "Labels, source names, file paths, captions and other local files are not sent.",
            "cases": audits,
        },
    )
    local_code = [HERE / name for name in ("study.py", "runner.py", "metrics.py", "acquire.py", "report.py")]
    baseline_code = list(BASELINE.glob("*.py")) + list((BASELINE / "tracks").rglob("*.py"))
    write_new(
        HERE / "protocol.json",
        {
            "version": 1,
            "frozen_at": datetime.now(UTC).isoformat(),
            "model": MODEL,
            "expected_revision": REVISION,
            "endpoint": DECISIONS,
            "request_headers": HEADERS,
            "maximum_attempts": 1600,
            "budget_usd": 0.25,
            "reserve_per_request_usd": 0.012,
            "concurrency": 4,
            "minimum_interval_seconds": 0.25,
            "manifest_sha256": sha(HERE / "manifest.json"),
            "prompt_sha256": sha(prompt_path),
            "execution_sha256": {str(path): sha(path) for path in sorted(local_code + baseline_code)},
            "source_inputs": {
                str(path): sha(path)
                for path in (
                    catalog_path,
                    PROMPT_SOURCE,
                    HERE / "acquisition-audit.json",
                    HERE / "payload-audit.json",
                )
            },
            "catalog_model": model,
            "dataset": "Rajarshi-Roy-research/Defactify_Image_Dataset",
            "dataset_revision": "787334f7857fa54f29027a7f09c30e895ad486ef",
            "splits": {
                "calibration": {"official_split": "validation", "real": 300, "ai": 300},
                "validation": {"official_split": "validation", "real": 500, "ai": 500},
            },
            "thresholds": [round(value / 100, 2) for value in range(50, 101)],
            "gates": ["none", "not_high"],
            "positive_comparison": "ai_probability >= threshold",
            "calibration_fpr_limit": 0.01,
            "source_split_correction": "Before inference, the official test split was found to repeat caption/label rows in five image-variant blocks. The first block has reduced brightness. Both study splits therefore use different caption families from the official validation pool; the discarded preflight cohort is retained only in ignored acquisition evidence.",
            "selection": "Maximum unconditional AI recall among rules with empirical calibration FPR <= 1%; tie: lower FPR, ungated rule, higher threshold. No feasible rule means no selected rule.",
            "validation": "All responses valid; selected rule has one-sided 95% exact binomial FPR upper bound <= 1% and observed unconditional AI recall >= 70%.",
            "recall_floor": 0.70,
            "recall_floor_basis": "Provisional operating assumption; not an approved production requirement or a confidence bound.",
            "baselines": [{"threshold": 0.5, "gate": "none"}, {"threshold": 0.8, "gate": "none"}],
            "unresolved": "Never assign score zero; retain every attempted case in operational denominators and report incomplete validation.",
            "retry_policy": "No automatic retry. Stop new dispatch on failed or unpriced call; drain in-flight calls and preserve every attempt.",
            "validation_isolation": "Freeze selection.json and its hash before any validation request. No threshold change based on validation results.",
            "probability_interpretation": "This selects decision thresholds; it does not calibrate numeric probabilities of correctness.",
            "view": "EXIF transpose, RGB, centered ImageOps.fit 512x512 LANCZOS, JPEG quality95 subsampling0; unchanged prior image-only prompt.",
        },
    )
    return {"status": "frozen", "cases": len(cases), "protocol_sha256": sha(HERE / "protocol.json")}


def select():
    import runner
    from metrics import evaluate, select_rule

    if (DATA / "validation.jsonl").exists() or (HERE / "selection.json").exists():
        raise ValueError("Selection must occur once before validation")
    _, all_cases, _, records = runner.load_study()
    cases = [c for c in all_cases if c["split"] == "calibration"]
    raw = records["calibration"]
    by_id = {r["row_id"]: r for r in raw}
    if len(raw) != len(cases) or set(by_id) != {c["row_id"] for c in cases}:
        raise ValueError("Calibration is incomplete")
    rows = []
    for case in cases:
        response = by_id[case["row_id"]]
        if runner.failed(response):
            raise ValueError("Calibration contains an unresolved response")
        rows.append({**case, **response["normalized"], "valid": True})
    rule = select_rule(rows)
    write_new(
        HERE / "selection.json",
        {
            "selected_at": datetime.now(UTC).isoformat(),
            "rule": rule,
            "protocol_sha256": sha(HERE / "protocol.json"),
            "calibration_ledger_sha256": sha(DATA / "calibration.jsonl"),
            "calibration_metrics": evaluate(rows, rule) if rule is not None else None,
            "calibration_cases": len(rows),
            "validation_attempts_at_selection": 0,
        },
    )
    write_new(HERE / "selection-lock.json", {"selection_sha256": sha(HERE / "selection.json")})
    return read(HERE / "selection.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "select"))
    args = parser.parse_args()
    print(json.dumps(freeze() if args.command == "freeze" else select(), indent=2))
