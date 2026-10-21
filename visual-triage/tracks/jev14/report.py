"""Compare Jev with frozen image benchmarks without pooling changed image views."""

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_analyze import (
    ai_metrics,
    combine_record,
    costs_for_comparison,
    numeric,
    operational_metrics,
    quality_metrics,
)
from bench_protocol import DECISIONS, digest
from tracks.jev14.client import make_payload, normalize

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("decider", "luna", "gemini")
TASKS = ("quality", "ai")
PHASES = ("primary", "compatibility", "size_probe")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_key(case):
    """A size probe may reuse its parent sample at several delivered sizes."""
    return case["phase"], case["task"], case["sample"], case["delivered_sha256"]


def valid_record(record):
    return bool(record and record.get("status") == 200 and not record.get("validation_error"))


def joined_row(sample, reference, record, task, arm):
    """Represent missing requests separately while retaining their scoring denominator."""
    item = dict(record or {})
    item.update(
        sample=sample["id"], task=task, arm=arm, benchmark=task, ledger=item.get("ledger", "not_attempted")
    )
    if not valid_record(record):
        item["normalized"] = None
    row = combine_record(item, {sample["id"]: reference}, {sample["id"]: sample})
    row["attempted"] = record is not None
    return row


def evaluate(samples, references, records, task, arm):
    """Score a fixed cohort; costs include every actual request, including failures."""
    by_id = {}
    for record in records:
        if record["sample"] in by_id:
            raise ValueError("Duplicate model attempt for a primary sample")
        by_id[record["sample"]] = record
    if set(by_id) - set(samples):
        raise ValueError("Unexpected primary sample")
    rows = [
        joined_row(sample, references.get(identifier, {}), by_id.get(identifier), task, arm)
        for identifier, sample in samples.items()
    ]
    attempted = [row for row in rows if row["attempted"]]
    metrics = quality_metrics(rows) if task == "quality" else ai_metrics(rows)
    if task == "ai":
        metrics["brier_all_expected_invalid_penalty1"] = (
            sum(
                (row["prediction"]["ai_probability"] - int(row["ai_generated"])) ** 2 if row["valid"] else 1
                for row in rows
            )
            / len(rows)
            if rows
            else None
        )
    return {
        "expected": len(rows),
        "unattempted": len(rows) - len(attempted),
        "operations": operational_metrics(attempted),
        "metrics": metrics,
    }, rows


def compare_primary(task, planned, records, baselines, samples, references):
    """Compute all-model and paired-valid comparisons on the exact same image hashes."""
    ids = [case["sample"] for case in planned]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate planned primary sample")
    selected = {identifier: samples[identifier] for identifier in ids}
    for case in planned:
        if case["delivered_sha256"] != selected[case["sample"]]["delivered_sha256"]:
            raise ValueError("Primary Jev view differs from the frozen benchmark")
    jev, rows = evaluate(selected, references, records, task, "jev14")
    valid_ids = {row["sample"] for row in rows if row["valid"]}
    paired = {key: value for key, value in selected.items() if key in valid_ids}
    comparisons, paired_baselines = {}, {}
    for arm in BASELINES:
        chosen = [row for row in baselines if row["arm"] == arm and row["sample"] in selected]
        if len(chosen) != len(selected):
            raise ValueError("Historical baseline does not cover the planned primary cohort")
        comparisons[arm], _ = evaluate(selected, references, chosen, task, arm)
        matched = [row for row in chosen if row["sample"] in paired]
        paired_baselines[arm], _ = evaluate(paired, references, matched, task, arm)
    return {
        "expected": len(selected),
        "jev14": jev,
        "baselines": comparisons,
        "paired_valid_ids": sorted(valid_ids),
        "paired_valid_baselines": paired_baselines,
    }, rows


def compact_case(case, record=None):
    """Retain measurable results and hashes, excluding raw responses and uploaded data."""
    record = record or {}
    valid = valid_record(record) and isinstance(record.get("normalized"), dict)
    prediction = record.get("normalized") if valid else {}
    usage = record.get("usage") or {}
    return {
        "phase": case["phase"],
        "task": case["task"],
        "sample": case["sample"],
        "disposition": "valid" if valid else "invalid" if record else "unattempted",
        "delivered_sha256": case["delivered_sha256"],
        "parent_delivered_sha256": case.get("parent_delivered_sha256"),
        "image_width": case.get("image_width"),
        "image_height": case.get("image_height"),
        "image_bytes": case.get("image_bytes"),
        "request_bytes": record.get("request_bytes"),
        "technical_quality": prediction.get("technical_quality"),
        "aesthetics": prediction.get("aesthetics"),
        "ai_probability": prediction.get("ai_probability"),
        "medium": prediction.get("medium"),
        "uncertainty": prediction.get("uncertainty"),
        "seconds": record.get("seconds"),
        **costs_for_comparison(record),
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "http_status": record.get("status"),
        "timestamp": record.get("timestamp"),
        "resolved_model": record.get("resolved_model"),
        "provider": record.get("provider"),
        "validation_error": record.get("validation_error"),
        "strict_validation_error": record.get("strict_validation_error"),
        "strict_parser_valid": record.get("strict_parser_valid"),
        "parser_review_applied": record.get("parser_review_applied", False),
        "request_sha256": record.get("request_sha256"),
    }


def _operations(records):
    rows = [
        {
            **costs_for_comparison(record),
            "seconds": record.get("seconds"),
            "valid": valid_record(record) and isinstance(record.get("normalized"), dict),
        }
        for record in records
    ]
    return operational_metrics(rows)


def _load_json(path):
    return json.loads(path.read_text())


def _sample_index(path):
    samples = _load_json(path)["samples"]
    result = {sample["id"]: sample for sample in samples}
    if len(result) != len(samples):
        raise ValueError("Duplicate source sample")
    return result


def _read_rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    )


def review_records(records, prompts, revision, normalizer):
    """Apply one reviewed parser to every response without changing the raw ledger."""
    from tracks.jev14.resume import ROUNDING_ERRORS

    effective, strict_rejected, recovered = [], 0, 0
    for original in records:
        strict_valid = valid_record(original) and original.get("normalized") is not None
        strict_rejected += original.get("status") == 200 and not strict_valid
        normalized, error = None, None
        try:
            if original.get("status") != 200 or original.get("resolved_model") != revision:
                raise ValueError("HTTP or model validation failed")
            if original.get("validation_error") not in ROUNDING_ERRORS:
                raise ValueError("Validation failure is outside probability rounding")
            normalized = normalizer(original["response"], prompts[original["task"]]["questions"])
        except (TypeError, ValueError, KeyError) as exc:
            error = type(exc).__name__ + ": " + str(exc)
        if "reviewed_normalized" in original and original["reviewed_normalized"] != normalized:
            raise ValueError("Recorded parser review differs from uniform response parsing")
        if "reviewed_validation_error" in original and original["reviewed_validation_error"] != error:
            raise ValueError("Recorded parser review error differs")
        row = {
            **original,
            "normalized": normalized,
            "strict_parser_valid": strict_valid,
            "strict_validation_error": original.get("validation_error"),
            "parser_review_applied": True,
        }
        row.pop("validation_error", None)
        if error:
            row["validation_error"] = error
        recovered += not strict_valid and normalized is not None
        effective.append(row)
    return effective, {
        "applied_to_responses": len(records),
        "original_strict_parser_rejections": strict_rejected,
        "recovered_strict_parser_rejections": recovered,
        "revised_valid": sum(row["normalized"] is not None for row in effective),
    }


def apply_amendment(root, protocol, protocol_hash, records, prompts, ledger_bytes):
    """Validate the preserved ledger prefix before applying a parser amendment."""
    track = root / "tracks/jev14"
    path = track / "parser-amendment.json"
    if not path.exists():
        if any("amendment_sha256" in row or "reviewed_normalized" in row for row in records):
            raise ValueError("Reviewed records require a frozen parser amendment")
        return records, None
    amendment, amendment_hash = _load_json(path), sha(path)
    count = amendment["prefix_rows"]
    if type(count) is not int or not 1 <= count <= len(records):
        raise ValueError("Invalid preserved ledger prefix length")
    prefix = b"".join(ledger_bytes.splitlines(keepends=True)[:count])
    if (
        amendment["protocol_sha256"] != protocol_hash
        or amendment["parser_sha256"] != sha(track / "rounding.py")
        or amendment["resume_sha256"] != sha(track / "resume.py")
        or amendment["prefix_sha256"] != hashlib.sha256(prefix).hexdigest()
        or amendment["maximum_attempts"] != protocol["maximum_attempts"]
    ):
        raise ValueError("Parser amendment or preserved ledger prefix changed")
    for row in records[count:]:
        if row.get("amendment_sha256") != amendment_hash or "reviewed_normalized" not in row:
            raise ValueError("Continued request lacks its parser amendment")
    from tracks.jev14.rounding import normalize as reviewed_normalize

    effective, audit = review_records(records, prompts, protocol["expected_revision"], reviewed_normalize)
    audit.update(
        amendment_sha256=amendment_hash,
        parser_sha256=amendment["parser_sha256"],
        preserved_prefix_rows=count,
        preserved_prefix_sha256=amendment["prefix_sha256"],
    )
    return effective, audit


def _validate_allocation(protocol, manifest, sources, development):
    cases = manifest["cases"]
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate planned case ID")
    counts = Counter(case["phase"] for case in cases)
    expected = {
        "primary": sum(protocol["primary_counts"].values()),
        "compatibility": protocol["compatibility_cases"],
        "size_probe": protocol["size_probe_cases"],
    }
    if any(counts[phase] != total for phase, total in expected.items()) or set(counts) - set(PHASES):
        raise ValueError("Frozen phase allocation differs")
    for task in TASKS:
        primary = [case for case in cases if case["phase"] == "primary" and case["task"] == task]
        if len(primary) != protocol["primary_counts"][task]:
            raise ValueError("Frozen primary allocation differs")
        excluded = [case for case in manifest.get("primary_exclusions", []) if case["task"] == task]
        allocated = [case["sample"] for case in [*primary, *excluded]]
        if len(set(allocated)) != len(allocated) or set(allocated) != set(sources[task]):
            raise ValueError("Primary allocation and exclusions do not cover the source panel")
    for case in [*cases, *manifest.get("primary_exclusions", [])]:
        parent = development if case["phase"] == "size_probe" else sources[case["task"]]
        source = parent.get(case["sample"])
        if source is None or case["parent_delivered_sha256"] != source["delivered_sha256"]:
            raise ValueError("Case does not match its frozen parent image")
        if case["phase"] == "primary" and case["delivered_sha256"] != source["delivered_sha256"]:
            raise ValueError("Primary image differs from its frozen view")


def _validate_attempt(root, row, case, protocol, protocol_hash, prompt, ordinal):
    copied = (
        "case_id",
        "phase",
        "task",
        "sample",
        "benchmark",
        "delivered_sha256",
        "parent_delivered_sha256",
        "image_width",
        "image_height",
        "image_bytes",
        "image_mime",
    )
    if any(row.get(field) != case[field] for field in copied) or row.get("attempt") != ordinal:
        raise ValueError("Ledger differs from the frozen case or request order")
    if (
        row.get("arm") != "jev14"
        or row.get("protocol_sha256") != protocol_hash
        or row.get("manifest_sha256") != protocol["manifest_sha256"]
        or row.get("prompt_sha256") != digest(prompt)
    ):
        raise ValueError("Ledger configuration hashes differ")
    payload = make_payload(root, case, prompt)
    request = {"endpoint": DECISIONS, "headers": protocol["request_headers"], "payload": payload}
    if (
        row.get("request_sha256") != digest(request)
        or row.get("endpoint") != DECISIONS
        or row.get("request_headers") != protocol["request_headers"]
        or row.get("request_bytes") != len(json.dumps(payload).encode())
    ):
        raise ValueError("Ledger request differs from frozen image and prompt")
    response = row.get("response")
    if isinstance(response, dict):
        usage = response.get("usage")
        if row.get("usage") != (usage if isinstance(usage, dict) else {}):
            raise ValueError("Ledger usage differs from its response")
        if row.get("resolved_model") != response.get("model") or row.get("provider") != response.get(
            "provider"
        ):
            raise ValueError("Ledger model or provider differs from its response")
        if numeric(row.get("cost_usd"), 0) and row["cost_usd"] != row["usage"].get("cost"):
            raise ValueError("Ledger charge differs from its response")
    if row.get("normalized") is not None:
        if (
            not valid_record(row)
            or row.get("resolved_model") != protocol["expected_revision"]
            or not isinstance(response, dict)
            or row["normalized"] != normalize(response, prompt["questions"])
        ):
            raise ValueError("Successful ledger result differs from its response")
    elif not row.get("validation_error"):
        raise ValueError("Missing prediction has no failure record")


def build_report(root=ROOT):
    """Verify immutable evidence, then report partial or complete results without API calls."""
    root = Path(root)
    track, data = root / "tracks/jev14", root / "data/jev14-v1"
    protocol_path = track / "protocol.json"
    protocol, protocol_hash = _load_json(protocol_path), sha(protocol_path)
    manifest_path = data / "manifest.json"
    if sha(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    for relative, pin in protocol["source_inputs"].items():
        path = root / relative
        if sha(path) != pin["sha256"] or path.stat().st_size != pin["bytes"]:
            raise ValueError("Frozen source changed")
    for relative, expected in protocol["execution_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError("Frozen execution code changed")
    manifests = {task: _sample_index(root / f"evidence/benchmarks/{task}.json") for task in TASKS}
    references = {task: _sample_index(root / f"evidence/assessments/{task}-60.json") for task in TASKS}
    prompts = {task: _load_json(root / f"prompts/final/{task}.json") for task in TASKS}
    baselines = {task: _read_rows(root / f"evidence/final/{task}.jsonl") for task in TASKS}
    for task, rows in baselines.items():
        for row in rows:
            row["ledger"] = f"evidence/final/{task}.jsonl"
    manifest = _load_json(manifest_path)
    _validate_allocation(protocol, manifest, manifests, _sample_index(root / "evidence/samples.json"))
    ledger_path = data / "attempts.jsonl"
    ledger_exists = ledger_path.exists()
    ledger_bytes = ledger_path.read_bytes() if ledger_exists else b""
    cases = manifest["cases"]
    records = [json.loads(line) for line in ledger_bytes.splitlines() if line.strip()]
    if len(records) > min(protocol["maximum_attempts"], len(cases)):
        raise ValueError("Too many Jev attempts")
    for index, (case, row) in enumerate(zip(cases, records), 1):
        _validate_attempt(root, row, case, protocol, protocol_hash, prompts[case["task"]], index)
    effective, parser_review = apply_amendment(root, protocol, protocol_hash, records, prompts, ledger_bytes)
    summary, compact = assemble_report(cases, effective, manifests, references, baselines)
    summary.update(
        protocol_sha256=protocol_hash,
        manifest_sha256=protocol["manifest_sha256"],
        source_inputs=protocol["source_inputs"],
        primary_exclusions=manifest.get("primary_exclusions", []),
        ledger_sha256=hashlib.sha256(ledger_bytes).hexdigest() if ledger_exists else None,
        model=protocol["model"],
        expected_revision=protocol["expected_revision"],
        context_tokens=protocol["context_tokens"],
        parser_review=parser_review,
    )
    primary_tokens = [
        row["input_tokens"]
        for row in compact
        if row["phase"] == "primary" and numeric(row["input_tokens"], 0)
    ]
    summary["primary_context"] = {
        "reported_token_count": len(primary_tokens),
        "max_input_tokens": max(primary_tokens) if primary_tokens else None,
        "max_context_fraction": max(primary_tokens) / protocol["context_tokens"] if primary_tokens else None,
    }
    return summary, compact


def assemble_report(cases, records, manifests, references, baselines):
    """Pure report assembly; fixed allocation and ledger checks happen before this call."""
    by_key = {}
    for record in records:
        key = case_key(record)
        if key in by_key:
            raise ValueError("Duplicate Jev case attempt")
        by_key[key] = record
    expected = {case_key(case) for case in cases}
    if len(expected) != len(cases) or set(by_key) - expected:
        raise ValueError("Duplicate planned case or unexpected Jev attempt")
    primary = {}
    for task in TASKS:
        planned = [case for case in cases if case["phase"] == "primary" and case["task"] == task]
        attempted = [record for record in records if record["phase"] == "primary" and record["task"] == task]
        primary[task], _ = compare_primary(
            task, planned, attempted, baselines[task], manifests[task], references[task]
        )
    per_image = [compact_case(case, by_key.get(case_key(case))) for case in cases]
    for row in per_image:
        if row["phase"] != "primary":
            continue
        task, identifier = row["task"], row["sample"]
        sample, reference = manifests[task][identifier], references[task][identifier]
        row.update(
            gold_ai_generated=sample.get("labels", {}).get("ai_generated"),
            human_quality_mos=sample.get("labels", {}).get("quality_mos"),
            assistant_technical_quality=reference.get("technical_quality"),
            assistant_aesthetics=reference.get("aesthetics"),
        )
        for historical in baselines[task]:
            if historical["sample"] != identifier:
                continue
            for axis in ("technical_quality", "aesthetics", "ai_probability"):
                row[f"{historical['arm']}_{axis}"] = historical.get("normalized", {}).get(axis)
    operations = _operations(records)
    status = (
        "partial"
        if len(records) != len(cases)
        else "complete_with_invalid"
        if operations["invalid"]
        else "complete"
    )
    summary = {
        "status": status,
        "expected": len(cases),
        "unattempted": len(cases) - len(records),
        "operations": operations,
        "failures": [
            {
                "case_id": row.get("case_id"),
                "sample": row["sample"],
                "phase": row["phase"],
                "http_status": row.get("status"),
                "validation_error": row.get("validation_error"),
            }
            for row in records
            if not valid_record(row) or row.get("normalized") is None
        ],
        "primary": primary,
        "phases": {
            phase: {
                "expected": sum(case["phase"] == phase for case in cases),
                **_operations([row for row in records if row["phase"] == phase]),
            }
            for phase in PHASES
        },
        "comparison_limits": [
            "Historical comparator calls were made on 11 October 2026; latency is not a simultaneous serving test.",
            "Primary images and prompts are unchanged. Two unsupported panoramas use separate compatibility calls.",
            "Quality MAE against the assistant measures agreement, not human ground truth.",
            "Human quality MOS was collected at 3840 pixels; models see 512-pixel views. No human aesthetic ratings exist.",
            "The AI panel includes 12 non-photo generated images and 30 real DOCCI photos; style can confound origin.",
            "Size probes measure cost, tokens and latency, not prediction accuracy or a fitted optimal resolution.",
        ],
    }
    return summary, per_image


def display(value, digits=3):
    return f"{value:.{digits}f}" if numeric(value) else "—"


def _operation_cells(evaluation):
    op = evaluation["operations"]
    return (
        f"{op['valid']}/{evaluation['expected']}",
        display(op["effective_provider_inference_cost_usd"], 6),
        f"{display(op['latency_p50_seconds'])} / {display(op['latency_p95_seconds'])}",
    )


def render_markdown(summary, cases):
    """Keep the two objectives and the input compatibility limits visible."""
    op = summary["operations"]
    primary_valid = sum(panel["jev14"]["operations"]["valid"] for panel in summary["primary"].values())
    primary_expected = sum(panel["expected"] for panel in summary["primary"].values())
    lines = [
        "# Jev 1.14: quality and AI detection",
        "",
        (
            f"**{summary['status'].replace('_', ' ').capitalize()}: {primary_valid}/{primary_expected} primary results are valid.** "
            "The primary comparison uses the same image bytes and prompts as the earlier benchmark. "
            "Two panoramas require separate compatibility views; their results are excluded from the primary scores."
        ),
        "",
        "## Quality and aesthetics",
        "",
        "| Model | Valid / planned | Quality MAE | Aesthetics MAE | Human MOS rank correlation | Cost USD | Latency p50 / p95 s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {
        "jev14": "Jev 1.14",
        "decider": "Decider v1.1",
        "luna": "GPT-6 Luna",
        "gemini": "Gemini 3.8 Flash",
    }
    quality = summary["primary"]["quality"]
    for arm, evaluation in [("jev14", quality["jev14"]), *quality["baselines"].items()]:
        metrics = evaluation["metrics"]
        valid, cost, latency = _operation_cells(evaluation)
        mae = [
            display(metrics["assistant_reference"][axis]["mae_valid"])
            for axis in ("technical_quality", "aesthetics")
        ]
        rank = display(metrics["human_raw_correlations"]["technical_quality"]["spearman"])
        lines.append(f"| {names[arm]} | {valid} | {mae[0]} | {mae[1]} | {rank} | {cost} | {latency} |")
    lines.extend(
        [
            "",
            (
                "Lower MAE is better. Score metrics use valid outputs. These errors measure agreement with the saved assistant assessments, "
                "not human ground truth. Rank correlation uses human quality MOS. Human aesthetic ratings are absent. "
                "UHD-IQA has no low-quality examples below 2.6/5 in this panel, and its human ratings used 3,840-pixel views. "
                "This test does not establish recall for rejecting poor images."
            ),
            "",
            "## AI detection",
            "",
            "| Model | Valid / planned | Accuracy | AI detected | Real falsely flagged | Brier | AUROC | Cost USD | Latency p50 / p95 s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    ai = summary["primary"]["ai"]
    for arm, evaluation in [("jev14", ai["jev14"]), *ai["baselines"].items()]:
        metrics = evaluation["metrics"]
        counts = metrics["confusion_valid_only"]
        valid, cost, latency = _operation_cells(evaluation)
        accuracy = metrics["accuracy_all_labeled_invalid_as_wrong"]
        detected = f"{counts['tp']}/{counts['tp'] + counts['fn']}"
        flagged = f"{counts['fp']}/{counts['fp'] + counts['tn']}"
        lines.append(
            f"| {names[arm]} | {valid} | {display(accuracy * 100 if accuracy is not None else None, 1)}% "
            f"| {detected} | {flagged} | {display(metrics['brier_valid_only'])} "
            f"| {display(metrics['auroc_valid_only'])} | {cost} | {latency} |"
        )
    lines.extend(
        [
            "",
            (
                "AI labels come from source provenance. The fixed threshold is 0.5. "
                "Accuracy counts invalid or missing outputs as wrong; detection counts, Brier and AUROC use valid outputs. "
                "The panel has 30 real photos and 30 generated images, including 12 graphics or renders. "
                "This source and style mix limits claims about general photo detection."
            ),
            "",
            (
                "The JSON report also compares each baseline on the exact subset with valid Jev results. "
                "No prompt, threshold or image allocation was tuned on these results."
            ),
            "",
            "## Separate compatibility and size checks",
            "",
            "| Phase | Image | Size | Quality / aesthetics | Input tokens | Cost USD | Latency s |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cases:
        if row["phase"] == "primary":
            continue
        size = f"{row['image_width']}×{row['image_height']}"
        scores = f"{display(row['technical_quality'])} / {display(row['aesthetics'])}"
        lines.append(
            f"| {row['phase']} | {row['sample']} | {size} | {scores} "
            f"| {row['input_tokens'] if row['input_tokens'] is not None else '—'} "
            f"| {display(row['effective_cost_usd'], 6)} | {display(row['seconds'])} |"
        )
    lines.extend(
        [
            "",
            (
                "Compatibility views add lossless padding to meet the minimum image dimension. "
                "They are not the original benchmark views. Size probes use two development images at three resolutions; "
                "they show observed token, cost and latency changes, not an accuracy result or an optimal resolution. "
                "The 1,024-pixel probes upscale the decoded 512-pixel source and add no source detail. "
                "Reported input tokens include text, questions and image processing; request bytes are retained separately."
            ),
            "",
            (
                f"All {op['attempted']} attempts have a known cost subtotal of "
                f"${display(op['known_effective_provider_inference_cost_usd'], 6)}. "
                f"Unknown-cost attempts: {op['unknown_cost_count']}. Unknown does not mean free. "
                "Costs include failed calls when charges are available. BYOK upstream inference is included."
            ),
            "",
            (
                "Comparator calls were made on 11 October 2026; Jev was tested later. "
                "Latency includes network time and is not a simultaneous serving comparison. "
                "Image-token accounting may differ by provider. The prompts were selected for Decider and reused without Jev optimisation."
            ),
            "",
            (
                "The [OpenRouter model catalogue](https://openrouter.ai/typesafe/jev-1.14) lists a 262,144-token context, "
                "$0.042 per million input tokens and no output-token charge. Measured costs above use response usage. "
                "Both experimental headers were sent; forwarding by OpenRouter was not independently confirmed. "
                "See [multimodal Decisions](https://openrouter.ai/docs/guides/community/multimodal-decisions) "
                "and [TypeSafe image constraints](https://docs.typesafe.ai/images)."
            ),
            "",
            (
                "[Per-image results](cases.csv) and [machine-readable summary](summary.json) retain the exact joins, "
                "coverage, error counts, hashes and paired-subset comparisons. Raw images and responses remain outside Git."
            ),
            "",
        ]
    )
    context = summary.get("primary_context", {})
    if context.get("max_input_tokens") is not None:
        lines.insert(
            -3,
            f"The largest primary request used {context['max_input_tokens']:,} reported input tokens, "
            f"or {100 * context['max_context_fraction']:.2f}% of the listed context window.\n",
        )
    if summary.get("failures"):
        counts = Counter(row["validation_error"] or "Unspecified failure" for row in summary["failures"])
        detail = "; ".join(f"{count} × {reason}" for reason, count in counts.items())
        lines[4:4] = [
            f"Invalid results were retained: {detail}. Failed or missing results are never treated as negative AI detections.",
            "",
        ]
    review = summary.get("parser_review")
    if review:
        lines[4:4] = [
            (
                f"The original parser rejected {review['original_strict_parser_rejections']} responses. "
                f"A fixed rounding rule recovered {review['recovered_strict_parser_rejections']} of them and was applied "
                f"to all {review['applied_to_responses']} responses. It preserves native scores and probabilities; "
                "no values were renormalised. The raw results and original parser errors remain intact."
            ),
            "",
        ]
    return "\n".join(lines)


def write_report(root, output):
    """Create an immutable compact snapshot; never replace an earlier report."""
    summary, cases = build_report(root)
    content = io.StringIO()
    fields = list(dict.fromkeys(key for row in cases for key in row))
    writer = csv.DictWriter(content, fieldnames=fields)
    writer.writeheader()
    writer.writerows(cases)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "cases.csv").write_text(content.getvalue())
    (output / "README.md").write_text(render_markdown(summary, cases))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = write_report(ROOT, args.output)
    print(f"Saved {summary['status']} report with {summary['operations']['attempted']} attempts")


if __name__ == "__main__":
    main()
