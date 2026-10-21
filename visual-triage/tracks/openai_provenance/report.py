"""Build an offline provenance report without converting missing signals to real labels."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_protocol import digest
from tracks.openai_provenance.client import _parse_signals

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("image_only", "provenance", "all_context")
SOURCES = ("real", "dalle2", "dalle3", "firefly", "midjourney-v5")
GUIDE = "https://developers.openai.com/api/docs/guides/content-provenance"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lower = math.floor(index)
    return values[lower] + (values[math.ceil(index)] - values[lower]) * (index - lower)


def timing(values):
    return {"count": len(values), "p50_seconds": percentile(values, 0.5),
            "p95_seconds": percentile(values, 0.95)}


def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite non-negative value")
    return value


def failure_details(rows):
    """Extract status and a bounded cooldown without publishing error bodies."""
    failures = []
    for row in rows:
        if row["success"]:
            continue
        error = row.get("error") or {}
        code = error.get("code")
        if not isinstance(code, str) or not re.fullmatch(r"[a-z_]{1,64}", code):
            code = None
        match = re.search(r"\btry again after ([0-9]{1,8}(?:\.[0-9]{1,6})?) seconds?\b",
                          error.get("detail", ""), flags=re.IGNORECASE) if row["http_status"] == 429 else None
        delay = float(match[1]) if match else None
        if delay is not None and delay > 604800:
            delay = None
        failures.append({"case": row["case"], "attempt": row["attempt"],
                         "http_status": row["http_status"], "error_kind": error.get("kind"), "error_code": code,
                         "server_cooldown_seconds": delay,
                         "cooldown_source": "error_message" if delay is not None else None,
                         "retry_after_header_present": bool(row.get("response_headers", {}).get("retry-after"))})
    return failures


def read_retries(track, data, rows, protocol, protocol_hash):
    path = data / "retries.jsonl"
    content = path.read_bytes() if path.exists() else b""
    retries = [json.loads(line) for line in content.splitlines() if line.strip()]
    amendment_path = track / "resume-amendment.json"
    if not amendment_path.exists():
        if retries or any(row.get("amendment_sha256") for row in rows):
            raise ValueError("Amended attempts require their frozen amendment")
        return retries, None, None
    amendment = json.loads(amendment_path.read_text())
    amendment_hash = sha(amendment_path)
    if (amendment["protocol_sha256"] != protocol_hash
            or amendment["resume_code_sha256"] != sha(track / "resume.py")
            or amendment["max_retries_per_case"] != 1 or amendment["eligible_status"] != 502):
        raise ValueError("Invalid frozen resume amendment")
    if retries and len(rows) != protocol["cases"]:
        raise ValueError("Retries must follow the complete primary cohort")
    by_case = {row["case"]: row for row in rows}
    for failure in amendment["observed_failures"]:
        parent = by_case.get(failure["case"])
        if (not parent or parent["success"] or parent["http_status"] != 502
                or digest(parent) != failure["record_sha256"] or parent["attempt"] != failure["attempt"]):
            raise ValueError("Reviewed failure differs from the primary ledger")
    last_reviewed = max(failure["attempt"] for failure in amendment["observed_failures"])
    for row in rows:
        if ((row["attempt"] > last_reviewed or "amendment_sha256" in row)
                and row.get("amendment_sha256") != amendment_hash):
            raise ValueError("Resumed primary request has another amendment")
    seen = set()
    for index, row in enumerate(retries, len(rows) + 1):
        parent = by_case.get(row["case"])
        if (row["case"] in seen or not parent or parent["success"] or parent["http_status"] != 502
                or row["retry_of_attempt"] != parent["attempt"]
                or row["retry_of_record_sha256"] != digest(parent)
                or row["input_sha256"] != parent["input_sha256"]
                or row["protocol_sha256"] != protocol_hash or row["amendment_sha256"] != amendment_hash
                or row["source"] != parent["source"] or row["endpoint"] != protocol["endpoint"]
                or row["attempt"] != index):
            raise ValueError("Retry does not match its unique original request")
        if row["success"] and (not row["schema_valid"] or _parse_signals(row["raw_response"]) != row["signals"]):
            raise ValueError("Successful retry has invalid signal evidence")
        seen.add(row["case"])
    if len(rows) + len(retries) > amendment["max_total_attempts"]:
        raise ValueError("Amended attempt limit exceeded")
    return retries, amendment_hash, hashlib.sha256(content).hexdigest() if content else None


def read_inputs(root):
    track, data = root / "tracks/openai_provenance", root / "data/openai-provenance-v1"
    protocol = json.loads((track / "protocol.json").read_text())
    pins = {data / "manifest.json": protocol["manifest_sha256"],
            data / "research/guide.md": protocol["guide_sha256"],
            root / "tracks/metadata_v3/dataset.json": protocol["source_dataset_sha256"],
            root / "tracks/metadata_v3/protocol.json": protocol["source_protocol_sha256"]}
    pins.update({root / path: expected for path, expected in protocol["execution_sha256"].items()})
    if any(sha(path) != expected for path, expected in pins.items()):
        raise ValueError("A frozen study input changed")
    cases = json.loads((data / "manifest.json").read_text())["cases"]
    if len(cases) != protocol["cases"] or len({case["case"] for case in cases}) != len(cases):
        raise ValueError("Invalid manifest case allocation")
    if dict(Counter(case["source"] for case in cases)) != protocol["source_counts"]:
        raise ValueError("Manifest source allocation changed")
    expected = {case["case"]: case for case in cases}
    ledger = data / "attempts.jsonl"
    ledger_bytes = ledger.read_bytes() if ledger.exists() else b""
    rows = [json.loads(line) for line in ledger_bytes.splitlines() if line.strip()]
    seen = set()
    protocol_hash = sha(track / "protocol.json")
    for index, row in enumerate(rows, 1):
        case = expected.get(row["case"])
        if not case or row["case"] in seen or row["input_sha256"] != case["native_sha256"]:
            raise ValueError("Unexpected, duplicate or changed ledger image")
        if row["protocol_sha256"] != protocol_hash or row["endpoint"] != protocol["endpoint"]:
            raise ValueError("Ledger protocol or endpoint differs")
        if row["source"] != case["source"] or row["attempt"] != index:
            raise ValueError("Ledger source or attempt order differs")
        if row["success"] and (not row["schema_valid"]
                               or _parse_signals(row["raw_response"]) != row["signals"]):
            raise ValueError("Successful response has invalid signal evidence")
        seen.add(row["case"])
    if len(rows) > protocol.get("maximum_attempts", protocol["cases"]):
        raise ValueError("Primary attempt limit exceeded")
    retries, amendment_hash, retry_hash = read_retries(track, data, rows, protocol, protocol_hash)
    return protocol, cases, rows, retries, {"protocol_sha256": protocol_hash,
            "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest() if ledger_bytes else None,
            "amendment_sha256": amendment_hash, "retry_ledger_sha256": retry_hash}


def detector_metrics(rows):
    valid = [row for row in rows if row["valid"] == "True"]
    probabilities = {row["case"]: number(row["prediction"]) for row in valid}
    if any(value > 1 for value in probabilities.values()):
        raise ValueError("Invalid Decider probability")
    gold = {row["case"]: int(row["gold"]) for row in rows}
    return {"images": len(rows), "valid": len(valid), "ai_images": sum(gold.values()),
            "real_images": len(rows) - sum(gold.values()),
            "correct": sum((value >= 0.5) == gold[case] for case, value in probabilities.items()),
            "ai_detected": sum(value >= 0.5 and gold[case] for case, value in probabilities.items()),
            "real_false_flags": sum(value >= 0.5 and not gold[case] for case, value in probabilities.items()),
            "brier": (sum((value - gold[case]) ** 2 for case, value in probabilities.items())
                      + len(rows) - len(valid)) / len(rows) if rows else None,
            "cost_usd": sum(number(row["effective_cost_usd"]) for row in rows),
            "latency": timing([number(row["seconds"]) for row in valid])}


def decider_comparison(root, cases, valid_api_cases):
    expected = {case["case"]: case for case in cases}
    with (root / "tracks/metadata_v3/results/cases.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["condition"] == "native" and row["prompt_variant"] == "selected" and row["arm"] in ARMS]
    metrics, matched, joined = {}, {}, {case: {} for case in expected}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if len(selected) != len(expected) or {row["case"] for row in selected} != set(expected):
            raise ValueError("Decider cohort does not match the provenance cohort")
        for row in selected:
            case = expected[row["case"]]
            if (row["native_sha256"] != case["native_sha256"]
                    or row["delivered_sha256"] != case["delivered_sha256"]
                    or int(row["gold"]) != int(case["gold_ai"])):
                raise ValueError("Decider image hash or gold label differs")
            joined[row["case"]][arm] = float(row["prediction"]) if row["valid"] == "True" else None
        metrics[arm] = detector_metrics(selected)
        matched[arm] = detector_metrics([row for row in selected if row["case"] in valid_api_cases])
        metrics[arm]["source_slices"] = {
            source: detector_metrics([row for row in selected if expected[row["case"]]["source"] == source])
            for source in SOURCES if any(case["source"] == source for case in cases)}
    return metrics, joined, matched


def historical_comparison(root, cases):
    path = root / "summaries/initial_benchmark.csv"
    if not path.exists():
        return {"status": "unavailable", "models": {}}
    with path.open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["panel"] == "ai" and row["phase"] == "primary"]
    hashes = {case[key] for case in cases for key in ("native_sha256", "delivered_sha256")}
    if any(row["delivered_sha256"] in hashes for row in rows):
        raise ValueError("Historical cohort unexpectedly overlaps with the native study")
    models = {}
    for model in ("decider", "luna", "gemini"):
        selected = [{**row, "case": row["sample"], "prediction": row["ai_probability"],
                     "gold": str(int(row["gold"].lower() in {"true", "1", "1.0"}))}
                    for row in rows if row["evaluation"] == model]
        if selected:
            models[model] = detector_metrics(selected)
    return {"status": "separate_cohort", "overlapping_image_hashes": 0, "models": models}


def build_report(root=ROOT):
    root = Path(root)
    protocol, cases, primary, retries, pins = read_inputs(root)
    rows = primary + retries
    failure_records = failure_details(rows)
    by_case = {row["case"]: row for row in primary}
    by_case.update({row["case"]: row for row in retries if row["success"]})
    valid_api_cases = {case for case, row in by_case.items() if row["success"]}
    decider, predictions, matched = decider_comparison(root, cases, valid_api_cases)
    compact = []
    for case in cases:
        row = by_case.get(case["case"])
        original = next((attempt for attempt in primary if attempt["case"] == case["case"]), None)
        success = bool(row and row["success"])
        signals = row["signals"] if success else {}
        detected = any(signal["outcome"] == "detected" for signal in signals.values()) if success else None
        compact.append({"case": case["case"], "source": case["source"], "gold_ai": case["gold_ai"],
                        "openai_origin": case["openai_origin"], "native_sha256": case["native_sha256"],
                        "delivered_sha256": case["delivered_sha256"], "protocol_sha256": pins["protocol_sha256"],
                        "attempted": row is not None, "success": success,
                        "effective_attempt": row["attempt"] if row else None,
                        "effective_retry_of_attempt": row.get("retry_of_attempt") if row else None,
                        "primary_success": original["success"] if original else None,
                        "primary_error_kind": (original.get("error") or {}).get("kind") if original else None,
                        "attempt_count": sum(attempt["case"] == case["case"] for attempt in rows),
                        "openai_signal_detected": detected,
                        "disposition": "openai_signal" if detected else "unresolved" if success else
                                       "failed" if row else "unattempted",
                        "c2pa_outcome": signals.get("c2pa", {}).get("outcome"),
                        "c2pa_validation_state": signals.get("c2pa", {}).get("validation_state"),
                        "c2pa_issuer": signals.get("c2pa", {}).get("issuer"),
                        "synthid_outcome": signals.get("synthid", {}).get("outcome"),
                        "seconds": row["elapsed_seconds"] if row else None,
                        "http_status": row["http_status"] if row else None,
                        "cost_usd": row.get("cost_usd") if row else None,
                        "error_kind": (row.get("error") or {}).get("kind") if row else None,
                        **{f"decider_{arm}_ai_probability": value for arm, value in predictions[case["case"]].items()}})
    slices = {}
    for source in SOURCES:
        subset = [row for row in compact if row["source"] == source]
        slices[source] = {"images": len(subset), "attempted": sum(row["attempted"] for row in subset),
                          "success": sum(row["success"] for row in subset),
                          "openai_signals": sum(row["openai_signal_detected"] is True for row in subset),
                          "unresolved": sum(row["disposition"] == "unresolved" for row in subset),
                          "failed_cases": sum(row["disposition"] == "failed" for row in subset),
                          "unattempted": sum(row["disposition"] == "unattempted" for row in subset),
                          "c2pa_states": dict(Counter(row["c2pa_validation_state"] for row in subset
                                                       if row["success"]))}
    successes = [row for row in rows if row["success"]]
    failures = [row for row in rows if not row["success"]]
    known_costs = [number(row["cost_usd"]) for row in rows if row.get("cost_usd") is not None]
    server_times = [number(row["response_headers"]["openai-processing-ms"]) / 1000 for row in successes
                    if row.get("response_headers", {}).get("openai-processing-ms") is not None]
    summary = {"version": 1, **pins,
               "decider_results_sha256": sha(root / "tracks/metadata_v3/results/cases.csv"),
               "status": "complete" if len(valid_api_cases) == len(cases) else "partial",
               "expected": len(cases), "attempted": len(rows), "attempted_cases": len(primary),
               "success": len(valid_api_cases), "failed": len(failures),
               "failed_cases": sum(not row["success"] for row in by_case.values()),
               "retry_attempts": len(retries), "recovered_cases": sum(row["success"] for row in retries),
               "failure_records": failure_records,
               "failed_http_status_counts": dict(Counter(str(row["http_status"]) for row in failures)),
               "unattempted": len(cases) - len(primary), "by_source": slices,
               "openai_origin_images": sum(case["openai_origin"] for case in cases),
               "openai_origin_successes": sum(row["openai_origin"] and row["success"] for row in compact),
               "openai_origin_detections": sum(row["openai_origin"] and row["openai_signal_detected"] is True
                                               for row in compact),
               "real_false_attributions": slices["real"]["openai_signals"],
               "real_successes": slices["real"]["success"],
               "other_ai_false_attributions": sum(slices[source]["openai_signals"]
                                                   for source in ("firefly", "midjourney-v5")),
               "other_ai_successes": sum(slices[source]["success"] for source in ("firefly", "midjourney-v5")),
               "successful_latency": timing([number(row["elapsed_seconds"]) for row in successes]),
               "failed_latency": timing([number(row["elapsed_seconds"]) for row in failures]),
               "successful_server_processing": timing(server_times),
               "known_cost_usd": sum(known_costs) if known_costs else None,
               "unknown_cost_attempts": len(rows) - len(known_costs),
               "total_cost_usd": sum(known_costs) if rows and len(known_costs) == len(rows) else None,
               "decider": decider, "decider_matched_valid_api": matched,
               "historical": historical_comparison(root, cases),
               "guide": GUIDE, "limitations": protocol["limitations"],
               "interpretation": "not_detected is unresolved, not a real-image prediction"}
    return {"summary": summary, "cases": compact}


def render_markdown(report):
    summary = report["summary"]
    latency = summary["successful_latency"]
    seconds = lambda value: "unavailable" if value is None else f"{value:.3f} s"
    lines = ["# OpenAI provenance: native benchmark check", "",
             (f"**{summary['status'].capitalize()}: valid results for {summary['success']}/{summary['expected']} images.** "
              f"Requests: {summary['attempted']}, including {summary['failed']} failed attempts and "
              f"{summary['retry_attempts']} retries. Recovered cases: {summary['recovered_cases']}; "
              f"unresolved failures: {summary['failed_cases']}; unattempted: {summary['unattempted']}."), "",
             (f"OpenAI signals were detected in {summary['openai_origin_detections']}/"
              f"{summary['openai_origin_successes']} successfully checked older DALL-E images "
              f"({summary['openai_origin_images']} planned). Missing signals remain unresolved. "
             "They do not establish camera origin or provide general AI-detection accuracy."), "",
             "| Source | Valid / tried / planned | Failed cases | Untried | OpenAI signals | C2PA states |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for source, values in summary["by_source"].items():
        states = ", ".join(f"{key}: {value}" for key, value in sorted(values["c2pa_states"].items())) or "—"
        lines.append(f"| {source} | {values['success']} / {values['attempted']} / {values['images']} | "
                     f"{values['failed_cases']} | {values['unattempted']} | {values['openai_signals']} | {states} |")
    lines += ["", (f"False OpenAI attributions among successful checks: {summary['real_false_attributions']}/"
              f"{summary['real_successes']} real photos and {summary['other_ai_false_attributions']}/"
              f"{summary['other_ai_successes']} images from other generators. C2PA validity and OpenAI attribution "
              "are separate: a valid Adobe manifest can return `not_detected` for an OpenAI signal."), "",
              (f"Successful request latency: median {seconds(latency['p50_seconds'])}, "
              f"P95 {seconds(latency['p95_seconds'])}. "
              f"API cost is {'unknown' if summary['total_cost_usd'] is None else '$' + format(summary['total_cost_usd'], '.6f')}; "
              f"{summary['unknown_cost_attempts']} attempts have no charge receipt. Unknown does not mean free."), "",
              "## Comparison with Decider", "",
              ("These are the same image identities. The API received native files; Decider saw 512-pixel JPEG "
              "crops and, for metadata arms, extracted text. Decider predicts broad AI origin. The API checks "
              "supported OpenAI provenance signals, so its unresolved results cannot be ranked as classifier errors."), "",
              "| Decider prompt | Accuracy | AI detected | Real false flags | Brier | Cost | Median |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    if summary["failure_records"]:
        failures = ", ".join(f"HTTP {status}: {count}" for status, count in
                             sorted(summary["failed_http_status_counts"].items()))
        delays = [row["server_cooldown_seconds"] for row in summary["failure_records"]
                  if row["server_cooldown_seconds"] is not None]
        detail = f"Request failures: {failures}."
        if delays:
            detail += (f" The server's error message asked for a {max(delays):.3f}-second "
                       f"wait ({max(delays) / 60:.1f} minutes). This records its response, not a live countdown.")
        lines[3:3] = ["", detail]
    for arm, values in summary["decider"].items():
        lines.append(f"| {arm} | {values['correct'] / values['images']:.1%} | "
                     f"{values['ai_detected']}/{values['ai_images']} | {values['real_false_flags']}/{values['real_images']} | "
                     f"{values['brier']:.4f} | ${values['cost_usd']:.6f} | {seconds(values['latency']['p50_seconds'])} |")
    lines += ["", (f"The table uses the full {summary['expected']}-image Decider reference. "
              f"The API has valid results on {summary['success']} of those same images.")]
    if 0 < summary["success"] < summary["expected"]:
        lines += ["", ("This small coverage sample cannot establish a ranking. The subset below contains only "
                  "images with valid API results; failed and untried files remain outside it."), "",
                  "| Decider on valid API cases only | Correct | AI detected | Real false flags |",
                  "| --- | ---: | ---: | ---: |"]
        for arm, values in summary["decider_matched_valid_api"].items():
            lines.append(f"| {arm} | {values['correct']}/{values['images']} | "
                         f"{values['ai_detected']}/{values['ai_images']} | "
                         f"{values['real_false_flags']}/{values['real_images']} |")
    lines += ["", "### Earlier models: separate images", "",
              ("The initial 60-image AI panel has no image-hash overlap with this trial. It provides context, "
              "not a ranking against native provenance checks."), "",
              "| Model | AI detected | Real false flags | Cost / 60 | Median |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for model, values in summary["historical"]["models"].items():
        label = {"decider": "Decider v1.1", "luna": "GPT-6 Luna", "gemini": "Gemini 3.8 Flash"}[model]
        lines.append(f"| {label} | {values['ai_detected']}/{values['ai_images']} | "
                     f"{values['real_false_flags']}/{values['real_images']} | ${values['cost_usd']:.6f} | "
                     f"{seconds(values['latency']['p50_seconds'])} |")
    lines += ["", ("The AI files come from the 2023 Synthbuster release. This cohort has no current GPT-image "
              "positive controls, so it cannot measure recall on current supported output. Use provenance as an "
              "early check and retain visual review when the result is unresolved."), "",
              (f"[API guide]({GUIDE}) · [Per-image results](results/cases.csv) · "
              "[Compact metrics](results/summary.json). Raw responses and image files remain outside Git."), ""]
    return "\n".join(lines)


def write_report(report, destination):
    destination = Path(destination)
    paths = [destination / "report.md", destination / "results/summary.json", destination / "results/cases.csv"]
    if any(path.exists() for path in paths):
        raise ValueError("Report snapshot exists; use a fresh destination to preserve it")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(report["cases"][0]))
    writer.writeheader()
    writer.writerows(report["cases"])
    contents = [render_markdown(report), json.dumps(report["summary"], indent=2, allow_nan=False) + "\n",
                stream.getvalue()]
    for path, content in zip(paths, contents, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="") as handle:
            handle.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if report["summary"]["status"] != "complete" and not (args.allow_partial and args.output_dir):
        raise ValueError("Partial reports require --allow-partial and a fresh --output-dir")
    write_report(report, args.output_dir or ROOT / "tracks/openai_provenance")


if __name__ == "__main__":
    main()
