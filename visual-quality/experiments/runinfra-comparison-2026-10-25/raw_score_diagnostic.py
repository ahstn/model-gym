"""Post-hoc numeric diagnostic; invalid label records remain invalid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from visual_quality.labels import LIMITATIONS, RATING_KEYS, parse_rating


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _module("raw_diagnostic_runner", "runner.py")
report = _module("raw_diagnostic_report", "report.py")
TOLERANCE = Decimal("1e-12")


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _json(text: str, *, decimal_integers: bool = False):
    return json.loads(
        text,
        parse_float=Decimal,
        parse_int=Decimal if decimal_integers else int,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )


def _validate_non_score_fields(rating: dict) -> None:
    """Check the original contract without replacing or rounding its score."""
    if not isinstance(rating, dict) or set(rating) != set(RATING_KEYS):
        raise ValueError("Rating fields differ from the original contract")
    if rating["task"] != "technical_quality" or rating["status"] not in {"ok", "unscorable"}:
        raise ValueError("Invalid task or status")
    evidence = rating["evidence"]
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence.split()) > 30:
        raise ValueError("Invalid evidence")
    limitations = rating["limitations"]
    if (
        not isinstance(limitations, list)
        or any(not isinstance(value, str) or value not in LIMITATIONS for value in limitations)
        or len(set(limitations)) != len(limitations)
    ):
        raise ValueError("Invalid limitations")


def classify_content(content: str) -> dict:
    """Return exact numeric text and eligibility, never a repaired label."""
    result = {
        "response_content": content,
        "included_in_numeric_diagnostic": False,
        "strict_label_valid": False,
        "precision_only_tenths_failure": False,
        "raw_score_decimal": None,
        "evidence": None,
        "exclusion_reason": None,
    }
    try:
        rating = _json(content, decimal_integers=True)
        if isinstance(rating, dict):
            if isinstance(rating.get("score"), Decimal):
                result["raw_score_decimal"] = str(rating["score"])
            if isinstance(rating.get("evidence"), str):
                result["evidence"] = rating["evidence"]
        _validate_non_score_fields(rating)
        score = rating["score"]
        if rating["status"] == "unscorable":
            if score is not None:
                raise ValueError("Unscorable rating has a numeric score")
            parse_rating(content)
            result.update(strict_label_valid=True, exclusion_reason="unscorable")
            return result
        if not isinstance(score, Decimal) or not score.is_finite() or not Decimal(1) <= score <= Decimal(5):
            raise ValueError("Score is not a finite number within 1..5")
        if score % Decimal("0.1") == 0:
            parse_rating(content)
            result.update(strict_label_valid=True, included_in_numeric_diagnostic=True)
            return result
        deviation = abs(score - score.quantize(Decimal("0.1")))
        if deviation <= TOLERANCE:
            result.update(precision_only_tenths_failure=True, included_in_numeric_diagnostic=True)
        else:
            result["exclusion_reason"] = "off_grid_beyond_precision_tolerance"
    except (ValueError, TypeError, ArithmeticError, KeyError, RecursionError):
        result["exclusion_reason"] = "invalid_rating_contract_or_json"
    return result


def diagnose_result(run: dict, result: dict) -> dict:
    """Require an intact successful envelope before inspecting the numeric token."""
    record = {
        "included_in_numeric_diagnostic": False,
        "strict_label_valid": False,
        "precision_only_tenths_failure": False,
        "raw_score_decimal": None,
        "response_content": None,
        "cost_usd": None,
        "exclusion_reason": None,
    }
    try:
        if not isinstance(result, dict) or result.get("transport_error") is not None:
            raise ValueError("Transport error or invalid result")
        if type(result.get("http_status")) is not int or result["http_status"] != 200:
            raise ValueError("HTTP status is not 200")
        body = _json(result["body"])
        if not isinstance(body, dict) or body.get("error"):
            raise ValueError("Provider error or invalid envelope")
        if body.get("model") not in {run["settings"]["model"], run["requested_model"]}:
            raise ValueError("Unexpected exact model identity")
        choices = body["choices"]
        if not isinstance(choices, list) or len(choices) != 1 or choices[0]["finish_reason"] != "stop":
            raise ValueError("Incomplete or ambiguous choice")
        message = choices[0]["message"]
        if not isinstance(message, dict) or message.get("refusal"):
            raise ValueError("Refusal or invalid message")
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError("Content must be JSON text")
        record["response_content"] = content
        cost = runner.cost(body)
        if cost is None:
            raise ValueError("Cost is unknown or inconsistent")
        record["cost_usd"] = str(cost)
        record.update(classify_content(content))
        record["model_returned"] = body["model"]
    except (ValueError, TypeError, ArithmeticError, KeyError, IndexError, AttributeError, RecursionError):
        record["exclusion_reason"] = "invalid_response_envelope_or_unknown_cost"
    return record


def _read_pinned(path: Path, pins: dict[Path, str]):
    data = path.read_bytes()
    pins[path] = hashlib.sha256(data).hexdigest()
    return _json(data.decode())


def load_journals(run: dict, journals: list[Path]) -> tuple[list[dict], dict[Path, str]]:
    """Read both journals once and reject duplicate requests or changed payload pins."""
    cases = {case["case_id"]: case for case in run["cases"]}
    records, pins, seen = [], {}, set()
    for root in journals:
        if not root.exists():
            continue
        for directory in sorted(root.iterdir()):
            case_id = directory.name
            if not directory.is_dir() or case_id not in cases or cases[case_id]["skip_reason"]:
                raise ValueError("Unexpected or size-excluded attempt directory")
            if case_id in seen:
                raise ValueError("Duplicate case across attempt journals")
            seen.add(case_id)
            case = cases[case_id]
            request = _read_pinned(directory / "request.json", pins)
            expected = {
                "case_id": case_id,
                "run_id": run["run_id"],
                "request_sha256": case["request_sha256"],
                "request_bytes": case["request_bytes"],
            }
            if any(request.get(key) != value for key, value in expected.items()):
                raise ValueError("Request metadata differs from the frozen run")
            body = runner.payload(run, case)
            if (
                runner.digest(body) != case["request_sha256"]
                or len(runner.encoded(body)) != case["request_bytes"]
            ):
                raise ValueError("Payload differs from the frozen request")
            path = directory / "result.json"
            if path.exists():
                try:
                    result = _read_pinned(path, pins)
                    record = diagnose_result(run, result)
                except (ValueError, TypeError, ArithmeticError, RecursionError):
                    record = {
                        "included_in_numeric_diagnostic": False,
                        "strict_label_valid": False,
                        "precision_only_tenths_failure": False,
                        "raw_score_decimal": None,
                        "response_content": None,
                        "cost_usd": None,
                        "exclusion_reason": "malformed_saved_result",
                    }
            else:
                record = {
                    "included_in_numeric_diagnostic": False,
                    "strict_label_valid": False,
                    "precision_only_tenths_failure": False,
                    "raw_score_decimal": None,
                    "response_content": None,
                    "cost_usd": None,
                    "exclusion_reason": "missing_saved_result",
                }
            record.update(
                case_id=case_id,
                journal=str(root.relative_to(PROJECT)),
                request_sha256=case["request_sha256"],
                view_sha256=case["view"]["view_sha256"],
            )
            records.append(record)
    return records, pins


def comparison(
    cases: list[dict], references: dict, baseline: list[dict], records: list[dict]
) -> tuple[dict, list[dict]]:
    """Use actual numeric values in metric rows, without creating label records."""
    rows = report.comparison_rows(cases, references, baseline, [])
    expected = {case["case_id"] for case in cases}
    indexed = report.original.indexed(records, "case_id", expected, "raw diagnostic case")
    for row in rows:
        record = indexed.get(row["case_id"], {})
        row["raw_score_decimal"] = record.get("raw_score_decimal")
        row["runinfra_evidence"] = record.get("evidence")
        row["strict_label_valid"] = record.get("strict_label_valid", False)
        row["precision_only_tenths_failure"] = record.get("precision_only_tenths_failure", False)
        row["diagnostic_exclusion_reason"] = record.get("exclusion_reason", "not_attempted")
        if not record.get("included_in_numeric_diagnostic"):
            continue
        score = float(Decimal(record["raw_score_decimal"]))
        interval = (
            [row["reference_lower"], row["reference_upper"]] if row["reference_lower"] is not None else None
        )
        row.update(
            runinfra_score=score,
            runinfra_status="raw_numeric_diagnostic",
            runinfra_distance=report.original.distance(score, interval),
            runinfra_critical=report.original.critical(score, interval),
            view_comparison="identical_bytes",
        )
        row["same_view_pair"] = row["baseline_distance"] is not None and row["runinfra_distance"] is not None
        if row["same_view_pair"]:
            row["score_delta_runinfra_minus_baseline"] = score - row["baseline_score"]
            row["interval_distance_improvement"] = row["baseline_distance"] - row["runinfra_distance"]
    paired = [row for row in rows if row["same_view_pair"]]
    strict = [row for row in paired if row["strict_label_valid"]]
    selected = {row["case_id"] for row in paired}
    planned_families = Counter(row["family_id"] for row in rows)
    paired_families = Counter(row["family_id"] for row in paired)
    complete = {family for family, count in planned_families.items() if paired_families[family] == count}
    summary = {
        "schema_version": "runinfra-raw-score-diagnostic-v1",
        "post_hoc": True,
        "invalid_labels_remain_invalid": True,
        "training_ready": False,
        "full_panel_improvement_claim_allowed": False,
        "explanation_support": {"status": "not_assessed", "unsupported_count": None},
        "precision_tolerance_for_diagnostic_only": str(TOLERANCE),
        "method": "Actual Decimal values converted to float only for metrics; no score is rounded or exported as a label",
        "same_view_numeric_subset": report._paired_metrics(paired),
        "strict_valid_score_subset": report._paired_metrics(strict),
        "complete_original_families_sensitivity": report._paired_metrics(
            [row for row in paired if row["family_id"] in complete]
        ),
        "incomplete_family_ids": sorted(
            family for family in planned_families if 0 < paired_families[family] < planned_families[family]
        ),
        "entirely_missing_family_ids": sorted(
            family for family in planned_families if paired_families[family] == 0
        ),
        "strata_same_view_numeric_subset": {
            source: report._paired_metrics([row for row in paired if row["stratum"] == source])
            for source in report.STRATA
        },
        "precision_only_case_ids": [row["case_id"] for row in paired if row["precision_only_tenths_failure"]],
        "not_compared_case_ids": [row["case_id"] for row in rows if row["case_id"] not in selected],
        "excluded_attempt_reasons": dict(
            Counter(
                record["exclusion_reason"]
                for record in records
                if not record["included_in_numeric_diagnostic"]
            )
        ),
        "attempts": records,
        "limitations": [
            "This analysis was added after the first decimal-step failure and is post hoc.",
            "Invalid responses remain excluded from validated observations and label promotion.",
            "A selected incomplete panel cannot establish accuracy on all 50 images.",
            "The references are assistant judgments, not human ground truth.",
            "Float approximations are used for metrics; exact Decimal strings and response content remain saved.",
        ],
    }
    return summary, rows


def build_saved() -> tuple[dict, list[dict], dict]:
    run = runner.read_run()
    data = runner.DATA
    journals = [data / "attempts", data / "continuation-01/attempts"]
    records, pins = load_journals(run, journals)
    manifest = _read_pinned(report.ORIGINAL / "manifest.json", pins)
    cases = manifest["stages"]["a"]["cases"]
    references, freeze = report.original.load_references("a", {case["case_id"] for case in cases})
    baseline_path = report.BASELINE / "observations.jsonl"
    baseline = report.original.read_rows(baseline_path)
    summary, rows = comparison(cases, references, baseline, records)
    observations = []
    for root in (data, data / "continuation-01"):
        path = root / "observations.jsonl"
        if path.exists():
            before = path.read_bytes()
            pins[path] = hashlib.sha256(before).hexdigest()
            observations.extend(_json(line) for line in before.decode().splitlines() if line.strip())
    # parse_rating expects normal JSON numeric types in existing validated exports.
    observations = json.loads(json.dumps(observations, default=float))
    strict_ids = {record["case_id"] for record in records if record["strict_label_valid"]}
    if {row["case_id"] for row in observations} != strict_ids:
        raise ValueError("Validated observation exports differ from strict-valid reply cases")
    validated, _ = report.build_report(cases, references, baseline, observations)
    for path in (
        data / "run.json",
        baseline_path,
        report.ORIGINAL / "references/a.jsonl",
        report.ORIGINAL / "references/a-freeze.json",
        Path(__file__).resolve(),
        HERE / "report.py",
    ):
        pins[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    for root in (data / "continuation-01",):
        for name in ("plan.json", "run.json"):
            path = root / name
            if path.exists():
                pins[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != expected for path, expected in pins.items()):
        raise ValueError("An input changed during analysis")
    summary["run_id"] = run["run_id"]
    summary["reference_freeze"] = freeze
    summary["input_sha256"] = {
        str(path.relative_to(PROJECT)): expected for path, expected in sorted(pins.items())
    }
    validated["input_sha256"] = summary["input_sha256"]
    return summary, rows, validated


def csv_text(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else ["case_id"])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_results(
    summary: dict,
    rows: list[dict],
    validated: dict,
    output: Path = HERE / "results",
    data: Path = runner.DATA,
) -> None:
    private = data / "results/raw-score-diagnostic-full.json"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    public = {
        **summary,
        "attempts": [
            {key: value for key, value in attempt.items() if key != "response_content"}
            for attempt in summary["attempts"]
        ],
        "private_full_diagnostic_path": str(private),
        "private_full_diagnostic_sha256": hashlib.sha256(private.read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw-score-diagnostic.json").write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    (output / "raw-score-diagnostic.csv").write_text(csv_text(rows))
    (output / "validated-summary.json").write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    summary, rows, validated = build_saved()
    if args.write:
        write_results(summary, rows, validated)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "post_hoc",
                    "invalid_labels_remain_invalid",
                    "same_view_numeric_subset",
                    "precision_only_case_ids",
                    "not_compared_case_ids",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
