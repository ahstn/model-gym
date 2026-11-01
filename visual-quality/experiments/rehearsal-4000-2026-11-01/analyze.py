"""Audit and export rehearsal attempts, labels, independent references and comparisons.

Run only after the 500-reference freeze. Missing model responses remain explicit rows.
The CSV contains every expected model/arm/image slot, references, and derived means.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import analysis_metrics as metrics
import api_contract
import api_runner
import lite_runner
import numpy as np
import reference_review

HERE = Path(__file__).resolve().parent
DATA = api_runner.DATA
MODELS = (*api_runner.ROUTES, "qrealign_lite")
COMPLETE_STATUSES = {"valid", "partial", "invalid"}
BENCHMARK = {"development", "representative", "challenge"}
CSV_COLUMNS = (
    "case_id",
    "model",
    "arm",
    "technical_score",
    "aesthetic_score",
    "latency_seconds",
    "price_usd",
    "price_basis",
    "status",
    "benchmark_partition",
    "view_sha256",
    "technical_score_native",
    "aesthetic_score_native",
    "review_elapsed_seconds",
    "technical_score_display",
    "aesthetic_score_display",
    "technical_score_valid",
    "aesthetic_score_valid",
    "family_id",
    "image_path",
    "source",
    "stratum",
    "operator",
    "severity",
    "content_cluster",
    "reference_eligibility",
    "scope_floor_applied",
    "excluded_probability",
    "model_returned",
    "provider_returned",
    "requested_reasoning",
    "returned_reasoning_effort",
    "reasoning_tokens",
    "attempt_id",
    "attempt_count",
    "request_started_at",
    "response_received_at",
    "scheduler_wait_seconds",
    "preparation_seconds",
    "total_attempt_latency_seconds",
    "total_attempt_price_usd",
    "total_attempt_known_price_usd",
    "unpriced_attempts",
    "latency_kind",
    "raw_response_path",
    "raw_response_sha256",
    "request_sha256",
    "stats_case_id",
    "validation_errors",
    "evidence",
    "limitations",
    "component_attempt_ids",
    "training_ready",
)


def read_lines(path: Path) -> list[dict]:
    return [api_contract.load_json(line) for line in path.read_text().splitlines() if line.strip()]


def valid_score(value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float} or not math.isfinite(value) or not 1 <= value <= 5:
        raise ValueError("Score outside finite 1–5 scale")
    return float(value)


def decimal_mean(values: list[float]) -> float:
    """Keep equal decimal chat-score means tied without rounding their outputs."""
    return float(sum((Decimal(str(value)) for value in values), Decimal(0)) / Decimal(len(values)))


def slots(cases: list[dict]) -> list[tuple[str, str, str]]:
    result = []
    for case in cases:
        identity = case["case_id"]
        result.extend((identity, model, "baseline") for model in MODELS)
        if case["benchmark_partition"] in BENCHMARK:
            result.extend(
                (identity, model, arm)
                for model in ("luna", "deepseek")
                for arm in ("own_stats", "swapped_stats")
            )
    return result


def base_row(case: dict, model: str, arm: str) -> dict:
    row = dict.fromkeys(CSV_COLUMNS)
    row.update(
        case_id=case["case_id"],
        model=model,
        arm=arm,
        status="missing",
        family_id=case["family_id"],
        benchmark_partition=case["benchmark_partition"],
        view_sha256=case["view_sha256"],
        image_path=case.get("image_path"),
        source=case.get("source"),
        stratum=case.get("stratum"),
        operator=case.get("operator"),
        severity=case.get("severity"),
        content_cluster=case.get("rehearsal_content_cluster"),
        attempt_count=0,
        technical_score_valid=False,
        aesthetic_score_valid=False,
        training_ready=False,
    )
    if model in {"qrealign_lite", "reference"}:
        row.update(price_usd="0.00", price_basis="user_assigned_zero; missing row does not imply a request")
    return row


def money(value: object) -> Decimal | None:
    return None if value is None else api_contract.amount(value)


def normalize_attempt(case: dict, attempt: dict) -> dict:
    row = base_row(case, attempt["model"], attempt["arm"])
    for key in CSV_COLUMNS:
        if key in attempt and key not in {"technical_score", "aesthetic_score"}:
            row[key] = attempt[key]
    row["latency_kind"] = (
        "http_request_through_complete_response"
        if (attempt.get("response_received_at") and type(attempt.get("http_status")) is int
            and 100 <= attempt["http_status"] <= 599 and not attempt.get("transport_error")
            and attempt.get("status") != "unknown")
        else "incomplete_or_unmeasured_http_attempt"
    )
    task_keys = ("quality", "aesthetics") if attempt["model"] == "qrealign_lite" else api_contract.TASKS
    for task, native_key in zip(metrics.TASKS, task_keys, strict=True):
        valid = bool(attempt.get("task_valid", {}).get(native_key, False))
        row[task + "_valid"] = valid
        row[task] = valid_score(attempt.get(task)) if valid else None
        row[task + "_native"] = valid_score(attempt.get(task + "_native"))
    row["price_usd"] = (
        str(money(attempt["price_usd"])) if money(attempt.get("price_usd")) is not None else None
    )
    if attempt["model"] == "qrealign_lite" and (
        money(attempt.get("price_usd")) != 0
        or not attempt.get("price_basis", "").startswith("user_assigned_zero")
    ):
        raise ValueError("Lite zero-price reporting override is missing")
    latency = attempt.get("latency_seconds")
    if latency is not None and (
        type(latency) not in {int, float} or not math.isfinite(latency) or latency < 0
    ):
        raise ValueError("Invalid measured HTTP latency")
    row["evidence"] = {k: v.get("evidence") for k, v in attempt.get("ratings", {}).items()}
    row["limitations"] = {k: v.get("limitations") for k, v in attempt.get("ratings", {}).items()}
    return row


def normalize_reference(case: dict, reference: dict | None) -> dict:
    row = base_row(case, "reference", "reference")
    row.update(latency_kind="not_applicable_assistant_review", price_basis="user_assigned_zero")
    if reference is None:
        return row
    if (
        reference["view_sha256"] != case["view_sha256"]
        or money(reference.get("price_usd")) != 0
        or reference.get("latency_seconds") is not None
    ):
        raise ValueError("Reference image or reporting contract mismatch")
    row.update(
        status="valid",
        reference_eligibility=reference["eligibility"],
        review_elapsed_seconds=reference.get("review_elapsed_seconds"),
        scope_floor_applied=reference["eligibility"] == "excluded",
        evidence={t: reference[t]["evidence"] for t in api_contract.TASKS},
        limitations={t: reference[t]["limitations"] for t in api_contract.TASKS},
    )
    for score, task in zip(metrics.TASKS, api_contract.TASKS, strict=True):
        row[score + "_valid"] = reference[task]["status"] in {"ok", "unscorable"}
        row[score] = valid_score(reference[task]["score"]) if reference[task]["status"] == "ok" else None
        row[score + "_native"] = row[score]
    return row


def validate_attempts(cases: list[dict], attempts: list[dict], frozen_at: str | None) -> None:
    by_id = {case["case_id"]: case for case in cases}
    expected = set(slots(cases))
    if len(by_id) != len(cases) or len({case["view_sha256"] for case in cases}) != len(cases):
        raise ValueError("Duplicate corpus identities or delivered views")
    seen = set()
    for attempt in attempts:
        key = attempt["case_id"], attempt["model"], attempt["arm"]
        if key not in expected or attempt["view_sha256"] != by_id[key[0]]["view_sha256"]:
            raise ValueError("Unexpected model/arm/image or changed view hash")
        identity = attempt["model"], attempt["attempt_id"]
        if identity in seen:
            raise ValueError("Duplicate attempt identity would double-count billing")
        seen.add(identity)
        if by_id[key[0]]["benchmark_partition"] in BENCHMARK:
            admitted = attempt.get("request_started_at") or attempt.get("admission_time")
            if not frozen_at or not admitted:
                raise ValueError("Benchmark attempt lacks a provable reference-freeze admission time")
            before, after = datetime.fromisoformat(frozen_at), datetime.fromisoformat(admitted)
            if before.tzinfo is None or after.tzinfo is None or after < before:
                raise ValueError("Benchmark request preceded the independent reference freeze")


def build_long(
    cases: list[dict], attempts: list[dict], references: list[dict], *, frozen_at: str | None
) -> list[dict]:
    validate_attempts(cases, attempts, frozen_at)
    by_id = {case["case_id"]: case for case in cases}
    grouped = defaultdict(list)
    for attempt in attempts:
        grouped[(attempt["case_id"], attempt["model"], attempt["arm"])].append(attempt)
    rows = []
    for identity, model, arm in slots(cases):
        history = grouped[(identity, model, arm)]
        completed = [r for r in history if r["status"] in COMPLETE_STATUSES]
        if len(completed) > 1:
            raise ValueError("Multiple completed responses for one image/arm require explicit adjudication")
        selected = (
            completed[0]
            if completed
            else max(history, key=lambda r: r.get("sequence", 0))
            if history
            else None
        )
        row = (
            normalize_attempt(by_id[identity], selected)
            if selected
            else base_row(by_id[identity], model, arm)
        )
        known = sum(
            (money(r["price_usd"]) for r in history if money(r.get("price_usd")) is not None), Decimal(0)
        )
        unknown = sum(r.get("price_usd") is None for r in history)
        row.update(
            attempt_count=len(history),
            unpriced_attempts=unknown,
            total_attempt_known_price_usd=str(known) if history else None,
            total_attempt_price_usd=str(known) if history and not unknown else None,
            total_attempt_latency_seconds=sum(r["latency_seconds"] for r in history)
            if history and all(r.get("latency_seconds") is not None for r in history)
            else None,
        )
        rows.append(row)
    reference_map = {r["case_id"]: r for r in references}
    expected = {c["case_id"] for c in cases if c["benchmark_partition"] in BENCHMARK}
    if len(reference_map) != len(references) or not set(reference_map) <= expected:
        raise ValueError("Duplicate or extraneous reference rows")
    rows.extend(
        normalize_reference(case, reference_map.get(case["case_id"]))
        for case in cases
        if case["case_id"] in expected
    )
    lookup = {(r["case_id"], r["model"], r["arm"]): r for r in rows}
    for case in cases:
        for arm in api_contract.ARMS if case["case_id"] in expected else ("baseline",):
            components = [lookup[(case["case_id"], model, arm)] for model in ("luna", "deepseek")]
            row = base_row(case, "luna_deepseek_mean", arm)
            row.update(
                price_basis="derived_no_independent_charge; see component attempts",
                latency_kind="derived_no_http_request",
                component_attempt_ids=[c["attempt_id"] for c in components],
            )
            for score in metrics.TASKS:
                row[score + "_valid"] = all(c[score + "_valid"] for c in components)
                row[score] = (
                    decimal_mean([c[score] for c in components])
                    if all(metrics.numeric(c, score) for c in components)
                    else None
                )
                native = [c[score + "_native"] for c in components]
                row[score + "_native"] = decimal_mean(native) if all(v is not None for v in native) else None
            valid = sum(row[t + "_valid"] for t in metrics.TASKS)
            row["status"] = "valid" if valid == 2 else "partial" if valid else "missing_or_invalid_components"
            row["scope_floor_applied"] = all(c.get("scope_floor_applied") is True for c in components)
            rows.append(row)
    for row in rows:
        for task in metrics.TASKS:
            row[task + "_display"] = f"{row[task]:.2f}" if row[task] is not None else None
    return rows


def latency_summary(values: list[float]) -> dict:
    return {
        "measured": len(values),
        "mean_seconds": float(np.mean(values)) if values else None,
        "p50_seconds": float(np.quantile(values, 0.5)) if values else None,
        "p90_seconds": float(np.quantile(values, 0.9)) if values else None,
        "p95_seconds": float(np.quantile(values, 0.95)) if values else None,
    }


def operational(rows: list[dict], attempts: list[dict]) -> dict:
    grouped = defaultdict(list)
    for attempt in attempts:
        grouped[(attempt["model"], attempt["arm"])].append(attempt)
    summaries = []
    for model, arm in sorted({(r["model"], r["arm"]) for r in rows if r["model"] != "luna_deepseek_mean"}):
        group = [r for r in rows if r["model"] == model and r["arm"] == arm]
        history = grouped[(model, arm)]
        known = sum(
            (money(r["price_usd"]) for r in history if money(r.get("price_usd")) is not None), Decimal(0)
        )
        unknown = sum(r.get("price_usd") is None for r in history)
        summaries.append(
            {
                "model": model,
                "arm": arm,
                "expected_images": len(group),
                "rows_with_attempts": sum(r["attempt_count"] > 0 for r in group),
                "status_counts": dict(Counter(r["status"] for r in group)),
                "attempts": len(history),
                "actual_http_starts_recorded": sum(r.get("request_started_at") is not None for r in history),
                "known_price_usd": str(known),
                "unpriced_attempts": unknown,
                "total_price_usd": str(known) if not unknown else None,
                "price_basis": "user_assigned_zero; infrastructure separate"
                if model == "qrealign_lite"
                else "user_assigned_zero; no external request"
                if model == "reference"
                else "all_attempt_receipts",
                "attempt_http_latency": latency_summary(
                    [r["latency_seconds"] for r in history if r.get("latency_seconds") is not None]
                ),
                "unmeasured_latency_attempts": sum(r.get("latency_seconds") is None for r in history),
                "selected_response_http_latency": latency_summary(
                    [
                        r["latency_seconds"]
                        for r in group
                        if r["status"] in COMPLETE_STATUSES and r.get("latency_seconds") is not None
                    ]
                ),
                "numeric_scores": {
                    task: sum(metrics.numeric(r, task) for r in group) for task in metrics.TASKS
                },
            }
        )
    totals = []
    for model in (*MODELS, "reference"):
        groups = [row for row in summaries if row["model"] == model]
        unknown = sum(row["unpriced_attempts"] for row in groups)
        known = sum((Decimal(row["known_price_usd"]) for row in groups), Decimal(0))
        totals.append(
            {
                "model": model,
                "attempts": sum(row["attempts"] for row in groups),
                "known_price_usd": str(known),
                "unpriced_attempts": unknown,
                "total_price_usd": str(known) if not unknown else None,
                "price_basis": "user_assigned_zero"
                if model in {"reference", "qrealign_lite"}
                else "all_attempt_receipts",
            }
        )
    return {
        "model_arms": summaries,
        "model_totals": totals,
        "derived_mean_billing": "Excluded; component attempts are counted once",
        "latency_caveat": "Different routes and fixed concurrency; not a controlled production-speed comparison",
        "lite_infrastructure_cost": "Accounted separately from requested zero displayed model prices",
    }


def load_attempts(
    data: Path, manifest_path: Path, cases: list[dict], *,
    postpass_allowlist: Path | None = None, postpass_allowlist_sha256: str | None = None,
) -> tuple[list[dict], dict]:
    if data.resolve() != DATA.resolve():
        raise ValueError("API journal reconstruction requires its frozen absolute data root")
    attempts, inputs = [], {}
    if bool(postpass_allowlist) != bool(postpass_allowlist_sha256):
        raise ValueError("Postpass reconstruction requires both allowlist path and its externally pinned SHA256")
    postpass, approval = None, None
    if postpass_allowlist:
        import postpass_recovery as postpass

        approval = postpass.read_allowlist(postpass_allowlist, postpass_allowlist_sha256)
        inputs[str(postpass_allowlist.resolve())] = postpass_allowlist_sha256
        inputs.update(approval["helper_sha256"])
        inputs[approval["policy_path"]] = approval["policy_sha256"]
    for model in api_runner.ROUTES:
        for phase in api_runner.PHASES:
            path = data / "results/api" / model / phase / "plan.json"
            if not path.exists():
                journal_root = path.parent / "attempts"
                if (
                    any(entry.is_file() for entry in path.parent.rglob("*"))
                    or journal_root.is_dir()
                    and any(journal_root.iterdir())
                ):
                    raise ValueError(f"Orphaned API journals without frozen plan: {model}/{phase}")
                continue
            plan = api_runner.read(path)
            api_runner.validate_plan(plan, check_payloads=True)
            inputs[str(path)] = api_contract.file_sha(path)
            records = postpass.reconstruct(plan, approval) if postpass else api_runner.reconstruct(plan)
            attempts.extend({**row, "admission_time": plan["created_at"]} for row in records)
    if postpass:
        for path in postpass.root().glob("*/attempt-metadata/**/admission.json"):
            inputs[str(path.resolve())] = api_contract.file_sha(path)
    by_id = {case["case_id"]: case for case in cases}
    for path in sorted((data / "results/lite").glob("**/attempts/*.json")):
        run_path = path.parent.parent / "run.json"
        run = api_runner.read(run_path)
        row = api_runner.read(path)
        configuration = run["configuration"]
        server_identity = api_runner.read(path.parent.parent / "server-identity.json")
        if (
            row["run_id"] != run["run_id"]
            or run["run_id"] != lite_runner.digest_bytes(lite_runner.canonical(configuration))
            or configuration["manifest_sha256"] != api_contract.file_sha(manifest_path)
            or configuration["client_sha256"] != api_contract.file_sha(lite_runner.__file__)
            or row["case_id"] not in configuration["case_ids"]
            or configuration["model"] != lite_runner.MODEL
            or configuration["revision"] != lite_runner.REVISION
            or server_identity.get("model") != lite_runner.MODEL
            or server_identity.get("revision") != lite_runner.REVISION
            or configuration["server_identity_sha256"]
            != lite_runner.digest_bytes(lite_runner.canonical(server_identity))
        ):
            raise ValueError("Lite journal configuration changed")
        image = Path(by_id[row["case_id"]]["image_path"]).read_bytes()
        payload = lite_runner.canonical({"image": base64.b64encode(image).decode("ascii")})
        if lite_runner.digest_bytes(payload) != row["request_sha256"]:
            raise ValueError("Lite frozen image request changed")
        row["admission_time"] = run["created_at"]
        raw = Path(row["raw_response_path"])
        if raw.exists():
            row["raw_response_sha256"] = api_contract.file_sha(raw)
        if row["status"] in COMPLETE_STATUSES and not raw.exists():
            raise ValueError("Completed Lite observation lacks raw evidence")
        if row["status"] in COMPLETE_STATUSES:
            outcome = api_runner.read(raw)
            if outcome["status"] != 200:
                raise ValueError("Lite completed label has non-200 raw response")
            normalized = lite_runner.normalize_response(api_contract.load_json(outcome["body"]))
            if any(row.get(key) != value for key, value in normalized.items()):
                raise ValueError("Lite label differs from independently reparsed raw response")
        inputs[str(path)] = api_contract.file_sha(path)
        inputs[str(run_path)] = api_contract.file_sha(run_path)
        attempts.append(row)
    return attempts, inputs


def write_exports(output: Path, rows: list[dict], analysis: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scores-long.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    api_runner.jsonl(output / "scores-long.jsonl", rows)
    api_runner.write(output / "analysis.json", analysis)
    with (output / "cost-per-model.csv").open("w", newline="") as stream:
        table = analysis["operations"]["model_totals"]
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--reference-freeze", type=Path, default=HERE / "references-freeze.json")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--stats", type=Path, default=HERE / "stats.jsonl")
    parser.add_argument("--qc-thresholds", type=Path, help="Frozen nonbenchmark threshold artifact, optional")
    parser.add_argument("--stats-percentiles", type=Path, default=HERE / "stats-percentiles.jsonl")
    parser.add_argument("--stats-reference", type=Path, default=HERE / "stats-reference.json")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--postpass-allowlist", type=Path)
    parser.add_argument("--postpass-allowlist-sha256")
    args = parser.parse_args()
    manifest = api_runner.read(args.manifest)
    pins = api_runner.frozen_references(args.manifest, manifest, args.reference_freeze)
    freeze = api_runner.read(args.reference_freeze)
    references = read_lines(Path(freeze["references_path"]))
    for reference in references:
        reference_review.validate_assessment(reference)
        reference_review.validate_timing(reference)
    cases = manifest["cases"]
    for case in cases:
        if api_contract.file_sha(case["image_path"]) != case["view_sha256"]:
            raise ValueError("Delivered image changed")
    attempts, journal_pins = load_attempts(
        DATA, args.manifest, cases,
        postpass_allowlist=args.postpass_allowlist, postpass_allowlist_sha256=args.postpass_allowlist_sha256,
    )
    rows = build_long(cases, attempts, references, frozen_at=freeze["frozen_at"])
    expected = len(slots(cases)) + len(references) + len(cases) + 2 * len(references)
    if len(rows) != expected:
        raise ValueError("Unexpected long-table coverage")
    analysis = {
        "schema": "rehearsal-analysis-v1",
        "created_at": api_runner.now(),
        "long_rows": len(rows),
        "reference_type": "Independent assistant judgments; not human MOS or human ground truth",
        "attempt_record_digest": api_contract.digest(attempts),
        "bootstrap_policy": {
            "repeats": args.bootstrap_repeats,
            "seed": metrics.SEED,
            "unit": "source family; paired contrasts retained",
            "primary_panel": "representative",
            "primary_model": "luna_deepseek_mean",
            "primary_arm": "own_stats",
            "primary_comparator": "baseline",
            "aesthetic_noninferiority_upper_margin": 0.03,
        },
        "inputs": {
            **pins,
            **journal_pins,
            str(args.manifest): api_contract.file_sha(args.manifest),
            str(Path(__file__)): api_contract.file_sha(__file__),
            str(HERE / "analysis_metrics.py"): api_contract.file_sha(HERE / "analysis_metrics.py"),
        },
        "operations": operational(rows, attempts),
        "descriptive": metrics.descriptive(rows),
        "between_model_agreement": metrics.between_models(rows),
        "accuracy": metrics.accuracy(rows),
        "baseline_comparisons": metrics.baseline_comparisons(rows, repeats=args.bootstrap_repeats),
        "treatments": metrics.treatment_effects(rows, repeats=args.bootstrap_repeats),
        "scope": metrics.scope_checks(rows),
        "ladders": metrics.ladders(cases, rows),
    }
    if args.qc_thresholds:
        thresholds = api_runner.read(args.qc_thresholds)
        calibration = set(thresholds["calibration_family_ids"])
        held_out = {case["family_id"] for case in cases if case["benchmark_partition"] in BENCHMARK}
        if not calibration or calibration & held_out:
            raise ValueError("QC thresholds include held-out benchmark families")
        if thresholds["reference_sha256"] != api_contract.file_sha(args.stats_reference):
            raise ValueError("QC percentile reference changed")
        raw = read_lines(args.stats)
        stats = {row["case_id"]: row.get("analysis", row) for row in raw}
        by_id = {case["case_id"]: case for case in cases}
        if len(raw) != len(stats) or set(stats) != set(by_id):
            raise ValueError("QC stats must cover each frozen image exactly once")
        for row in raw:
            if row["sha256"] != by_id[row["case_id"]]["view_sha256"]:
                raise ValueError("QC stats image hash mismatch")
        percentiles = read_lines(args.stats_percentiles)
        if len(percentiles) != len(by_id) or {row["case_id"] for row in percentiles} != set(by_id):
            raise ValueError("QC percentile coverage mismatch")
        for row in percentiles:
            if row["view_sha256"] != by_id[row["case_id"]]["view_sha256"]:
                raise ValueError("QC percentile image hash mismatch")
            if row["reference_sha256"] != thresholds["reference_sha256"]:
                raise ValueError("QC percentiles use a different statistical reference")
            stats[row["case_id"]]["percentiles"] = {
                "sharpness_peak": api_contract.number(row["sharpness_peak_percentile"], 0, 100),
                "noise_sigma": api_contract.number(row["noise_sigma_percentile"], 0, 100),
            }
        analysis["qc"] = metrics.qc_review(rows, stats, thresholds)
        for path in (args.qc_thresholds, args.stats, args.stats_percentiles, args.stats_reference):
            analysis["inputs"][str(path)] = api_contract.file_sha(path)
    else:
        analysis["qc"] = {
            "status": "not_computed",
            "reason": "No frozen nonbenchmark QC threshold artifact supplied",
        }
    write_exports(args.output, rows, analysis)
    print(
        json.dumps(
            {
                "long_rows": len(rows),
                "output": str(args.output),
                "attempts": len(attempts),
                "reference_rows": len(references),
            }
        )
    )


if __name__ == "__main__":
    main()
