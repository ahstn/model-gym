"""Offline native-feature, prompt-search and counterfactual reports for metadata v3."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench_analyze import numeric
from bench_protocol import digest
from tracks.image_context import CAMERA_TAGS
from tracks.metadata_v3.budget_reallocation import reallocation_summary
from tracks.metadata_v3.cost_recovery import allowance_summary
from tracks.metadata_v3.features import ARMS, assemble_context
from tracks.metadata_v3.http_error_recovery import http_summary
from tracks.metadata_v3.shared_allowance_pool import shared_pool_summary
from tracks.metadata_v3.timeout_recovery import timeout_summary
from tracks.metrics import ai_metrics, summarize

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261012
CSV_FIELDS = (
    "case",
    "condition",
    "base_sample_id",
    "arm",
    "prompt_variant",
    "reused_baseline",
    "split",
    "gold",
    "prediction",
    "correct",
    "attempted",
    "valid",
    "uncertainty",
    "request_sha256",
    "features_sha256",
    "context_sha256",
    "prompt_sha256",
    "delivered_sha256",
    "native_sha256",
    "protocol_sha256",
    "model",
    "provider",
    "cost_usd",
    "effective_cost_usd",
    "seconds",
    "preprocessing_seconds",
    "source",
    "camera_group",
    "generator",
    "group_id",
    "metadata_availability",
    "camera_numeric_present",
    "c2pa_status",
    "c2pa_manifest_present",
    "c2pa_trust_status",
    "c2pa_ai_generation_claim",
    "failure",
)


def _json(path):
    return json.loads(Path(path).read_bytes())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ledger(path):
    return (
        [json.loads(line) for line in Path(path).read_bytes().splitlines() if line]
        if Path(path).exists()
        else []
    )


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def expense_summary(records):
    """Count actual attempts once; an unknown charge is never a zero charge."""
    costs = [row.get("effective_cost_usd") for row in records]
    known = [value for value in costs if numeric(value, 0)]
    charges = []
    for row in records:
        response = row.get("response")
        response = response if isinstance(response, dict) else {}
        usage = row.get("usage") or response.get("usage") or {}
        usage = usage if isinstance(usage, dict) else {}
        charge = row.get("cost_usd", usage.get("cost"))
        charges.append(charge)
    known_charges = [value for value in charges if numeric(value, 0)]
    return {
        "attempts": len(records),
        "effective_cost_usd": math.fsum(known) if len(known) == len(records) else None,
        "known_effective_cost_usd": math.fsum(known),
        "unknown_cost_count": len(records) - len(known),
        "openrouter_charge_usd": math.fsum(known_charges) if len(known_charges) == len(records) else None,
        "known_openrouter_charge_usd": math.fsum(known_charges),
        "unknown_router_charge_count": len(records) - len(known_charges),
        "basis": "Recorded effective cost includes BYOK upstream and any separate router charge; not invoice reconciliation",
    }


def _availability(sample):
    camera = sample["features"]["camera"]
    return {
        "camera_present": camera["exif_availability"] == "present",
        "camera_numeric_present": any(camera.get(key) is not None for key in CAMERA_TAGS),
    }


def coverage_summary(samples):
    """Expose source/metadata confounding before any performance comparison."""
    splits = {}
    for split in sorted({sample["split"] for sample in samples}):
        classes = {}
        for gold, name in ((0, "real"), (1, "generated")):
            members = [sample for sample in samples if sample["split"] == split and sample["gold"] == gold]
            classes[name] = {
                "count": len(members),
                **{
                    key: sum(_availability(sample)[key] for sample in members)
                    for key in ("camera_present", "camera_numeric_present")
                },
                "sources": dict(
                    Counter(sample.get("source", {}).get("dataset", "unknown") for sample in members)
                ),
                "groups": len({sample["group_id"] for sample in members}),
            }
        splits[split] = classes
    dev = [sample for sample in samples if sample["split"] == "dev"]
    diagnostics = {}
    for field, name in (
        ("camera_present", "availability_only_dev"),
        ("camera_numeric_present", "numeric_camera_only_dev"),
    ):
        correct = [int(not _availability(sample)[field]) == sample["gold"] for sample in dev]
        diagnostics[name] = {
            "rule": f"Predict real when {field} is true; otherwise predict generated. Fixed diagnostic, not fitted.",
            "n": len(dev),
            "accuracy": _ratio(sum(correct), len(dev)),
            "perfect_label_separation": bool(dev)
            and len({sample["gold"] for sample in dev}) == 2
            and all(correct),
            "not_a_vision_detector": True,
        }
    return {
        "by_split": splits,
        **diagnostics,
        "limit": "Native metadata availability and source identity can predict labels without detecting image synthesis",
    }


def provenance_summary(samples):
    claims = [sample for sample in samples if sample["features"]["provenance"]["ai_generation_claim"] is True]
    validated = [
        sample
        for sample in claims
        if all(
            sample["features"]["provenance"].get(key) is True
            for key in ("signature_valid", "hard_binding_valid")
        )
    ]
    return {
        "n": len(samples),
        "status": dict(Counter(sample["features"]["provenance"]["status"] for sample in samples)),
        "trust_status": dict(Counter(sample["features"]["provenance"]["trust_status"] for sample in samples)),
        "manifest_present": sum(
            sample["features"]["provenance"]["manifest_present"] is True for sample in samples
        ),
        "explicit_ai_claims": len(claims),
        "signature_and_binding_valid_ai_claims": len(validated),
        "explicit_ai_claims_agreeing_with_source": sum(sample["gold"] == 1 for sample in claims),
        "validated_ai_claims_agreeing_with_source": sum(sample["gold"] == 1 for sample in validated),
        "interpretation": "Claim agreement is separate from vision accuracy. A valid or trusted signature does not prove truth; an absent AI claim does not assert camera origin.",
    }


def _groups_by_class(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    strata = {0: [], 1: []}
    for members in groups.values():
        labels = {int(row["gold"]) for row in members}
        if len(labels) != 1:
            raise ValueError("Metadata source groups must contain one origin class")
        strata[next(iter(labels))].append(members)
    return strata


def _percentiles(values):
    values = sorted(value for value in values if value is not None)
    return (
        [values[int(0.025 * (len(values) - 1))], values[math.ceil(0.975 * (len(values) - 1))]]
        if values
        else None
    )


def paired_class_balanced_delta(left_rows, right_rows, replicates=1000, require_both_classes=True):
    left, right = ({row["id"]: row for row in rows} for rows in (left_rows, right_rows))
    if set(left) != set(right) or len(left) != len(left_rows) or len(right) != len(right_rows):
        raise ValueError("Paired comparison requires identical unique case IDs")
    differences = []
    for identifier, baseline in left.items():
        selected = right[identifier]
        if any(baseline[key] != selected[key] for key in ("group_id", "gold", "split", "delivered_sha256")):
            raise ValueError("Paired labels, groups, splits or image hashes differ")
        differences.append({**baseline, "reward": selected["reward"] - baseline["reward"]})
    strata = _groups_by_class(differences)
    values = {
        label: [mean(row["reward"] for row in group) for group in groups] for label, groups in strata.items()
    }
    included = values if require_both_classes else {label: items for label, items in values.items() if items}
    complete = bool(included) and all(included.values())
    delta = mean(mean(items) for items in included.values()) if complete else None
    draws = []
    if complete and replicates > 0:
        rng = random.Random(SEED)
        draws = [
            mean(mean(rng.choices(items, k=len(items))) for items in included.values())
            for _ in range(replicates)
        ]
    return {
        "delta": delta,
        "interval": _percentiles(draws),
        "n_images": len(left),
        "n_groups": sum(map(len, strata.values())),
        "class_group_counts": {str(k): len(v) for k, v in strata.items()},
        "replicates": len(draws),
        "seed": SEED,
        "single_class_subset": sum(bool(items) for items in values.values()) == 1,
        "estimand": "Selected minus reference reward, equal represented origin-class weight then equal source-group weight; invalid/unattempted reward zero",
        "limit": "Source-group bootstrap excludes source bias, metadata confounding and selection uncertainty",
    }


def _metrics(rows):
    result = ai_metrics(rows)
    classes = {gold: [row for row in rows if int(row["gold"]) == gold] for gold in (0, 1)}
    rates = {
        gold: _ratio(sum(row["correct"] is True for row in members), len(members))
        for gold, members in classes.items()
    }
    result.update(
        class_counts_expected={"real": len(classes[0]), "generated": len(classes[1])},
        balanced_accuracy_all_expected=mean(rates.values())
        if all(value is not None for value in rates.values())
        else None,
        recall_all_expected=rates[1],
        false_rejection_or_failure_rate_all_real=1 - rates[0] if rates[0] is not None else None,
        review_or_failure_fraction_all_expected=_ratio(
            result["review_count_valid"] + len(rows) - result["n_valid"], len(rows)
        ),
    )
    return result


def _intervals(rows, replicates):
    strata = _groups_by_class(rows)
    values = defaultdict(list)
    if not all(strata.values()) or replicates <= 0:
        return {"replicates": 0, "intervals": {}}
    rng = random.Random(SEED)
    for _ in range(replicates):
        sampled = [
            row for groups in strata.values() for group in rng.choices(groups, k=len(groups)) for row in group
        ]
        metric = _metrics(sampled)
        for key in (
            "accuracy_all_expected_invalid_as_wrong",
            "balanced_accuracy_all_expected",
            "recall_all_expected",
            "brier_invalid_or_unattempted_penalty1",
            "auroc_valid",
        ):
            values[key].append(metric[key])
        values["false_positive_rate_valid_only"].append(metric["valid_only"]["false_positive_rate"])
    return {
        "replicates": replicates,
        "seed": SEED,
        "intervals": {key: _percentiles(value) for key, value in values.items()},
        "method": "Resample source groups within each origin class; retain all images in each sampled group",
    }


def _clean_record(record):
    """Normalize malformed containers for offline metrics, without changing raw evidence."""
    clean = dict(record)
    response = clean.get("response")
    response = response if isinstance(response, dict) else {}
    usage = clean.get("usage") or response.get("usage") or {}
    usage = dict(usage) if isinstance(usage, dict) else {}
    if not isinstance(usage.get("cost_details"), dict):
        usage["cost_details"] = {}
    clean.update(usage=usage, response={**response, "usage": usage})
    return clean


def _preprocessing_time(sample):
    # Control builders copy native timing; they do not measure extra control work.
    if sample.get("base_sample_id") is not None:
        return None
    return sample.get("preprocess_seconds", sample.get("preprocessing_seconds"))


def _preprocessing(samples):
    times = [_preprocessing_time(sample) for sample in samples]
    return {
        "cases": len(samples),
        "seconds": math.fsum(times) if all(numeric(value, 0) for value in times) else None,
        "known_seconds": math.fsum(value for value in times if numeric(value, 0)),
        "unknown_count": sum(not numeric(value, 0) for value in times),
        "note": "One-time per-case extraction time; do not add it again for each arm or prompt",
    }


def summarise_cases(samples, records, bootstrap_replicates=0):
    metric_samples = [{**sample, "labels": {"ai_generated": bool(sample["gold"])}} for sample in samples]
    if any(type(sample["gold"]) not in (bool, int) or sample["gold"] not in (0, 1) for sample in samples):
        raise ValueError("Source-origin gold must be boolean or zero/one")
    result = summarize("ai_detection", metric_samples, [_clean_record(row) for row in records])
    result["summary"]["metrics"] = _metrics(result["per_image"])
    result["summary"]["group_bootstrap"] = _intervals(result["per_image"], bootstrap_replicates)
    result["summary"]["expenses"] = expense_summary(records)
    result["summary"]["preprocessing"] = _preprocessing(samples)
    return result


def _validate_records(records, samples, arm, prompt, protocol_hash):
    expected = {sample["id"]: sample for sample in samples}
    for row in records:
        if row.get("sample") not in expected:
            raise ValueError("Ledger contains a case outside the intended evaluation")
        sample = expected[row["sample"]]
        checks = {
            "metadata_arm": arm,
            "protocol_sha256": protocol_hash,
            "prompt_sha256": digest(prompt),
            "features_sha256": sample["features_sha256"],
            "delivered_sha256": sample["delivered_sha256"],
            "context_sha256": digest(assemble_context(sample["features"], arm)),
            "task": "ai",
            "arm": "decider",
        }
        if any(row.get(key) != value for key, value in checks.items()):
            raise ValueError("Ledger hashes, task or arm differ from frozen inputs")


def _compact(sample, row, raw, arm, variant, reused, protocol_hash, condition="native"):
    prediction = row.get("prediction") or {}
    source = sample.get("source", {})
    provenance = sample["features"]["provenance"]
    return {
        "case": sample["id"],
        "condition": condition,
        "base_sample_id": sample.get("base_sample_id"),
        "arm": arm,
        "prompt_variant": variant,
        "reused_baseline": reused,
        "split": sample["split"],
        "gold": int(sample["gold"]),
        "prediction": prediction.get("ai_probability"),
        "correct": row["correct"],
        "attempted": row["attempted"],
        "valid": row["valid"],
        "uncertainty": prediction.get("uncertainty"),
        **{key: raw.get(key) for key in ("request_sha256", "context_sha256", "prompt_sha256")},
        "features_sha256": sample["features_sha256"],
        "delivered_sha256": sample["delivered_sha256"],
        "native_sha256": sample.get("native_sha256"),
        "protocol_sha256": protocol_hash,
        "model": row.get("model"),
        "provider": row.get("provider"),
        "cost_usd": raw.get("cost_usd"),
        "effective_cost_usd": raw.get("effective_cost_usd"),
        "seconds": row.get("seconds"),
        "preprocessing_seconds": _preprocessing_time(sample),
        "source": source.get("dataset"),
        "camera_group": source.get("camera_group"),
        "generator": source.get("generator"),
        "group_id": sample["group_id"],
        "metadata_availability": sample["features"]["camera"]["exif_availability"],
        "camera_numeric_present": _availability(sample)["camera_numeric_present"],
        "c2pa_status": provenance["status"],
        "c2pa_manifest_present": provenance["manifest_present"],
        "c2pa_trust_status": provenance["trust_status"],
        "c2pa_ai_generation_claim": provenance["ai_generation_claim"],
        "failure": raw.get("validation_error")
        or raw.get("transport_error")
        or ("unattempted" if not raw else None),
    }


def _source_slices(samples, records):
    slices = {}
    for field in ("dataset", "camera_group", "generator"):
        values = sorted(
            {
                str(sample.get("source", {}).get(field))
                for sample in samples
                if sample.get("source", {}).get(field) is not None
            }
        )
        for value in values:
            subset = [sample for sample in samples if str(sample.get("source", {}).get(field)) == value]
            ids = {sample["id"] for sample in subset}
            slices[f"{field}:{value}"] = summarise_cases(
                subset, [row for row in records if row["sample"] in ids]
            )["summary"]
    return slices


def _metadata_slices(samples, records):
    predicates = {
        "exif_present": lambda sample: sample["features"]["camera"]["exif_availability"],
        "numeric_camera_present": lambda sample: (
            "present" if _availability(sample)["camera_numeric_present"] else "absent"
        ),
        "c2pa_ai_claim": lambda sample: (
            "present" if sample["features"]["provenance"]["ai_generation_claim"] is True else "absent"
        ),
    }
    slices = {}
    for field, state in predicates.items():
        states = {"present", "absent"} | {state(sample) for sample in samples}
        for value in sorted(states):
            subset = [sample for sample in samples if state(sample) == value]
            ids = {sample["id"] for sample in subset}
            slices[f"{field}:{value}"] = summarise_cases(
                subset, [row for row in records if row["sample"] in ids]
            )["summary"]
    return slices


def _evaluate(samples, ledger, prompt, arm, protocol_hash, replicates):
    records = _ledger(ledger)
    _validate_records(records, samples, arm, prompt, protocol_hash)
    result = summarise_cases(samples, records, replicates)
    result["summary"]["source_slices"] = _source_slices(samples, records)
    result["summary"]["metadata_slices"] = _metadata_slices(samples, records)
    result["summary"].update(
        ledger_sha256=_sha(ledger) if ledger.exists() else None, prompt_sha256=digest(prompt)
    )
    timestamps = [row["timestamp"] for row in records if row.get("timestamp")]
    result["summary"].update(
        first_attempt_at=min(timestamps) if timestamps else None,
        last_attempt_at=max(timestamps) if timestamps else None,
    )
    return result, {row["sample"]: row for row in records}


def _expense_inventory(runs_root):
    inventory, records = [], []
    by_phase, by_arm = defaultdict(list), defaultdict(list)
    for path in sorted(runs_root.rglob("*.jsonl")):
        rows = _ledger(path)
        relative = path.relative_to(runs_root)
        phase = (
            "reflection"
            if path.name == "reflection.jsonl"
            else "controls"
            if "controls" in relative.parts
            else "final"
            if "final" in relative.parts
            else "optimisation"
        )
        for row in rows:
            by_phase[phase].append(row)
            by_arm[row.get("metadata_arm", relative.parts[0])].append(row)
        records.extend(rows)
        inventory.append({"path": str(relative), "sha256": _sha(path), "attempts": len(rows)})
    return {
        **expense_summary(records),
        "by_phase": {key: expense_summary(value) for key, value in by_phase.items()},
        "by_arm": {key: expense_summary(value) for key, value in by_arm.items()},
        "ledger_inventory": inventory,
    }


def _recovery_accounting(
    expenses,
    runs_root,
    amendment_path,
    protocol_hash,
    project_root,
    reallocation_path=None,
    timeout_extension_path=None,
    shared_pool_path=None,
    http_extension_path=None,
):
    validated = (
        http_summary(runs_root, http_extension_path)
        if http_extension_path is not None
        else shared_pool_summary(runs_root, shared_pool_path)
        if shared_pool_path is not None
        else timeout_summary(runs_root, timeout_extension_path)
        if timeout_extension_path is not None
        else reallocation_summary(runs_root, reallocation_path)
        if reallocation_path is not None
        else allowance_summary(runs_root, amendment_path)
    )
    amendment, costs = validated["amendment"], validated["costs"]
    if (
        amendment["original_protocol_sha256"] != protocol_hash
        or costs["unknown_cost_count"] != expenses["unknown_cost_count"]
        or not math.isclose(costs["known_reported_usd"], expenses["known_effective_cost_usd"], abs_tol=1e-12)
    ):
        raise ValueError("Recovery allowance accounting differs from the raw study evidence")
    receipts = []
    for path in sorted((runs_root / "recovery-receipts").glob("*.json")):
        receipt = _json(path)
        receipts.append({"path": str(path.relative_to(runs_root)), "sha256": _sha(path), **receipt})
    if len(receipts) != costs["unknown_cost_count"]:
        raise ValueError("Recovery receipt inventory differs from the unknown-cost attempts")
    for key in ("reserved_unverified_usd", "budget_accounted_usd"):
        expenses[key] = costs[key]
        for arm, values in expenses["by_arm"].items():
            values[key] = costs["arms"][arm][key]
    expenses["recovery"] = {
        "amendment_path": os.path.relpath(amendment_path, project_root),
        "amendment_sha256": _sha(amendment_path),
        **{
            key: amendment[key]
            for key in (
                "frozen_at",
                "review_reason",
                "allowance_usd",
                "actual_cost_remains_unknown",
                "no_retry",
                "pricing_assumption",
            )
        },
        "receipt_inventory": receipts,
        "interpretation": "Unverified allowances reserve budget; they are not charged amounts or proven billing limits",
    }
    if reallocation_path is not None:
        reallocation = validated["reallocation"]
        if reallocation["first_amendment_sha256"] != _sha(amendment_path):
            raise ValueError("Reallocation does not reference the supplied recovery amendment")
        if shared_pool_path is None:
            expenses["guard_budget_by_arm"] = costs["guard_budget_by_arm"]
            for arm, values in expenses["by_arm"].items():
                values["guard_budget_usd"] = costs["arms"][arm]["guard_budget_usd"]
        expenses["recovery"]["reallocation"] = {
            "path": os.path.relpath(reallocation_path, project_root),
            "sha256": _sha(reallocation_path),
            **{
                key: reallocation[key]
                for key in (
                    "frozen_at",
                    "review_reason",
                    "global_budget_usd",
                    "original_run_receipt_arm_usd",
                    "guard_budget_by_arm",
                    "reallocated_unused_global_headroom_usd",
                    "no_retry",
                    "new_unknown_batch_action",
                )
            },
        }
    if timeout_extension_path is not None:
        extension = validated["timeout_extension"]
        if reallocation_path is None or extension["reallocation_sha256"] != _sha(reallocation_path):
            raise ValueError("Timeout extension does not reference the supplied reallocation")
        expenses["recovery"]["timeout_extension"] = {
            "path": os.path.relpath(timeout_extension_path, project_root),
            "sha256": _sha(timeout_extension_path),
            **{
                key: extension[key]
                for key in (
                    "frozen_at",
                    "review_reason",
                    "eligible_transport_errors",
                    "added_transport_error",
                    "allowance_usd",
                    "actual_cost_remains_unknown",
                    "no_retry",
                    "new_unknown_batch_action",
                )
            },
        }
    if shared_pool_path is not None:
        shared = validated["shared_pool_amendment"]
        if timeout_extension_path is None or shared["timeout_extension_sha256"] != _sha(
            timeout_extension_path
        ):
            raise ValueError("Shared pool does not reference the supplied timeout extension")
        expenses.update(
            known_spend_budget_by_arm=costs["known_spend_budget_by_arm"],
            known_spend_budget_per_arm_usd=costs["known_spend_budget_per_arm_usd"],
            shared_unknown_allowance_pool_usd=costs["shared_unknown_allowance_pool_usd"],
            budget_guard_scope=costs["budget_guard_scope"],
            global_budget_usd=costs["global_budget_usd"],
            reserved_inflight_usd=costs["reserved_inflight_usd"],
            global_guard_accounted_usd=costs["budget_accounted_usd"] + costs["reserved_inflight_usd"],
        )
        for arm, values in expenses["by_arm"].items():
            values.update(
                known_spend_budget_usd=costs["arms"][arm]["known_spend_budget_usd"],
                guard_scope=costs["arms"][arm]["guard_scope"],
            )
        expenses["recovery"]["shared_pool_amendment"] = {
            "path": os.path.relpath(shared_pool_path, project_root),
            "sha256": _sha(shared_pool_path),
            **{
                key: shared[key]
                for key in (
                    "frozen_at",
                    "review_reason",
                    "global_budget_usd",
                    "known_spend_budget_per_arm_usd",
                    "unknown_allowance_usd",
                    "unknown_allowance_scope",
                    "global_guard_scope",
                    "arm_guard_scope",
                    "actual_cost_remains_unknown",
                    "no_retry",
                    "transport_policy_unchanged",
                    "circuit_stop_unchanged",
                )
            },
        }
    if http_extension_path is not None:
        extension = validated["http_extension"]
        if shared_pool_path is None or extension["shared_pool_sha256"] != _sha(shared_pool_path):
            raise ValueError("HTTP extension does not reference the supplied shared pool")
        expenses["recovery"]["http_extension"] = {
            "path": os.path.relpath(http_extension_path, project_root),
            "sha256": _sha(http_extension_path),
            **{
                key: extension[key]
                for key in (
                    "frozen_at",
                    "review_reason",
                    "added_http_statuses",
                    "error_body_required",
                    "response_contract",
                    "allowance_usd",
                    "actual_cost_remains_unknown",
                    "no_retry",
                    "circuit_stop_unchanged",
                )
            },
        }


def _control_report(cases, native_samples, native_rows, selections, runs_root, protocol_hash, replicates):
    summaries, compact = {}, []
    sources = {sample["id"]: sample for sample in native_samples}
    for case in cases:
        base = sources.get(case.get("base_sample_id"))
        if base and case["delivered_sha256"] != base["delivered_sha256"]:
            raise ValueError("A metadata control changed the delivered pixels")
    for condition in sorted({case["condition"] for case in cases}):
        summaries[condition] = {}
        for arm in ARMS:
            subset = [case for case in cases if case["condition"] == condition and arm in case["arms"]]
            if not subset:
                continue
            prompt = selections[arm]["selected_prompt"]
            result, raw = _evaluate(
                subset,
                runs_root / arm / "controls" / f"{condition}.jsonl",
                prompt,
                arm,
                protocol_hash,
                replicates,
            )
            checks = [
                case["features"]["provenance"]["status"] == case["expected_c2pa_status"]
                and (
                    case.get("control_kind") != "native_byte_intervention"
                    or case.get("c2pa_validation_pass") is True
                )
                for case in subset
                if case.get("expected_c2pa_status") is not None
            ]
            result["summary"]["integrity_fixtures"] = {
                "asserted": len(checks),
                "passed": sum(checks),
                "not_origin_accuracy": True,
            }
            baseline = {row["id"]: row for row in native_rows[arm]["selected"]}
            shifts, flips, paired_native, paired_control = [], [], [], []
            for case, row in zip(subset, result["per_image"], strict=True):
                base = baseline.get(case.get("base_sample_id"))
                if base:
                    paired_native.append(base)
                    paired_control.append({**row, "id": base["id"]})
                if base and base["valid"] and row["valid"]:
                    before, after = base["prediction"]["ai_probability"], row["prediction"]["ai_probability"]
                    shifts.append(after - before)
                    flips.append((after >= 0.5) != (before >= 0.5))
            result["summary"]["same_pixels_vs_native_selected"] = {
                "paired_valid": len(shifts),
                "mean_probability_change": mean(shifts) if shifts else None,
                "mean_absolute_probability_change": mean(map(abs, shifts)) if shifts else None,
                "decision_flips": sum(flips),
                "not_a_native_accuracy_estimate": True,
            }
            delta = paired_class_balanced_delta(
                paired_native, paired_control, replicates, require_both_classes=False
            )
            result["summary"]["paired_native_subset"] = {
                "n_unpaired_intended": len(subset) - len(paired_native),
                "native_metrics": _metrics(paired_native),
                "control_metrics": _metrics(paired_control),
                "reward_change": delta,
                "brier_change": -delta["delta"] if delta["delta"] is not None else None,
                "brier_change_interval": [-delta["interval"][1], -delta["interval"][0]]
                if delta["interval"] is not None
                else None,
                "interpretation": "Control minus exact native-subset Brier; positive is worse. Failures retain loss 1. Class/group weights match the paired reward estimand; single-class controls do not estimate performance on the absent class.",
            }
            summaries[condition][arm] = result["summary"]
            compact.extend(
                _compact(
                    sample, row, raw.get(sample["id"], {}), arm, "selected", False, protocol_hash, condition
                )
                for sample, row in zip(subset, result["per_image"], strict=True)
            )
    return summaries, compact


def build_report(
    dataset_path,
    protocol_path,
    runs_root,
    controls_path=None,
    project_root=None,
    bootstrap_replicates=1000,
    amendment_path=None,
    reallocation_path=None,
    timeout_extension_path=None,
    shared_pool_path=None,
    http_extension_path=None,
):
    project_root, runs_root = Path(project_root or ROOT).resolve(), Path(runs_root).resolve()
    dataset, protocol, protocol_hash = _json(dataset_path), _json(protocol_path), _sha(protocol_path)
    if _sha(dataset_path) != protocol["dataset_sha256"]:
        raise ValueError("Dataset differs from the frozen protocol")
    samples = dataset["samples"]
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Dataset contains duplicate case IDs")
    groups = {}
    for sample in samples:
        if digest(sample["features"]) != sample["features_sha256"]:
            raise ValueError("Frozen sample features changed")
        if sample["group_id"] in groups and groups[sample["group_id"]] != sample["split"]:
            raise ValueError("A source group crosses dataset splits")
        groups[sample["group_id"]] = sample["split"]
    test = [sample for sample in samples if sample["split"] == "test"]
    selections, native, rows, compact = {}, {}, {}, []
    baseline_hash = None
    for arm in ARMS:
        selection = _json(runs_root / arm / "selection.json")
        if (
            selection.get("protocol_sha256") != protocol_hash
            or selection.get("metadata_arm") != arm
            or selection.get("selection_split") != "dev"
            or selection.get("test_used") is not False
        ):
            raise ValueError("Selection is not frozen from development data for this arm/protocol")
        paths = {variant: Path(selection[f"{variant}_prompt_path"]) for variant in ("baseline", "selected")}
        prompts = {
            key: _json(path if path.is_absolute() else project_root / path) for key, path in paths.items()
        }
        if digest(prompts["selected"]) != selection["selected_prompt_sha256"]:
            raise ValueError("Selected prompt changed")
        current = digest(prompts["baseline"])
        if protocol.get("baseline_prompt_sha256", current) != current:
            raise ValueError("Baseline prompt differs from the frozen protocol")
        if baseline_hash is not None and current != baseline_hash:
            raise ValueError("Common-prompt feature comparison requires the same baseline prompt across arms")
        baseline_hash = current
        selections[arm] = {**selection, "selected_prompt": prompts["selected"]}
        reuse = digest(prompts["selected"]) == current
        native[arm], rows[arm] = {}, {}
        evaluated = {}
        for variant in ("baseline", "selected"):
            actual = "baseline" if variant == "selected" and reuse else variant
            if actual not in evaluated:
                evaluated[actual] = _evaluate(
                    test,
                    runs_root / arm / "final" / f"{actual}.jsonl",
                    prompts[variant],
                    arm,
                    protocol_hash,
                    bootstrap_replicates,
                )
            result, raw = evaluated[actual]
            native[arm][variant] = {**result["summary"], "reused_baseline": variant == "selected" and reuse}
            rows[arm][variant] = result["per_image"]
            compact.extend(
                _compact(
                    sample,
                    row,
                    raw.get(sample["id"], {}),
                    arm,
                    variant,
                    variant == "selected" and reuse,
                    protocol_hash,
                )
                for sample, row in zip(test, result["per_image"], strict=True)
            )
    comparisons = {
        "common_prompt_feature_effect": {
            arm: paired_class_balanced_delta(
                rows["image_only"]["baseline"], rows[arm]["baseline"], bootstrap_replicates
            )
            for arm in ARMS
        },
        "selected_pipeline_effect": {
            arm: paired_class_balanced_delta(
                rows["image_only"]["selected"], rows[arm]["selected"], bootstrap_replicates
            )
            for arm in ARMS
        },
        "within_arm_gepa_effect": {
            arm: paired_class_balanced_delta(
                rows[arm]["baseline"], rows[arm]["selected"], bootstrap_replicates
            )
            for arm in ARMS
        },
    }
    controls, control_rows, control_cases = {}, [], []
    if controls_path is not None:
        if _sha(controls_path) != protocol.get("controls_sha256"):
            raise ValueError("Control cases differ from the frozen protocol")
        control_cases = _json(controls_path)["cases"]
        controls, control_rows = _control_report(
            control_cases, test, rows, selections, runs_root, protocol_hash, bootstrap_replicates
        )
    selection_times = [selected.get("selected_at") for selected in selections.values()]
    first_calls = [
        value["first_attempt_at"]
        for arm in native.values()
        for value in arm.values()
        if value["first_attempt_at"]
    ]
    first_calls += [
        value["first_attempt_at"]
        for arm in controls.values()
        for value in arm.values()
        if value["first_attempt_at"]
    ]
    latest = max(selection_times) if all(selection_times) else None
    earliest = min(first_calls) if first_calls else None
    timing_ok = (
        datetime.fromisoformat(latest) <= datetime.fromisoformat(earliest) if latest and earliest else None
    )
    if timing_ok is False:
        raise ValueError("Final or control calls preceded the last arm selection")
    expenses = _expense_inventory(runs_root)
    if amendment_path is None:
        candidate = Path(protocol_path).resolve().parent / "cost-recovery-amendment.json"
        amendment_path = candidate if candidate.exists() else None
    if reallocation_path is None:
        candidate = Path(protocol_path).resolve().parent / "cost-recovery-reallocation.json"
        reallocation_path = candidate if candidate.exists() else None
    if timeout_extension_path is None:
        candidate = Path(protocol_path).resolve().parent / "cost-recovery-timeouts.json"
        timeout_extension_path = candidate if candidate.exists() else None
    if shared_pool_path is None:
        candidate = Path(protocol_path).resolve().parent / "cost-recovery-shared-pool.json"
        shared_pool_path = candidate if candidate.exists() else None
    if http_extension_path is None:
        candidate = Path(protocol_path).resolve().parent / "cost-recovery-http-errors.json"
        http_extension_path = candidate if candidate.exists() else None
    if amendment_path is not None:
        _recovery_accounting(
            expenses,
            runs_root,
            amendment_path,
            protocol_hash,
            project_root,
            reallocation_path,
            timeout_extension_path,
            shared_pool_path,
            http_extension_path,
        )
    elif any(
        path is not None
        for path in (reallocation_path, timeout_extension_path, shared_pool_path, http_extension_path)
    ):
        raise ValueError("Operational extensions require the recovery amendment")
    elif any((runs_root / "recovery-receipts").glob("*.json")):
        raise ValueError("Recovery receipts require the operational amendment")
    return {
        "version": 1,
        "dataset_sha256": _sha(dataset_path),
        "protocol_sha256": protocol_hash,
        "coverage": coverage_summary(samples),
        "native_c2pa_claims": provenance_summary(test),
        "native": native,
        "freeze_timing": {
            "latest_selection_at": latest,
            "first_evaluation_at": earliest,
            "all_selections_preceded_evaluation": timing_ok,
        },
        "comparisons": comparisons,
        "controls": controls,
        "selection": {
            arm: {key: value for key, value in selected.items() if key != "selected_prompt"}
            for arm, selected in selections.items()
        },
        "expenses": expenses,
        "preprocessing": {
            "native_unique_cases": _preprocessing(samples),
            "control_unique_cases": _preprocessing(control_cases),
        },
        "cases": compact + control_rows,
        "limits": [
            "Native source/metadata availability may encode origin labels",
            "Controls are not pooled with native accuracy",
            "GEPA pipeline effects combine context and independently selected prompts",
            "Reused baseline rows are repeated references, not additional paid attempts",
            "Unknown costs are null; acquisition, storage and local compute charges are excluded",
            "No C2PA assertion is treated as guaranteed origin truth",
            "Intervals are descriptive and do not correct for multiple arm comparisons or deployment prevalence",
        ],
    }


def export_selected_prompts(selections, output_root, project_root=None):
    """Publish frozen prompt bodies and portable paths without changing raw receipts."""
    project_root = Path(project_root or ROOT).resolve()
    output_root = Path(output_root).resolve()
    prompts = {}
    for arm, selection in selections.items():
        source = Path(selection["selected_prompt_path"])
        prompt = _json(source if source.is_absolute() else project_root / source)
        if digest(prompt) != selection["selected_prompt_sha256"]:
            raise ValueError("Selected prompt changed before export")
        prompts[arm] = prompt
    directory = output_root / "prompts"
    directory.mkdir(parents=True, exist_ok=True)
    exported = {}
    for arm, prompt in prompts.items():
        path = directory / f"{arm}.json"
        path.write_text(json.dumps(prompt, indent=2, allow_nan=False) + "\n")
        selection = dict(selections[arm])
        for key, value in selection.items():
            if key.endswith("_path") and isinstance(value, str) and Path(value).is_absolute():
                selection[key] = os.path.relpath(value, project_root)
        selection["selected_export_path"] = os.path.relpath(path, project_root)
        selection["selected_prompt_path"] = selection["selected_export_path"]
        selection["selected_prompt_id"] = prompt.get("id")
        exported[arm] = selection
    return exported


def _fmt(value, places=4):
    return "unknown" if value is None else f"{value:.{places}f}"


def _percent(value):
    return "unknown" if value is None else f"{value * 100:.1f}%"


def _gain(delta):
    interval = delta["interval"]
    ci = "unknown" if interval is None else f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"
    return f"{_fmt(delta['delta'])} {ci}"


def _cost(expenses):
    if expenses["unknown_cost_count"]:
        return (
            f"{_fmt(expenses['known_effective_cost_usd'], 6)} known; {expenses['unknown_cost_count']} unknown"
        )
    return _fmt(expenses["effective_cost_usd"], 6)


def markdown_report(report):
    diagnostic = report["coverage"]["availability_only_dev"]
    numeric_diagnostic = report["coverage"]["numeric_camera_only_dev"]
    provenance = report["native_c2pa_claims"]
    image_only = report["native"]["image_only"]["selected"]["metrics"]
    all_context = report["native"]["all_context"]["selected"]["metrics"]
    ai_count = report["coverage"]["by_split"]["test"]["generated"]["count"]
    ai_claims = provenance["validated_ai_claims_agreeing_with_source"]
    lead = (
        f"On {image_only['n_expected']} held-out images, selected image-only Decider reached {_percent(image_only['accuracy_all_expected_invalid_as_wrong'])} accuracy. All context reached {_percent(all_context['accuracy_all_expected_invalid_as_wrong'])}. AI detections were {image_only['valid_only']['confusion']['tp']}/{ai_count} for image-only and {all_context['valid_only']['confusion']['tp']}/{ai_count} for all context."
        if image_only["n_valid"] and all_context["n_valid"]
        else "Held-out results are incomplete; missing and failed cases remain in the accuracy denominator."
    )
    lines = [
        "# Decider: native metadata and controls",
        "",
        lead,
        "",
        f"C2PA coverage is a separate limit: {ai_claims}/{ai_count} AI images had an AI claim with a valid signature and image binding; {ai_count - ai_claims}/{ai_count} lacked such a claim. Missing credentials do not prove camera origin. Native metadata can also identify source datasets, which limits how well these results can transfer.",
        "",
        "See the [methods](README.md#frozen-experiment) and [paper review](../../research/metadata-2026-10-12/paper-and-design.md). Source notes cover [camera images](../../research/metadata-2026-10-12/real-sources.md), [generated images](../../research/metadata-2026-10-12/ai-sources.md) and [content credentials](../../research/metadata-2026-10-12/c2pa-sources.md).",
        "",
        "<details>",
        "<summary>Source coverage and label shortcuts</summary>",
        "",
        "| Split | Class | Images | EXIF present | Numeric camera fields | Groups |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split, classes in report["coverage"]["by_split"].items():
        for label, value in classes.items():
            lines.append(
                f"| {split} | {label} | {value['count']} | {value['camera_present']} | {value['camera_numeric_present']} | {value['groups']} |"
            )
    lines += [
        "",
        f"On development data, a rule based only on EXIF presence scored {_percent(diagnostic['accuracy'])} accuracy. A rule based on numeric camera settings scored {_percent(numeric_diagnostic['accuracy'])}. These scores show how metadata availability can identify the source dataset.",
        "",
        f"Across all {provenance['n']} native test images, {provenance['manifest_present']} had manifests and {provenance['explicit_ai_claims']} had explicit AI claims; {provenance['signature_and_binding_valid_ai_claims']} claims passed signature and binding checks. The fixed trust check marked {provenance['trust_status'].get('trusted', 0)} trusted and {provenance['trust_status'].get('untrusted', 0)} untrusted. Signature validity and signer trust are separate checks; neither guarantees true content.",
        "",
        "</details>",
        "",
    ]
    for variant, title in (
        ("selected", "Independently selected GEPA pipelines"),
        ("baseline", "Common prompt: fixed feature effect"),
    ):
        if variant == "baseline":
            lines += ["<details>", "<summary>Common prompt and paired comparisons</summary>", ""]
        lines += [
            f"## {title}",
            "",
            "| Arm | Accuracy (%) | AI recall (%) | Real FPR (%) | Brier (0–1) | Cost USD | p50 s | p95 s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        counts, review_rates = {}, []
        for arm, results in report["native"].items():
            value = results[variant]
            metric, ops = value["metrics"], value["operations"]
            counts[arm] = (metric["n_valid"], metric["n_expected"])
            review_rates.append(metric["review_or_failure_fraction_all_expected"])
            lines.append(
                f"| {arm}{' (reused)' if value['reused_baseline'] else ''} | {_percent(metric['accuracy_all_expected_invalid_as_wrong'])} | {_percent(metric['recall_all_expected'])} | {_percent(metric['valid_only']['false_positive_rate'])} | {_fmt(metric['brier_invalid_or_unattempted_penalty1'])} | {_cost(value['expenses'])} | {_fmt(ops['latency_p50_seconds'], 3)} | {_fmt(ops['latency_p95_seconds'], 3)} |"
            )
        same_counts = len(set(counts.values())) == 1
        note = (
            f"Each arm returned {next(iter(counts.values()))[0]}/{next(iter(counts.values()))[1]} valid outputs."
            if same_counts
            else "Valid/intended outputs: "
            + "; ".join(f"{arm} {valid}/{expected}" for arm, (valid, expected) in counts.items())
            + "."
        )
        available = [value for value in review_rates if value is not None]
        if available:
            note += f" Review or failure rates range from {_percent(min(available))} to {_percent(max(available))}; per-arm values are in JSON."
        lines += ["", note, ""]
    lines += [
        "Accuracy and AI recall count invalid or missing outputs as failures. Brier assigns these outputs loss 1; lower scores are better. Real FPR uses valid outputs only. Cost cells show the known subtotal and the number of unknown charges. Latency uses recorded request durations. JSON retains balanced accuracy, AUROC and failure rates.",
        "",
        "## Paired reward changes",
        "",
        "| Arm | Fixed feature effect vs image only | Selected pipeline vs image only | GEPA effect within arm |",
        "| --- | --- | --- | --- |",
    ]
    for arm in report["selection"]:
        gains = [
            _gain(report["comparisons"][key][arm])
            for key in ("common_prompt_feature_effect", "selected_pipeline_effect", "within_arm_gepa_effect")
        ]
        lines.append(f"| {arm} | {' | '.join(gains)} |")
    lines += [
        "",
        "Values are changes in 1−Brier with 95% intervals from source-group resampling, with equal class weights. Selected-pipeline effects include prompt search. Intervals are descriptive, with no correction for multiple comparisons. Proposal counts are in the restored JSON; a positive estimate alone does not establish an improvement.",
        "",
        "</details>",
        "",
        "## Separate controls",
        "",
    ]
    if report["controls"]:
        stripped = report["controls"].get("c2pa_stripped", {})
        if stripped:
            changes = []
            for arm, result in stripped.items():
                paired = result["paired_native_subset"]
                native, control = paired["native_metrics"], paired["control_metrics"]
                count = native["class_counts_expected"]["generated"]
                changes.append(
                    f"{arm}: {native['valid_only']['confusion']['tp']}/{count} to {control['valid_only']['confusion']['tp']}/{count}"
                )
            lines += [
                "Removing C2PA changed AI detections on the same images (" + "; ".join(changes) + "). These controls show how the model used credentials. They contain only generated images, so they cannot measure false positives on real photos.",
                "",
            ]
        lines += [
            "| Condition | Arm | Native / control Brier | Paired ΔBrier (95% CI) | Valid / intended | Integrity passed / asserted |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for condition, arms in report["controls"].items():
            for arm, result in arms.items():
                metric, checks = result["metrics"], result["integrity_fixtures"]
                paired = result["paired_native_subset"]
                before = paired["native_metrics"]["brier_invalid_or_unattempted_penalty1"]
                after = paired["control_metrics"]["brier_invalid_or_unattempted_penalty1"]
                change = _gain({"delta": paired["brier_change"], "interval": paired["brier_change_interval"]})
                lines.append(
                    f"| {condition} | {arm} | {_fmt(before)} / {_fmt(after)} | {change} | {metric['n_valid']} / {metric['n_expected']} | {checks['passed']} / {checks['asserted']} |"
                )
    else:
        lines.append("No approved control results were supplied.")
    expense = report["expenses"]
    lines += [
        "",
        "Controls use the same native images, including failed or missing predictions. Positive ΔBrier means worse performance; represented classes and source groups have equal weight. The restored JSON retains accuracy, probability shifts, decision flips and class counts. Integrity checks test the validator, not origin detection; missing or trusted claims do not guarantee an origin label.",
        "",
        f"Actual total cost across all raw trials, reflection, final requests and controls: **{('unknown' if expense['effective_cost_usd'] is None else '$' + _fmt(expense['effective_cost_usd'], 6))}**. Known reported cost: ${_fmt(expense['known_effective_cost_usd'], 6)}; unknown-cost attempts: {expense['unknown_cost_count']}. Baseline reuse adds no calls. Do not sum costs across repeated reference rows.",
        "",
        f"Native image preparation took {_fmt(report['preprocessing']['native_unique_cases']['seconds'], 3)} seconds in total; {report['preprocessing']['native_unique_cases']['unknown_count']} images have no recorded time. Extra preparation time for the controls was not measured.",
        "",
        "[Per-image results](results/cases.csv) and [selected prompts](prompts/) remain in Git. [Restore the local evidence](README.md#run-and-verify) for `results/summary.json`: source slices, class counts, intervals, extraction times, selection records and hashes. Raw responses and detailed evidence remain outside Git.",
        "",
    ]
    if expense.get("recovery"):
        lines += [
            "<details>",
            "<summary>Spending guard and unknown charges</summary>",
            "",
            f"Unverified allowances reserve **${_fmt(expense['reserved_unverified_usd'], 6)}**; the guard-accounted total is **${_fmt(expense['budget_accounted_usd'], 6)}**. Allowances are not charged amounts or proven billing limits. Missing costs remain unknown, with no retries. [Restored amendment and receipt evidence](README.md#run-and-verify) records each allowance.",
            "",
        ]
        shared = expense["recovery"].get("shared_pool_amendment")
        if shared:
            lines += [
                f"The current guard caps known spending at ${_fmt(shared['known_spend_budget_per_arm_usd'], 2)} per arm. All unknown allowances share the ${_fmt(shared['global_budget_usd'], 2)} global cap, which includes known costs and in-flight reservations. The global guard accounts for ${_fmt(expense['global_guard_accounted_usd'], 6)}, including ${_fmt(expense['reserved_inflight_usd'], 6)} in flight. The [restored amendment chain](README.md#run-and-verify) preserves earlier limits; requests and retry rules are unchanged.",
                "",
            ]
        elif expense["recovery"].get("reallocation"):
            reallocation = expense["recovery"]["reallocation"]
            revised = "; ".join(
                f"{arm} ${_fmt(cap, 2)}"
                for arm, cap in reallocation["guard_budget_by_arm"].items()
                if cap != reallocation["original_run_receipt_arm_usd"]
            )
            lines += [
                f"The second operational amendment sets {revised}, using ${_fmt(reallocation['reallocated_unused_global_headroom_usd'], 2)} of unused headroom; the global cap stays ${_fmt(reallocation['global_budget_usd'], 2)}. Original run receipts remain unchanged.",
                "",
            ]
        if not shared and expense["recovery"].get("timeout_extension"):
            extension = expense["recovery"]["timeout_extension"]
            lines += [
                f"The [restored timeout extension](README.md#run-and-verify) also covers Decider read timeouts with the same ${_fmt(extension['allowance_usd'], 2)} reserve and no retries; each new unknown batch stops for network review.",
                "",
            ]
        lines += ["</details>", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = ROOT / "tracks/metadata_v3"
    parser.add_argument("--dataset", type=Path, default=directory / "dataset.json")
    parser.add_argument("--protocol", type=Path, default=directory / "protocol.json")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "data/metadata-v3/runs")
    parser.add_argument("--controls", type=Path)
    parser.add_argument(
        "--recovery-amendment", type=Path, help="Defaults to the protocol's sibling amendment, if present"
    )
    parser.add_argument(
        "--budget-reallocation", type=Path, help="Defaults to the protocol's sibling reallocation, if present"
    )
    parser.add_argument(
        "--timeout-extension",
        type=Path,
        help="Defaults to the protocol's sibling timeout extension, if present",
    )
    parser.add_argument(
        "--shared-pool-amendment",
        type=Path,
        help="Defaults to the protocol's sibling shared-pool amendment, if present",
    )
    parser.add_argument(
        "--http-extension",
        type=Path,
        help="Defaults to the protocol's sibling HTTP-error extension, if present",
    )
    parser.add_argument("--output-root", type=Path, default=directory)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if not 0 <= args.bootstrap_replicates <= 10000:
        parser.error("Use 0..10000 bootstrap replicates")
    result = build_report(
        args.dataset,
        args.protocol,
        args.runs_root,
        args.controls,
        bootstrap_replicates=args.bootstrap_replicates,
        amendment_path=args.recovery_amendment,
        reallocation_path=args.budget_reallocation,
        timeout_extension_path=args.timeout_extension,
        shared_pool_path=args.shared_pool_amendment,
        http_extension_path=args.http_extension,
    )
    result["selection"] = export_selected_prompts(result["selection"], args.output_root)
    output = args.output_root / "results"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result["cases"])
    compact_summary = {key: value for key, value in result.items() if key != "cases"}
    (output / "summary.json").write_text(json.dumps(compact_summary, indent=2, allow_nan=False) + "\n")
    (args.output_root / "report.md").write_text(markdown_report(result))
    print(f"Saved {len(result['cases'])} case rows and {len(result['native'])} native arm comparisons")


if __name__ == "__main__":
    main()
