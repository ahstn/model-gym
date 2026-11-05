"""Track metrics, grouped uncertainty intervals, and trace feedback; no API calls."""

import math
import random
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_analyze import (
    auc,
    costs_for_comparison,
    numeric,
    operational_metrics,
    pearson,
    ranks,
    wilson_interval,
)
from tracks.settings import task_for

QUALITY_BAD_THRESHOLD = 2.6
AI_THRESHOLD = 0.5
REVIEW_BAND = (0.2, 0.8)
QUALITY_BINS = (1.8, 2.6, 3.4, 4.2)


def average(values):
    return mean(values) if values else None


def confusion(labels, predictions):
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for gold, prediction in zip(labels, predictions, strict=True):
        key = ("t" if gold == prediction else "f") + ("p" if prediction else "n")
        counts[key] += 1
    positives, negatives = counts["tp"] + counts["fn"], counts["tn"] + counts["fp"]
    positive_predictions = counts["tp"] + counts["fp"]
    recall = counts["tp"] / positives if positives else None
    specificity = counts["tn"] / negatives if negatives else None
    return {
        "confusion": counts,
        "accuracy": (counts["tp"] + counts["tn"]) / len(labels) if labels else None,
        "balanced_accuracy": (recall + specificity) / 2 if positives and negatives else None,
        "precision": counts["tp"] / positive_predictions if positive_predictions else None,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": counts["fp"] / negatives if negatives else None,
        "accuracy_wilson95_naive_independent_images": wilson_interval(
            counts["tp"] + counts["tn"], len(labels)
        ),
        "recall_wilson95": wilson_interval(counts["tp"], positives),
        "false_positive_rate_wilson95": wilson_interval(counts["fp"], negatives),
        "interval_note": "Accuracy Wilson interval is naive and assumes independent images; use whole-group bootstrap for grouped accuracy. Recall/FPR Wilson intervals assume independent observations within each class; sampling and source bias remain",
    }


def prediction_valid(track, record):
    prediction = record.get("normalized")
    if record.get("status") != 200 or record.get("validation_error") or not isinstance(prediction, dict):
        return False
    if task_for(track) == "quality":
        return all(numeric(prediction.get(key), 1, 5) for key in ("technical_quality", "aesthetics"))
    uncertainty = prediction.get("uncertainty")
    return (
        numeric(prediction.get("ai_probability"), 0, 1)
        and isinstance(prediction.get("medium"), str)
        and bool(prediction["medium"])
        and (numeric(uncertainty, 0, 1) or uncertainty in ("low", "medium", "high"))
    )


def trace_row(track, sample, record=None):
    valid = record is not None and prediction_valid(track, record)
    prediction = record.get("normalized") if valid else None
    gold_key = "quality_mos_1to5" if task_for(track) == "quality" else "ai_generated"
    gold = sample["labels"][gold_key]
    row = {
        "id": sample["id"],
        "group_id": sample["group_id"],
        "split": sample["split"],
        "image": sample["image"],
        "delivered_sha256": sample["delivered_sha256"],
        "source": sample.get("source"),
        "gold": gold,
        "gold_field": gold_key,
        "attempted": record is not None,
        "valid": valid,
        "prediction": prediction,
        "reward": 0.0,
        "correct": None,
        "error": None,
        "cost_usd": None,
        "effective_cost_usd": None,
        "cost_basis": "not_attempted",
        "seconds": None,
    }
    if record is not None:
        row.update(
            **costs_for_comparison(record),
            seconds=record.get("seconds"),
            status=record.get("status"),
            model=record.get("resolved_model") or record.get("response", {}).get("model"),
            provider=record.get("provider") or record.get("response", {}).get("provider"),
            request_sha256=record.get("request_sha256"),
            validation_error=record.get("validation_error") or record.get("transport_error"),
        )
    if valid:
        value = prediction["technical_quality" if task_for(track) == "quality" else "ai_probability"]
        error = abs(value - gold) if task_for(track) == "quality" else (value - int(gold)) ** 2
        row.update(
            error=error,
            reward=1 - error / 4 if task_for(track) == "quality" else 1 - error,
            correct=(value < QUALITY_BAD_THRESHOLD) == (gold < QUALITY_BAD_THRESHOLD)
            if task_for(track) == "quality"
            else (value >= AI_THRESHOLD) == gold,
        )
    row["feedback"] = {
        "status": "valid" if valid else "invalid" if record else "unattempted",
        "target": {gold_key: gold},
        "prediction": prediction,
        "loss": row["error"],
        "reward": row["reward"],
        "error": row.get("validation_error"),
        "instruction": "Use visible evidence; keep quality independent of aesthetics."
        if task_for(track) == "quality"
        else "Assess image provenance cues; an attractive or rendered appearance alone is not proof of AI origin.",
    }
    return row


def quality_metrics(rows):
    valid = [row for row in rows if row["valid"]]
    gold = [row["gold"] for row in valid]
    prediction = [row["prediction"]["technical_quality"] for row in valid]
    errors = [left - right for left, right in zip(prediction, gold, strict=True)]
    bins = {}
    for index, label in enumerate(("bad", "poor", "fair", "good", "excellent")):
        members = [row for row in rows if bisect_right(QUALITY_BINS, row["gold"]) == index]
        bins[label] = {
            "n": len(members),
            "valid": sum(row["valid"] for row in members),
            "mae_valid": average([row["error"] for row in members if row["valid"]]),
        }
    binary = confusion(
        [value < QUALITY_BAD_THRESHOLD for value in gold],
        [value < QUALITY_BAD_THRESHOLD for value in prediction],
    )
    return {
        "n_expected": len(rows),
        "n_valid": len(valid),
        "mae_valid": average([abs(error) for error in errors]),
        "rmse_valid": math.sqrt(mean(error * error for error in errors)) if errors else None,
        "spearman_valid": pearson(ranks(gold), ranks(prediction)),
        "mae_invalid_or_unattempted_penalty4": average([row["error"] if row["valid"] else 4 for row in rows]),
        "bad_detection": {
            "rule": "quality < 2.6",
            "positive_class": "low_quality",
            "valid_only": binary,
            "accuracy_all_expected_invalid_as_wrong": sum(row["correct"] is True for row in rows) / len(rows)
            if rows
            else None,
        },
        "native_mos_bins": bins,
        "aesthetics": {"scored_against_gold": False, "reason": "No human aesthetic labels"},
    }


def ai_metrics(rows):
    valid = [row for row in rows if row["valid"]]
    labels = [row["gold"] for row in valid]
    probabilities = [row["prediction"]["ai_probability"] for row in valid]
    counts = confusion(labels, [value >= AI_THRESHOLD for value in probabilities])
    review = sum(REVIEW_BAND[0] < value < REVIEW_BAND[1] for value in probabilities)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    pairs = [
        members
        for members in grouped.values()
        if len(members) == 2 and {row["gold"] for row in members} == {False, True}
    ]
    valid_pairs = [pair for pair in pairs if all(row["valid"] for row in pair)]
    correct_pairs = sum(all(row["correct"] for row in pair) for pair in valid_pairs)
    ordered_pairs = sum(
        next(row["prediction"]["ai_probability"] for row in pair if row["gold"])
        > next(row["prediction"]["ai_probability"] for row in pair if not row["gold"])
        for pair in valid_pairs
    )
    return {
        "n_expected": len(rows),
        "n_valid": len(valid),
        "threshold": AI_THRESHOLD,
        "valid_only": counts,
        "accuracy_all_expected_invalid_as_wrong": sum(row["correct"] is True for row in rows) / len(rows)
        if rows
        else None,
        "auroc_valid": auc(probabilities, labels),
        "brier_valid": average([row["error"] for row in valid]),
        "brier_invalid_or_unattempted_penalty1": average(
            [row["error"] if row["valid"] else 1 for row in rows]
        ),
        "review_band_open": list(REVIEW_BAND),
        "review_count_valid": review,
        "review_fraction_valid": review / len(valid) if valid else None,
        "automatic_coverage_all_expected": (len(valid) - review) / len(rows) if rows else None,
        "pairs": {
            "expected": len(pairs),
            "valid": len(valid_pairs),
            "both_correct": correct_pairs,
            "both_correct_fraction_valid": correct_pairs / len(valid_pairs) if valid_pairs else None,
            "both_correct_wilson95": wilson_interval(correct_pairs, len(valid_pairs)),
            "ai_probability_higher_than_real": ordered_pairs,
            "ordering_fraction_valid": ordered_pairs / len(valid_pairs) if valid_pairs else None,
        },
    }


def objective_score(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row["reward"])
    return average([mean(values) for values in groups.values()])


def bootstrap_intervals(track, rows, replicates=500):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(groups) < 2 or replicates < 1:
        return {"group_count": len(groups), "replicates": 0, "intervals": {}}
    rng = random.Random(20261011)
    names = sorted(groups)
    values = defaultdict(list)
    for _ in range(replicates):
        selected = rng.choices(names, k=len(names))
        sampled = [
            dict(row, group_id=f"bootstrap-{index}")
            for index, name in enumerate(selected)
            for row in groups[name]
        ]
        result = quality_metrics(sampled) if task_for(track) == "quality" else ai_metrics(sampled)
        binary = (
            result["bad_detection"]["valid_only"] if task_for(track) == "quality" else result["valid_only"]
        )
        metrics = {
            "group_balanced_reward": objective_score(sampled),
            "accuracy_valid": binary["accuracy"],
            "balanced_accuracy_valid": binary["balanced_accuracy"],
        }
        if task_for(track) == "quality":
            metrics.update(
                mae_valid=result["mae_valid"],
                spearman_valid=result["spearman_valid"],
            )
        else:
            metrics.update(
                brier_valid=result["brier_valid"],
                auroc_valid=result["auroc_valid"],
            )
        for key, value in metrics.items():
            if value is not None:
                values[key].append(value)
    intervals = {}
    for key, items in values.items():
        items.sort()
        intervals[key] = {
            "percentile95": [
                items[int(0.025 * (len(items) - 1))],
                items[math.ceil(0.975 * (len(items) - 1))],
            ],
            "valid_replicates": len(items),
        }
    return {
        "group_count": len(groups),
        "replicates": replicates,
        "seed": 20261011,
        "method": "Resample whole source groups or matched pairs; intervals do not model source bias",
        "intervals": intervals,
    }


def paired_reward_delta(baseline, selected, replicates=500):
    """Estimate prompt gain using the same resampled source groups for both arms."""
    left = {row["id"]: row for row in baseline}
    right = {row["id"]: row for row in selected}
    if set(left) != set(right) or len(left) != len(baseline) or len(right) != len(selected):
        raise ValueError("Paired comparison requires the same unique image IDs")
    groups = defaultdict(list)
    for identifier, base in left.items():
        candidate = right[identifier]
        if any(base[key] != candidate[key] for key in ("group_id", "delivered_sha256", "gold", "split")):
            raise ValueError("Paired comparison inputs or groups differ")
        groups[base["group_id"]].append(candidate["reward"] - base["reward"])
    deltas = [mean(values) for values in groups.values()]
    interval = None
    if len(deltas) > 1 and replicates > 0:
        rng = random.Random(20261011)
        estimates = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(replicates))
        interval = [estimates[int(0.025 * (replicates - 1))], estimates[math.ceil(0.975 * (replicates - 1))]]
    return {
        "n_images": len(left),
        "n_groups": len(groups),
        "baseline_reward": objective_score(baseline),
        "selected_reward": objective_score(selected),
        "selected_minus_baseline": average(deltas),
        "paired_group_bootstrap95": interval,
        "replicates": replicates if interval is not None else 0,
        "seed": 20261011,
        "note": "The 0.005 development selection margin is not a test of statistical significance; bootstrap intervals do not cover source bias",
    }


def summarize(track, samples, records, bootstrap_replicates=0):
    by_id = {sample["id"]: sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError("Duplicate sample IDs")
    attempts = {}
    for record in records:
        identifier = record["sample"]
        if identifier not in by_id:
            continue
        if identifier in attempts:
            raise ValueError("Multiple attempts for one sample in a single prompt evaluation")
        if record.get("delivered_sha256") != by_id[identifier]["delivered_sha256"]:
            raise ValueError("Recorded delivered hash differs from the dataset")
        attempts[identifier] = record
    rows = [trace_row(track, sample, attempts.get(sample["id"])) for sample in samples]
    valid_records = list(attempts.values())
    identities = {
        (record.get("task"), record.get("arm"), record.get("prompt_sha256")) for record in valid_records
    }
    if len(identities) > 1:
        raise ValueError("A summary must contain one model, task and prompt")
    operations = operational_metrics([row for row in rows if row["attempted"]])
    return {
        "summary": {
            "track": track,
            "expected": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "unattempted": len(rows) - len(attempts),
            "operations": operations,
            "objective": {
                "name": "group_balanced_mean_reward",
                "score": objective_score(rows),
                "reward": "1 - absolute quality error / 4"
                if task_for(track) == "quality"
                else "1 - Brier loss",
                "invalid_or_unattempted_reward": 0,
            },
            "metrics": quality_metrics(rows) if task_for(track) == "quality" else ai_metrics(rows),
            "group_bootstrap": bootstrap_intervals(track, rows, bootstrap_replicates),
        },
        "per_image": rows,
    }
