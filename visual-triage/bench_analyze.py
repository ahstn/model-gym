"""Summarize frozen benchmark ledgers without fitting thresholds or changing labels."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

AXES = ("technical_quality", "aesthetics")
GROUP_FIELDS = ("benchmark", "task", "arm", "prompt_id", "requested_reasoning", "provider", "partition")


def numeric(value, low=-math.inf, high=math.inf):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and low <= value <= high
    )


def average(values):
    return mean(values) if values else None


def ranks(values):
    """Return one-based midranks, including ties, in the original order."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in ordered[start:end]:
            result[position] = rank
        start = end
    return result


def pearson(left, right):
    if len(left) != len(right):
        raise ValueError("Correlation vectors have different lengths")
    if len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    x = [value - left_mean for value in left]
    y = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in x) * sum(value * value for value in y))
    return sum(a * b for a, b in zip(x, y)) / denominator if denominator else None


def correlations(pairs):
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    return {"n": len(pairs), "pearson": pearson(left, right), "spearman": pearson(ranks(left), ranks(right))}


def auc(probabilities, labels):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    rank_sum = sum(rank for rank, label in zip(ranks(probabilities), labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def wilson_interval(successes, total):
    """Two-sided 95% binomial interval; sampling bias and dependence remain unmodeled."""
    if total == 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def sample_group(sample):
    source = sample.get("source", {})
    for owner, keys in (
        (sample, ("sample_group", "group_id", "parent_id")),
        (source, ("group_id", "parent_id", "split_group")),
    ):
        for key in keys:
            if owner.get(key):
                return str(owner[key])
    # The original AGIQA panel used one subject prompt across all generators.
    if source.get("component") == "agiqa":
        labels = source.get("source_labels", {})
        fields = [str(labels.get(key, "")) for key in ("prompt", "adj1", "adj2", "style")]
        if fields[0].strip():
            return "agiqa-prompt:" + "|".join(" ".join(value.split()).lower() for value in fields)
    return str(source.get("id") or sample["id"])


def ai_label(sample):
    label = sample.get("labels", {}).get("ai_generated")
    if isinstance(label, bool):
        return label
    label = sample.get("source", {}).get("known_ai_generated")
    return label if isinstance(label, bool) else None


def human_target(sample, axis):
    """Read only explicitly mapped task labels; never infer a MOS scale or construct."""
    label = sample.get("labels", {}).get(axis)
    if numeric(label, 1, 5):
        return {"normalized": label, "raw": label}
    if isinstance(label, dict):
        normalized = label.get("normalized_1_to_5")
        raw = label.get("value")
        if not numeric(normalized, 1, 5) and label.get("scale") == [1, 5]:
            normalized = raw
        return {
            "normalized": normalized if numeric(normalized, 1, 5) else None,
            "raw": raw if numeric(raw) else None,
        }
    return {"normalized": None, "raw": None}


def valid_prediction(record):
    prediction = record.get("normalized")
    if record.get("validation_error") or not isinstance(prediction, dict):
        return False
    if record["task"] == "quality":
        return all(numeric(prediction.get(axis), 1, 5) for axis in AXES)
    if record["task"] == "ai":
        uncertainty = prediction.get("uncertainty")
        return (
            numeric(prediction.get("ai_probability"), 0, 1)
            and (numeric(uncertainty, 0, 1) or uncertainty in ("low", "medium", "high"))
            and isinstance(prediction.get("medium"), str)
            and bool(prediction["medium"])
        )
    raise ValueError(f"Unsupported benchmark task: {record['task']}")


def scalar_metrics(rows, axis, reference_key):
    referenced = [row for row in rows if numeric(row[reference_key].get(axis), 1, 5)]
    pairs, errors, penalized, groups = [], [], [], defaultdict(list)
    for row in referenced:
        target = row[reference_key][axis]
        error = 4.0
        if row["valid"]:
            prediction = row["prediction"][axis]
            error = abs(prediction - target)
            pairs.append((prediction, target))
            errors.append(error)
        penalized.append(error)
        groups[row["sample_group"]].append(error)
    return {
        "referenced": len(referenced),
        "valid_referenced": len(pairs),
        "mae_valid": average(errors),
        "mae_invalid_penalty_4": average(penalized),
        "group_balanced_mae_invalid_penalty_4": average([mean(values) for values in groups.values()]),
        "reference_groups": len(groups),
        **correlations(pairs),
    }


def quality_metrics(rows):
    result = {}
    for key in ("assistant_reference", "human_reference"):
        axes = {axis: scalar_metrics(rows, axis, key) for axis in AXES}
        group_errors = [axes[axis]["group_balanced_mae_invalid_penalty_4"] for axis in AXES]
        valid_errors = [axes[axis]["mae_valid"] for axis in AXES]
        result[key] = {
            **axes,
            "mean_axis_mae_valid": mean(valid_errors)
            if all(value is not None for value in valid_errors)
            else None,
            "mean_axis_group_balanced_mae_invalid_penalty_4": mean(group_errors)
            if all(value is not None for value in group_errors)
            else None,
        }
    result["human_raw_correlations"] = {
        axis: correlations(
            [
                (row["prediction"][axis], row["human_raw"][axis])
                for row in rows
                if row["valid"] and numeric(row["human_raw"].get(axis))
            ]
        )
        for axis in AXES
    }
    return result


def ai_metrics(rows):
    labeled = [row for row in rows if isinstance(row["ai_generated"], bool)]
    valid = [row for row in labeled if row["valid"]]
    labels = [int(row["ai_generated"]) for row in valid]
    probabilities = [row["prediction"]["ai_probability"] for row in valid]
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for probability, label in zip(probabilities, labels):
        counts[
            ("t" if (probability >= 0.5) == bool(label) else "f") + ("p" if probability >= 0.5 else "n")
        ] += 1
    positive_count = counts["tp"] + counts["fn"]
    negative_count = counts["tn"] + counts["fp"]
    recall = counts["tp"] / positive_count if positive_count else None
    specificity = counts["tn"] / negative_count if negative_count else None
    valid_rows = [row for row in rows if row["valid"]]
    review_count = sum(0.2 < row["prediction"]["ai_probability"] < 0.8 for row in valid_rows)
    assistant_pairs = [
        (row["prediction"]["ai_probability"], row["assistant_reference"]["ai_probability"])
        for row in valid_rows
        if numeric(row["assistant_reference"].get("ai_probability"), 0, 1)
    ]
    return {
        "labeled": len(labeled),
        "valid_labeled": len(valid),
        "invalid_labeled": len(labeled) - len(valid),
        "unlabeled": len(rows) - len(labeled),
        "threshold": 0.5,
        "confusion_valid_only": counts,
        "ai_recall_valid_only": recall,
        "ai_recall_wilson95_valid_only": wilson_interval(counts["tp"], positive_count),
        "false_positive_rate_valid_only": 1 - specificity if specificity is not None else None,
        "false_positive_rate_wilson95_valid_only": wilson_interval(counts["fp"], negative_count),
        "balanced_accuracy_valid_only": (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None,
        "accuracy_valid_only": (counts["tp"] + counts["tn"]) / len(valid) if valid else None,
        "accuracy_wilson95_valid_only": wilson_interval(counts["tp"] + counts["tn"], len(valid)),
        "accuracy_all_labeled_invalid_as_wrong": (counts["tp"] + counts["tn"]) / len(labeled)
        if labeled
        else None,
        "brier_valid_only": average(
            [(probability - label) ** 2 for probability, label in zip(probabilities, labels)]
        ),
        "auroc_valid_only": auc(probabilities, labels),
        "review_interval_open": [0.2, 0.8],
        "review_count_valid": review_count,
        "review_fraction_valid": review_count / len(valid_rows) if valid_rows else None,
        "automatic_coverage_all_attempts": (len(valid_rows) - review_count) / len(rows) if rows else None,
        "assistant_probability_mae_valid": average([abs(left - right) for left, right in assistant_pairs]),
        "assistant_probability_pairs": len(assistant_pairs),
    }


def operational_metrics(rows):
    costs = [row["cost_usd"] for row in rows if numeric(row["cost_usd"], 0)]
    effective = [row["effective_cost_usd"] for row in rows if numeric(row["effective_cost_usd"], 0)]
    latencies = sorted(row["seconds"] for row in rows if numeric(row["seconds"], 0))
    valid_count = sum(row["valid"] for row in rows)
    complete = len(effective) == len(rows)
    cost = sum(costs)
    effective_cost = sum(effective)
    return {
        "attempted": len(rows),
        "valid": valid_count,
        "invalid": len(rows) - valid_count,
        "valid_fraction": valid_count / len(rows) if rows else None,
        "reported_cost_usd": cost,
        "openrouter_charged_cost_usd": cost,
        "openrouter_cost_complete": len(costs) == len(rows),
        "effective_provider_inference_cost_usd": effective_cost if complete else None,
        "known_effective_provider_inference_cost_usd": effective_cost,
        "cost_complete": complete,
        "unknown_cost_count": len(rows) - len(effective),
        "cost_per_attempt_usd": effective_cost / len(rows) if complete and rows else None,
        "cost_per_valid_result_usd": effective_cost / valid_count if complete and valid_count else None,
        "cost_basis_counts": {
            basis: sum(row["cost_basis"] == basis for row in rows)
            for basis in sorted({row["cost_basis"] for row in rows})
        },
        "latency_count": len(latencies),
        "latency_p50_seconds": median(latencies) if latencies else None,
        "latency_p95_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
    }


def costs_for_comparison(record):
    """Separate provider inference from OpenRouter charges without assuming additive fees."""
    usage = record.get("usage") or record.get("response", {}).get("usage") or {}
    charged = record.get("cost_usd")
    if not numeric(charged, 0):
        charged = usage.get("cost")
    charged = charged if numeric(charged, 0) else None
    if usage.get("is_byok") is True:
        upstream = (usage.get("cost_details") or {}).get("upstream_inference_cost")
        effective = upstream if numeric(upstream, 0) else None
        basis = "byok_upstream_inference" if effective is not None else "unknown_byok_upstream_cost"
    else:
        effective, basis = charged, "openrouter_usage_cost"
    return {"cost_usd": charged, "effective_cost_usd": effective, "cost_basis": basis}


def load_samples(paths, snapshots=None):
    samples = {}
    for path in paths:
        document = json.loads(snapshots[path] if snapshots is not None else path.read_bytes())
        for sample in document["samples"]:
            sample = {**sample, "_benchmark": document.get("benchmark", path.stem)}
            if sample["id"] in samples and samples[sample["id"]] != sample:
                raise ValueError(f"Conflicting sample metadata: {sample['id']}")
            samples[sample["id"]] = sample
    return samples


def combine_record(record, references, samples):
    identifier = record["sample"]
    sample = samples.get(identifier, {"id": identifier})
    reference = references.get(identifier, {})
    hashes = {
        item["delivered_sha256"] for item in (record, sample, reference) if item.get("delivered_sha256")
    }
    if len(hashes) > 1:
        raise ValueError(f"Delivered image hash does not match assessment or manifest: {identifier}")
    if samples and identifier not in samples:
        raise ValueError(f"Ledger sample absent from supplied manifests: {identifier}")
    group = {field: record.get(field) for field in GROUP_FIELDS if field != "partition"}
    group["partition"] = reference.get("development_partition")
    targets = {axis: human_target(sample, axis) for axis in AXES}
    source = sample.get("source", {})
    return {
        **group,
        "sample": identifier,
        "sample_group": sample_group(sample),
        "source_stratum": source.get("source_kind") or source.get("component") or source.get("dataset"),
        "delivered_sha256": next(iter(hashes), None),
        "valid": valid_prediction(record),
        "prediction": record.get("normalized"),
        "assistant_reference": {key: reference[key] for key in (*AXES, "ai_probability") if key in reference},
        "assistant_uncertainty": {
            key: reference[key] for key in ("confidence", "ai_probability_interval") if key in reference
        },
        "human_reference": {axis: target["normalized"] for axis, target in targets.items()},
        "human_raw": {axis: target["raw"] for axis, target in targets.items()},
        "human_label_metadata": {axis: sample.get("labels", {}).get(axis) for axis in AXES},
        "ai_generated": ai_label(sample),
        **costs_for_comparison(record),
        "seconds": record.get("seconds"),
        "validation_error": record.get("validation_error"),
        "ledger": record["ledger"],
    }


def summarize(records, references=None, samples=None):
    references, samples = references or {}, samples or {}
    grouped, rows, seen = defaultdict(list), [], set()
    for record in records:
        row = combine_record(record, references, samples)
        group = {field: row[field] for field in GROUP_FIELDS}
        key = json.dumps(group, sort_keys=True)
        identity = (key, row["sample"])
        if identity in seen:
            raise ValueError(f"Duplicate sample attempt for the same configuration: {row['sample']}")
        seen.add(identity)
        grouped[key].append(row)
        rows.append(row)
    groups = []
    for key, members in sorted(grouped.items()):
        metadata = json.loads(key)
        summary = {**metadata, **operational_metrics(members)}
        summary["sample_ids"] = sorted(row["sample"] for row in members)
        summary["ledger_files"] = sorted({row["ledger"] for row in members})
        expected = {
            identifier
            for identifier, sample in samples.items()
            if sample.get("_benchmark", metadata["benchmark"]) == metadata["benchmark"]
            and references.get(identifier, {}).get("development_partition") == metadata["partition"]
        }
        summary["manifest_expected_count"] = len(expected) if samples else None
        summary["manifest_unattempted_ids"] = sorted(expected - set(summary["sample_ids"]))
        if metadata["task"] == "quality":
            summary["quality"] = quality_metrics(members)
        else:
            summary["ai"] = ai_metrics(members)
            strata = sorted({row["source_stratum"] for row in members if row["source_stratum"]})
            summary["ai_by_source_stratum"] = {
                stratum: ai_metrics([row for row in members if row["source_stratum"] == stratum])
                for stratum in strata
            }
        groups.append(summary)
    return {
        "protocol": {
            "assistant_reference_is_human_gold": False,
            "invalid_quality_penalty": 4,
            "ai_threshold": 0.5,
            "review_interval_open": [0.2, 0.8],
            "human_labels": "Explicit labels.technical_quality/aesthetics only; no implicit MOS conversion",
            "auc_ties": "Mann-Whitney midranks; null when either class is absent",
            "ai_rate_intervals": "Two-sided 95% Wilson binomial intervals; convenience sampling, source bias and correlated images are not covered",
            "latency_p95": "Nearest rank over attempted calls with reported latency",
            "provider_policy": "Each returned provider is a separate result group",
            "cost_comparison": "BYOK: upstream inference only; otherwise OpenRouter usage.cost. OpenRouter charges reported separately; not assumed additive.",
        },
        "total": operational_metrics(rows),
        "manifest_samples": len(samples),
        "referenced_samples": len(references),
        "manifest_unattempted_ids": sorted(set(samples) - {row["sample"] for row in rows}),
        "groups": groups,
        "per_image": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledgers", nargs="+", type=Path, required=True)
    parser.add_argument("--references", nargs="+", type=Path, default=[])
    parser.add_argument("--manifest", nargs="+", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = (*args.ledgers, *args.references, *args.manifest)
        snapshots = {path: path.read_bytes() for path in paths}
        records = []
        for path in args.ledgers:
            records.extend(
                {**json.loads(line), "ledger": str(path)} for line in snapshots[path].splitlines() if line
            )
        result = summarize(
            records, load_samples(args.references, snapshots), load_samples(args.manifest, snapshots)
        )
        result["inputs"] = [
            {
                "path": str(path),
                "sha256": hashlib.sha256(snapshots[path]).hexdigest(),
                "bytes": len(snapshots[path]),
            }
            for path in paths
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"Saved {len(result['groups'])} result groups for {result['total']['attempted']} attempts")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
