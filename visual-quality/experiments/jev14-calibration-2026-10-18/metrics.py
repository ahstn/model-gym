"""Select an AI flagging rule on calibration data, then assess a frozen rule.

Each row has expected_ai, ai_probability, uncertainty, and valid. Non-flagged
cases require review; they are not classified as verified camera photos.
"""

import math
from collections import defaultdict
from functools import lru_cache

THRESHOLDS = tuple(value / 100 for value in range(50, 101))
GATES = ("none", "not_high")
MAX_FPR = 0.01
MIN_RECALL = 0.70


@lru_cache(maxsize=32)
def _log_coefficients(n):
    return tuple(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(n + 1))


def _binomial_cdf(k, n, probability):
    if probability == 0 or k == n:
        return 1.0
    if probability == 1:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    logs = [
        coefficient + i * log_p + (n - i) * log_q
        for i, coefficient in enumerate(_log_coefficients(n)[: k + 1])
    ]
    maximum = max(logs)
    return min(1.0, math.exp(maximum) * math.fsum(math.exp(value - maximum) for value in logs))


def _invert_cdf(k, n, target):
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        if _binomial_cdf(k, n, middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _check_counts(k, n):
    if type(k) is not int or type(n) is not int or not 0 <= k <= n:
        raise ValueError("Counts must be integers with 0 <= k <= n")


@lru_cache(maxsize=4096, typed=True)
def exact_upper_95(k, n):
    """One-sided 95% Clopper-Pearson upper bound; None for no cases."""
    _check_counts(k, n)
    if n == 0:
        return None
    return 1.0 if k == n else _invert_cdf(k, n, 0.05)


@lru_cache(maxsize=4096, typed=True)
def exact_interval_95(k, n):
    """Two-sided 95% Clopper-Pearson interval; None for no cases."""
    _check_counts(k, n)
    if n == 0:
        return None
    low = 0.0 if k == 0 else _invert_cdf(k - 1, n, 0.975)
    high = 1.0 if k == n else _invert_cdf(k, n, 0.025)
    return (low, high)


def _valid_probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _check_rule(rule):
    if rule is None:
        return
    if not _valid_probability(rule.get("threshold")) or rule.get("gate") not in GATES:
        raise ValueError("Rule needs a threshold in [0, 1] and gate none or not_high")


def evaluate(rows, rule):
    """Evaluate a rule using all cases; invalid responses remain unresolved.

    gate_exclusions counts otherwise eligible flags blocked by the uncertainty
    gate. A None rule flags nothing and is never eligible for validation.
    """
    _check_rule(rule)
    counts = dict.fromkeys(
        (
            "total",
            "real",
            "ai",
            "valid",
            "unresolved",
            "tp",
            "fp",
            "tn",
            "fn",
            "flagged",
            "review",
            "gate_exclusions",
        ),
        0,
    )
    for row in rows:
        expected = row.get("expected_ai")
        if type(expected) is not bool:
            raise ValueError("Every case needs a Boolean expected_ai provenance label")
        counts["total"] += 1
        counts["ai" if expected else "real"] += 1
        probability = row.get("ai_probability")
        valid = row.get("valid") is True and _valid_probability(probability)
        counts["valid" if valid else "unresolved"] += 1
        above = valid and rule is not None and probability >= rule["threshold"]
        excluded = above and rule["gate"] == "not_high" and row.get("uncertainty") not in ("low", "medium")
        flagged = above and not excluded
        counts["gate_exclusions"] += int(excluded)
        counts["flagged" if flagged else "review"] += 1
        counts[("tp" if flagged else "fn") if expected else ("fp" if flagged else "tn")] += 1
    recall_interval = exact_interval_95(counts["tp"], counts["ai"])
    return {
        "rule": dict(rule) if rule is not None else None,
        "counts": counts,
        "fpr": counts["fp"] / counts["real"] if counts["real"] else None,
        "fpr_upper_95": exact_upper_95(counts["fp"], counts["real"]),
        "recall": counts["tp"] / counts["ai"] if counts["ai"] else None,
        "recall_ci_95": list(recall_interval) if recall_interval is not None else None,
        "review_share": counts["review"] / counts["total"] if counts["total"] else None,
        "unresolved": counts["unresolved"],
        "gate_exclusions": counts["gate_exclusions"],
    }


def candidate_results(rows):
    """Fixed candidate grid. Calling this on validation is descriptive only."""
    rows = list(rows)
    return [
        evaluate(rows, {"threshold": threshold, "gate": gate}) for gate in GATES for threshold in THRESHOLDS
    ]


def select_rule(rows):
    """Select on calibration only; do not substitute a validation winner."""
    feasible = [
        result
        for result in candidate_results(rows)
        if result["fpr"] is not None and result["recall"] is not None and result["fpr"] <= MAX_FPR
    ]
    if not feasible:
        return None
    selected = max(
        feasible,
        key=lambda result: (
            result["recall"],
            -result["fpr"],
            result["rule"]["gate"] == "none",
            result["rule"]["threshold"],
        ),
    )
    return selected["rule"]


def validation_gate(rows, rule):
    """Check the frozen rule; the recall requirement uses its observed value."""
    result = evaluate(rows, rule)
    checks = {
        "rule_selected": rule is not None,
        "both_classes_present": bool(result["counts"]["real"] and result["counts"]["ai"]),
        "all_valid": result["counts"]["total"] > 0 and result["unresolved"] == 0,
        "fpr_upper_within_budget": result["fpr_upper_95"] is not None and result["fpr_upper_95"] <= MAX_FPR,
        "recall_meets_floor": result["recall"] is not None and result["recall"] >= MIN_RECALL,
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": result}


def comparators(rows):
    rows = list(rows)
    return {
        str(threshold): evaluate(rows, {"threshold": threshold, "gate": "none"}) for threshold in (0.5, 0.8)
    }


def grouped_metrics(rows, rule, group_key="generator"):
    """Descriptive subgroup metrics, without a separate pass/fail claim."""
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "unknown")].append(row)
    return {name: evaluate(group, rule) for name, group in sorted(groups.items())}
