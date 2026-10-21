"""Tests for exact bounds, frozen selection, and unresolved-case handling."""

import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("calibration_metrics", Path(__file__).with_name("metrics.py"))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def case(expected, probability, uncertainty="low", valid=True, **extra):
    return {
        "expected_ai": expected,
        "ai_probability": probability,
        "uncertainty": uncertainty,
        "valid": valid,
        **extra,
    }


class ExactBoundsTests(unittest.TestCase):
    def test_zero_false_flags_has_nonzero_upper_bound(self):
        self.assertAlmostEqual(metrics.exact_upper_95(0, 500), 1 - 0.05 ** (1 / 500), places=12)
        one_false_flag = metrics.exact_upper_95(1, 500)
        self.assertAlmostEqual(one_false_flag, 0.009452282208, places=10)
        self.assertAlmostEqual(
            (1 - one_false_flag) ** 500 + 500 * one_false_flag * (1 - one_false_flag) ** 499, 0.05, places=12
        )
        self.assertGreater(metrics.exact_upper_95(2, 500), 0.01)

    def test_recall_interval_and_endpoints(self):
        low, high = metrics.exact_interval_95(5, 10)
        self.assertAlmostEqual(low, 0.187086028447, places=10)
        self.assertAlmostEqual(high, 0.812913971553, places=10)
        self.assertEqual(metrics.exact_interval_95(0, 10)[0], 0)
        self.assertEqual(metrics.exact_interval_95(10, 10)[1], 1)
        self.assertIsNone(metrics.exact_interval_95(0, 0))
        self.assertIsNone(metrics.exact_upper_95(0, 0))
        self.assertEqual(metrics.exact_upper_95(5, 5), 1)

    def test_invalid_counts(self):
        metrics.exact_upper_95(1, 10)
        metrics.exact_upper_95(0, 10)
        for counts in ((-1, 10), (11, 10), (0, -1), (True, 10), (0.0, 10)):
            with self.assertRaises(ValueError):
                metrics.exact_upper_95(*counts)


class SelectionTests(unittest.TestCase):
    def test_one_percent_limit_is_inclusive(self):
        rows = [case(False, 0.1)] * 99 + [case(False, 1.0), case(True, 0.7)]
        self.assertEqual(metrics.select_rule(rows), {"threshold": 0.7, "gate": "none"})
        rows = [case(False, 0.1)] * 98 + [case(False, 1.0)] * 2 + [case(True, 0.7)]
        self.assertIsNone(metrics.select_rule(rows))

    def test_maximize_recall_before_other_ties(self):
        rows = [case(False, 0.61)] * 100 + [case(True, 0.72), case(True, 0.91)]
        self.assertEqual(metrics.select_rule(rows), {"threshold": 0.72, "gate": "none"})

    def test_lower_false_positive_rate_breaks_recall_tie(self):
        rows = [case(False, 0.1)] * 99 + [case(False, 0.8, "high"), case(True, 0.9)]
        self.assertEqual(metrics.select_rule(rows), {"threshold": 0.9, "gate": "none"})

    def test_prefer_ungated_when_metrics_tie(self):
        rows = [case(False, 0.1)] * 100 + [case(True, 0.9)]
        self.assertEqual(metrics.select_rule(rows), {"threshold": 0.9, "gate": "none"})

    def test_uncertainty_gate_can_preserve_recall_at_lower_fpr(self):
        rows = [case(False, 0.9, "high")] * 100 + [case(True, 0.7)]
        self.assertEqual(metrics.select_rule(rows), {"threshold": 0.7, "gate": "not_high"})

    def test_no_feasible_rule_is_not_replaced_with_fallback(self):
        rows = [case(False, 1.0)] * 100 + [case(True, 1.0)]
        self.assertIsNone(metrics.select_rule(rows))
        self.assertIsNone(metrics.select_rule([]))
        self.assertIsNone(metrics.select_rule([case(True, 0.9)]))


class EvaluationTests(unittest.TestCase):
    def test_report_mutation_does_not_change_cached_interval(self):
        rows = [case(True, 0.9)]
        rule = {"threshold": 0.8, "gate": "none"}
        first = metrics.evaluate(rows, rule)
        first["recall_ci_95"][0] = 1.0
        self.assertLess(metrics.evaluate(rows, rule)["recall_ci_95"][0], 1.0)

    def test_invalid_and_gated_cases_remain_in_denominators(self):
        rows = [
            case(True, 0.9),
            case(True, 0.9, valid=False),
            case(True, 0.9, "high"),
            case(False, 0.9, None),
            case(False, 0.9, "medium"),
        ]
        result = metrics.evaluate(rows, {"threshold": 0.8, "gate": "not_high"})
        self.assertEqual(result["counts"]["tp"], 1)
        self.assertEqual(result["counts"]["fp"], 1)
        self.assertEqual(result["recall"], 1 / 3)
        self.assertEqual(result["fpr"], 1 / 2)
        self.assertEqual(result["review_share"], 3 / 5)
        self.assertEqual(result["unresolved"], 1)
        self.assertEqual(result["gate_exclusions"], 2)

    def test_invalid_numeric_probabilities_cannot_flag(self):
        rows = [case(True, value) for value in (None, math.nan, math.inf, -0.1, 1.1, True, "0.9")]
        result = metrics.evaluate(rows, {"threshold": 0.5, "gate": "none"})
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["unresolved"], len(rows))

    def test_validation_requires_all_valid_and_confidence_bound(self):
        rule = {"threshold": 0.8, "gate": "none"}
        real = [case(False, 0.1)] * 499 + [case(False, 0.9)]
        ai = [case(True, 0.9)] * 350 + [case(True, 0.1)] * 150
        self.assertTrue(metrics.validation_gate(real + ai, rule)["passed"])
        self.assertFalse(metrics.validation_gate(real[:-1] + [case(False, 0.9)] * 2 + ai, rule)["passed"])
        self.assertFalse(
            metrics.validation_gate(real + ai[:-1] + [case(True, None, valid=False)], rule)["passed"]
        )
        self.assertFalse(metrics.validation_gate(real + ai, None)["passed"])

    def test_comparators_and_groups_use_fixed_rules(self):
        rows = [case(True, 0.7, generator="generator_a"), case(False, 0.6, source="camera")]
        self.assertEqual(metrics.comparators(rows)["0.5"]["counts"]["flagged"], 2)
        self.assertEqual(metrics.comparators(rows)["0.8"]["counts"]["flagged"], 0)
        grouped = metrics.grouped_metrics(rows, {"threshold": 0.5, "gate": "none"})
        self.assertEqual(set(grouped), {"generator_a", "unknown"})
        self.assertIsNone(grouped["generator_a"]["fpr"])

    def test_provenance_and_rules_must_be_valid(self):
        with self.assertRaises(ValueError):
            metrics.evaluate([case("real", 0.9)], {"threshold": 0.8, "gate": "none"})
        with self.assertRaises(ValueError):
            metrics.evaluate([], {"threshold": math.nan, "gate": "none"})


if __name__ == "__main__":
    unittest.main()
