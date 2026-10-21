"""Check reporting failure cases and separation of selection from validation."""

import importlib.util
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("calibration_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
metrics = report.load_module("metrics")
MODEL = "typesafe/jev-test"


def case(row_id, expected, split="calibration"):
    return {"row_id": row_id, "expected_ai": expected, "split": split, "generator": "test"}


def response(row_id, probability, **updates):
    return {
        "row_id": row_id,
        "status": 200,
        "resolved_model": MODEL,
        "normalized": {"ai_probability": probability, "uncertainty": "low"},
        "effective_cost_usd": 0.001,
        "seconds": 0.1,
        **updates,
    }


class ReportingTests(unittest.TestCase):
    def test_missing_failed_and_unpriced_cases_remain_unresolved(self):
        cases = [case(str(index), bool(index % 2)) for index in range(5)]
        records = [
            response("0", 0.1),
            response("1", 0.9, status=500),
            response("2", 0.1, effective_cost_usd=None),
            response("3", 0.9, resolved_model="wrong"),
        ]
        rows = report.metric_rows(cases, records, MODEL)
        result = metrics.evaluate(rows, {"threshold": 0.5, "gate": "none"})
        self.assertEqual(result["counts"]["total"], 5)
        self.assertEqual(result["counts"]["valid"], 1)
        self.assertEqual(result["unresolved"], 4)
        self.assertEqual(result["counts"]["review"], 5)
        self.assertFalse(rows[-1]["attempted"])
        operations = report.operations(records, len(cases))
        self.assertEqual(operations["cost_unknown_calls"], 1)
        self.assertEqual(operations["unattempted_calls"], 1)
        self.assertEqual(operations["tokens"]["input_tokens"]["unknown_calls"], 4)

    def test_source_allocation_uses_corrected_protocol(self):
        protocol = {
            "splits": {
                "calibration": {"official_split": "validation", "real": 300, "ai": 300},
                "validation": {"official_split": "validation", "real": 500, "ai": 500},
            }
        }
        text = report.source_allocation(protocol)
        self.assertEqual(text.count("official `validation`"), 2)
        self.assertIn("300 real and 300 generated", text)
        self.assertIn("500 real and 500 generated", text)
        self.assertNotIn("official `test`", text)
        self.assertIn("not a cross-source validation", text)

    def test_duplicate_or_unplanned_responses_fail(self):
        planned = [case("a", True)]
        for responses in ([response("a", 0.9), response("a", 0.8)], [response("b", 0.9)]):
            with self.assertRaises(ValueError):
                report.metric_rows(planned, responses, MODEL)

    def test_zero_false_flags_does_not_hide_precision_sensitivity(self):
        result = metrics.evaluate(
            [{"expected_ai": False, "ai_probability": 0.1, "valid": True}] * 500
            + [{"expected_ai": True, "ai_probability": 0.9, "valid": True}] * 500,
            {"threshold": 0.8, "gate": "none"},
        )
        scenario = report.precision_scenarios(result)[0]
        self.assertEqual(scenario["precision_using_observed_rates"], 1)
        self.assertLess(scenario["precision_using_fpr_upper_95_and_observed_recall"], 0.65)
        self.assertGreater(scenario["precision_using_fpr_upper_95_and_observed_recall"], 0.60)

    def fixture(self, rule):
        cases = [
            case("c-real", False),
            case("c-ai", True),
            case("v-real", False, "validation"),
            case("v-ai", True, "validation"),
        ]
        records = {
            "calibration": [response("c-real", 0.1), response("c-ai", 0.9)],
            "validation": [response("v-real", 0.99), response("v-ai", 0.2)],
        }
        rows = report.metric_rows(cases[:2], records["calibration"], MODEL)
        selection = {"rule": rule, "calibration_metrics": metrics.evaluate(rows, rule)}
        return {"expected_revision": MODEL}, cases, records, selection, {}, {}

    def test_validation_does_not_select_or_refit_rule(self):
        frozen = {"threshold": 0.9, "gate": "none"}
        with (
            patch.object(report, "load_verified_evidence", return_value=self.fixture(frozen)),
            patch.object(report, "load_module", return_value=metrics),
            patch.object(metrics, "select_rule", wraps=metrics.select_rule) as selector,
            patch.object(metrics, "candidate_results", wraps=metrics.candidate_results) as candidates,
        ):
            summary, cases, _ = report.build_report()
        self.assertEqual(summary["selected_rule"], frozen)
        self.assertFalse(summary["validation_gate"]["passed"])
        self.assertEqual(selector.call_count, 1)
        self.assertTrue(all(row["split"] == "calibration" for row in selector.call_args.args[0]))
        for call in candidates.call_args_list:
            self.assertTrue(all(row["split"] == "calibration" for row in call.args[0]))
        self.assertEqual(next(row for row in cases if row["row_id"] == "v-ai")["routing"], "review")

    def test_calibration_figure_is_valid_svg_and_marks_frozen_rule(self):
        curve = [
            {"threshold": 0.8, "gate": "none", "fpr": 0.01, "recall": 0.7},
            {"threshold": 0.8, "gate": "not_high", "fpr": 0, "recall": 0.5},
        ]
        svg = report.calibration_svg(curve, {"threshold": 0.8, "gate": "none"})
        root = ET.fromstring(svg)
        self.assertEqual(len(root.findall(".//{http://www.w3.org/2000/svg}circle")), 1)
        self.assertIn("Calibration data only", svg)
        self.assertIn("Real-photo false-positive rate (%)", svg)
        with self.assertRaisesRegex(ValueError, "missing from"):
            report.calibration_svg(curve, {"threshold": 0.9, "gate": "none"})

    def test_frozen_selection_mismatch_is_rejected(self):
        with (
            patch.object(
                report,
                "load_verified_evidence",
                return_value=self.fixture({"threshold": 0.8, "gate": "none"}),
            ),
            patch.object(report, "load_module", return_value=metrics),
            self.assertRaisesRegex(ValueError, "Frozen selection differs"),
        ):
            report.build_report()


if __name__ == "__main__":
    unittest.main()
