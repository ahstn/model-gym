"""Regression checks for partial validity, fair comparison cohorts and family statistics."""

import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location("analysis", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analysis)


def rating(technical, aesthetic, invalid=()):
    result = {"validation_errors": {task: "invalid evidence" for task in invalid}}
    for task, value in zip(analysis.TASKS, (technical, aesthetic), strict=True):
        result[task] = {
            "status": "ok" if value is not None else "unreadable",
            "score": value,
            "limitations": [],
        }
    return result


class AnalysisTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_constant_offsets_and_excludes_invalid_task(self):
        refs = {"A": rating(3, 3), "B": rating(1, 1), "C": rating(1, 1)}
        observed = {
            "deepseek": {"A": rating(5, 5), "B": rating(4, 4), "C": rating(5, 5, ("aesthetics",))},
            "luna": {"A": rating(3, 3), "B": rating(2, 2), "C": rating(1, 1)},
        }
        predicted = analysis.predictions(observed, list(refs))
        families = {case_id: "family-" + case_id for case_id in refs}
        result = analysis.paired_bootstrap(predicted, refs, list(refs), families, seed=123, draws=250)
        self.assertEqual(result["common_valid_case_ids"], ["A", "B"])
        self.assertEqual(result["excluded_case_ids"], ["C"])
        self.assertEqual(result["mean_absolute_error"], {"deepseek": 2.5, "luna": 0.5, "average": 1.5})
        self.assertEqual(
            result["average_minus_individual"]["deepseek"],
            {"mean_error_delta": -1.0, "percentile_95_interval": [-1.0, -1.0]},
        )
        self.assertEqual(
            result["average_minus_individual"]["luna"],
            {"mean_error_delta": 1.0, "percentile_95_interval": [1.0, 1.0]},
        )
        self.assertEqual(result["resamples_drawn"], 250)
        self.assertEqual(
            result, analysis.paired_bootstrap(predicted, refs, list(refs), families, seed=123, draws=250)
        )
        families["B"] = families["A"]
        with self.assertRaisesRegex(ValueError, "distinct families"):
            analysis.paired_bootstrap(predicted, refs, list(refs), families, seed=123, draws=250)

    def test_scope_boundary_sensitivity_only_removes_declared_case(self):
        ordinary = [f"S{index:04d}" for index in range(70)]
        case_ids = [*ordinary, "T0632", "T2000"]
        cases = [{"case_id": case_id, "known_anchor": case_id == "T2000"} for case_id in case_ids]
        refs = {case_id: {**rating(3, 3), "eligibility": "eligible"} for case_id in ordinary}
        refs.update({case_id: {**rating(1, 1), "eligibility": "excluded"} for case_id in ("T0632", "T2000")})
        frozen_copy = deepcopy(refs)
        cohorts = analysis.reference_cohorts(cases, refs)
        self.assertEqual(len(cohorts["all_72"]), 72)
        self.assertEqual(len(cohorts["fresh_71"]), 71)
        self.assertIn("T0632", cohorts["fresh_71"])
        self.assertEqual(cohorts["fresh_70_excluding_scope_boundary"], ordinary)
        self.assertEqual(
            set(cohorts["fresh_71"]) - set(cohorts["fresh_70_excluding_scope_boundary"]), {"T0632"}
        )
        observed = {model: {case_id: rating(3, 3) for case_id in ordinary} for model in analysis.MODELS}
        observed["deepseek"]["T0632"] = rating(5, 5)
        observed["luna"]["T0632"] = rating(1, 1)
        predicted = analysis.predictions(observed, case_ids)
        for task in analysis.TASKS:
            primary = analysis.reference_comparison(predicted, refs, cohorts["fresh_71"], task)
            secondary = analysis.reference_comparison(
                predicted, refs, cohorts["fresh_70_excluding_scope_boundary"], task
            )
            self.assertAlmostEqual(primary["common_valid"]["deepseek"]["mae"], 4 / 71)
            self.assertEqual(secondary["common_valid"]["deepseek"]["mae"], 0.0)
            self.assertEqual(secondary["common_valid"]["average"]["n"], 70)
        self.assertEqual(refs, frozen_copy)

    def test_billing_counts_rejections_and_preserves_unknown_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = []
            for index, (status, selected, usage) in enumerate(
                (
                    (429, False, {}),
                    (
                        200,
                        True,
                        {
                            "cost": 0.004,
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                            "completion_tokens_details": {"reasoning_tokens": 15},
                        },
                    ),
                )
            ):
                path = Path(directory) / f"{index}.json"
                path.write_text(
                    json.dumps({"http_status": status, "headers": {}, "body": json.dumps({"usage": usage})})
                )
                ledger.append(
                    {
                        "case_id": "A",
                        "attempt_id": str(index),
                        "stage": "original" if not selected else "continuation",
                        "result_path": str(path),
                        "raw_result_sha256": analysis.digest(path),
                        "http_status": status,
                        "selected": selected,
                        "status": "valid" if selected else "rejected_retryable",
                        "cost_usd": "0.004" if selected else None,
                        "cost_source": "usage.cost" if selected else "unknown",
                    }
                )
            report = {
                "all_attempts": ledger,
                "attempts": [ledger[1]],
                "http_attempts": 2,
                "selected_response_calls": 1,
                "known_cost_usd": "0.004",
                "unpriced_attempts": 1,
                "unpriced_rejected_attempts": 1,
                "unpriced_liability_reserve_usd": "0.1",
                "admission_debit_usd": "0.104",
            }
            audited = analysis.audit_all_attempts(report, {"reasoning_effort": "high"}, "0.1")
            self.assertEqual(audited["calls"], 2)
            self.assertEqual(audited["selected_response_calls"], 1)
            self.assertEqual(audited["known_cost_usd"], "0.004")
            self.assertEqual(audited["unpriced_liability_reserve_usd"], "0.1")
            self.assertFalse(audited["cost_complete"])
            self.assertEqual(audited["usage"]["reasoning_tokens"], 15)
            per_image = analysis.billing_by_case(audited["attempts"])["A"]
            self.assertIsNone(per_image["cost_usd"])
            self.assertEqual(per_image["known_cost_usd"], "0.004")
            self.assertEqual(per_image["http_attempts"], 2)
            self.assertEqual(per_image["unpriced_http_attempts"], 1)
            self.assertEqual(per_image["selected_response_cost_usd"], "0.004")
            ledger[0]["cost_usd"] = "0"
            with self.assertRaisesRegex(ValueError, "Raw billing differs"):
                analysis.audit_all_attempts(report, {}, "0.1")
            ledger[0]["cost_usd"] = None
            report["admission_debit_usd"] = "0.004"
            with self.assertRaisesRegex(ValueError, "Held reservations"):
                analysis.audit_all_attempts(report, {}, "0.1")

    def test_reference_manifest_selection_and_input_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = ("manifest.json", "collage-selection.json", "references.jsonl", "review.md")
            for name in files:
                (root / name).write_text("original")
            frozen = {
                "manifest_sha256": analysis.digest(root / "manifest.json"),
                "selection_sha256": analysis.digest(root / "collage-selection.json"),
                "references_sha256": analysis.digest(root / "references.jsonl"),
                "inputs": {str(root / "review.md"): analysis.digest(root / "review.md")},
            }
            with mock.patch.object(analysis, "HERE", root):
                analysis.validate_reference_inputs(frozen)
                for name in files:
                    with self.subTest(name=name):
                        (root / name).write_text("changed")
                        with self.assertRaisesRegex(ValueError, "Reference freeze input changed"):
                            analysis.validate_reference_inputs(frozen)
                        (root / name).write_text("original")

    def test_invalid_aesthetic_does_not_discard_valid_technical(self):
        observed = {"deepseek": {"A": rating(4, 3, ("aesthetics",))}, "luna": {"A": rating(2, 5)}}
        predicted = analysis.predictions(observed, ["A"])
        self.assertEqual(predicted["deepseek"]["technical_quality"]["A"], Decimal(4))
        self.assertEqual(predicted["average"]["technical_quality"]["A"], Decimal(3))
        self.assertIsNone(predicted["average"]["aesthetics"]["A"])
        self.assertEqual(predicted["luna"]["aesthetics"]["A"], Decimal(5))

    def test_reference_comparison_uses_identical_common_ids(self):
        observed = {
            "deepseek": {"A": rating(4, 4), "B": rating(5, 5)},
            "luna": {"A": rating(2, 2), "C": rating(1, 1)},
        }
        predicted = analysis.predictions(observed, ["A", "B", "C"])
        refs = {"A": rating(3, 3), "B": rating(1, 1), "C": rating(1, 1)}
        result = analysis.reference_comparison(predicted, refs, ["A", "B", "C"], "technical_quality")
        self.assertEqual(result["common_valid_case_ids"], ["A"])
        self.assertEqual(result["individual_coverage"]["deepseek"]["mae"], 2.5)
        self.assertEqual(result["individual_coverage"]["luna"]["mae"], 0.5)
        self.assertEqual(result["common_valid"]["deepseek"]["mae"], 1.0)
        self.assertEqual(result["common_valid"]["luna"]["mae"], 1.0)
        self.assertEqual(result["common_valid"]["average"]["mae"], 0.0)
        self.assertEqual({tuple(value["case_ids"]) for value in result["common_valid"].values()}, {("A",)})

    def test_scope_floor_requires_flags_and_keeps_partial_errors(self):
        refs = {"A": {**rating(2, 2), "eligibility": "eligible"}}
        row = rating(1, 1)
        result = analysis.scope_audit({"A": row}, refs)
        self.assertEqual(result["issue_counts"], {})
        self.assertEqual(result["complete_floor_case_ids"], [])
        row["technical_quality"]["limitations"] = ["unsupported_domain"]
        row["validation_errors"]["aesthetics"] = "too long"
        result = analysis.scope_audit({"A": row}, refs)
        self.assertEqual(result["issue_counts"], {"invalid_or_unscored_rating": 1, "false_exclusion": 1})

    def test_family_comparison_excludes_incomplete_families_symmetrically(self):
        cases = [
            {
                "case_id": f"{family}{severity}",
                "family_id": family,
                "stratum": "controlled",
                "operator": "gaussian_blur",
                "severity": severity or None,
            }
            for family in ("A", "B")
            for severity in range(4)
        ]
        observed = {
            model: {
                case["case_id"]: rating(5 - (case["severity"] or 0), 2 + (case["severity"] or 0))
                for case in cases
            }
            for model in analysis.MODELS
        }
        observed["deepseek"]["B2"]["validation_errors"]["technical_quality"] = "invalid"
        predicted = analysis.predictions(observed, [case["case_id"] for case in cases])
        result = analysis.controlled_analysis(cases, predicted, observed)
        technical = result["tasks"]["technical_quality"]["groups"]["all_operators"]
        self.assertEqual(technical["individual_coverage"]["deepseek"]["complete_families"], 1)
        self.assertEqual(technical["individual_coverage"]["luna"]["complete_families"], 2)
        for values in technical["common_valid"].values():
            self.assertEqual(values["complete_family_ids"], ["A"])
            self.assertEqual(values["mean_parent_to_severe_delta"], -3.0)
            self.assertEqual(values["non_increasing_families"], 1)
        aesthetic = result["tasks"]["aesthetics"]["groups"]["all_operators"]["common_valid"]["average"]
        self.assertEqual(aesthetic["complete_families"], 2)
        self.assertEqual(aesthetic["mean_parent_to_severe_delta"], 3.0)
        self.assertEqual(aesthetic["non_increasing_families"], 0)

    def test_missing_scores_are_not_counted_as_zero(self):
        result = analysis.distribution([Decimal(4), None], 2)
        self.assertEqual(result["mean"], 4.0)
        self.assertEqual(result["numeric_n"], 1)
        self.assertEqual(result["unscored_n"], 1)
        self.assertIsNone(analysis.distribution([None], 1)["mean"])


if __name__ == "__main__":
    unittest.main()
