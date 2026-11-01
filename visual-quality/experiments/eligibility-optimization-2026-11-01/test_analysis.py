"""Verify error directions, coverage and development selection without model calls."""

import unittest

import analyze


class AnalysisTests(unittest.TestCase):
    def test_counts_false_floors_separately_from_missed_exclusions(self):
        rows = [
            {
                "case_id": "a",
                "group": "poster",
                "reference_eligibility": "excluded",
                "status": "valid",
                "scope_floor_applied": False,
            },
            {
                "case_id": "b",
                "group": "context",
                "reference_eligibility": "eligible",
                "status": "valid",
                "scope_floor_applied": True,
            },
            {
                "case_id": "c",
                "group": "render",
                "reference_eligibility": "excluded",
                "status": "valid",
                "scope_floor_applied": True,
            },
            {
                "case_id": "d",
                "group": "scene",
                "reference_eligibility": "eligible",
                "status": "missing",
                "scope_floor_applied": None,
            },
        ]
        x = analyze.evaluate(rows)
        self.assertEqual((x["missed_exclusions"], x["false_floors"], x["correct_exclusions"]), (1, 1, 1))
        self.assertEqual(x["missing_ids"], ["d"])
        self.assertEqual(x["by_group"]["poster"]["errors"], 1)

    def test_selection_rejects_false_floor_increase_above_guard(self):
        def metrics(fn, fp):
            return {
                "valid": 40,
                "eligible_valid": 20,
                "excluded_valid": 20,
                "missed_exclusions": fn,
                "false_floors": fp,
            }

        baseline = metrics(4, 0)
        versions = {}
        for key, fn, fp in [("r01", 2, 0), ("r02", 0, 2), ("r03", 4, 0)]:
            cand = metrics(fn, fp)
            versions[key] = {
                "metrics": cand,
                "paired": {"baseline_paired": baseline, "candidate_paired": cand},
            }
        dev = {m: {"baseline": baseline, "rounds": versions} for m in analyze.MODELS}
        result = analyze.choose(dev, {"r01": "one", "r02": "two", "r03": "three"})
        self.assertEqual(result["jev"]["candidate_round"], "r01")
        self.assertTrue(result["jev"]["advance_from_development"])

    def test_no_improvement_keeps_baseline(self):
        b = {
            "valid": 40,
            "eligible_valid": 20,
            "excluded_valid": 20,
            "missed_exclusions": 0,
            "false_floors": 0,
        }
        versions = {"r01": {"metrics": b, "paired": {"baseline_paired": b, "candidate_paired": b}}}
        dev = {m: {"baseline": b, "rounds": versions} for m in analyze.MODELS}
        self.assertEqual(
            analyze.choose(dev, {"r01": "x"})["liquid"]["working_default_after_development"], "baseline"
        )


if __name__ == "__main__":
    unittest.main()
