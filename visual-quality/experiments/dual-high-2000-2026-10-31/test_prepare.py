"""Protect corpus overlap exclusions and independent-family review selection."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("cohort", Path(__file__).with_name("prepare.py"))
cohort = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cohort)


class PreparationTests(unittest.TestCase):
    def test_index_matches_prior_overlap_rule(self):
        previous = [
            {
                "split_group": "family-1",
                "sha256": "bytes-1",
                "decoded_sha256": "pixels-1",
                "dhash": "0000000000000000",
            }
        ]
        candidates = [
            {"split_group": "family-1"},
            {"sha256": "bytes-1"},
            {"decoded_sha256": "pixels-1"},
            {"dhash": "0000000000000003"},
            {"dhash": "0000000000000007"},
            {"dhash": "ffffffffffffffff"},
            {"split_group": "new-family"},
        ]
        index = cohort.ExposureIndex(previous)
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assertEqual(index.overlaps([candidate]), cohort.prior.overlaps([candidate], previous))

    def test_selected_families_become_excluded(self):
        index = cohort.ExposureIndex([])
        row = {"split_group": "new-family", "dhash": "00000000000000ff"}
        self.assertFalse(index.overlaps([row]))
        index.extend([row])
        self.assertTrue(index.overlaps([{"split_group": "new-family"}]))
        self.assertTrue(index.overlaps([{"dhash": "00000000000000fe"}]))

    def test_audit_strata_and_families_are_fixed_without_scores(self):
        selections = []

        def add(role, operator=None, severity=None, defect=None):
            number = len(selections) + 1
            selections.append(
                {
                    "case_id": f"T{number:04d}",
                    "role": role,
                    "operator": operator,
                    "severity": severity,
                    "vizwiz_stratum": defect,
                    "row": {"id": f"image-{number}", "split_group": f"family-{number}"},
                }
            )

        for _ in range(41):
            add("standalone_megalith")
        for defect in cohort.VIZWIZ_QUOTAS:
            for _ in range(3):
                add("vizwiz", defect=defect)
        for operator in cohort.prior.OPERATORS:
            for severity in (None, 1, 2, 3):
                add("source_parent" if severity is None else "variant", operator, severity)
        chosen = cohort.choose_audit(selections)
        self.assertEqual(len(chosen), 72)
        self.assertEqual(chosen, cohort.choose_audit(list(reversed(selections))))
        self.assertEqual(set(chosen), {item["case_id"] for item in selections} | {"T2000"})
        selections[-1]["row"]["split_group"] = selections[-2]["row"]["split_group"]
        with self.assertRaisesRegex(ValueError, "independent-family"):
            cohort.choose_audit(selections)


if __name__ == "__main__":
    unittest.main()
