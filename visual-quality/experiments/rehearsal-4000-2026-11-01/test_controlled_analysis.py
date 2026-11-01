"""Synthetic tests only: no rehearsal model outputs are opened."""

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import controlled_analysis as analysis
import controlled_review as review


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    asset = root / "asset.bin"
    asset.write_bytes(b"synthetic source/view/crop/contact-sheet identity")
    digest = review.sha(asset)
    cases, packet_families = [], []
    for family_index in range(100):
        members = []
        for severity in range(4):
            case = {"case_id": f"F{family_index:03d}-{severity}", "family_id": f"family-{family_index:03d}",
                    "image_id": f"image-{family_index}-{severity}", "image_path": str(asset), "source_path": str(asset),
                    "view_sha256": digest, "source_sha256": digest,
                    "benchmark_partition": "rehearsal", "stratum": "controlled", "operator": "blur" if family_index < 50 else "noise",
                    "role": "source_parent" if severity == 0 else "variant", "severity": severity or None}
            cases.append(case)
            members.append({**review.identity(case), "crop_path": str(asset), "crop_sha256": digest})
        packet_families.append({"family_id": f"family-{family_index:03d}", "review_family_id": f"F{family_index:03d}",
                                "contact_sheet_path": str(asset), "contact_sheet_sha256": digest, "members": members})
    cases.extend({"case_id": f"other-{i}"} for i in range(3600))
    manifest = root / "manifest.json"
    write(manifest, {"cases": cases})
    directory = root / "controlled-review"
    index = {"manifest_sha256": review.sha(manifest), "families": 100, "derivatives": 300,
             "comparison_entries": 600, "unique_image_pairs": 500, "packets": []}
    for start in range(0, 100, 10):
        packet_id = f"controlled-{start // 10 + 1:02d}"
        packet = directory / f"{packet_id}.json"
        families = packet_families[start:start + 10]
        write(packet, {"packet_id": packet_id, "manifest_sha256": review.sha(manifest), "families": families})
        completed = {"packet_sha256": review.sha(packet), "reviewer": "synthetic only",
                     "families": [review.review_template(f) for f in families]}
        for family in completed["families"]:
            family["reviewed_full_frames_and_native_crops"] = True
            for comparison in family["comparisons"]:
                for relation in analysis.RELATIONS:
                    comparison[relation].update(classification="clear_degradation", evidence="Synthetic visible degradation evidence.")
        if start == 0:
            first = completed["families"][0]["comparisons"][0]
            for relation in analysis.RELATIONS:
                first[relation].update(classification="not_clearly_worse", evidence="Synthetic imperceptible first change.")
            completed["families"][1]["comparisons"][1]["versus_previous"].update(classification="unjudgeable", evidence="Synthetic ambiguous adjacent change.")
            completed["families"][2]["comparisons"][2]["versus_parent"].update(classification="not_clearly_worse", evidence="Synthetic severe change not established.")
        write(directory / f"{packet_id}-reviewed.json", completed)
        index["packets"].append({"packet_id": packet_id, **analysis.binding(packet)})
    write(directory / "index.json", index)
    return manifest, directory, cases


def rows_for(cases, scores_by_family):
    rows = []
    for case in cases:
        if case.get("stratum") != "controlled":
            continue
        severity = case["severity"] or 0
        for model in analysis.MODELS:
            scores = scores_by_family.get((case["family_id"], model), (4, 3, 2, 1))
            value = scores[severity]
            rows.append({**case, "model": model, "arm": "baseline", "technical_score": value,
                         "technical_score_valid": value is not None, "scope_floor_applied": value == 1,
                         "technical_score_display": f"{value:.2f}" if value is not None else None})
    return rows


class ControlledAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest, self.directory, self.cases = fixture(self.root)
        with redirect_stdout(StringIO()):
            self.audit = analysis.build_audit(self.manifest, self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_coverage_and_distinct_review_subsets(self):
        self.assertEqual(self.audit["coverage"], {"manifest_cases": 4000, "packets": 10, "families": 100,
                                              "members": 400, "comparison_entries": 600, "unique_image_pairs": 500})
        self.assertEqual(self.audit["classifications"]["entries"], {"clear_degradation": 596, "not_clearly_worse": 3, "unjudgeable": 1})
        self.assertEqual(self.audit["classifications"]["unique_pairs"], {"clear_degradation": 497, "not_clearly_worse": 2, "unjudgeable": 1})
        subsets = self.audit["subsets"]
        self.assertEqual({key: len(value) for key, value in subsets.items()}, {
            "all_three_vs_parent_verified_family_ids": 98, "fully_adjacent_verified_family_ids": 98,
            "severe_vs_parent_verified_family_ids": 99})
        self.assertIn("family-001", subsets["all_three_vs_parent_verified_family_ids"])
        self.assertNotIn("family-001", subsets["fully_adjacent_verified_family_ids"])
        self.assertIn("family-002", subsets["fully_adjacent_verified_family_ids"])
        self.assertNotIn("family-002", subsets["severe_vs_parent_verified_family_ids"])
        self.assertEqual(self.audit["families"][1]["unverified_comparisons"][0]["evidence"], "Synthetic ambiguous adjacent change.")

    def test_missing_intermediate_scores_do_not_remove_severe_pairs(self):
        rows = rows_for(self.cases, {("family-003", "luna"): (4, None, 2, 1)})
        rows = [r for r in rows if (r["case_id"], r["model"]) != ("F004-3", "luna")]
        result = analysis.analyze_controlled(self.cases, rows, self.audit)
        severe = next(r for r in result["severe_parent_comparisons"]["visually_verified"] if r["model"] == "luna" and r["operator"] == "all")
        self.assertEqual((severe["expected_pairs"], severe["complete_pairs"], severe["missing_pairs"], severe["pairs_with_absent_rows"]), (99, 98, 1, 1))
        ladders = next(r for r in result["all_family_ladders"]["summary"] if r["model"] == "luna" and r["operator"] == "all")
        self.assertEqual((ladders["families"], ladders["complete_families"], ladders["missing_families"]), (100, 98, 2))
        pair = next(p for p in result["pair_details"] if p["case_id"] == "F003-3" and p["comparison_case_id"] == "F003-0" and p["model"] == "luna")
        self.assertEqual((pair["outcome"], pair["technical_drop"]), ("strict_decrease", 3))
        self.assertTrue(result["all_families_preserved"])
        self.assertEqual(len({f["family_id"] for f in result["all_family_ladders"]["families"]}), 100)

    def test_floor_ties_and_native_precision_inversions_are_separate(self):
        rows = rows_for(self.cases, {("family-003", "luna"): (1, 1, 1, 1),
                                    ("family-004", "luna"): (3, 3 + 5e-9, 3, 3.00002)})
        result = analysis.analyze_controlled(self.cases, rows, self.audit)
        selected = {p["case_id"]: p for p in result["pair_details"] if p["model"] == "luna" and p["comparison_case_id"] in {"F003-0", "F004-0"}}
        self.assertEqual(selected["F003-3"]["outcome"], "tie")
        self.assertTrue(selected["F003-3"]["both_technical_scores_exactly_1"])
        self.assertEqual(selected["F004-1"]["outcome"], "tie")
        self.assertFalse(selected["F004-1"]["both_technical_scores_exactly_1"])
        self.assertEqual(selected["F004-3"]["outcome"], "strict_increase")
        severe = next(r for r in result["severe_parent_comparisons"]["all"] if r["model"] == "luna" and r["operator"] == "all")
        self.assertEqual((severe["strict_decrease"], severe["ties"], severe["strict_increase"], severe["floor_1_pairs"]), (98, 1, 1, 1))
        self.assertIn("policy-correct", analysis.render_report(result))

    def test_scores_reject_duplicates_bad_identity_and_invalid_numbers(self):
        cases = [c for c in self.cases if c.get("stratum") == "controlled"]
        rows = rows_for(self.cases, {})
        variants = [rows + [rows[0]], [{**rows[0], "view_sha256": "changed"}],
                    [{**rows[0], "technical_score": float("nan")}],
                    [{**rows[0], "technical_score": 5.1}],
                    [{**rows[0], "technical_score_valid": False}],
                    [{**rows[0], "technical_score": True}]]
        for changed in variants:
            with self.subTest(changed=changed[0]), self.assertRaises(ValueError):
                analysis.score_rows(cases, changed)

    def test_empty_verified_operators_keep_explicit_zero_denominators(self):
        cases = [c for c in self.cases if c.get("stratum") == "controlled"]
        _, rows = analysis.score_rows(cases, rows_for(self.cases, {}))
        subset = analysis.ladder_subset(cases, rows, [], verified=True)
        self.assertEqual(len(subset["summary"]), 18)
        self.assertTrue(all(r["families"] == r["complete_families"] == r["missing_families"] == 0 for r in subset["summary"]))
        self.assertTrue(all(r["mean_parent_minus_severity3"] is None for r in subset["summary"]))

    def test_manifest_packet_member_and_cross_packet_coverage_binding(self):
        second = self.directory / "controlled-02.json"
        second_review = self.directory / "controlled-02-reviewed.json"
        index_path = self.directory / "index.json"
        packet = json.loads(second.read_text())
        complete = json.loads(second_review.read_text())
        packet["families"][0] = json.loads((self.directory / "controlled-01.json").read_text())["families"][0]
        complete["families"][0] = json.loads((self.directory / "controlled-01-reviewed.json").read_text())["families"][0]
        write(second, packet)
        complete["packet_sha256"] = review.sha(second)
        write(second_review, complete)
        index = json.loads(index_path.read_text())
        index["packets"][1]["sha256"] = review.sha(second)
        write(index_path, index)
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "exactly once"):
            analysis.build_audit(self.manifest, self.directory)
        packet["families"][0]["members"][0]["image_id"] = "wrong-manifest-member"
        write(second, packet)
        complete["packet_sha256"] = review.sha(second)
        write(second_review, complete)
        index["packets"][1]["sha256"] = review.sha(second)
        write(index_path, index)
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "member identity"):
            analysis.build_audit(self.manifest, self.directory)

    def test_hash_binding_and_reconstructed_audit_reject_tampering(self):
        audit_path = self.directory / "audit.json"
        write(audit_path, self.audit)
        with redirect_stdout(StringIO()):
            self.assertEqual(analysis.load_audit(audit_path, self.manifest), self.audit)
        altered = copy.deepcopy(self.audit)
        altered["comparisons"][0]["evidence"] = "Changed only the derived audit."
        write(audit_path, altered)
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "reconstructed"):
            analysis.load_audit(audit_path, self.manifest)
        for key in ("script", "validator", "metrics", "manifest"):
            changed = copy.deepcopy(self.audit)
            changed["inputs"][key]["sha256"] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                analysis.check_bindings(changed, self.manifest)
        write(audit_path, self.audit)
        reviewed_path = self.directory / "controlled-01-reviewed.json"
        reviewed = json.loads(reviewed_path.read_text())
        reviewed["families"][0]["comparisons"][0]["versus_parent"]["evidence"] = "Changed review evidence."
        write(reviewed_path, reviewed)
        with self.assertRaisesRegex(ValueError, "Audited input changed"):
            analysis.load_audit(audit_path, self.manifest)

    def test_full_manifest_and_source_asset_bytes_are_required(self):
        short = self.root / "short-manifest.json"
        write(short, {"cases": self.cases[:-1]})
        with self.assertRaisesRegex(ValueError, "4000-case"):
            analysis.build_audit(short, self.directory)
        (self.root / "asset.bin").write_bytes(b"changed bytes")
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "changed"):
            analysis.build_audit(self.manifest, self.directory)


if __name__ == "__main__":
    unittest.main()
