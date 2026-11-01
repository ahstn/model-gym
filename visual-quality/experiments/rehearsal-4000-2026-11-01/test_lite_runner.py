"""Checks that prevent corrupted scores, accidental resends and benchmark leakage."""

import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("lite_runner.py")
SPEC = importlib.util.spec_from_file_location("rehearsal_lite", MODULE_PATH)
lite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lite)


def task(probabilities=(0.2, 0.2, 0.2, 0.2, 0.2)):
    native = sum(p * w for p, w in zip(probabilities, lite.WEIGHTS))
    return {
        "native_0_1": native,
        "score_1_5": 1 + 4 * native,
        "probabilities": list(probabilities),
        "input_tokens": 100,
    }


def frozen_corpus(root):
    cases = [
        {
            "case_id": f"R{i:04}",
            "family_id": f"family-{i}",
            "view_sha256": f"image-{i}",
            "benchmark_partition": "development"
            if i < 100
            else "representative"
            if i < 400
            else "challenge"
            if i < 500
            else "rehearsal",
        }
        for i in range(4000)
    ]
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases}))
    references = root / "references.jsonl"
    references.write_text(
        "".join(
            json.dumps({"case_id": c["case_id"], "view_sha256": c["view_sha256"]}) + "\n" for c in cases[:500]
        )
    )
    receipt = root / "freeze.json"
    value = {
        "frozen_at": "2026-01-01T00:00:00+00:00",
        "references_path": references.name,
        "references_sha256": lite.digest_bytes(references.read_bytes()),
        "case_ids": [c["case_id"] for c in cases[:500]],
        "manifest_sha256": lite.digest_bytes(manifest.read_bytes()),
    }
    receipt.write_text(json.dumps(value))
    return cases, manifest, references, receipt, value


class LiteValidationTests(unittest.TestCase):
    def test_preserves_unrounded_expected_scores(self):
        value = task((0.12345, 0.23456, 0.11111, 0.34567, 0.18521))
        normalized = lite.normalize_response({"quality": value, "aesthetics": task()})
        self.assertEqual(normalized["technical_score"], value["score_1_5"])
        self.assertEqual(normalized["technical_score_native"], value["score_1_5"])
        self.assertEqual(normalized["status"], "valid")

    def test_probability_inconsistency_cannot_become_valid_score(self):
        value = task()
        value["native_0_1"] = 0.9
        value["score_1_5"] = 4.6
        normalized = lite.normalize_response({"quality": value, "aesthetics": task()})
        self.assertEqual(normalized["status"], "partial")
        self.assertIsNone(normalized["technical_score"])
        self.assertEqual(normalized["aesthetic_score"], 3.0)

    def test_nonfinite_and_boolean_values_are_rejected(self):
        for invalid in (float("nan"), float("inf"), True, "0.5"):
            with self.subTest(invalid=invalid):
                value = task()
                value["native_0_1"] = invalid
                with self.assertRaises(ValueError):
                    lite.validate_task(value)

    def test_partition_does_not_leak_benchmark_into_rehearsal(self):
        cases = [
            {"case_id": str(i), "benchmark_partition": p}
            for i, p in enumerate((None, "rehearsal", "development", "representative", "challenge"))
        ]
        self.assertEqual({case["case_id"] for case in lite.select_cases(cases, "rehearsal")}, {"0", "1"})
        self.assertEqual({case["case_id"] for case in lite.select_cases(cases, "benchmark")}, {"2", "3", "4"})

    def test_duplicate_ids_rejected_before_any_request(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            lite.select_cases([{"case_id": "a"}, {"case_id": "a"}], "all")

    def test_unresolved_attempt_never_automatically_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempt.json"
            case = {"case_id": "a", "view_sha256": "image"}
            for status in ("reserved", "in_flight", "http_error", "transport_or_parse_error"):
                path.write_text(json.dumps({"run_id": "run", "view_sha256": "image", "status": status}))
                with self.subTest(status=status), self.assertRaisesRegex(ValueError, "Unresolved"):
                    lite.resume_record(path, case, "run")

    def test_completed_attempt_requires_same_inputs_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempt.json"
            raw = Path(tmp) / "raw.json"
            raw.write_text("{}")
            case = {"case_id": "a", "view_sha256": "image"}
            record = {
                "run_id": "run",
                "view_sha256": "image",
                "status": "valid",
                "raw_response_path": str(raw),
            }
            path.write_text(json.dumps(record))
            self.assertEqual(lite.resume_record(path, case, "run"), record)
            with self.assertRaisesRegex(ValueError, "different frozen"):
                lite.resume_record(path, case, "other-run")
            raw.unlink()
            with self.assertRaisesRegex(ValueError, "missing its raw"):
                lite.resume_record(path, case, "run")

    def test_hash_mismatch_stops_before_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.png"
            image.write_bytes(b"changed-image")
            case = {"case_id": "a", "image_path": str(image), "view_sha256": "wrong", "family_id": "family"}
            with self.assertRaisesRegex(ValueError, "hash changed"):
                lite.prepare_attempt(case, "run", Path(tmp) / "output")
            self.assertFalse((Path(tmp) / "output").exists())

    def test_benchmark_admission_needs_matching_frozen_references(self):
        cases = [{"case_id": "a", "benchmark_partition": "representative"}]
        with self.assertRaisesRegex(ValueError, "require a frozen"):
            lite.reference_admission(cases, None)
        with tempfile.TemporaryDirectory() as tmp:
            cases, manifest, references, receipt, _ = frozen_corpus(Path(tmp))
            with self.assertRaisesRegex(ValueError, "full corpus manifest"):
                lite.reference_admission(cases[:1], receipt)
            admission = lite.reference_admission(cases[:1], receipt, manifest)
            self.assertTrue(admission["required"])
            self.assertEqual(admission["benchmark_cases"], 500)
            self.assertEqual(admission["benchmark_families"], 500)
            self.assertEqual(admission["manifest_sha256"], lite.digest_bytes(manifest.read_bytes()))
            references.write_text(json.dumps({"case_id": "changed"}) + "\n")
            with self.assertRaisesRegex(ValueError, "changed since freeze"):
                lite.reference_admission(cases[:1], receipt, manifest)

    def test_one_case_subset_still_rejects_partial_duplicate_and_extra_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases, manifest, references, receipt, value = frozen_corpus(Path(tmp))
            original = [json.loads(line) for line in references.read_text().splitlines()]
            for rows in (
                original[:1],
                original[:-1],
                original[:-1] + [original[0]],
                original + [{"case_id": "extra", "view_sha256": "extra"}],
            ):
                with self.subTest(count=len(rows)):
                    references.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    value.update(
                        references_sha256=lite.digest_bytes(references.read_bytes()),
                        case_ids=[row["case_id"] for row in rows],
                    )
                    receipt.write_text(json.dumps(value))
                    with self.assertRaisesRegex(ValueError, "All 500 matching"):
                        lite.reference_admission(cases[:1], receipt, manifest)

    def test_changed_reference_image_and_duplicate_receipt_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases, manifest, references, receipt, value = frozen_corpus(Path(tmp))
            value["case_ids"].append(value["case_ids"][0])
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "All 500 matching"):
                lite.reference_admission(cases[:1], receipt, manifest)
            value["case_ids"].pop()
            rows = [json.loads(line) for line in references.read_text().splitlines()]
            rows[-1]["view_sha256"] = "changed nonselected benchmark image"
            references.write_text("".join(json.dumps(row) + "\n" for row in rows))
            value["references_sha256"] = lite.digest_bytes(references.read_bytes())
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "reference image differs"):
                lite.reference_admission(cases[:1], receipt, manifest)

    def test_full_manifest_identity_family_and_partition_contract_is_required(self):
        changes = (
            lambda cases: cases.pop(),
            lambda cases: cases[1].update(case_id=cases[0]["case_id"]),
            lambda cases: cases[1].update(family_id=cases[0]["family_id"]),
            lambda cases: cases[1].update(view_sha256=cases[0]["view_sha256"]),
            lambda cases: cases[500].update(family_id=cases[0]["family_id"]),
            lambda cases: cases[1].update(benchmark_partition="challenge"),
        )
        for change in changes:
            with tempfile.TemporaryDirectory() as tmp:
                cases, manifest, _, receipt, value = frozen_corpus(Path(tmp))
                change(cases)
                manifest.write_text(json.dumps({"cases": cases}))
                value["manifest_sha256"] = lite.digest_bytes(manifest.read_bytes())
                receipt.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "500 distinct benchmark families"):
                    lite.reference_admission(cases[:1], receipt, manifest)

    def test_selected_case_and_manifest_hash_must_match_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases, manifest, _, receipt, _ = frozen_corpus(Path(tmp))
            for field in ("case_id", "family_id", "view_sha256", "benchmark_partition"):
                changed = {**cases[0], field: "wrong"}
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Selected case differs"):
                    lite.reference_admission([changed], receipt, manifest)
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "different corpus manifest"):
                lite.reference_admission(cases[:1], receipt, manifest)

    def test_nonbenchmark_partition_does_not_require_reference_freeze(self):
        self.assertEqual(
            lite.reference_admission([{"benchmark_partition": "rehearsal"}], None), {"required": False}
        )

    def test_cli_subset_cannot_contact_server_before_complete_benchmark_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases, manifest, references, receipt, value = frozen_corpus(root)
            token = root / "test-token"
            token.write_text("test-only")
            arguments = [
                "lite_runner.py",
                "--manifest",
                str(manifest),
                "--output",
                str(root / "output"),
                "--references-freeze",
                str(receipt),
                "--partition",
                "benchmark",
                "--case-ids",
                cases[0]["case_id"],
                "--token-file",
                str(token),
            ]
            full_references = references.read_bytes()
            full_receipt = receipt.read_bytes()
            references.write_text(references.read_text().splitlines()[0] + "\n")
            value.update(
                references_sha256=lite.digest_bytes(references.read_bytes()), case_ids=[cases[0]["case_id"]]
            )
            receipt.write_text(json.dumps(value))
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(lite.requests, "get") as health:
                with self.assertRaisesRegex(ValueError, "All 500 matching"):
                    lite.main()
                health.assert_not_called()
            references.write_bytes(full_references)
            receipt.write_bytes(full_receipt)
            response = mock.Mock()
            response.json.return_value = {"model": lite.MODEL, "revision": lite.REVISION}
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(lite.requests, "get", return_value=response),
                mock.patch.object(lite, "execute", return_value={"status": "complete"}) as execute,
                mock.patch("builtins.print"),
            ):
                lite.main()
            self.assertEqual(execute.call_args.args[0], cases[:1])
            self.assertEqual(
                execute.call_args.kwargs["config"]["reference_admission"]["benchmark_cases"], 500
            )

    def test_exclusive_writes_sync_new_directories_and_file_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "run" / "attempts" / "reserved.json"
            events = []
            real_fsync = os.fsync

            def observe(descriptor):
                events.append(
                    (
                        stat.S_ISDIR(os.fstat(descriptor).st_mode),
                        Path(os.readlink(f"/proc/self/fd/{descriptor}")),
                    )
                )
                return real_fsync(descriptor)

            with mock.patch.object(lite.os, "fsync", side_effect=observe):
                lite.write_json(path, {"status": "reserved"}, exclusive=True)
            self.assertEqual(events[-1], (True, path.parent))
            file_index = events.index((False, path))
            for directory in (root, root / "run", path.parent):
                self.assertIn((True, directory), events[:file_index])
            with self.assertRaises(FileExistsError):
                lite.write_json(path, {"status": "replacement"}, exclusive=True)
            self.assertEqual(json.loads(path.read_text()), {"status": "reserved"})

    def test_atomic_replacement_syncs_new_file_then_parent_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempt.json"
            lite.write_json(path, {"status": "reserved"}, exclusive=True)
            events = []
            real_fsync, real_replace = os.fsync, Path.replace

            def observe(descriptor):
                events.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
                return real_fsync(descriptor)

            def replace(source, target):
                events.append("replace")
                return real_replace(source, target)

            with (
                mock.patch.object(lite.os, "fsync", side_effect=observe),
                mock.patch.object(Path, "replace", replace),
            ):
                lite.write_json(path, {"status": "in_flight"})
            self.assertEqual(events[-3:], ["file", "replace", "directory"])
            self.assertEqual(json.loads(path.read_text()), {"status": "in_flight"})

    def test_directory_sync_failure_prevents_dispatch_and_keeps_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            image.write_bytes(b"test image")
            case = {
                "case_id": "case",
                "family_id": "family",
                "benchmark_partition": "rehearsal",
                "image_path": str(image),
                "view_sha256": lite.digest_bytes(image.read_bytes()),
            }
            directory = root / "output"
            reservation = directory / "attempts" / "case.json"
            real_sync = lite.fsync_directory

            def fail_after_reservation(path):
                if path == reservation.parent and reservation.exists():
                    raise OSError("simulated directory fsync failure")
                return real_sync(path)

            with (
                mock.patch.object(lite, "fsync_directory", side_effect=fail_after_reservation),
                mock.patch.object(lite, "send_attempt") as send,
            ):
                with self.assertRaisesRegex(OSError, "simulated directory"):
                    lite.execute(
                        [case],
                        directory=directory,
                        config={"concurrency": 1, "url": "https://unused.invalid"},
                        token="test-only",
                        max_seconds=60,
                    )
                send.assert_not_called()
            record = json.loads(reservation.read_text())
            self.assertEqual(record["status"], "reserved")
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                lite.resume_record(reservation, case, record["run_id"])


if __name__ == "__main__":
    unittest.main()
