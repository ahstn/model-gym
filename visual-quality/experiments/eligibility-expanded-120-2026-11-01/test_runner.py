"""Tests for job identity, retry eligibility, floor application and cost accounting."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import runner


class Response:
    def __init__(self, status, body):
        self.status_code = status
        self.text = json.dumps(body)
        self.headers = {}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / "view.png"
        self.image.write_bytes(b"exact-view")
        self.case = {
            "case_id": "C0001",
            "view_sha256": hashlib.sha256(b"exact-view").hexdigest(),
            "image_path": str(self.image),
        }
        runner.DATA = self.root
        runner.STOP.clear()
        runner.LEDGER = {
            "jev": {"known": Decimal(0), "unknown": 0},
            "liquid": {"known": Decimal(0), "unknown": 0},
        }
        runner.OUTSTANDING = {"jev": 0, "liquid": 0}
        self.routes = runner.read(runner.OLD / "api-routes.json")
        self.prompts = runner.read(runner.OLD / "api-decision-prompts.json")

    def tearDown(self):
        self.tmp.cleanup()

    def success(self, p=0.7):
        return {
            "model": "typesafe/jev-1.14-20261014",
            "provider": "TypeSafe",
            "usage": {"cost": 0.0001},
            "answers": {
                "excluded": {"type": "noul", "noul": p},
                "technical_quality": {"type": "score", "score": 2.5},
                "aesthetics": {"type": "score", "score": 3.0},
            },
        }

    def test_retry_only_verified_503_then_floor_and_never_replay(self):
        err = Response(503, {"error": {"message": "Image processing is unavailable."}})
        success = Response(200, self.success())
        calls = []

        def send(*a, **kw):
            calls.append(kw)
            return [err, success][len(calls) - 1]

        with patch.object(runner.requests, "post", side_effect=send), patch.object(runner.time, "sleep"):
            self.assertEqual(
                runner.job("r01", "jev", self.case, "explicit rule", self.routes, self.prompts, "dummy"),
                "valid",
            )
            self.assertEqual(
                runner.job("r01", "jev", self.case, "explicit rule", self.routes, self.prompts, "dummy"),
                "existing_valid",
            )
        self.assertEqual(len(calls), 2)
        body = json.loads(calls[0]["data"])
        self.assertEqual(body["questions"]["excluded"]["instructions"], "explicit rule")
        self.assertEqual(
            body["questions"]["technical_quality"],
            self.prompts["prompts"]["typesafe/jev-1.14"]["questions"]["technical_quality"],
        )
        result = runner.read(self.root / "attempts/r01/jev/C0001/02/result.json")
        self.assertEqual(result["parsed"]["ratings"]["technical_quality"]["score"], 1.0)
        self.assertEqual(runner.LEDGER["jev"]["unknown"], 1)
        self.assertEqual(runner.LEDGER["jev"]["known"], Decimal("0.0001"))

    def test_interrupted_admission_never_replayed(self):
        req = self.root / "attempts/r01/jev/C0001/01/request.json"
        runner.atomic(req, {"started_at": "unknown"})
        with patch.object(runner.requests, "post") as send:
            self.assertEqual(
                runner.job("r01", "jev", self.case, "x", self.routes, self.prompts, "dummy"),
                "unknown_no_replay",
            )
            send.assert_not_called()

    def test_non_503_error_is_not_replayed(self):
        with patch.object(
            runner.requests, "post", return_value=Response(500, {"error": {"message": "unrelated"}})
        ) as send:
            self.assertEqual(
                runner.job("r01", "jev", self.case, "x", self.routes, self.prompts, "dummy"), "error"
            )
            self.assertEqual(
                runner.job("r01", "jev", self.case, "x", self.routes, self.prompts, "dummy"), "existing_error"
            )
            self.assertEqual(send.call_count, 1)

    def test_caps_include_outstanding_and_unpriced_reservations(self):
        runner.LEDGER["jev"]["known"] = Decimal("9.995")
        with self.assertRaisesRegex(RuntimeError, "cap"):
            runner.acquire("jev")
        self.assertTrue(runner.STOP.is_set())

    def test_changed_view_prevents_request(self):
        self.image.write_bytes(b"changed")
        with (
            patch.object(runner.requests, "post") as send,
            self.assertRaisesRegex(ValueError, "Image changed"),
        ):
            runner.job("r01", "jev", self.case, "x", self.routes, self.prompts, "dummy")
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
