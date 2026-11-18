"""Protect readback integrity when an R2 response stream is interrupted."""

import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.exceptions import ResponseStreamingError
from transfer import transfer, verify_readback


class ReadbackTests(unittest.TestCase):
    def test_interrupted_stream_restarts_hash_and_closes_bodies(self):
        def interrupted(**kwargs):
            yield b"partial"
            raise ResponseStreamingError(error="connection reset")

        failed = Mock()
        failed.iter_chunks.side_effect = interrupted
        complete = Mock()
        complete.iter_chunks.return_value = iter([b"exact", b" bytes"])
        client = Mock()
        client.get_object.side_effect = [{"Body": failed}, {"Body": complete}]
        with patch("transfer.time.sleep"):
            count = verify_readback(client, "bucket", "key", hashlib.sha256(b"exact bytes").hexdigest(), 11)
        self.assertEqual(count, 11)
        self.assertEqual(client.get_object.call_count, 2)
        failed.close.assert_called_once()
        complete.close.assert_called_once()

    def test_retries_are_bounded(self):
        client = Mock()
        client.get_object.side_effect = ResponseStreamingError(error="connection reset")
        with patch("transfer.time.sleep"), self.assertRaises(ResponseStreamingError):
            verify_readback(client, "bucket", "key", "unused", 1)
        self.assertEqual(client.get_object.call_count, 5)

    def test_completed_corruption_is_not_retried(self):
        body = Mock()
        body.iter_chunks.return_value = iter([b"bad"])
        client = Mock()
        client.get_object.return_value = {"Body": body}
        with self.assertRaisesRegex(ValueError, "Remote readback failed"):
            verify_readback(client, "bucket", "key", hashlib.sha256(b"good").hexdigest(), 4)
        client.get_object.assert_called_once()
        body.close.assert_called_once()


class StagedVerificationTests(unittest.TestCase):
    def setUp(self):
        from test_snapshot import MemoryS3

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.client = MemoryS3()
        get = patch.object(self.client, "get_object", wraps=self.client.get_object)
        self.get = get.start()
        self.addCleanup(get.stop)
        head = self.client.head_object
        self.modified = datetime.now(UTC) - timedelta(seconds=1)
        self.client.head_object = lambda **kwargs: {**head(**kwargs), "LastModified": self.modified}
        client = patch("transfer.boto3.client", return_value=self.client)
        client.start()
        self.addCleanup(client.stop)
        self.credentials = self.root / "credentials.json"
        self.credentials.write_text(json.dumps({"access_key_id": "test", "secret_access_key": "test"}))

        def item(key, data):
            path = self.root / key
            path.write_bytes(data)
            return {
                "key": key,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "local_path": str(path),
            }

        image = item("image.png", b"image")
        labels = item("records", b"labels")
        manifest = item("snapshot.json", json.dumps({"objects": [image, labels]}).encode())
        self.plan = self.root / "plan.json"
        self.plan.write_text(json.dumps([image, labels, manifest]))
        self.stage = self.root / "stage.json"
        self.stage.write_text(json.dumps([image]))
        self.session = self.root / "stage-start.json"
        self.session.write_text(
            json.dumps(
                {
                    "account": "account",
                    "bucket": "bucket",
                    "prefix": "snapshot/",
                    "started_at": datetime.now(UTC).isoformat(),
                    "plan_sha256": hashlib.sha256(self.stage.read_bytes()).hexdigest(),
                }
            )
        )
        transfer(self.stage, self.credentials, "account", "bucket", "snapshot", 1)
        self.get.reset_mock()

    def finish(self):
        transfer(
            self.plan, self.credentials, "account", "bucket", "snapshot", 1, True, self.stage, self.session
        )

    def test_fresh_get_only_for_objects_without_matching_earlier_readback(self):
        self.finish()
        keys = [call.kwargs["Key"] for call in self.get.call_args_list]
        self.assertCountEqual(
            keys, ["snapshot/records", "snapshot/snapshot.json", "snapshot/verification.json"]
        )
        receipt = json.loads((self.root / "verification.json").read_bytes())
        self.assertEqual(receipt["objects_verified"], 3)
        self.assertEqual(receipt["stage_readbacks_reused"], 1)
        self.assertEqual(
            receipt["stage_verification_sha256"],
            hashlib.sha256(self.stage.with_suffix(".verification.json").read_bytes()).hexdigest(),
        )

    def test_stage_destination_mismatch_and_old_receipts_are_rejected(self):
        session = json.loads(self.session.read_bytes())
        self.session.write_text(json.dumps({**session, "account": "other-account"}))
        with self.assertRaisesRegex(ValueError, "different plan or destination"):
            self.finish()
        self.session.write_text(json.dumps(session))
        ledger = self.root / "transfer-receipts.jsonl"
        row = json.loads(ledger.read_text())
        ledger.write_text(json.dumps({**row, "verified_at": self.modified.isoformat()}) + "\n")
        with self.assertRaisesRegex(ValueError, "Missing full GET"):
            self.finish()
        self.assertNotIn("snapshot/verification.json", self.client.objects)

    def test_replaced_remote_object_gets_new_readback(self):
        self.modified = datetime.now(UTC) + timedelta(seconds=1)
        self.finish()
        self.get.assert_any_call(Bucket="bucket", Key="snapshot/image.png")
        self.assertEqual(
            json.loads((self.root / "verification.json").read_bytes())["stage_readbacks_reused"], 0
        )

    def test_changed_local_data_fails_even_with_completed_stage_receipt(self):
        (self.root / "image.png").write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "Changed local file"):
            self.finish()
        self.assertNotIn("snapshot/verification.json", self.client.objects)

    def test_missing_stage_summary_cannot_certify_snapshot(self):
        self.stage.with_suffix(".verification.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.finish()
        self.assertNotIn("snapshot/verification.json", self.client.objects)


if __name__ == "__main__":
    unittest.main()
