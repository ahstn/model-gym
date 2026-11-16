"""Protect readback integrity when an R2 response stream is interrupted."""

import hashlib
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ResponseStreamingError
from transfer import verify_readback


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


if __name__ == "__main__":
    unittest.main()
