"""Exercise the complete prepare/upload/restore contract using a tiny snapshot."""

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from prepare import digest, finish_snapshot, write_json
from restore import restore
from transfer import transfer


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.race = False

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        body, metadata = self.objects[Key]
        return {
            "ContentLength": len(body),
            "Metadata": metadata,
            "ETag": hashlib.md5(body, usedforsecurity=False).hexdigest(),
        }

    def put_object(self, Bucket, Key, Body, IfNoneMatch, **kwargs):
        assert IfNoneMatch == "*"
        if self.race or Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        body = Body.read() if hasattr(Body, "read") else Body
        self.objects[Key] = (body, kwargs.get("Metadata", {}))
        return {"ETag": hashlib.md5(body, usedforsecurity=False).hexdigest()}

    def get_object(self, Bucket, Key):
        body = self.objects[Key][0]
        return {"Body": StreamingBody(io.BytesIO(body), len(body))}

    def get_paginator(self, name):
        def paginate(Bucket, Prefix):
            return [
                {
                    "Contents": [
                        {
                            "Key": key,
                            "Size": len(body),
                            "ETag": hashlib.md5(body, usedforsecurity=False).hexdigest(),
                        }
                        for key, (body, _) in self.objects.items()
                        if key.startswith(Prefix)
                    ]
                }
            ]

        return Mock(paginate=paginate)


@unittest.skipUnless(shutil.which("zstd"), "zstd is required for snapshot preparation")
class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "prepared"
        self.output.mkdir()
        source = self.root / "original"
        image = source / "views/image.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"exact scored image bytes")
        labels = source / "labeling-20000-2026-11-16/exports/labelled-images.jsonl"
        labels.parent.mkdir(parents=True)
        labels.write_text(json.dumps({"image_path": str(image), "view_sha256": digest(image)}) + "\n")
        label_key = "data/" + labels.relative_to(source).as_posix()
        inventory = [{"path": label_key, "bytes": labels.stat().st_size, "sha256": digest(labels)}]
        write_json(self.output / "metadata-inventory.json", inventory)
        write_json(self.output / "image-index.json", [])
        tarpath = self.root / "metadata.tar"
        with tarfile.open(tarpath, "w") as archive:
            archive.add(labels, arcname=label_key)
            archive.add(self.output / "metadata-inventory.json", arcname="metadata-inventory.json")
        with (self.output / "metadata.tar.zst").open("wb") as destination:
            subprocess.run(["zstd", "-q", "-c", str(tarpath)], stdout=destination, check=True)
        paths = [(image, "data/views/image.png")] + [
            (self.output / n, n) for n in ("metadata.tar.zst", "image-index.json", "metadata-inventory.json")
        ]
        self.objects = [
            {"local_path": str(p), "key": key, "bytes": p.stat().st_size, "sha256": digest(p)}
            for p, key in paths
        ]
        finish_snapshot(self.output, self.objects, {"source_data_root": str(source), "labels": 1}, labels)
        self.plan = self.output / "upload-plan.json"
        self.credentials = self.root / "credentials.json"
        write_json(self.credentials, {"access_key_id": "test", "secret_access_key": "test"})
        self.client = MemoryS3()

    def upload(self, complete=True):
        with patch("transfer.boto3.client", return_value=self.client):
            transfer(self.plan, self.credentials, "account", "bucket", "snapshot", 2, complete)

    def test_complete_snapshot_can_restore_and_resume(self):
        self.upload()
        receipt = (self.output / "verification.json").read_bytes()
        self.upload()
        self.assertEqual(receipt, (self.output / "verification.json").read_bytes())
        target = self.root / "restore"
        for key, (body, _) in self.client.objects.items():
            path = target / key.removeprefix("snapshot/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        # Ensure restore cannot accidentally depend on files at the old paths.
        shutil.rmtree(self.root / "original")
        restore(target)
        record = json.loads((target / "labelled-images.portable.jsonl").read_text())
        self.assertEqual(Path(record["image_path"]), target / "data/views/image.png")

    def test_partial_plan_cannot_certify_or_overwrite_completion(self):
        self.plan.write_text(json.dumps(self.objects[:1]))
        with self.assertRaisesRegex(ValueError, "requires snapshot.json"):
            self.upload()
        self.assertEqual(self.client.objects, {})
        receipt = self.output / "verification.json"
        receipt.write_text("historical receipt")
        self.upload(complete=False)
        self.assertEqual(receipt.read_text(), "historical receipt")
        self.assertNotIn("snapshot/verification.json", self.client.objects)

    def test_manifest_in_partial_plan_is_not_sufficient(self):
        objects = json.loads(self.plan.read_text())
        self.plan.write_text(json.dumps(objects[1:]))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            self.upload()
        self.assertEqual(self.client.objects, {})

    def test_unexpected_remote_object_prevents_completion(self):
        self.client.objects["snapshot/unexpected"] = (b"extra", {})
        with self.assertRaisesRegex(ValueError, "Unexpected objects"):
            self.upload()
        self.assertNotIn("snapshot/verification.json", self.client.objects)

    def test_changed_local_object_prevents_completion(self):
        Path(self.objects[0]["local_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Unexpected file size"):
            self.upload()
        self.assertNotIn("snapshot/verification.json", self.client.objects)

    def test_racing_writer_is_never_overwritten(self):
        self.client.race = True
        with self.assertRaises(ClientError):
            self.upload()
        self.assertEqual(self.client.objects, {})


if __name__ == "__main__":
    unittest.main()
