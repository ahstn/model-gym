"""Upload an explicit immutable snapshot plan, then read back every byte."""

import argparse
import base64
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)


def verify_readback(client, bucket, key, expected_sha, expected_bytes):
    """Retry interrupted streams; never retry a completed checksum mismatch."""
    for attempt in range(5):
        response = None
        try:
            response = client.get_object(Bucket=bucket, Key=key)
            remote_sha, remote_bytes = hashlib.sha256(), 0
            for chunk in response["Body"].iter_chunks(chunk_size=8 * 1024**2):
                remote_sha.update(chunk)
                remote_bytes += len(chunk)
            if remote_sha.hexdigest() != expected_sha or remote_bytes != expected_bytes:
                raise ValueError(f"Remote readback failed: {key}")
            return remote_bytes
        except (ConnectionClosedError, IncompleteReadError, ReadTimeoutError, ResponseStreamingError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)
        finally:
            if response is not None:
                response["Body"].close()
    raise RuntimeError("Unreachable readback state")


def validate_complete_plan(objects):
    """A partial plan must never certify a complete snapshot."""
    manifest = next((row for row in objects if row["key"] == "snapshot.json"), None)
    if manifest is None:
        raise ValueError("Complete verification requires snapshot.json in the plan")
    path = Path(manifest["local_path"])
    raw = path.read_bytes()
    if len(raw) != manifest["bytes"] or hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("Snapshot manifest changed")
    expected = json.loads(raw)["objects"] + [manifest]

    def identities(rows):
        keys = [row["key"] for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate snapshot keys")
        return {(row["key"], row["bytes"], row["sha256"]) for row in rows}

    if identities(expected) != identities(objects):
        raise ValueError("Upload plan does not cover the complete snapshot")


def transfer(plan, credentials, account, bucket, prefix, workers, complete_snapshot=False):
    objects = json.loads(plan.read_text())
    if len({entry["key"] for entry in objects}) != len(objects):
        raise ValueError("Duplicate upload keys")
    if complete_snapshot:
        validate_complete_plan(objects)
    credentials = json.loads(credentials.read_text())
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        region_name="auto",
        aws_access_key_id=credentials["access_key_id"],
        aws_secret_access_key=credentials["secret_access_key"],
        config=Config(
            max_pool_connections=workers,
            retries={"mode": "standard", "max_attempts": 5},
            connect_timeout=20,
            read_timeout=180,
            request_checksum_calculation="when_required",
        ),
    )
    prefix = prefix.rstrip("/") + "/"
    lock = threading.Lock()
    receipt = plan.parent / "transfer-receipts.jsonl"

    def one(entry):
        path = Path(entry["local_path"])
        key = prefix + entry["key"]
        # Single-part uploads keep ETag comparable to the content MD5. All files
        # in this snapshot are below 5 GiB; fail before an unsupported upload.
        if path.stat().st_size != entry["bytes"] or entry["bytes"] >= 5 * 1024**3:
            raise ValueError(f"Unexpected file size: {entry['key']}")
        sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024**2), b""):
                sha.update(chunk)
                md5.update(chunk)
        if sha.hexdigest() != entry["sha256"]:
            raise ValueError(f"Changed local file: {entry['key']}")
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
            head = None
        if head is not None:
            if (
                head["ContentLength"] != entry["bytes"]
                or head["Metadata"].get("sha256") != entry["sha256"]
                or head["ETag"].strip('"') != md5.hexdigest()
            ):
                raise ValueError(f"Refusing to overwrite a different remote object: {key}")
        else:
            with path.open("rb") as source:
                uploaded = client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=source,
                    ContentLength=entry["bytes"],
                    ContentMD5=base64.b64encode(md5.digest()).decode(),
                    Metadata={"sha256": entry["sha256"]},
                    IfNoneMatch="*",
                )
            if uploaded["ETag"].strip('"') != md5.hexdigest():
                raise ValueError(f"Upload ETag mismatch: {key}")
        # Full GET verification, including objects found during resume. Neither
        # caller-supplied metadata nor HEAD alone proves remote byte identity.
        remote_bytes = verify_readback(client, bucket, key, entry["sha256"], entry["bytes"])
        row = {
            "key": key,
            "sha256": entry["sha256"],
            "bytes": remote_bytes,
            "etag": md5.hexdigest(),
            "reused": head is not None,
            "verified_at": datetime.now(UTC).isoformat(),
        }
        with lock, receipt.open("a") as output:
            output.write(json.dumps(row) + "\n")
        return row

    completed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, entry) for entry in objects]
        try:
            for future in as_completed(futures):
                completed.append(future.result())
                if len(completed) % 250 == 0:
                    print(
                        f"Uploaded and readback-verified {len(completed)}/{len(objects)} objects", flush=True
                    )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    listed = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        listed.update(
            {item["Key"]: (item["Size"], item["ETag"].strip('"')) for item in page.get("Contents", [])}
        )
    for row in completed:
        if listed.get(row["key"]) != (row["bytes"], row["etag"]):
            raise ValueError(f"Final remote inventory mismatch: {row['key']}")
    if complete_snapshot and set(listed) - {prefix + "verification.json"} != {r["key"] for r in completed}:
        raise ValueError("Unexpected objects in snapshot prefix")
    summary = {
        "bucket": bucket,
        "prefix": prefix,
        "objects_verified": len(completed),
        "bytes_verified": sum(row["bytes"] for row in completed),
        "verification": "SHA-256 of full GET for every uploaded object",
        "completed_at": datetime.now(UTC).isoformat(),
        "scope": "complete_snapshot" if complete_snapshot else "transfer_plan_only",
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
    }
    # Partial transfers have their own receipts and cannot overwrite completion.
    receipt_path = plan.with_suffix(".verification.json")
    receipt_path.write_text(json.dumps(summary, indent=2) + "\n")
    if complete_snapshot:
        summary["snapshot_sha256"] = next(r["sha256"] for r in objects if r["key"] == "snapshot.json")
        key = prefix + "verification.json"
        body = (json.dumps(summary, indent=2) + "\n").encode()
        try:
            client.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("412", "PreconditionFailed"):
                raise
            response = client.get_object(Bucket=bucket, Key=key)
            try:
                prior = response["Body"].read()
            finally:
                response["Body"].close()
            saved = json.loads(prior)
            for field in (
                "bucket",
                "prefix",
                "snapshot_sha256",
                "objects_verified",
                "bytes_verified",
                "scope",
            ):
                if saved.get(field) != summary[field]:
                    raise ValueError("Existing completion receipt describes a different snapshot") from error
            body = prior
        verify_readback(client, bucket, key, hashlib.sha256(body).hexdigest(), len(body))
        (plan.parent / "verification.json").write_bytes(body)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--complete-snapshot",
        action="store_true",
        help="Require the full manifest and publish its completion receipt",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")
    transfer(
        args.plan,
        args.credentials,
        args.account,
        args.bucket,
        args.prefix,
        args.workers,
        args.complete_snapshot,
    )
