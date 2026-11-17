# Private R2 dataset backup

The user approved backing up all 20,000 exact scored views alongside the labels on
2026-11-16. Use Cloudflare R2 Standard in the private `model-gym-backups` bucket,
under `visual-quality/labeling-20000-2026-11-16/`. Account:
`2b0d43c32a6f1fb83147092522f3c1bb`. Public `r2.dev` access is disabled and no custom
domain is attached. The location hint is Western Europe; this is not an EU
jurisdiction guarantee.

Completed and verified on 2026-11-17: **20,011 objects, 38,360,238,571 bytes
(35.73 GiB)**, including all 20,000 scored images. Every object was downloaded and
hashed; the complete remote inventory matched, and the full local restore test
passed. The temporary upload token was revoked. See the
[completion receipt](receipt-20000-2026-11-16.json) for snapshot checksums and evidence.

## Contents

- `labelled-images.jsonl.zst`: convenient compressed canonical 20K image-level
  export. It retains the original paths and all scores, latency, price, image
  statistics, provenance and training masks.
- `metadata.tar.zst`: canonical exports, manifests, statistics, embeddings,
  related-image/duplicate review data, reference records, raw inference results
  including failed attempts, audit reports and source/configuration snapshots.
- `data/.../views/...`: all 20,000 exact scored image files, preserving original
  bytes and their relative data-directory paths. Images span three historical
  directories; copying only the latest run's views is insufficient.
- `image-index.json`: case ID to object key, byte count and SHA-256 mapping.
- `metadata-inventory.json`: hashes of the unpacked metadata/evidence files.
- `snapshot.json`: object checksums, source root, counts and snapshot provenance.
- `restore.py`: portable restore helper; requires Python 3.12+ and `zstd`.
- `verification.json`: completed verification receipt, published after success.

This is a scored-dataset backup, not a mirror of the workspace. It excludes
credentials, unselected acquisition candidates, caches, unscored source originals,
temporary operations and redundant intermediate exports. Source URLs, licenses
and original-source hashes remain as provenance. The exact delivered views are
self-contained; original `source_path` values may still refer to files outside
this backup. Historical raw responses and unknown prices remain unchanged.

The dataset has 20,000 image records and 19,997 complete ensemble score pairs.
The three missing pairs are preserved, not filled or silently reweighted. The
ensemble remains 37.5% Luna + 37.5% DeepSeek + 25% Liquid for each outcome. Training
masks remain disabled. This private backup does not change the Git publication
allowlist of the first 1,000 examples.

## Restore

Create a bucket-scoped **Object Read** R2 credential and configure a local AWS
profile named `model-gym-r2`. Do not paste credentials into a chat, script or Git.
Use a new empty restore directory with at least 45 GiB free; raw evidence includes
many small files. From that directory:

```sh
aws --profile model-gym-r2 \
  --endpoint-url https://2b0d43c32a6f1fb83147092522f3c1bb.r2.cloudflarestorage.com \
  s3 sync s3://model-gym-backups/visual-quality/labeling-20000-2026-11-16/ .
python3 restore.py .
```

The helper verifies each remote snapshot file against `snapshot.json`, safely
extracts the metadata archive, verifies its internal inventory, and writes
`labelled-images.portable.jsonl`. This derived file rewrites paths only for backed
up files that exist under the new restore directory. It retains unavailable source
paths as provenance. The original canonical JSONL remains byte-identical in
`data/labeling-20000-2026-11-16/exports/labelled-images.jsonl`.

For labels alone, download `labelled-images.jsonl.zst` and run
`zstd -d labelled-images.jsonl.zst`. Images and raw evidence require the full restore.

## Backup procedure

`prepare.py` is the explicit allowlist for this snapshot. It verifies every image
against the combined manifest and archives the selected metadata. It deliberately
does not recursively copy the entire data directory. The `source_commit` identifies
the labeling implementation; individual source files in the archive have their
own hashes. Backup helper revisions are also stored separately at the snapshot root.

`transfer.py` accepts an explicit local upload plan and a mode-0600 credential JSON
containing `access_key_id` and `secret_access_key`. It checks local SHA-256, supplies
Content-MD5, checks the returned ETag, and performs a full GET/SHA-256 verification
for **every** object. Existing matching objects can be resumed; differing objects
are refused rather than overwritten. Use a new dated prefix for a changed snapshot.
Files must be smaller than 5 GiB for this single-part uploader.

The uploader uses `boto3` (validated with 1.44.4); the preparation and restore
helpers use the Python standard library plus the `zstd` executable. Run the
readback retry tests from the repository root with:

```sh
uv run --no-project --with boto3==1.44.4 python -m unittest discover \
  -s visual-quality/labeling/backups -p 'test_*.py'
```

Use a temporary bucket-scoped Object Read & Write token, then revoke it after
verification. Never include its credential file in the upload plan. Keep plans,
receipts and per-image inventories in ignored `visual-quality/data/`. Publish a
completion receipt only after every planned object has passed verification.

R2 Standard storage is $0.015/GB-month before the account's shared free allowance;
request charges are separate. See [R2 pricing](https://developers.cloudflare.com/r2/pricing/).
The full snapshot is approximately 38.4 GB (35.7 GiB), or about $0.58/month before
the free allowance, subject to Cloudflare's billing rounding and other account use.
