# Evidence storage and restore

Git retains code, dependency locks, protocols, selected prompts, source notes and compact results. Raw responses and image files stay outside Git. The earlier [compact results](summaries/) retain 1,600 per-image rows, aggregate metrics, costs, latency and all 43 candidate prompts. These exports support comparison; they do not replace raw responses for an audit or a resumed run.

## Native metadata trial

The [metadata report](tracks/metadata_v3/report.md) and [per-image results](tracks/metadata_v3/results/cases.csv) are tracked. Its full dataset, control manifest and summary remain ignored, with raw calls and source captures in a local checksummed archive. The [receipt](tracks/metadata_v3/evidence-bundle.json) pins the archive and index. This archive is unpublished; a fresh clone must obtain both files from local storage before restoration. Image bytes are excluded and need the separate image cache.

Follow the [metadata restore commands](tracks/metadata_v3/README.md#run-and-verify). Evidence-only restoration preserves current code and reports. Full snapshot restoration needs an empty directory. Frozen receipts retain their original absolute paths, so another checkout can inspect the evidence but cannot assume that all historical commands will run unchanged.

## Earlier benchmark archive

The earlier cleanup reduced the tracked directory from 456 files and 35.0 MB to 154 files and 5.4 MB. The local evidence files remain present and ignored.

The [versioned release](https://github.com/ahstn/model-gym/releases/tag/visual-triage-evidence-2026-10-11) preserves all 456 original files from commit `b70ecfdfa5f09d2de92e214d910ad2696b2f7834`. This includes the original code, source captures, raw API attempts, reflection records and full reports. The archive is 5,133,104 bytes. Its SHA-256 is:

```text
5addb4ad543e7ea0c6782bf018a310e3523c241e791a5370f048e159b7b6687f
```

The checked-in [archive index](archive-index.json) pins the download URL, archive size and hash, and every file hash. The release also supplies `SHA256SUMS` for the archive and index. All archived content was already in the public commit. Downloaded image bytes were never tracked and are not in this archive.

## Restore evidence

Run from `visual-triage`. Restoration uses only the Python standard library and makes no paid model calls:

```sh
python3 evidence_archive.py restore
python3 evidence_archive.py verify
```

The default command restores the 321 files removed from Git. It checks the complete archive before writing, skips identical files and rejects changed files, unsafe paths and links. Restored evidence stays ignored by Git. Reports and runners that use historical evidence check its hashes before they read results or calculate spending. Unknown charges remain unknown; a budget reservation is not a reported charge.

For an offline restore, add `--archive /absolute/path/visual-triage-b70ecfd-evidence.tar.gz`. To restore the original code and all evidence, use an empty directory:

```sh
python3 evidence_archive.py restore --snapshot --destination /private/tmp/triage-evidence-b70ecfd
python3 evidence_archive.py verify --snapshot --destination /private/tmp/triage-evidence-b70ecfd
```

Use the full snapshot for an exact historical report rebuild: current Markdown links now point to compact exports. The restore command never overwrites changed code. On macOS, use `/private/tmp`, since `/tmp` is a symbolic link and restore rejects links in destination paths.

## Reproduce and extend

After restoration, use the existing [track commands](tracks/README.md) to rebuild full reports. Regenerate compact exports with:

```sh
python3 scripts/export_results.py
```

Image validation still needs the local `data/` cache or acquisition through the dataset builders. Frozen manifests contain the selected IDs and hashes; each track's source notes state its access and licence limits.

For new experiments, put bulky output in ignored `data/` or the existing ignored trial directories. Review and commit code, frozen inputs, selected prompts and compact results. Publish a new archive and index before removing unique evidence. `scripts/build_evidence_archive.py` builds deterministic snapshots from a pinned commit and refuses to overwrite an existing archive or index.

PR #6's branch history and the release tag were rewritten to remove the archived evidence from normal Git clones. The release assets, archive checksum and restore commands are unchanged. The source commit in the index is a provenance record; restoration does not require that commit to exist in the clone. The release's automatic source-code download contains the clean tree; use the named evidence archive for the original snapshot.

Existing clones can retain the old objects and tags. Use a fresh clone to obtain the smaller history. GitHub can retain internal PR references, so this change is not a guarantee that old objects have been deleted from its servers.
