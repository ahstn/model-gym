# Consolidation review and cleanup

Inventory taken on 4 November 2026 at `e51fab149e13ce6cde421b3cdae5be11709cce32`, against `origin/main` at `83a9f1efb7436b9b6f9adf0428cc9f3c11b1ece7`. This is a scope and dependency assessment, not a completed correctness review or merge approval. Sizes below are tracked working-tree bytes in decimal MB, not compressed Git storage.

The subsequent [PR #7 review](pr-7-review-2026-11-04.md) records the correctness fixes and completed workspace checks. The [10K handover prompt](handover-photo-labeling-10000-2026-11-04.md) preserves the configuration and continuation instructions. The inventory and staged cleanup history below remain as originally assessed.

## Second cleanup completed

The follow-up pass archived and removed 13 older experiment trees: 1,794 tracked files and 334,193,783 bytes before retained-code extraction. Only the completed 10,000-image run remains as a tracked experiment. Shared response, reference, Lite and rendering helpers now live in `visual-quality/src/visual_quality/scoring/`; their regression tests moved with them. The exact Lite GPU server and health identity live in `visual-quality/labeling/runtime/`. Runtime imports no longer require code in removed experiment folders. Historical corpus/reuse inputs resolve through ignored restored data, with an environment override for other machines.

The [experiment catalogue](../visual-quality/archives/experiments.md) preserves the findings, archive hashes, pinned Git revision and recovery instructions. The existing uncommitted clustering experiment was preserved. Source receipts and completed datasets were not rewritten. The sections below record the initial assessment and first cleanup; their removal prerequisites have now been addressed.

## First cleanup completed

Following the user's approval, the three bulk artifact targets below were archived and removed from the tracked tree: 4,096 files, 449,353,429 bytes. Small archive notices replace gallery entry points and directory indexes. The [recovery guide](../visual-quality/archives/README.md) supplies a local tar, pinned Git-history retrieval commands and all member checksums. A fresh-directory restore verified every file's size and SHA256 before removal. The existing local gallery, full 10,000-image dataset and committed 1,000-example dataset remain intact.

The scripts and historical manifests/score files still imported by the active workflow remain in place. Their extraction is a separate structural change, not a prerequisite for this completed media/receipt cleanup. Original report conclusions and validation receipts remain historical evidence; archive notices do not claim to be the original validated gallery pages.

PRs [#3](https://github.com/ahstn/model-gym/pull/3) and [#6](https://github.com/ahstn/model-gym/pull/6) are now closed as superseded by [draft PR #7](https://github.com/ahstn/model-gym/pull/7), after confirming both tips are ancestors of the consolidation branch. Their branches were preserved. PR #5 remains separate and open.

Post-cleanup checks: 385 visual-quality/current-workflow tests and four subtests passed. Documentation recovery links and exact archive member hashes were checked. This does not replace the broader correctness review of the consolidation PR.

## Review scope

The branch contains 40 commits beyond main and changes 6,359 files: 6,049 in visual-quality, 256 in visual-triage, 50 in image-stats, three in docs and one gitignore. Most of the file count is experiment evidence and generated media. Main is already an ancestor, so no main merge is needed at this snapshot.

There was no open PR for `codex/consolidate-model-gym-2026-10-21` at initial inspection; draft PR #7 was subsequently created. PR #3 (visual-quality pilot) and #6 (visual-triage) have their tips included in this branch. Their historical descriptions do not describe the completed consolidation. PR #5 (`feat/vq-baseline`) is separate: its tip is not included.

No `.github` workflow directory exists in this checkout. The saved 10,000-image validation receipt records the earlier test and artifact checks; it is not CI evidence for the entire consolidation branch. Review the three workspaces separately, then run their documented offline checks on the final proposed merge revision.

## Keep as the active entry points

- [Current labeling contract and records](../visual-quality/labeling/README.md): two scores, model routes, shared ensemble, first 1,000 benchmark examples and publication validator.
- [10,000-image run](../visual-quality/experiments/labeling-10000-2026-11-01/README.md): final report, validation receipt, runbooks and four bounded collage pages. This directory remains an active reproduction workflow despite living under `experiments/`.
- [Image statistics](../image-stats/README.md), including its source, tests and control evidence.
- [Visual triage](../visual-triage/README.md), including its existing evidence archive index and restore tooling.
- Source terms, manifests, exact view hashes, reference freezes, cost receipts and label-review state. Historical raw evidence is not interchangeable with fresh model observations.

The full 10,000-image export stays in ignored `visual-quality/data/labeling-10000-2026-11-01/exports/`. The tracked `visual-quality/labeling/records/` contains the first 1,000 benchmark images by explicit order. Neither is a cleanup target. Training remains disabled pending review.

## Cleanup candidates

| Area | Tracked size | Recommended treatment |
|---|---:|---|
| Earlier 4,000-image experiment | 4,944 files / 556.44 MB | Retain its compact report/protocol and required shared code; archive bulk evidence and generated views before removing them from the active tree. |
| Its `gallery/` | 4,016 files / 227.77 MB | Largest file-count reduction. Move historical generated media to a checksummed evidence bundle with an explicit restore command; preserve the running local gallery and update documentation links. |
| Its `execution-records/` | 8 files / 95.49 MB | Move compressed response archives and their index together to durable evidence storage. Verify restored member hashes and receipt accounting before removal. |
| Its top-level files | 211 files / 215.18 MB | Mixed code and evidence: do not remove as a directory. Includes 62.34 MB `scores-long.jsonl`, 29.87 MB CSV, 16.05 MB candidate manifest, 15.73 MB candidate embeddings and 12.97 MB manifest. Some are direct inputs to the expansion. |
| Earlier photo-scoring experiment | 98 files / 155.19 MB | Archive the 72 exact PNG views (126.10 MB) and redundant presentation pages after recording restoration and regeneration paths. Keep the compact score/provenance evidence supporting its report. |
| Prompt and eligibility experiments | Several smaller dated directories | Mark historical and archive bulk outputs only after checking references. Negative results explain why stats prompting and alternative eligibility wording were rejected; retain their protocols and conclusions. |

The two largest historical experiments account for about 711.63 MB together; that is an upper bound on material to inspect, not a promised deletion total. Their gallery, execution archives and 72 original PNGs alone account for about 449.36 MB of candidate media/evidence storage. These categories are disjoint, but each still needs a restore/dependency check.

Deleting files in a new commit reduces the checked-out tree and PR file content; it does not remove blobs from Git history or guarantee smaller clones. Do not rewrite published history for this cleanup.

## Dependencies that block blanket deletion

The 10,000-image workflow directly loads files from `rehearsal-4000-2026-11-01`:

| Current consumer | Historical dependency | Required preparation |
|---|---|---|
| `api_runner.py` | `api_contract.py` | Extract the shared response contract into the package, retain an import compatibility path, and test validation/cost behavior. |
| `gallery_run.py` | `gallery.py`, `gallery_png.py`, `validate_gallery.py` | Extract shared rendering/validation together; preserve displayed values, image identity checks and browser checks. |
| `lite_run.py` | `lite_runner.py`, `manifest.json`, `scores-long.jsonl` | Separate the client from historical evidence; make restored evidence paths explicit while preserving all reuse hashes and native scores. |
| `reference_work.py` | `reference_review.py`, `references.jsonl` | Preserve the review contract and frozen original-reference identity. |
| `corpus_prepare.py` / `corpus_common.py` | `candidate-manifest.json` and other prior corpus inputs | Use an explicit archived-evidence root and validate hashes before rebuilding. |
| `api_profile.py` | `api-teacher-prompt.json` | Keep the historical prompt used to establish profile provenance. |

The existing `experiments/README.md` called every experiment historical, although the latest workflow is still active and imports the earlier one. The index is corrected with this assessment. Age and folder names are insufficient evidence that a script is dead.

## Recommended order before merge

1. Review the current contract, ensemble arithmetic, publication boundary and final dataset. Keep these fixed while cleaning presentation files.
2. Extract the still-used response, Lite and gallery helpers without changing scoring behavior. Retain the executed source snapshots/hashes as historical evidence.
3. Build a checksummed historical evidence bundle and test restoration into an empty directory. Use the visual-triage archive tooling as a precedent, not as proof that visual-quality evidence is already archived remotely.
4. Remove verified archived bulk outputs from the tracked tree; update report links and reconstruction commands in the same change. Keep a compact archive index with storage location, revision, hashes and source terms. Check documentation links and offline regeneration before committing removals.
5. Review and run the documented offline checks for visual-quality, image-stats and visual-triage on the cleaned branch. Reconcile PR #3/#6, then perform the final merge review. PR #5 needs a separate integration decision.

The initial assessment made no model calls, dataset changes or evidence deletions. The authorized cleanup is recorded above. Existing uncommitted embedding documentation and experiment files were left untouched; they need their own scope decision before the working directory can be clean.
