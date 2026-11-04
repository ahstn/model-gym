# Historical artifact recovery

The 4 November cleanup removed 4,096 historical artifact files (449,353,429 bytes before replacement notices) from the tracked experiment tree. Current scoring code, current labels, exact-view manifests, final 10,000-image results and the current 72-image collage remain in place.

[Archive index](historical-artifacts-2026-11-04.json) · [Per-file SHA256 checksums](historical-artifacts-2026-11-04.sha256)

The archived paths are the earlier 4,000-image `gallery/` and `execution-records/`, plus the 72 `photo-scoring-2026-10-31/images/` PNGs. A complete tar was built from commit `4c644bbbb96db5145c1ebff51313a87e7394f106`, restored into an empty directory, and every member's size and SHA256 verified before removal. The separate local canonical gallery and receipt copies were left intact.

## Restore from the local copy

Run from the repository root. This restores into a new directory and does not overwrite current code or reports:

```sh
REPOSITORY_ROOT="$PWD"
RESTORE_ROOT=$(mktemp -d /tmp/model-gym-history.XXXXXX)
sha256sum visual-quality/data/historical-cleanup-2026-11-04/historical-artifacts.tar
# Expected: abef5cf5f3ed5992c0d50aaddcac60249de74f8fb2b77a3ac70c08bd43b6a8e4
tar -xf visual-quality/data/historical-cleanup-2026-11-04/historical-artifacts.tar -C "$RESTORE_ROOT"
(cd "$RESTORE_ROOT" && sha256sum --quiet --check "$REPOSITORY_ROOT/visual-quality/archives/historical-artifacts-2026-11-04.sha256")
```

The local tar is ignored by Git. It includes historical score-bearing gallery data and receipts and must not be added to the current first-1,000 example export.

## Fetch the exact bytes from Git history

The source commit remains in the normal ancestry of the consolidation branch. A normal fetch preserves access; shallow clones may need `git fetch --unshallow origin`. If the local tar is unavailable, substitute this for the `tar -xf` command above, then run the same checksum check:

```sh
git archive 4c644bbbb96db5145c1ebff51313a87e7394f106 -- \
  visual-quality/experiments/rehearsal-4000-2026-11-01/gallery \
  visual-quality/experiments/rehearsal-4000-2026-11-01/execution-records \
  visual-quality/experiments/photo-scoring-2026-10-31/images \
  | tar -x -C "$RESTORE_ROOT"
```

This recovers the exact evaluated bytes without depending on upstream image URLs remaining available. The restored gallery is a historical snapshot; its original-image links still require the original local delivered views. Receipt archives should be extracted only into another empty directory, preserving their original path and timestamp provenance.

For complete historical scripts, inputs and reports together, use a detached checkout at that revision, for example `git worktree add --detach /tmp/model-gym-historical-review 4c644bbbb96db5145c1ebff51313a87e7394f106`. Follow that checkout's reproduction notes; API re-execution is unnecessary for viewing or regenerating saved results.

## Viewing current results

Use the [current 72-image collage](../experiments/labeling-10000-2026-11-01/collage/README.md) or the [full local 10,000-image gallery](../data/labeling-10000-2026-11-01/gallery/index.html). The existing 4,000-image local service continues to use its independent ignored data copy.

The two older static comparison galleries now provide archive notices because their inline photos were archived. Their compact score files, original rendering scripts and historical results remain available. Restored images and generated historical outputs are ignored to prevent accidental recommits. No Git history was rewritten; this cleanup reduces the tracked checkout and review diff, not historical clone storage.
