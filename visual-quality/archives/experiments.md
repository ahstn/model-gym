# Historical experiments

The tracked experiments directory retains the completed 10,000-image run. These 13 earlier trees were archived at commit `ddcbc4d422b4d4b931bf01b2f00b7938935052fe`: **1,794 files / 334,193,783 bytes**. Every file was restored into ignored storage and checked by size and SHA256 before removal. Historical labels and conclusions were not rewritten.

[Archive index](experiments-2026-11-04.json) · [Member checksums](experiments-2026-11-04.sha256)

| Archived experiment | Retained evidence |
|---|---|
| [dual-score-trial-2026-10-27](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/dual-score-trial-2026-10-27) | Early separate technical/aesthetic schema transport; shared validation was extracted into the scoring package. |
| [eligibility-expanded-120-2026-11-01](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/eligibility-expanded-120-2026-11-01) | Expanded eligibility panels and alternative wording; selected candidates remained experimental. Polarity and photographed-artwork boundaries need care. |
| [eligibility-optimization-2026-11-01](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/eligibility-optimization-2026-11-01) | Earlier decision-model eligibility wording trials. Current deployed questions are identified by the labeling profile. |
| [jev14-2026-10-18](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/jev14-2026-10-18) | Early Jev benchmark and uncertainty evidence. |
| [jev14-calibration-2026-10-18](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/jev14-calibration-2026-10-18) | Historical Jev calibration study, not the current label contract. |
| [photo-scoring-2026-10-31](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/photo-scoring-2026-10-31) | Earlier 2,000-image trial and 72-image model comparison; superseded for current reporting by the 10,000-image run. |
| [prompt-ablation-2026-10-30](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/prompt-ablation-2026-10-30) | Individual and combined prompt additions did not establish reliable improvement; preserve this negative finding. |
| [prompt-calibration-2026-10-18](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/prompt-calibration-2026-10-18) | Early prompt calibration and comparisons. |
| [qalign-inference-2026-10-31](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/qalign-inference-2026-10-31) | Earlier Q-Align inference comparison and runtime investigation. |
| [qrealign-lite-72-2026-10-31](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/qrealign-lite-72-2026-10-31) | Pinned Lite trial and native score mapping. The server remains in labeling/runtime; Lite has no photography-scope gate. |
| [rehearsal-4000-2026-11-01](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/rehearsal-4000-2026-11-01) | Prior corpus and three-arm stats trial. Own-image stats failed the predefined improvement target; swapped stats worsened errors. Stats stay outside default grader prompts. |
| [runinfra-comparison-2026-10-25](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/runinfra-comparison-2026-10-25) | Initial RunInfra model comparison. |
| [runinfra-string-trial-2026-10-26](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/runinfra-string-trial-2026-10-26) | Earlier constrained score-string transport investigation. |

## Recovery

The local archive is `visual-quality/data/experiment-history-2026-11-04/experiments.tar`. A verified extracted copy is already under `visual-quality/data/experiment-history-2026-11-04/tree/`. Current corpus and reuse tools resolve historical inputs there through `visual_quality.scoring.history`. Set `VISUAL_QUALITY_HISTORY` to an alternative directory containing the experiment folders if needed.

From the repository root, restore into an empty destination and check every member:

```sh
REPOSITORY_ROOT="$PWD"
HISTORY_TREE=$(mktemp -d /tmp/model-gym-experiments.XXXXXX)
tar -xf visual-quality/data/experiment-history-2026-11-04/experiments.tar -C "$HISTORY_TREE"
(cd "$HISTORY_TREE" && sha256sum --quiet --check "$REPOSITORY_ROOT/visual-quality/archives/experiments-2026-11-04.sha256")
export VISUAL_QUALITY_HISTORY="$HISTORY_TREE/visual-quality/experiments"
```

Without the local tar, fetch the pinned revision through normal Git history (unshallow a shallow clone if needed), then replace the extraction command with:

```sh
git archive ddcbc4d422b4d4b931bf01b2f00b7938935052fe -- \
  visual-quality/experiments/dual-score-trial-2026-10-27 \
  visual-quality/experiments/eligibility-expanded-120-2026-11-01 \
  visual-quality/experiments/eligibility-optimization-2026-11-01 \
  visual-quality/experiments/jev14-2026-10-18 \
  visual-quality/experiments/jev14-calibration-2026-10-18 \
  visual-quality/experiments/photo-scoring-2026-10-31 \
  visual-quality/experiments/prompt-ablation-2026-10-30 \
  visual-quality/experiments/prompt-calibration-2026-10-18 \
  visual-quality/experiments/qalign-inference-2026-10-31 \
  visual-quality/experiments/qrealign-lite-72-2026-10-31 \
  visual-quality/experiments/rehearsal-4000-2026-11-01 \
  visual-quality/experiments/runinfra-comparison-2026-10-25 \
  visual-quality/experiments/runinfra-string-trial-2026-10-26 \
  | tar -x -C "$HISTORY_TREE"
```

Run the same checksum check afterward. This restores exact evaluated bytes without depending on upstream image URLs. The [first cleanup bundle](README.md) separately contains media and receipts removed before this snapshot. For a complete pre-cleanup checkout, use its pinned `4c644bb` revision.

## Preserved decisions

- Current ensemble: 37.5% Luna + 37.5% DeepSeek + 25% Liquid independently per outcome. Historical two-model metrics describe their original studies.
- Statistics remain debugging metadata. Own/swapped-stats results do not justify feeding them to the default grader or overwriting scores.
- Eligibility distinguishes flat reproductions from photographed objects with physical context. Experimental question variants are not deployed automatically.
- The 10,000-image report is the current comparison; assistant references are not human ratings. Training masks remain disabled.

Runtime helpers and their regression tests moved to the package. The Lite server bytes and health identity were retained. Superseded launchers, variants, media, receipts, candidate data and old tests are recoverable here. Raw-response paths and source hashes still describe their original execution; use a pinned historical checkout for exact replay.

Existing uncommitted EmbeddingGemma work was preserved. Other workspaces and model-training evidence were not deleted.
