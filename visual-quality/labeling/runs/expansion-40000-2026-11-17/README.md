# Expansion to 40,000 labeled images

Decision accepted 2026-11-17: add 20,000 images to the completed 20K corpus, retaining **37.5% Luna + 37.5% DeepSeek + 25% Liquid** independently for technical quality and aesthetics. The [training research and decision](../../../docs/training/qwen-adaptation-review-2026-11-17/README.md) is saved separately. This run labels data; it does not start training or enable training masks.

Status: acquisition and embedding preparation are in progress. Do not interpret this document as a completed 40K dataset. The final aggregate report and audit will record actual coverage and costs.

Use the unchanged versioned v2 profile with Luna high, DeepSeek high, Liquid d1.1 and Jev 1.14. All four receive the same exact delivered view; image statistics and existing reference scores are not prompt inputs. No new Q-ReAlign Lite calls or assistant reference assessments are made. Preserve the original 1,500 references, historical Lite comparisons, and first-1,000 Git publication examples unchanged.

## Data locations

Full artifacts live outside Git at `/home/ahstn/git/model-gym/visual-quality/data/labeling-40000-2026-11-17/`. Once finalization succeeds:

- `exports/labelled-images.jsonl` contains the combined 40K image-level dataset, preserving the prior 20K canonical records byte-for-byte as its prefix.
- `exports/new-labelled-images.jsonl` contains the added 20K records.
- `exports/scores-long.jsonl` and `.csv` contain 211,500 assessor slots: four graders and the ensemble for 40K images, plus the original 10K Lite and 1,500 reference slots. Missing assessments stay missing.
- `manifest.json` identifies the new 20K; `combined-manifest.json` identifies all 40K.
- `stats.jsonl` contains new exact-view statistics, joined by case ID and image hash and stored separately from model responses.
- `related-images.json`, `possible-duplicates.json`, `content-coverage.json` and `review/index.html` support content and duplicate review.
- `results/api/` retains immutable plans, request/response attempts, latency, cost and observations.

The Git publication set remains `labeling/records/`. Do not copy new per-image artifacts into tracked reports, galleries or archives. The completed 20K private R2 snapshot remains unchanged; this new directory is not implicitly a completed remote backup.

## Corpus and execution

Retain the pinned EmbeddingGemma 2 checkpoint, 768-dimensional normalized vectors and 64 original content prototypes. Verify 64 exact-view embeddings against the prior cache before mixing implementations. Exact-view reuse requires matching hashes; controlled variants may use their parent's content vector, explicitly marked as family representatives. Each grader and image-stats still inspects the actual variant.

Sampling does not consult quality scores or captions. Exclude prior source identities, exact images, protected evaluation families and conservative perceptual matches. Use content quotas that account for previous coverage. Source targets are 15,000 Megalith photos, 3,000 VizWiz photos, 500 nonphotographic controls and 1,500 controlled variants from 500 included parents. If fewer controls survive exclusion, fill that shortfall with Megalith photos. Similarity candidates require review; semantic proximity does not prove duplication or equal quality.

The old 22K-row Megalith metadata sample cannot supply enough fresh photos. Widen the candidate pool from the same pinned CC0 repository revision into new-run files, retaining old source manifests and selection evidence. Cached source downloads remain reusable with checksum validation.

After final acquisition, run preparation, embeddings and selection sequentially; overlapping earlier preparation/embedding passes are cache warmups only. Then freeze one API plan per model. A small `run --limit 4 --workers 2` checks each route before a full run; valid smoke observations are reused. Full runs use 16 workers per route. Explicit recovery phases can increase bounded attempt/liability limits without rewriting frozen plans or retrying terminal content denials.

Run the existing `stats_run.py` with this checkout's `image-stats/src` and `visual-quality/src` on `PYTHONPATH`. Use `related.py` for cross-family retrieval, then `finalize.py` after scoring. Finalization locks all routes, validates current plan IDs and complete traversal, exports, and audits raw responses and historical invariance. Traversal completion is distinct from complete numerical labels.

Projected API fees are about **$33.87**, based on the previous added 10K, before unknown charges or changed provider prices. This is an estimate, not the internal conservative liability guard. Actual model fees, unknown attempt charges, HTTP latencies and embedding infrastructure costs must be reported separately. Never count derived ensemble component costs twice.
