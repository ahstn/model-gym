# Expansion to 40,000 labeled images

Decision accepted 2026-11-17: add 20,000 images to the completed 20K corpus, retaining **37.5% Luna + 37.5% DeepSeek + 25% Liquid** independently for technical quality and aesthetics. The [training research and decision](../../../docs/training/qwen-adaptation-review-2026-11-17/README.md) is saved separately. This run labels data; it does not start training or enable training masks.

Completed and audited on 2026-11-17: all 20,000 new images have numerical Luna, DeepSeek, Liquid and ensemble score pairs; Jev has 19,998 pairs and two retained dimension errors. The combined dataset has **39,997 complete ensemble pairs across 40,000 images**, preserving three earlier gaps. The [aggregate report](REPORT.md) records coverage, latency, costs and validation. [Content coverage](CONTENT-COVERAGE.md) records the source mix, theme balance and duplicate-review limitations. Labels remain unapproved for training.

Use the unchanged versioned v2 profile with Luna high, DeepSeek high, Liquid d1.1 and Jev 1.14. All four receive the same exact delivered view; image statistics and existing reference scores are not prompt inputs. No new Q-ReAlign Lite calls or assistant reference assessments are made. Preserve the original 1,500 references, historical Lite comparisons, and first-1,000 Git publication examples unchanged.

## Data locations

Full artifacts live outside Git at `/home/ahstn/git/model-gym/visual-quality/data/labeling-40000-2026-11-17/`:

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

Embedding clients accept `--workers 1..8` (default 4) and an optional immutable `--frame` snapshot. A bounded 64-view probe on the H100 measured 6.08 images/second with eight total clients, versus 2.80 with four, without errors or local-transport fallbacks. This short comparison used different images and ordered-ledger counts, so it is operational evidence rather than a controlled hardware benchmark. This run resumed with eight embedding clients after archiving the incomplete four-client pass; API grader concurrency remains 16. Each completed embedding receipt records the worker count and verifies that its source hash stayed unchanged during execution. Detailed probe and interruption evidence remains in the ignored `embedding-operations/` directory.

Run the existing `stats_run.py` with this checkout's `image-stats/src` and `visual-quality/src` on `PYTHONPATH`. Use `related.py` for cross-family retrieval, then `finalize.py` after scoring. Finalization locks all routes, validates current plan IDs and complete traversal, exports, and audits raw responses and historical invariance. Traversal completion is distinct from complete numerical labels.

Known grading fees are **$33.902338754**, including all baseline attempts and the five bounded DeepSeek recovery calls. The separate compatibility check cost **$0.007057011**, bringing known API fees to **$33.909395765**; 14 failed attempts have unknown charges. The 16 new-image smoke calls were reused in the baseline and are already included. Runpod's latest posted checkpoint is **$6.1674128406448287** for embedding compute and storage; this is not an independently reconciled final account invoice. Both owned temporary pods and their retained disks were deleted with explicit user approval and verified absent. Never count derived ensemble component costs twice.

## Why full runs use 16 workers per route

Sixteen workers is an empirically safer default from the prior 20K expansion, not a published provider limit or a measured CPU/RAM ceiling. Its request receipts show:

| Model | Workers | Attempts | Retryable failures | HTTP 429 responses | Valid-response latency, mean / p95 |
|---|---:|---:|---:|---:|---:|
| Luna, initial phase | 64 | 635 | 162 (25.5%) | 0 | 26.63s / 36.86s |
| Luna, recovery | 16 | 9,530 | 3 (0.031%) | 0 | 13.96s / 22.20s |
| DeepSeek, initial phase | 64 | 715 | 245 (34.3%) | 79 | 32.47s / 48.46s |
| DeepSeek, recovery | 16 | 9,533 | 3 (0.031%) | 0 | 15.24s / 28.15s |

The 429 responses are included in the retryable-failure totals; the remaining failures were transport errors. Observed peak concurrency matched the configured limits. The initial teacher phases left 407 unpriced attempts, versus six across the much larger recovery phases. Liquid and Jev's later 16-worker runs each completed 2,855 attempts without transport errors or 429s; there is no comparable 32-worker measurement for those routes.

Keep 16 for this run and update the completion estimate from production throughput. These initial/recovery periods were not randomized, so they do not establish a hard provider limit or isolate shared client/network pressure. A future bounded 32-worker test could use still-unscored cases under a separately frozen policy, preserving the planned unique assessments, but retries could increase actual attempts and cost. Do not change an active frozen plan to run that experiment. Full aggregate evidence is retained locally at `operations/prior-concurrency-evidence.json`.

## Live compatibility check

Four historical rehearsal views, chosen by source category without consulting their scores, were assessed independently by all four current routes. All 16 observations were valid, with 16 HTTP attempts, no content denials and no unknown charges. Total known cost was **$0.007057011**. Luna and DeepSeek requested high reasoning and returned nonzero reasoning-token usage; providers did not separately echo the effort setting. Existing labels were not modified. This small operational check establishes route/schema compatibility, not accuracy or a guarantee of later request success.

After finalization, `publish_aggregate.py --data <full-local-run> --output-directory <tracked-run-directory>` creates the Git-safe report and summary. It binds publication to the matching export audit and explicitly projects aggregate fields. Never copy the full local summary directly: it includes per-image duplicate-review evidence.

## One retry for an empty token-limit completion

`recover_empty.py` is a separate recovery phase for Luna or DeepSeek after that model's baseline finishes completely. Eligibility requires HTTP 200, the expected model/provider, `finish_reason=length`, no visible answer or refusal, and all 16,384 completion tokens reported as reasoning. It excludes content denials, invalid numeric assessments, unscorable answers, truncated visible answers and any case with a prior valid score. Ordinary baseline retry policy remains unchanged.

The `empty-length-once` plan freezes the eligible subset, original request and response hashes, baseline observation copies, and an exact total-attempt limit per image. Every noneligible observation is hash-pinned. The recovery sends the identical request bytes with the same image, profile, high reasoning and token budget. It appends one attempt and retains every earlier charge and measured latency. Even a recorded transport failure consumes that one attempt; restarting cannot send a second recovery request. An admitted request without a response receipt stops for manual reconciliation and retains an unknown charge.

From this run directory, after reviewing the completed baseline's missing outcomes:

```sh
python recover_empty.py freeze --model deepseek
python recover_empty.py check --model deepseek
# With the already-authorized route credentials sourced privately:
python recover_empty.py run --model deepseek
python finalize.py --phase deepseek=empty-length-once
```

Freezing and checking do not make provider requests. A model with no eligible cases gets no recovery plan and keeps its normal finalization phase. Recovery runs sequentially under the model lock and the existing liability guard; the four ordinary graders keep their frozen 16-worker limits. A repeated cutoff remains a missing assessment, rather than a substituted score. The final report must separate recovery outcomes from baseline outcomes and count all attempts once.

This runner expects the original `expansion` baseline without an intervening recovery phase. If exhausted transport failures also need recovery, review phase ordering and evidence pins before freezing either recovery plan; do not retrofit a frozen plan.

## Observed exceptions

Five DeepSeek baseline responses exhausted the 16,384-token budget entirely on reasoning and returned no visible answer. All five returned valid scores on their single permitted, identical recovery request. The final export keeps both attempts, their charges and their measured HTTP latencies. No successful score was resampled.

Two Jev requests ended with HTTP 400 `Invalid image dimensions`. Both delivered PNGs decode locally and match their frozen hashes. Successful requests include smaller pixel areas and wider aspect ratios, so the available evidence does not establish a reliable size threshold. These two assessments remain missing; images were not resized or resubmitted to work around the rejection. The three ensemble components succeeded for both images.

The code verification at commit `ec233cd890ce35afb42effd6463c01100efb06d1` passed 600 tests and 20 subtests, Ruff lint and formatting, and publication checks. The receipt in `operations/final-code-checks.json` pins the tested sources. The final publisher additionally accepts the existing historical `missing_or_invalid_components` status without altering it; all 12 publisher tests and lint/format checks passed after that change.

Final dataset validation reparsed all 80,000 new raw observations, checked all 20,000 new image/statistics joins, and verified the prior 20K export prefix and original reference/publication files unchanged. The local `audit.json` records the passing checks. The combined `exports/labelled-images.jsonl` SHA256 is `8bae5443fe57a0adc4268a30f0c296108df1cd9b3071d4f31abac0ee9818ee95`. This validates provenance and consistency, not accuracy against new human ratings; no new reference assessments were made.
