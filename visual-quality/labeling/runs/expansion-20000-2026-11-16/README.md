# Expansion to 20,000 labeled images

Decision accepted 2026-11-16. Retain the current ensemble: **37.5% Luna + 37.5% DeepSeek + 25% Liquid**, independently for technical and aesthetic scores. The [500-image comparison](../../trials/decision-comparison-500-2026-11-13/report.md) is a disagreement-enriched diagnostic against assistant references, not an unbiased population benchmark. Neither new Decisions model demonstrated grounds for replacement: Perplexity overestimated technical quality, while Gemini underestimated aesthetics.

Add 10,000 previously unlabeled images with EmbeddingGemma 2 content clustering, content-balanced sampling and related/duplicate review. Score the exact delivered views with Luna high, DeepSeek high, Liquid d1.1 and Jev 1.14 using the unchanged versioned v2 profile. Retain all initial 10K observations, including Q-ReAlign Lite, for historical comparisons; make no additional Q-ReAlign calls. Keep the existing 1,500 assistant references and the first-1,000 publication allowlist unchanged.

Store image-stats measurements separately from model responses; do not feed them to graders or use them to change labels. No training masks are enabled. All new per-image data stays under ignored `visual-quality/data/labeling-20000-2026-11-16/` in the primary checkout. Source code and aggregate reports are tracked here.

Status: all four graders completed the new 10,000-image traversal. The 20K corpus and 40,000 new response audit passed. There are **17,144 complete ensemble score pairs across 20,000 images**; missing provider outcomes remain missing. See [report.md](report.md), [summary.json](summary.json), and [audit.json](audit.json). The temporary embedding pod and retained disk were deleted after explicit user approval (HTTP 204).

## Reproduction and data boundaries

The run scripts bootstrap the frozen acquisition/rendering adapters with an explicit new output root. Existing 10K records are never edited. Full artifacts live at `/home/ahstn/git/model-gym/visual-quality/data/labeling-20000-2026-11-16/`:

- `exports/labelled-images.jsonl`: combined 20K image-level records, once complete.
- `exports/new-labelled-images.jsonl`: only the added 10K records.
- `exports/scores-long.jsonl` and `.csv`: combined assessor rows.
- `manifest.json`: new images; `combined-manifest.json`: original plus new images.
- `stats.jsonl`: new exact-view statistics, separately from inference responses.
- `related-images.json`, `possible-duplicates.json`, and `review/index.html`: cross-family content retrieval and review candidates.
- `results/api/`: frozen plans, attempt receipts and observations for the four graders.

The committed first-1,000 records remain in `labeling/records/`. Do not copy expanded per-image outputs into Git.

Use `corpus.py acquire --source megalith --count 11000 --workers 64` and `--source vizwiz --count 3000 --workers 16` to overdraw candidates; then `corpus.py prepare --workers 12`. `embed.py --endpoint <authenticated-local-tunnel>` uses the pinned worker and a 64-image numerical-overlap check. Run `selection.py` after all selected candidates have vectors. Source targets are 7,500 Megalith photos, 1,500 VizWiz photos, 250 controls and 750 controlled variants from 250 included parents. If fewer controls survive exclusions, the predeclared fallback fills their quota with Megalith photos. Controlled variants share their parent's content embedding, explicitly identified as family representatives; each grader and the statistics package receives the actual variant view.

Selection assigns candidates to normalized mean prototypes of the original 64 content groups. Square-root quotas account for initial content counts as well as the candidate pool. It does not inspect quality scores, captions or reference labels. Conservative exact/perceptual exclusions happen before selection. Similarity pairs remain review candidates, not confirmed duplicate labels.

After freezing selection, run `score.py freeze --model <luna|deepseek|liquid|jev>` and `score.py run --model <model>` with credentials sourced quietly. An initial `--limit 8` checks each frozen route; rerunning resumes without repeating completed calls. The unchanged `stats_run.py` accepts `--manifest`, `--output`, `--cache` and `--workers`, with this checkout's `image-stats/src` and `visual-quality/src` on `PYTHONPATH`. Finish with `export.py`, `related.py`, then `audit.py`.

For this run's recovery phases below, `finalize.py --wait` holds off until all four workers release their locks and report complete traversal. It then exports and audits locally, without making API calls. `operations/finalization-status.json` records whether that step is waiting, audited or failed. An exhausted transport failure remains an explicitly missing assessment; a pending or unfinalized observation prevents export. The final report's means and latency medians use available outcome pairs, and unknown API charges remain unknown.

## Compatibility check

Sixteen preliminary requests reused four initial-run views without changing their stored labels. Luna and DeepSeek returned four valid results each. Liquid and Jev returned three each and denied one inline public-photo payload with HTTP 403 (`Request blocked by content filter`). These rejected attempts are retained with unknown cost, not assigned zero. [OpenRouter's documented image format](https://openrouter.ai/docs/guides/community/multimodal-decisions) requires inline data for these two models; external image URLs are not supported for them.

A content-filter denial is terminal for that image and is not retried or bypassed. It no longer masquerades as a shared credential failure that stops unrelated images. Credential/billing errors still stop the runner; the aggregate failure circuit breaker remains enabled. Missing components always leave the corresponding ensemble outcome missing. Session accounting includes the compatibility check separately from dataset labeling.

The first selected-image checks confirmed the same rejection on 3/8 Liquid cases; Jev's original shared failure circuit stopped after 35 observations, including six image denials. `decision_continuation.py` freezes a separate operational phase with identical image bytes, rubrics, model settings and outcome parsing. It reuses every completed result, does not retry a denied image, and excludes explicit per-image denials from the shared service-outage circuit. Other credential, billing and service-failure stops remain active. Full coverage may therefore contain missing Liquid/Jev outcomes and missing ensembles; report those counts rather than claiming complete labels. The original phase receipts and source pins remain immutable.

The initial 64-worker teacher phase produced connection errors and RunInfra HTTP429 responses. Conservative guards stopped it because unknown charges reserve the maximum allowed request liability, not because known charges approached the guard. `teacher_continuation.py` freezes `workers16-recovery` with the historical 16-worker concurrency, retaining all attempts and valid results. Previously exhausted transport failures receive at most two further attempts (five total for Luna, seven for DeepSeek); valid/invalid assessments and explicit content denials are never retried. Internal liability guards are $120 per teacher; this is not a claim about actual spending. Report known receipts and unknown attempts separately. Latency observations mix operational phases and are not an isolated model-speed benchmark.

## Secret-guardrail recovery

The user disabled their secret-redaction guardrail. A recovery script is prepared for the 2,855 previously blocked requests per decision model, preserving image bytes, prompts, successful assessments and all original receipts. Earlier exports are retained in `snapshots/before-guardrail-recovery/`. Exact rejection categories are 2,757 compressed Bitcoin-key patterns, 85 uncompressed Bitcoin-key patterns and 13 Google API-key patterns. The user explicitly approved resending these flagged payloads after automatic review requested clarification. Recovery is in progress. `guardrail_recovery.py freeze` requires `resubmission_approved=true` in the local authorization record before creating a plan. A fresh 403 stops the recovery.
