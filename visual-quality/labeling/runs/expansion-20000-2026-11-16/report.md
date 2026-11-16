# 20,000-image labeling expansion

The existing ensemble remains 37.5% Luna, 37.5% DeepSeek and 25% Liquid. The new 10,000 images use the unchanged v2 profile. The initial 10K observations are appended byte-for-byte; no additional Q-ReAlign inference or reference assessments were requested. The existing 1,500 assistant references and first-1,000 publication allowlist remain unchanged.

All new per-image data is local and ignored by Git. Image statistics are metadata only. Training masks remain disabled.

## Corpus and review

The added set contains 7,500 Megalith images, 1,500 VizWiz images, 250 non-photographic controls and 750 controlled variants from 250 included parents. Exact delivered-image hashes are unique across all 20,000 records; new source families are disjoint from the original set and protected evaluation families were excluded before selection.

EmbeddingGemma 2 revision `f9972ac22e696a105c482202646979e33e4b5a1e` assigns new candidates to the original 64 content prototypes. Square-root sampling quotas account for existing cluster coverage without reading quality labels. Of the new records, 9,111 use exact-view embeddings and 889 explicitly reuse a source-family embedding for a resized or distorted view. Graders and image-stats always receive the actual delivered view.

Six cross-family content neighbors are surfaced per image where available. There are 339 high-similarity pair candidates across the combined corpus; they are unconfirmed review candidates, not automatic duplicate labels. The local `review/index.html` shows content counts and the highest-similarity pairs. All candidates remain in `possible-duplicates.json`.

The original 1,500 assistant references remain the benchmark. No new reference judgments were requested, and no additional Q-ReAlign calls were made. The first 1,000 committed examples remain unchanged.

## Incremental model coverage

| Model | Valid score pairs | Mean tech | Mean aest | Median valid HTTP seconds |
|---|---:|---:|---:|---:|
| luna | 10,000 / 10,000 | 3.546 | 3.378 | 15.28 |
| deepseek | 9,998 / 10,000 | 3.679 | 3.293 | 15.41 |
| liquid | 10,000 / 10,000 | 3.301 | 3.004 | 9.90 |
| jev | 9,999 / 10,000 | 3.472 | 2.986 | 9.50 |
| photo_ensemble | 9,998 / 10,000 | 3.535 | 3.253 | — |

These are score distributions, not accuracy measurements: no new independent reference scores were created. Means use each model's available score pairs, so differing missing coverage also changes the evaluated population. The original benchmark remains the comparison baseline. Missing or invalid outcomes remain missing; the ensemble never renormalizes its weights.

Latency is the final successful HTTP attempt through complete response receipt. It excludes payload preparation and earlier retries. Raw records retain every attempt's latency and costs. Concurrency and service conditions changed during this run, so these measurements are operational observations rather than a controlled speed comparison.

## Operational limitations

DeepSeek returned 9,998 numeric pairs, one schema-valid unscorable response, and one invalid aesthetic response retaining its technical score. The added set has 9,998 complete ensemble pairs; the combined set has 19,997.

The initial teacher phase encountered connection errors and rate limits. A separately frozen continuation reduced concurrency to 16 and allowed bounded additional transient retries while preserving valid responses and all unknown-charge reserves. The rubric, image bytes, high reasoning and response validation were unchanged.

The first pass retained 2,855 shared OpenRouter secret-scanner denials per decision model: 2,757 compressed Bitcoin-key patterns, 85 uncompressed Bitcoin-key patterns and 13 Google API-key patterns. The user subsequently disabled their secret-redaction guardrail and explicitly approved resending these flagged image payloads. The recovery uses identical images and prompts, keeps all original receipts, preserves other assessments, and stops on a new 403. Earlier exports remain in the local pre-recovery snapshot. Unknown original charges remain unknown.

Recovery results: liquid: complete, {'valid': 2854, 'failed': 1}, additional known cost $0.149796; jev: complete, {'valid': 2855}, additional known cost $0.541120

A separate, single-request HTTP520 recovery is recorded where needed: liquid: complete, {'valid': 1}

Jev's original HTTP400 `Invalid image dimensions` rejection is outside the secret-scanner recovery and remains missing. The exact view stays unchanged across graders. Current coverage and any additional failures are retained in the aggregate JSON.

## Files and infrastructure

Full local data: `/home/ahstn/git/model-gym/visual-quality/data/labeling-20000-2026-11-16`. The combined image records are `exports/labelled-images.jsonl`; `exports/new-labelled-images.jsonl` contains only the added 10K. `exports/scores-long.jsonl` and `.csv` contain assessor rows. `stats.jsonl` contains new image measurements, also joined separately into image-level exports. None of the added per-image labels is committed to Git.

The owned embedding GPU was stopped after vectors were saved. Runpod reported $0.627130 for that pod as of 2026-11-16 21:15:10 UTC; billing may settle later. This infrastructure charge is separate from the API table below and does not change the historical Q-ReAlign/reference zero-price reporting rule.
The temporary embedding pod and disk have been removed after explicit authorization; all required outputs remain local.

## Total API cost by model

Known subtotals include all recorded attempts and retries: initial 10K, added 10K and the 16-call compatibility probe. Unpriced attempts are excluded from these subtotals, so a total with unpriced calls is a lower bound. The ensemble has no independent charge. Embedding infrastructure is reported separately.

| Model | Initial 10K known USD | Added 10K known USD | Probe known USD | Combined known USD | Unpriced attempts |
|---|---:|---:|---:|---:|---:|
| luna | $8.093621 | $8.123164 | $0.003304 | $16.220088 | 165 |
| deepseek | $6.552428 | $6.489999 | $0.004032 | $13.046459 | 329 |
| liquid | $0.502199 | $0.503338 | $0.000171 | $1.005708 | 2896 |
| jev | $1.813157 | $1.816232 | $0.000620 | $3.630009 | 4877 |
| Q-ReAlign Lite (historical only) | $0.00 | — | — | $0.00 | 0 |
| Reference (existing 1,500) | $0.00 | — | — | $0.00 | 0 |
