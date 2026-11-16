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
| liquid | 7,145 / 10,000 | 3.270 | 2.975 | 11.75 |
| jev | 7,144 / 10,000 | 3.443 | 2.958 | 11.39 |
| photo_ensemble | 7,145 / 10,000 | 3.505 | 3.225 | — |

These are score distributions, not accuracy measurements: no new independent reference scores were created. Means use each model's available score pairs, so differing missing coverage also changes the evaluated population. The original benchmark remains the comparison baseline. Missing or invalid outcomes remain missing; the ensemble never renormalizes its weights.

Latency is the final successful HTTP attempt through complete response receipt. It excludes payload preparation and earlier retries. Raw records retain every attempt's latency and costs. Concurrency and service conditions changed during this run, so these measurements are operational observations rather than a controlled speed comparison.

DeepSeek produced 9,999 schema-valid responses: 9,998 numeric pairs and one unscorable response. One additional response failed aesthetic evidence validation while retaining its technical score. Both affected images also lack Liquid scores, so the added set has 7,145 complete ensemble pairs; the combined set has 17,144.

## Operational limitations

The initial teacher phase encountered connection errors and rate limits. A separately frozen continuation reduced concurrency to 16 and allowed bounded additional transient retries while preserving valid responses and all unknown-charge reserves. The rubric, image bytes, high reasoning and response validation were unchanged.

OpenRouter returned per-image HTTP403 content-filter denials for some Liquid/Jev inline-image requests. These were retained as missing assessments and not retried or bypassed. Their counts are included in the aggregate JSON. Other credential, billing and service-failure stops remained active. A missing Liquid outcome also leaves its ensemble outcome missing.

The completed Decisions failure audit found 2,855 identical image denials in each model. Jev additionally rejected one delivered view with HTTP400, `Invalid image dimensions`. The exact view remains unchanged across graders, and that Jev assessment is missing.

## Files and infrastructure

Full local data: `/home/ahstn/git/model-gym/visual-quality/data/labeling-20000-2026-11-16`. The combined image records are `exports/labelled-images.jsonl`; `exports/new-labelled-images.jsonl` contains only the added 10K. `exports/scores-long.jsonl` and `.csv` contain assessor rows. `stats.jsonl` contains new image measurements, also joined separately into image-level exports. None of the added per-image labels is committed to Git.

The owned embedding GPU was stopped after vectors were saved. Runpod reported $0.627130 for that pod as of 2026-11-16 16:52:12 UTC; billing may settle later. This infrastructure charge is separate from the API table below and does not change the historical Q-ReAlign/reference zero-price reporting rule.
The pod's disk remains retained while permission for permanent deletion is pending; storage charges may continue. Automatic approval review rejected irreversible disk deletion without explicit authorization.

## Total API cost by model

Known subtotals include all recorded attempts and retries: initial 10K, added 10K and the 16-call compatibility probe. Unpriced attempts are excluded from these subtotals, so a total with unpriced calls is a lower bound. The ensemble has no independent charge. Embedding infrastructure is reported separately.

| Model | Initial 10K known USD | Added 10K known USD | Probe known USD | Combined known USD | Unpriced attempts |
|---|---:|---:|---:|---:|---:|
| luna | $8.093621 | $8.123164 | $0.003304 | $16.220088 | 165 |
| deepseek | $6.552428 | $6.489999 | $0.004032 | $13.046459 | 329 |
| liquid | $0.502199 | $0.353542 | $0.000171 | $0.855912 | 2895 |
| jev | $1.813157 | $1.275112 | $0.000620 | $3.088890 | 4877 |
| Q-ReAlign Lite (historical only) | $0.00 | — | — | $0.00 | 0 |
| Reference (existing 1,500) | $0.00 | — | — | $0.00 | 0 |
