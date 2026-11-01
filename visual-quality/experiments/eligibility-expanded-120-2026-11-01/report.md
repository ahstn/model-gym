# Expanded eligibility prompt optimisation: 120 + 120 views

Ten additional eligibility questions were evaluated on 120 development views per model. Each model-specific selection was frozen before the 120-view held-out comparison. On the holdout, the original question, the previously selected prompt, and the new selection were evaluated separately. The quality questions, typed decision schema and 0.50 exclusion threshold were unchanged. Reference eligibility labels are assistant visual assessments, not human ground truth.

Each panel contains 60 eligible photographs and 60 excluded views. Per panel: 30 physical-context photos, 30 other photo scenes, 10 historical flat works, two public-domain museum works, 10 charts, 10 renders, and 28 synthetic screenshots derived from photos in the same panel. Families are split between panels. Screenshot styles are generated from two UI templates per panel and are paired with their source photos; counts are not independent observations and neither the set nor its 50% exclusion share represents real-world prevalence.

The ten development historical cases were part of a previous eligibility trial and include known failures. The ten held-out historical cases were model-scored in the previous 4,000-view corpus, but were not part of the earlier eligibility optimization; they were selected from prior reference classifications rather than model scores. The other sampled families were absent from those runs, but family separation does not prove absence of visually related images. [Met Open Access API](https://metmuseum.github.io/) retrieval encountered HTTP 403s, so only four distinct public-domain museum images were usable. See [protocol](protocol.md), [reference manifest](reference-manifest.json), [prompt variants](variants.json) and [previous prompts](previous-prompts.json).

The first holdout coordinator matched the wrong partition name and scheduled zero jobs. After confirming no holdout requests had been made, the driver filter was corrected without changing the already frozen selection; the corrected stage then attempted all 120 expected images. See [the driver amendment](holdout-driver-amendment.json) and service logs.

## Holdout including historical challenge

The 120 held-out views include the ten previously scored natural cases; the table is intentionally split by category below. A missing or invalid result is never counted as correct.

| Model | Question | Valid | Missed exclusions / valid | Exclusion recall | False floors / valid | False-floor rate |
| --- | --- | --- | --- | --- | --- | --- |
| Liquid | Original | 120/120 | 36/60 | 40.0% | 0/60 | 0.0% |
| Liquid | Previous | 120/120 | 3/60 | 95.0% | 0/60 | 0.0% |
| Liquid | Selected previous | 120/120 | 3/60 | 95.0% | 0/60 | 0.0% |
| Jev | Original | 119/120 | 36/59 | 39.0% | 0/60 | 0.0% |
| Jev | Previous | 119/120 | 5/60 | 91.7% | 0/59 | 0.0% |
| Jev | Selected r14 | 118/120 | 5/59 | 91.5% | 0/59 | 0.0% |

### Paired change against the previous prompt

| Model | Paired cases | Missed exclusions | False floors | Δ misses / Δ false (paired) |
| --- | --- | --- | --- | --- |
| Liquid | 120 | 3 → 3 | 0 → 0 | +0 / +0 (120) |
| Jev | 117 | 4 → 5 | 0 → 0 | +1 / +0 (117) |

For **Liquid**, the selected and previous arms use the same question in separate held-out requests. Any difference between them reflects repeat-request variability or provider behavior, not an improvement from a new prompt.

**Recommendation:** keep Liquid's previous revision for further validation. Jev's r14 improved development errors but did not improve the paired holdout over its previous revision, so retain the previous Jev prompt for now. Neither result is sufficient to deploy an automatic floor without broader independent review.

### Held-out breakdown by reference category

| Model | Category | Original errors / valid | Previous errors / valid | Selected errors / valid | Expected |
| --- | --- | --- | --- | --- | --- |
| Liquid | chart | 0/10 | 0/10 | 0/10 | 10 |
| Liquid | historical_flat | 6/10 | 3/10 | 3/10 | 10 |
| Liquid | museum_flat | 2/2 | 0/2 | 0/2 | 2 |
| Liquid | photo_scene | 0/30 | 0/30 | 0/30 | 30 |
| Liquid | physical_context | 0/30 | 0/30 | 0/30 | 30 |
| Liquid | render | 0/10 | 0/10 | 0/10 | 10 |
| Liquid | synthetic_screenshot | 28/28 | 0/28 | 0/28 | 28 |
| Jev | chart | 0/10 | 0/10 | 0/10 | 10 |
| Jev | historical_flat | 6/9 | 5/10 | 5/9 | 10 |
| Jev | museum_flat | 2/2 | 0/2 | 0/2 | 2 |
| Jev | photo_scene | 0/30 | 0/29 | 0/30 | 30 |
| Jev | physical_context | 0/30 | 0/30 | 0/29 | 30 |
| Jev | render | 0/10 | 0/10 | 0/10 | 10 |
| Jev | synthetic_screenshot | 28/28 | 0/28 | 0/28 | 28 |

For excluded categories an error is a missed exclusion; for eligible photographs it is a false floor. Historical works are a selected previously scored challenge. Synthetic UI templates, chart and render families limit generalization. Even zero false floors among 60 eligible photos is not proof that the rate is negligible.

### Boundary-label sensitivity

Before seeing holdout responses, three historical cases (R0226, R0317 and R0356) were flagged as potentially eligible photos of physical printed material with surrounding context. The main table preserves the prior excluded labels. This secondary calculation treats those three as eligible, leaving 57 excluded and 63 eligible reference cases; it is an alternative interpretation, not a changed ground truth. See [boundary review](boundary-sensitivity.json).

| Model | Prompt | Missed exclusions / valid | False floors / valid |
| --- | --- | --- | --- |
| Liquid | original | 33/57 | 0/63 |
| Liquid | previous | 1/57 | 1/63 |
| Liquid | selected | 1/57 | 1/63 |
| Liquid | paired Δ selected vs previous | +0 / +0 (120) | — |
| Jev | original | 33/56 | 0/63 |
| Jev | previous | 2/57 | 0/62 |
| Jev | selected | 3/57 | 0/61 |
| Jev | paired Δ selected vs previous | +1 / +0 (117) | — |

## Development: ten additional rounds

| Round | Liquid misses / false (valid) | Liquid Δ vs previous (paired) | Liquid qualifies | Jev misses / false (valid) | Jev Δ vs previous (paired) | Jev qualifies |
| --- | --- | --- | --- | --- | --- | --- |
| r11 | 2 / 0 (120/120) | +1 / +0 (120) | no | 3 / 0 (120/120) | -1 / +0 (120) | yes |
| r12 | 3 / 0 (120/120) | +2 / +0 (120) | no | 3 / 0 (120/120) | -1 / +0 (120) | yes |
| r13 | 3 / 0 (120/120) | +2 / +0 (120) | no | 7 / 0 (120/120) | +3 / +0 (120) | no |
| r14 | 1 / 0 (120/120) | +0 / +0 (120) | no | 2 / 0 (120/120) | -2 / +0 (120) | yes |
| r15 | 59 / 59 (120/120) | +58 / +59 (120) | no | 59 / 56 (120/120) | +55 / +56 (120) | no |
| r16 | 2 / 0 (120/120) | +1 / +0 (120) | no | 2 / 0 (119/120) | -2 / +0 (119) | yes |
| r17 | 4 / 0 (120/120) | +3 / +0 (120) | no | 3 / 0 (119/120) | -1 / +0 (119) | yes |
| r18 | 4 / 0 (120/120) | +3 / +0 (120) | no | 31 / 60 (120/120) | +27 / +60 (120) | no |
| r19 | 3 / 0 (120/120) | +2 / +0 (120) | no | 3 / 0 (120/120) | -1 / +0 (120) | yes |
| r20 | 2 / 0 (120/120) | +1 / +0 (120) | no | 37 / 10 (120/120) | +33 / +10 (120) | no |

**Polarity failure:** r15 produced 59 missed exclusions and 59 false floors for Liquid, and 59 missed exclusions and 56 false floors for Jev. Its wording begins by asking whether the image is a photographic scene, then assigns YES to exclusion. The near reversal is consistent with the models following the opening question's polarity instead of the later mapping, but the trial does not establish the cause. Jev's r18 and r20 also regressed sharply. Keep the decision question's polarity explicit and test it; do not assume extra instructions are harmless.

**Liquid:** selected **previous**; 0 new variants met the development guard. Eligibility required at least 114 valid and 57 valid per class, no extra paired false floors, and fewer paired total errors than the previous prompt. Missing candidate responses on previous-valid cases count as errors for ranking. If no variant qualified the previous prompt was retained. The complete frozen selection and timings are in [winner.json](winner.json) and [winner-freeze.json](winner-freeze.json).

**Jev:** selected **r14**; 6 new variants met the development guard. Eligibility required at least 114 valid and 57 valid per class, no extra paired false floors, and fewer paired total errors than the previous prompt. Missing candidate responses on previous-valid cases count as errors for ranking. If no variant qualified the previous prompt was retained. The complete frozen selection and timings are in [winner.json](winner.json) and [winner-freeze.json](winner-freeze.json).

## Quality-score drift on eligible photos

These are changes in native 1–5 scores before applying any floor. No independent technical or aesthetic quality scores were collected here; drift is not a measure of accuracy and may include model stochasticity.

| Model | Score | Paired photos | Mean signed change | Mean absolute change | Changes ≥0.5 |
| --- | --- | --- | --- | --- | --- |
| Liquid | technical_native | 60 | -0.006 | 0.069 | 0 |
| Liquid | aesthetic_native | 60 | 0.013 | 0.063 | 0 |
| Jev | technical_native | 58 | 0.005 | 0.023 | 0 |
| Jev | aesthetic_native | 58 | 0.002 | 0.021 | 0 |

## Latency and cost

Latency is elapsed end-to-end request/response time for valid calls; concurrency was six combined workers. All attempts are counted, including retries. Unpriced failures are unknown rather than zero. Provider-parsed JSON and request/response hashes are preserved; exact original wire text is not retained.

| Model | Attempts | Median valid seconds | Known cost subtotal | Unpriced attempts | Total cost |
| --- | --- | --- | --- | --- | --- |
| Liquid | 1800 | 1.11 | $0.07945284 | 0 | $0.07945284 |
| Jev | 2129 | 0.94 | $0.285588954 | 336 | unknown |

Statuses: retryable_503: 336, valid: 3593.

## Limitations and next step

Do not treat these deliberately balanced, partly synthetic assistant-labeled panels as an estimate of deployment error rates. Before promoting a prompt into automatic labeling, use an independent human-reviewed sample containing more diverse real screenshots, posters, cropped artwork, photographed packaging and ambiguous physical objects; examine both missed exclusions and false floors. Keep original scores and decision evidence available for review.
