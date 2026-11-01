## Diagnostic review of large baseline disagreements

This posthoc review inspected the eight largest representative and four largest challenge disagreements in baseline ensemble technical score, with case ID breaking ties. All 12 exact delivered originals were opened. Selection used the frozen reference and model scores; it is not a fresh evaluation set, and comparisons here do not establish an overall model winner. Statistics were consulted after the image observations were saved. Primary references and model scores remain unchanged.

Ten reference defect or scope interpretations were visibly supported; the two largest disagreements, **R0317 and R0356**, remain ambiguous at the boundary between an artwork reproduction and a photographed physical object. “Supported” concerns the visible interpretation, not proof of the precise numeric reference score. Both ambiguous cases retain their original excluded/1.00 labels in every primary result.

Scores below are **technical / aesthetic**. Native Liquid and Lite values are rounded for display only. Jev is omitted here because its recovery was still pending in the pinned snapshot.

| Case | Reference | Luna + DeepSeek | Luna | DeepSeek | Liquid | Lite |
| --- | --- | --- | --- | --- | --- | --- |
| R0317 | 1.00 / 1.00 | 3.95 / 3.55 | 3.90 / 3.40 | 4.00 / 3.70 | 3.32 / 2.92 | 4.37 / 2.68 |
| R0356 | 1.00 / 1.00 | 3.45 / 3.30 | 3.50 / 3.60 | 3.40 / 3.00 | 3.47 / 2.85 | 3.97 / 1.86 |
| R0357 | 1.00 / 1.50 | 3.35 / 3.60 | 2.90 / 3.70 | 3.80 / 3.50 | 1.29 / 1.86 | 1.59 / 1.48 |
| R0302 | 2.70 / 3.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 3.31 / 3.18 | 3.99 / 2.20 |
| R0206 | 2.20 / 3.20 | 3.75 / 3.90 | 3.50 / 3.80 | 4.00 / 4.00 | 3.89 / 3.86 | 4.00 / 2.56 |
| R0260 | 2.50 / 3.10 | 3.75 / 3.55 | 3.50 / 3.40 | 4.00 / 3.70 | 2.90 / 2.98 | 3.19 / 1.47 |
| R0200 | 2.80 / 2.80 | 3.90 / 3.80 | 3.80 / 3.60 | 4.00 / 4.00 | 3.22 / 3.02 | 3.03 / 2.02 |
| R0109 | 2.40 / 2.20 | 3.40 / 2.70 | 3.10 / 2.60 | 3.70 / 2.80 | 2.96 / 2.57 | 3.01 / 1.98 |
| R0463 | 1.00 / 1.00 | 2.90 / 2.75 | 2.50 / 2.70 | 3.30 / 2.80 | 2.87 / 2.15 | 1.92 / 1.02 |
| R0444 | 3.00 / 3.70 | 4.05 / 4.15 | 4.00 / 4.10 | 4.10 / 4.20 | 3.30 / 3.48 | 4.02 / 2.01 |
| R0481 | 1.80 / 2.40 | 2.80 / 3.25 | 2.60 / 3.20 | 3.00 / 3.30 | 1.70 / 2.01 | 1.46 / 1.07 |
| R0490 | 3.10 / 3.70 | 4.00 / 3.65 | 3.60 / 3.50 | 4.40 / 3.80 | 3.89 / 3.52 | 4.00 / 1.96 |

**Scope needs human boundary examples.** R0317 shows a cyan graphic on visibly bent, torn cardboard; R0356 shows a diagonal cartoon card with wood and reflected light around it. The near-full-frame reproduction clause and photographed-object exception support competing interpretations. R0302 is clearer: the lettering protrudes and casts shadows, and the star is visibly dimensional, yet both teachers apply the non-photographic floor. R0463 is dominated by a software interface and supports the frozen near-full-frame interface rule, while its bezel makes it another useful boundary-calibration example.

**Recognizing the subject still masks lost detail.** R0260 and R0444 have readable locomotive lettering but soft mechanical detail; R0200 has a dark, soft pilot face; R0490 has soft porcelain decoration. In R0109 the tree is textured while the focal bird is blurred. These observations support inspecting the important subject rather than treating scene recognition or a sharp incidental region as sufficient for a good technical score.

**Not every large error is blur.** R0206 retains crisp flower outlines but has harsh tonal and color transitions and lost petal/stem detail. Its sharpness peak is at the 90.43rd percentile, so the low-sharpness cue misses it. R0357 is almost entirely a clipped white glow and darkness; both teachers describe abstraction while assigning relatively high technical scores. The existing uniform-frame gate flags it. R0481 has visible grain and motion softness, yet its noise statistic is only at the 42.15th percentile; its low-sharpness flag catches it instead.

The unchanged QC rule flags **7 of these 12 selected disagreements**, but the full-panel 10% review budgets select only **3**: R0357 and R0200 through defect flags, plus R0317 through the reserved random unflagged audit. These selected-extreme counts are not global precision or recall. The sharp tree in R0109 keeps peak sharpness above the bottom quarter; the erroneous 1.00 floor on R0302 bypasses the “score at least 2” softness condition. Stats therefore need semantic review and a random audit allocation.

| Case | Peak-sharpness percentile | Noise percentile | Shadow QC flag | Included in full-panel review budget |
| --- | ---: | ---: | --- | --- |
| R0317 | 38.48 | 78.38 | None | Yes: random unflagged audit |
| R0356 | 65.55 | 34.99 | None | No |
| R0357 | 2.00 | 13.01 | gate, soft | Yes: flagged |
| R0302 | 19.19 | 60.72 | None | No |
| R0206 | 90.43 | 36.78 | None | No |
| R0260 | 8.90 | 16.41 | soft | No |
| R0200 | 3.61 | 47.47 | soft | Yes: flagged |
| R0109 | 34.09 | 72.06 | None | No |
| R0463 | 60.26 | 95.21 | noisy | No |
| R0444 | 22.27 | 39.57 | soft | No |
| R0481 | 20.29 | 42.15 | soft | No |
| R0490 | 10.55 | 67.22 | soft | No |

Before a larger labelling run, obtain human decisions on physical-card/sign/screen versus reproduction examples, retain focal-region and tonal-damage cases in the review set, and keep the original benchmark labels frozen for this report. Do not fit a new threshold to these 12 extremes. Full image evidence, reference records, native scores, source hashes and QC memberships are in [baseline-error-review.json](baseline-error-review.json).
