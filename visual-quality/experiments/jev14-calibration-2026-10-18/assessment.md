# Decision after held-out validation

Keep Jev as an advisory signal. The selected rule does not establish the requested 1% false-positive limit with 95% confidence, and it detects only about half of the generated images. Do not promote it as a reliable filter for the pilot.

The run completed 600 calibration calls and 1,000 held-out validation calls. The two sets use different caption families from MS COCOAI. All 1,600 responses were valid and priced, for a total cost of **$0.110866392**.

| Rule | False flags among 500 source-real images | One-sided 95% FPR upper bound | AI images detected | AI recall |
| --- | ---: | ---: | ---: | ---: |
| Fixed probability ≥ 0.50 | 167 (33.4%) | 37.04% | 480/500 | 96.0% |
| Selected probability ≥ 0.78 | 3 (0.6%) | 1.54% | 259/500 | 51.8% |
| Fixed probability ≥ 0.80 | 2 (0.4%) | 1.25% | 198/500 | 39.6% |

The 0.78 rule was selected on calibration alone and locked before validation began. It fails both the exact false-positive bound and the provisional 70% recall requirement. The 0.50 and 0.80 results are predeclared comparators; neither was chosen as a replacement after validation. The observed 0.6% rate must not be presented as proof that the true rate is below 1%.

Performance also differs by generator. At the selected threshold, recall ranges from **27% for Midjourney-v6** to **89% for DALL-E3**. The overall result depends on this five-generator mixture. It does not establish performance on new generators or a different photo source.

## What happened to the confidence filter

Requiring uncertainty to be `low` or `medium` did not improve the best calibration rule. At threshold 0.78, that condition removed 18 correct AI detections and left all three calibration false flags.

The three held-out false flags also reported `low` or `medium` uncertainty:

| Case | AI probability | Reported uncertainty | Native confidence in that uncertainty choice |
| --- | ---: | --- | ---: |
| `validation/mscocoai-0010-real` | 0.80 | low | 0.06 |
| `validation/mscocoai-0128-real` | 0.78 | medium | 0.13 |
| `validation/mscocoai-0380-real` | 0.80 | low | 0.11 |

Thus, excluding only `high` uncertainty would not remove these new errors. Their low native choice-confidence values are a possible feature for another study. They are not probabilities that the AI-origin decision is correct. Choosing a new cutoff from these validation errors would be further tuning and would need fresh validation data.

## Source-label limit found during error review

Visual inspection of the three false flags found one black-and-white sheep scene with colored geometric overlays, a church facade, and an unfinished interior with toilets. The first is visibly edited or composited. The dataset's source-real class must therefore not be read as a guarantee of untouched camera photographs. The rates above use the author's origin labels and keep all cases in the original denominators.

This inspection did not change labels or remove errors. Appearance alone cannot prove how an image was made. The [error audit](false-positive-audit.json) preserves the three native answers, image hashes, and descriptive observations. The frozen report uses the shorter term “real photos” for the source-real label; this distinction limits that wording.

## Recorded evidence and next decision

The [full report](results/README.md) includes exact intervals, generator results, cost, and latency. The [calibration operating points](calibration-tradeoffs.md) compare 0.5%, 1%, and 2% empirical targets. The [plot](results/calibration-curve.svg) shows the calibration tradeoff.

The next useful test is a different detector or an independently planned combination of scores, using a new held-out cohort. Raising Jev's probability threshold alone reduces coverage sharply on this set. Keep origin screening separate from technical quality and aesthetics labels.
