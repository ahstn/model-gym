**Keep the baseline Luna/DeepSeek average as the working labeler. Do not add the tested statistics text to its prompt.** The average has the lowest combined-error point estimate on the fresh representative panel, including the 299 images where all five scorers returned both scores. Its advantage over Liquid and Jev remains uncertain; this trial supports a practical default, not a definitive winner over every alternative.

The table shows agreement with the frozen assistant references on the 300-image representative panel. Lower MAE is better. Each model's technical/aesthetic columns use its available cases; the final column uses the same 299 cases for everyone. Scores use the 1–5 scale; aggregate errors retain extra precision for comparison.

| Baseline | Available cases | Technical MAE | Aesthetic MAE | Combined MAE, common 299 |
| --- | ---: | ---: | ---: | ---: |
| Luna + DeepSeek mean | 300 | 0.365 | 0.274 | 0.318 |
| Luna | 300 | 0.351 | 0.327 | 0.337 |
| DeepSeek | 300 | 0.407 | 0.281 | 0.344 |
| Liquid d1.1 | 300 | 0.380 | 0.297 | 0.338 |
| Jev 1.14 | 299 | 0.366 | 0.305 | 0.336 |
| Q-ReAlign Lite | 300 | 0.595 | 1.082 | 0.839 |

Luna has the lowest technical point estimate; averaging improves its aesthetic agreement. The mean's combined MAE is 0.020 lower than Luna and 0.025 lower than DeepSeek on their 300 paired cases. Those exploratory paired intervals exclude zero, but the corresponding intervals for Liquid and Jev include zero. These are small, uncorrected comparisons conditional on assistant judgments. The 100 challenge images and 100 development images are reported separately. “Representative” refers to the declared source frame, not an estimate of every deployment image population.

**The own-stats treatment did not meet the predefined improvement target.** For the mean, technical MAE changes by **+0.014**, with 95% family-bootstrap interval **[−0.013, +0.041]**, against a required reduction of at least 0.050. Aesthetic noninferiority meets its numerical margin, but that does not rescue the failed technical target or the visible regressions. Swapped stats worsen technical MAE by **+0.124 [+0.091, +0.157]**. This demonstrates sensitivity to misleading statistics; it does not establish that models ignored the image.

The exploratory sharpness slice explains part of the tradeoff. Among 69 reference-eligible representative photos below the existing sharpness cutoff, own stats reduce the mean's technical MAE from 0.506 to 0.437. On the other 207 photos, MAE rises from 0.322 to 0.365. This follow-up was added after seeing Luna's results and is descriptive. Low measured sharpness is not itself proof of unwanted blur. Visual adjudication confirms regressions on deliberately selective-focus flowers and a dramatic dark cloud/silhouette scene, alongside some justified improvements.

**Use the statistics for model-specific review triage.** At the fixed 30-image budget, the baseline mean's QC queue finds 8 of 29 large technical errors; random selection would find 2.9 in expectation. Recall is only 28%. The same rule performs poorly for Liquid and Lite, whose error patterns differ. Preserve random unflagged audits and inspect the image before changing a label. Sharp backgrounds can conceal a blurred focal subject, and sharp edges can coexist with clipping or excessive processing.

Q-ReAlign Lite's current outputs need separate scope handling and calibration before they can supply labels under this rubric. Its missing photography-scope gate explains some errors, but not all: on the 276 eligible representative photographs, technical MAE remains 0.481 and aesthetic MAE 1.094, with aesthetic bias −1.089. Its raw scores should not be substituted for the teacher average on the strength of speed or the requested $0.00 reporting override.

Lite does show strong degradation ordering: it scores all 100 severe variants below their parents, and 70 of the 73 fully verified ladders decrease strictly at every step. That supports retaining it as a calibration or training candidate. These controlled comparisons measure response to distortion, separately from agreement with the absolute reference scale and the photography-scope rule.

Before a larger labeling run:

1. Resolve photographic-scope boundaries with human examples, especially photographed flat artwork/cards versus dimensional objects or packaging. The two largest selected technical disagreements include ambiguous reference interpretations; primary labels remain unchanged. A 50-image blind assistant repeat measured mean absolute differences of 0.144 technical and 0.146 aesthetic points, which describes assistant repeatability rather than human agreement.
2. Test the full image plus native-resolution detail crops on a fresh, family-held-out sample, including sharp-background/soft-subject cases and intentional effects. Use the development set to draft changes, freeze them, and validate on fresh images before adoption. Keep the tested stats-text treatment out of the default prompt.
3. Keep EmbeddingGemma 2 clustering, related-image review and family-level separation for corpus organization. Use model disagreement, the QC flags and random audits to choose human review work; retain raw labels and review decisions separately.

The corpus contains 4,000 views from 3,700 source families, with 40 content clusters and 500 independently assessed benchmark images. [Content coverage](content-coverage.md), [the complete gallery](gallery/index.html), [protected-case adjudication](protected-review/report.md) and [selected baseline disagreements](baseline-error-review.md) provide the supporting examples. None of these experimental labels is automatically approved for training.
