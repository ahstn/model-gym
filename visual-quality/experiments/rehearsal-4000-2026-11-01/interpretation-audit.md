# Interpretation audit

Completed 2026-11-01T16:22:22.980717+00:00 against the frozen final benchmark snapshot while nonbenchmark recovery continued. This audit does not certify final operational coverage, final billing, publication artifacts or completion of the full run.

**No numerical or interpretation blocker found in the audited findings and report renderer.** No scoring call was made and no primary label, model output or analysis implementation was changed. This is a read-only interpretation audit with this document as its sole output.

## Direct checks

- Verified the final snapshot SHA256, 6,500 unique rows, 500 reference cases, 300 distinct representative source families and the 299-case intersection with complete scores from all five scorers.
- Recomputed all six headline baseline MAEs and common-case combined errors from the raw score rows using Decimal arithmetic. The reported table agrees at its displayed precision. The mean is the exact 50/50 arithmetic mean of Luna and DeepSeek for every benchmark case, arm and task.
- Recomputed the five representative baseline-versus-mean combined paired bootstrap intervals independently from raw rows, using 10,000 resamples and the declared family ordering/seed. The intervals match the saved analysis. Gains over Luna and DeepSeek exclude zero; Liquid and Jev include zero. The report calls these small, exploratory, uncorrected comparisons conditional on assistant labels.
- Recomputed the mean's own-stats and swapped-stats technical/aesthetic paired intervals and one-sided aesthetic upper bounds. The technical target fails for own stats; aesthetic noninferiority meets its numerical margin. The positive swapped technical interval supports sensitivity to misleading measurements, without proving that the image was ignored.
- Confirmed all 6,000 non-Jev benchmark rows match the initial snapshot exactly. Thus the earlier Luna/DeepSeek/Liquid/Lite, derived-mean and reference diagnostic inputs remain applicable to the final benchmark.
- Directly counted 29 representative baseline-mean technical disagreements of at least 0.75 points. The frozen QC selection contains 30 images and eight such disagreements, including six reserved random unflagged audits. The random-selection expectation is 30 × 29 / 300 = 2.9. The report preserves the low recall and model-specific limits instead of claiming comprehensive or universal QC benefit.
- Verified 276 reference-eligible representative photos, split into 69 at or below the existing sharpness percentile cutoff and 207 above it. Mean technical MAE changes 0.505797→0.436957 in the lower band and 0.322222→0.364734 elsewhere. The report explicitly identifies this follow-up as posthoc/descriptive and avoids equating low measured sharpness with unintended blur.
- Confirmed the Lite eligible-photo figures: technical MAE 0.481422, aesthetic MAE 1.094247, aesthetic bias −1.088923. The renderer distinguishes native-on-eligible results from all-image scope performance and does not call these task-available subsets an all-model common ranking.
- Checked all 76 protected-review baseline/treatment score pairs against final raw benchmark rows. Verdict totals are 17 visibly justified, 13 regressions and 46 ambiguous. The report identifies nonblind opinions, dependence of derived-mean comparisons, uncertain scope cases and unchanged primary labels.
- Confirmed corpus counts of 4,000 views, 3,700 families and 40 full-corpus clusters. The source method records the representative 80%/16%/4% source frame and separates the 100 challenge and 100 development cases; the findings do not claim deployment-population representativeness.

## Interpretation boundaries retained

The recommendation to keep the baseline average is presented as a working default, not proof of superiority over every competitor. The proposed crop experiment is future work, not a tested improvement. The 50-image repeat is described as assistant repeatability rather than human validation. The selected-error review retains R0317/R0356 as ambiguous scope interpretations while leaving their original labels unchanged. Neither posthoc extreme-case selection nor completed protected review is treated as a new unbiased evaluation set or approval for training.

Final assembly must still join the protected adjudication against the complete export and retain the explicit two exhausted Jev benchmark gaps. Later nonbenchmark recovery cannot justify changing these frozen benchmark conclusions.

## Audited input hashes

| Input | SHA256 |
| --- | --- |
| /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01/final-findings.md | `45b66225f597c99e67d1eb2531cc3e1aea28fca57eccf3d9726406f5cc8245fa` |
| /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01/report.py | `6cbdfade3f1fbed64a4272429f999678d4ad3fbd35481013ba4875b3fefddd2b` |
| /home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/results/api/postpass/final-benchmark-snapshot/scores-long.jsonl | `5e83ad19bd55e5a7aac36fef1e92c974a658f15005f22558c9e80e517c5d07cf` |
| /home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/results/api/postpass/initial-snapshot/scores-long.jsonl | `d225fc99d2b32b365d9dab961f1d3e0154c441319c0e79988f2e3a2838513059` |
| /home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/progress-snapshots/final-benchmark-analysis.json | `2033ff2dba338d6acaa7e700dd70855a524a9a1317ed55e232b07394749281d9` |
| /home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/progress-snapshots/initial-benchmark-diagnostics.json | `db3b6e06f4b78707e50f3c0f114572837cd309622add81549b80a1a70a7eebce` |
| /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01/baseline-error-review.md | `d4ba044264361f37b19922867c14848c93c156fb707c33bf1593c577444acaf5` |
| /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01/baseline-error-review.json | `16dd727e379525093622f5818c3fa251a33ce59ccb2478109514baeaf0883d4d` |
| /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01/protected-review-initial-snapshot/report.md | `aa442299d51a70847c1bc83e82112ae9fbd63a7a6bf5fccaa0648023e0d10740` |
| /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01/protected-adjudication/luna-reviewed.json | `7f144280d3c4be802946a53eca4f1563b7b842bb597893c36fa9cec206a1433a` |
| /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01/protected-adjudication/deepseek-mean-reviewed.json | `e96be71eaff0f165c7471ae6c798adce0f9465cc76dd6f185cd945af7b7d180f` |
| /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01/reference-reliability/analysis.json | `7b5e4853b26c7a301f457bf56efe11e66eb45d35779a763b3a664325f653b63f` |
| /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01/corpus-method.md | `68b0cdbb9321f133e6e741aaf059941bf4c42884146148134c00ac430c09686f` |
| /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01/protocol.md | `78d72eb887feb52dc169e2f7cd06db3167f0d933415620cc036f197b2c4af1f0` |
