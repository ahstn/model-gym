# Four-thousand-image rehearsal results

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

Analysis generated 2026-11-01T16:25:47.469387+00:00. The frozen corpus has 4,000 image views across 3,700 source families. There are 300 representative, 100 challenge and 100 development reference cases. Reference rows recorded as valid: 500 / 500.

Recorded completed scoring responses: **21,986 / 22,000** expected model/arm/image slots. Completion includes invalid schema responses; numeric-label coverage is shown below. 14 slots lack a completed scoring response. The CSV contains 27,500 rows including explicit missing slots and derived means.

Reference scores are independent assistant visual judgments, not human MOS or verified human ground truth. Lower MAE means closer agreement with those judgments. All labels remain experimental. The Luna/DeepSeek mean is derived from its two components.

[Full scores](scores-long.csv) · [Detailed analysis](analysis.json) · [Per-model costs](cost-per-model.csv)

## Completion and validity

| Model | Arm | Expected | Attempted images | Completed responses | Numeric tech | Numeric aest | Missing/error | All attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | Baseline | 4000 | 4000 | 3999 | 3997 | 3997 | 1 | 4001 |
| DeepSeek V4.1 Flash 1017 | Own stats | 500 | 500 | 500 | 500 | 500 | 0 | 500 |
| DeepSeek V4.1 Flash 1017 | Swapped stats | 500 | 500 | 500 | 500 | 500 | 0 | 500 |
| Jev 1.14 | Baseline | 4000 | 4000 | 3987 | 3987 | 3987 | 13 | 4736 |
| Liquid d1.1 | Baseline | 4000 | 4000 | 4000 | 4000 | 4000 | 0 | 4004 |
| GPT-6 Luna | Baseline | 4000 | 4000 | 4000 | 4000 | 4000 | 0 | 4016 |
| GPT-6 Luna | Own stats | 500 | 500 | 500 | 500 | 500 | 0 | 500 |
| GPT-6 Luna | Swapped stats | 500 | 500 | 500 | 500 | 500 | 0 | 500 |
| Q-ReAlign Lite | Baseline | 4000 | 4000 | 4000 | 4000 | 4000 | 0 | 4000 |

Each task retains its own valid cases. Combined MAE uses cases with both scores available. The additional five-baseline common subset supports a like-for-like comparison; it does not change the primary paired treatment population. Signed bias is prediction minus reference. Spearman ρ is rank agreement.

## Representative benchmark (300 images)

| Model | Tech MAE (n) | Aest MAE (n) | Tech bias | Aest bias | ρ tech / aest | Common-task MAE (n) | Five-baseline common MAE (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | 0.35 (300) | 0.33 (300) | 0.16 | 0.22 | 0.89 / 0.84 | 0.34 (300) | 0.34 (299) |
| DeepSeek V4.1 Flash 1017 | 0.41 (300) | 0.28 (300) | 0.17 | 0.05 | 0.83 / 0.83 | 0.34 (300) | 0.34 (299) |
| Jev 1.14 | 0.37 (299) | 0.31 (299) | 0.10 | -0.17 | 0.85 / 0.87 | 0.34 (299) | 0.34 (299) |
| Liquid d1.1 | 0.38 (300) | 0.30 (300) | -0.04 | -0.14 | 0.86 / 0.86 | 0.34 (300) | 0.34 (299) |
| Q-ReAlign Lite | 0.60 (300) | 1.08 (300) | 0.51 | -0.93 | 0.83 / 0.67 | 0.84 (300) | 0.84 (299) |
| Luna + DeepSeek mean | 0.37 (300) | 0.27 (300) | 0.16 | 0.13 | 0.89 / 0.87 | 0.32 (300) | 0.32 (299) |

On the task-specific all-five common subsets, an invalid aesthetic score does not remove an otherwise valid technical score (and vice versa):

| Model | Common tech MAE (n) | Common aest MAE (n) | Tech bias | Aest bias | ρ tech / aest |
| --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | 0.35 (299) | 0.32 (299) | 0.15 | 0.21 | 0.89 / 0.84 |
| DeepSeek V4.1 Flash 1017 | 0.41 (299) | 0.28 (299) | 0.17 | 0.05 | 0.83 / 0.83 |
| Jev 1.14 | 0.37 (299) | 0.31 (299) | 0.10 | -0.17 | 0.85 / 0.87 |
| Liquid d1.1 | 0.38 (299) | 0.30 (299) | -0.04 | -0.14 | 0.86 / 0.86 |
| Q-ReAlign Lite | 0.60 (299) | 1.08 (299) | 0.51 | -0.93 | 0.83 / 0.67 |
| Luna + DeepSeek mean | 0.37 (299) | 0.27 (299) | 0.16 | 0.13 | 0.89 / 0.87 |

## Challenge benchmark (100 images)

| Model | Tech MAE (n) | Aest MAE (n) | Tech bias | Aest bias | ρ tech / aest | Common-task MAE (n) | Five-baseline common MAE (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | 0.33 (100) | 0.29 (100) | 0.15 | 0.23 | 0.94 / 0.94 | 0.31 (100) | 0.31 (99) |
| DeepSeek V4.1 Flash 1017 | 0.36 (100) | 0.23 (100) | 0.19 | 0.07 | 0.88 / 0.94 | 0.30 (100) | 0.29 (99) |
| Jev 1.14 | 0.33 (99) | 0.24 (99) | 0.11 | -0.15 | 0.92 / 0.93 | 0.28 (99) | 0.28 (99) |
| Liquid d1.1 | 0.33 (100) | 0.23 (100) | 0.01 | -0.08 | 0.91 / 0.93 | 0.28 (100) | 0.28 (99) |
| Q-ReAlign Lite | 0.76 (100) | 1.13 (100) | 0.71 | -0.68 | 0.77 / 0.44 | 0.95 (100) | 0.95 (99) |
| Luna + DeepSeek mean | 0.33 (100) | 0.23 (100) | 0.17 | 0.15 | 0.94 / 0.95 | 0.28 (100) | 0.28 (99) |

On the task-specific all-five common subsets, an invalid aesthetic score does not remove an otherwise valid technical score (and vice versa):

| Model | Common tech MAE (n) | Common aest MAE (n) | Tech bias | Aest bias | ρ tech / aest |
| --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | 0.32 (99) | 0.29 (99) | 0.16 | 0.24 | 0.95 / 0.94 |
| DeepSeek V4.1 Flash 1017 | 0.36 (99) | 0.23 (99) | 0.20 | 0.08 | 0.89 / 0.94 |
| Jev 1.14 | 0.33 (99) | 0.24 (99) | 0.11 | -0.15 | 0.92 / 0.93 |
| Liquid d1.1 | 0.33 (99) | 0.23 (99) | 0.01 | -0.08 | 0.92 / 0.94 |
| Q-ReAlign Lite | 0.77 (99) | 1.12 (99) | 0.71 | -0.67 | 0.77 / 0.44 |
| Luna + DeepSeek mean | 0.32 (99) | 0.23 (99) | 0.18 | 0.16 | 0.94 / 0.96 |

Development results (100 images), pooled evaluation results and full-corpus score agreement are retained in analysis.json. Agreement on the 3,500 nonbenchmark images does not measure accuracy because they have no independent reference labels.

## Exploratory baseline comparisons

Each difference is the individual baseline's MAE minus the fixed Luna/DeepSeek mean's MAE, on the same valid cases. Negative values favor the individual point estimate; positive values favor the ensemble. Cells show ΔMAE, paired source-family 95% interval and paired image count. Technical and aesthetic comparisons retain their own valid cases; combined error requires both tasks. These contrasts are exploratory and uncorrected for multiple comparisons, conditional on assistant labels. Small point-estimate gaps alone do not establish a meaningful winner; intervals spanning zero leave the direction unresolved.

| Panel | Individual baseline | Tech ΔMAE [CI] (n) | Aest ΔMAE [CI] (n) | Combined ΔMAE [CI] (n) |
| --- | --- | --- | --- | --- |
| Representative | GPT-6 Luna | -0.014 [-0.033, 0.005] (300) | 0.053 [0.034, 0.073] (300) | 0.020 [0.004, 0.036] (300) |
| Representative | DeepSeek V4.1 Flash 1017 | 0.042 [0.023, 0.061] (300) | 0.007 [-0.012, 0.026] (300) | 0.025 [0.009, 0.039] (300) |
| Representative | Jev 1.14 | 0.001 [-0.035, 0.041] (299) | 0.033 [-0.005, 0.073] (299) | 0.017 [-0.015, 0.052] (299) |
| Representative | Liquid d1.1 | 0.015 [-0.026, 0.055] (300) | 0.023 [-0.018, 0.063] (300) | 0.019 [-0.018, 0.055] (300) |
| Representative | Q-ReAlign Lite | 0.230 [0.158, 0.307] (300) | 0.808 [0.737, 0.877] (300) | 0.519 [0.461, 0.577] (300) |
| Challenge | GPT-6 Luna | -0.000 [-0.034, 0.035] (100) | 0.062 [0.032, 0.094] (100) | 0.031 [0.003, 0.060] (100) |
| Challenge | DeepSeek V4.1 Flash 1017 | 0.033 [-0.001, 0.065] (100) | -0.001 [-0.031, 0.027] (100) | 0.016 [-0.009, 0.039] (100) |
| Challenge | Jev 1.14 | 0.001 [-0.046, 0.047] (99) | 0.008 [-0.052, 0.068] (99) | 0.005 [-0.037, 0.046] (99) |
| Challenge | Liquid d1.1 | 0.006 [-0.043, 0.054] (100) | -0.004 [-0.057, 0.050] (100) | 0.001 [-0.043, 0.045] (100) |
| Challenge | Q-ReAlign Lite | 0.435 [0.244, 0.635] (100) | 0.895 [0.757, 1.020] (100) | 0.665 [0.525, 0.806] (100) |

## Native scores on reference-eligible photographs

These descriptive results restrict the reference to eligible photographs and use each model's native scores before any separate scope adjustment. They separate photographic quality agreement from failures to reject excluded graphics. Each task retains its own available cases, shown in parentheses; these are not all-five common-case rankings. Q-ReAlign Lite has no photography-scope gate, so its all-image error includes a different failure mode from photographic quality error. No labels are altered by this diagnostic.

| Panel | Model | Tech MAE (n) | Aest MAE (n) | Tech bias | Aest bias | ρ tech / aest |
| --- | --- | --- | --- | --- | --- | --- |
| Representative | GPT-6 Luna | 0.35 (276) | 0.32 (276) | 0.13 | 0.20 | 0.87 / 0.82 |
| Representative | DeepSeek V4.1 Flash 1017 | 0.42 (276) | 0.28 (276) | 0.16 | 0.04 | 0.79 / 0.79 |
| Representative | Jev 1.14 | 0.35 (275) | 0.30 (275) | 0.06 | -0.22 | 0.83 / 0.86 |
| Representative | Liquid d1.1 | 0.37 (276) | 0.29 (276) | -0.09 | -0.18 | 0.84 / 0.83 |
| Representative | Q-ReAlign Lite | 0.48 (276) | 1.09 (276) | 0.39 | -1.09 | 0.83 / 0.74 |
| Representative | Luna + DeepSeek mean | 0.37 (276) | 0.27 (276) | 0.15 | 0.12 | 0.87 / 0.85 |
| Challenge | GPT-6 Luna | 0.36 (83) | 0.32 (83) | 0.16 | 0.24 | 0.91 / 0.90 |
| Challenge | DeepSeek V4.1 Flash 1017 | 0.40 (83) | 0.25 (83) | 0.20 | 0.07 | 0.81 / 0.90 |
| Challenge | Jev 1.14 | 0.37 (82) | 0.27 (82) | 0.11 | -0.20 | 0.87 / 0.90 |
| Challenge | Liquid d1.1 | 0.37 (83) | 0.26 (83) | -0.02 | -0.11 | 0.86 / 0.89 |
| Challenge | Q-ReAlign Lite | 0.44 (83) | 1.07 (83) | 0.37 | -1.07 | 0.90 / 0.83 |
| Challenge | Luna + DeepSeek mean | 0.36 (83) | 0.25 (83) | 0.18 | 0.16 | 0.90 / 0.93 |

## Primary stats comparison

The primary comparison is own stats versus baseline for the fixed 50/50 Luna/DeepSeek mean on the representative panel. Negative differences mean lower MAE against the assistant references. Each task uses its own valid paired cases; an unrelated challenger's missing label does not remove a case.

| Treatment | Tech paired n | Families | Δ tech MAE | 95% family CI | Δ aest MAE | 95% family CI | Aest one-sided 95% upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Own stats | 300 | 300 | 0.014 | [-0.013, 0.041] | -0.006 | [-0.027, 0.015] | 0.012 |
| Swapped stats | 300 | 300 | 0.124 | [0.091, 0.157] | 0.012 | [-0.009, 0.033] | 0.030 |

| Treatment | Task | Expected | Numeric reference | Paired | Coverage fraction | Baseline unusable | Treatment unusable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Own stats | Technical | 300 | 300 | 300 | 1.00 | 0 | 0 |
| Own stats | Aesthetic | 300 | 300 | 300 | 1.00 | 0 | 0 |
| Swapped stats | Technical | 300 | 300 | 300 | 1.00 | 0 | 0 |
| Swapped stats | Aesthetic | 300 | 300 | 300 | 1.00 | 0 | 0 |

Coverage and missingness gate: **Met**. Each task requires at least 95% paired coverage and no increase in nonnumeric or invalid scores versus baseline. Incomplete comparisons retain their numerical estimates for inspection but cannot support advancement.

Numerical technical target: **Not met** (MAE reduction ≥0.05 and two-sided 95% upper bound below zero). Numerical aesthetic noninferiority: **Met** (one-sided 95% upper bound ≤+0.03). These numerical tests are separate from the coverage gate. The numerical analysis does not adjudicate protected slices; any completed visual review is reported separately below. Neither a numerical pass nor completion of that review automatically approves labels for training.

Bootstrap resamples whole source families: repeats 10000; seed(s) 20261101. The intervals are conditional on assistant labels. Swapped stats are an exploratory sensitivity check; sensitivity does not prove the image was ignored. Individual-model, challenge and development contrasts remain secondary.

Individual-model effects below are secondary comparisons. Failure to establish aesthetic noninferiority is not itself evidence of aesthetic harm; the estimates and intervals must also be considered.

| Model | Arm | Tech ΔMAE [95% CI] | Aest ΔMAE [95% CI] | Aest noninferiority |
| --- | --- | --- | --- | --- |
| GPT-6 Luna | Own stats | 0.008 [-0.021, 0.036] | -0.009 [-0.033, 0.015] | Met |
| GPT-6 Luna | Swapped stats | 0.087 [0.056, 0.119] | 0.010 [-0.014, 0.034] | Met |
| DeepSeek V4.1 Flash 1017 | Own stats | 0.011 [-0.024, 0.047] | 0.008 [-0.022, 0.036] | Not met |
| DeepSeek V4.1 Flash 1017 | Swapped stats | 0.151 [0.108, 0.194] | 0.015 [-0.014, 0.043] | Not met |

## Scope diagnostics

The reference photography-scope rule floors clearly excluded content at 1.00 for both dimensions. Declared floors on reference-eligible images and scores above 1.00 on reference-excluded images are shown separately. Missing scores are not evidence that the floor was respected. Lite retains its native outputs and has no added eligibility floor.

| Panel | Model | Declared floor / eligible | Either >1 / excluded | Both numeric / excluded |
| --- | --- | --- | --- | --- |
| Representative | GPT-6 Luna | 1 / 276 | 3 / 18 | 18 / 18 |
| Representative | DeepSeek V4.1 Flash 1017 | 1 / 276 | 3 / 18 | 18 / 18 |
| Representative | Jev 1.14 | 0 / 276 | 6 / 18 | 18 / 18 |
| Representative | Liquid d1.1 | 0 / 276 | 6 / 18 | 18 / 18 |
| Representative | Q-ReAlign Lite | 0 / 276 | 18 / 18 | 18 / 18 |
| Representative | Luna + DeepSeek mean | 1 / 276 | 4 / 18 | 18 / 18 |
| Challenge | GPT-6 Luna | 0 / 83 | 1 / 15 | 15 / 15 |
| Challenge | DeepSeek V4.1 Flash 1017 | 0 / 83 | 1 / 15 | 15 / 15 |
| Challenge | Jev 1.14 | 0 / 83 | 1 / 15 | 15 / 15 |
| Challenge | Liquid d1.1 | 0 / 83 | 1 / 15 | 15 / 15 |
| Challenge | Q-ReAlign Lite | 0 / 83 | 15 / 15 | 15 / 15 |
| Challenge | Luna + DeepSeek mean | 0 / 83 | 1 / 15 | 15 / 15 |

Reference-uncertain cases and native-score agreement on eligible images are preserved in analysis.json. These scope counts do not establish performance on every intentional-effects or other protected slice.

## QC review trial

QC flags select images for review without changing a label. The fixed budget is 10% of the panel, capped by valid cases with stats. Twenty percent of review slots (rounded up) are reserved for random unflagged audits; slots move between pools only when a pool cannot fill its allocation. A large error is absolute technical error ≥0.75. Percentiles come from the frozen nonbenchmark calibration families. The table below is the representative baseline panel; challenge and treatment diagnostics remain in analysis.json.

| Model | Valid cases | Used / planned reviews | Large errors found | Precision | Recall | Random expected hits | Reserved / actual audits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | 300 | 30 / 30 | 8 | 0.27 | 0.20 | 4.00 | 6 / 6 |
| Jev 1.14 | 299 | 30 / 30 | 4 | 0.13 | 0.12 | 3.31 | 6 / 6 |
| Liquid d1.1 | 300 | 30 / 30 | 1 | 0.03 | 0.03 | 3.00 | 6 / 6 |
| GPT-6 Luna | 300 | 30 / 30 | 10 | 0.33 | 0.36 | 2.80 | 6 / 6 |
| Luna + DeepSeek mean | 300 | 30 / 30 | 8 | 0.27 | 0.28 | 2.90 | 6 / 6 |
| Q-ReAlign Lite | 300 | 30 / 30 | 3 | 0.10 | 0.04 | 7.60 | 6 / 6 |

## Controlled degradation ladders

All controlled families are retained. These diagnostics describe whether technical scores decrease through the three severities; aesthetic monotonicity is not required. Ties and strict inversions are separate counts. Some mild changes are not visibly worse, and a photography-scope floor can legitimately produce ties. Positive parent-minus-severity-3 values mean the original scored higher. Independently verified comparisons, when available, appear in the supplemental review below.

| Model | Complete / families | Non-increasing | Adjacent ties | Strict inversions | Severe ≥ parent | Mean parent − severe |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | 100 / 100 | 79 | 39 | 22 | 6 | 1.28 |
| DeepSeek V4.1 Flash 1017 | 100 / 100 | 62 | 38 | 43 | 5 | 1.43 |
| Jev 1.14 | 100 / 100 | 67 | 4 | 34 | 0 | 1.46 |
| Liquid d1.1 | 100 / 100 | 65 | 0 | 39 | 2 | 1.35 |
| Q-ReAlign Lite | 100 / 100 | 95 | 0 | 6 | 0 | 2.26 |
| Luna + DeepSeek mean | 100 / 100 | 72 | 19 | 30 | 4 | 1.35 |

This table retains the complete family set independently of the supplemental visual review. Parent-to-distortion comparison is not PACE ground-truth-threshold Holistic Override Rate.

## End-to-end request latency

Seconds from HTTP invocation through the complete response include upload, network/proxy, provider queue and processing. Local scheduler waiting and payload preparation are separate. This table uses completed scoring responses; all-attempt distributions and failures remain in analysis.json. Routes used different concurrency, so this is not a controlled production-speed comparison. Reference review duration is separate from HTTP latency; derived means have no new request.

| Model | Arm | Measured responses | Mean s | Median s | P95 s | Unmeasured attempts |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | Baseline | 3999 | 11.05 | 9.73 | 21.54 | 0 |
| DeepSeek V4.1 Flash 1017 | Own stats | 500 | 10.56 | 8.49 | 24.05 | 0 |
| DeepSeek V4.1 Flash 1017 | Swapped stats | 500 | 11.00 | 9.00 | 24.75 | 0 |
| Jev 1.14 | Baseline | 3987 | 1.58 | 1.28 | 3.96 | 1 |
| Liquid d1.1 | Baseline | 4000 | 3.10 | 3.00 | 5.49 | 4 |
| GPT-6 Luna | Baseline | 4000 | 8.77 | 8.69 | 12.15 | 16 |
| GPT-6 Luna | Own stats | 500 | 8.92 | 9.02 | 12.14 | 0 |
| GPT-6 Luna | Swapped stats | 500 | 9.14 | 9.27 | 12.58 | 0 |
| Q-ReAlign Lite | Baseline | 4000 | 2.58 | 2.04 | 5.16 | 0 |

## Limitations and next decisions

The reference sample is selected from this corpus and represents assistant preferences. The bootstrap conditions on those labels and does not measure human-rating uncertainty. Challenge and development panels are separate from the representative primary population. Secondary comparisons are exploratory; the report does not claim that every quality issue or protected slice is unaffected. Missing or invalid labels reduce coverage and are not silently replaced. Controlled ladder results do not by themselves verify that every delivered distortion is perceptually worse. A larger labeling decision should consider the primary paired effect, aesthetic noninferiority, scope regressions, error review and coverage together.

## Blind repeat of 50 assistant reference assessments

A fresh reviewer inspected 50 individual images using the same rubric, without primary labels, model scores or image statistics. The fixed hash selection comprises 10 development, 30 representative and 10 challenge cases. All original primary labels remain unchanged. This measures assistant repeatability, not agreement with human ratings.

| Panel | Images | Technical mean absolute difference | Aesthetic mean absolute difference | Repeat technical bias | Repeat aesthetic bias | Eligibility agreement |
|---|---:|---:|---:|---:|---:|---:|
| all | 50 | 0.144 | 0.146 | -0.072 | -0.110 | 50/50 |
| development | 10 | 0.150 | 0.140 | -0.110 | -0.060 | 10/10 |
| representative | 30 | 0.150 | 0.143 | -0.043 | -0.110 | 30/30 |
| challenge | 10 | 0.120 | 0.160 | -0.120 | -0.160 | 10/10 |

There were no photography-floor decision disagreements and no per-task absolute differences of 0.75 points or more. Median absolute disagreement was 0.10 on both tasks; the 90th percentiles were 0.31 technical and 0.30 aesthetic. Repeats tended to score lower, especially for aesthetics (−0.11 on average). The small panel and assistant-only reviewers limit any claim about reference reliability. These estimates do not adjust or subtract from grader error.

Review elapsed time is recorded separately from HTTP latency, which is not applicable to assistant visual assessment. Each reference has the requested $0.00 price.

## Exploratory sharpness slices

This descriptive follow-up was added after inspecting Luna's completed results. It uses the existing frozen QC cutoff, without tuning a new threshold. The tables restrict the representative panel to reference-eligible photographs; all-case, challenge, development and aesthetic results remain in sharpness-slices.json. Low measured detail can reflect shallow focus, silhouettes or smooth subjects, so the lower band is not a verified blur label.

| Baseline | Lower-band n | Tech MAE | Tech bias | Other n | Tech MAE | Tech bias |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | 69 | 0.490 | 0.371 | 207 | 0.298 | 0.055 |
| DeepSeek V4.1 Flash 1017 | 69 | 0.548 | 0.484 | 207 | 0.378 | 0.052 |
| Jev 1.14 | 69 | 0.451 | 0.321 | 206 | 0.321 | -0.027 |
| Liquid d1.1 | 69 | 0.331 | 0.222 | 207 | 0.388 | -0.187 |
| Q-ReAlign Lite | 69 | 0.445 | 0.404 | 207 | 0.494 | 0.388 |
| Luna + DeepSeek mean | 69 | 0.506 | 0.428 | 207 | 0.322 | 0.053 |

Paired technical MAE changes below are treatment minus baseline; negative values are improvements. These subgroup point estimates have no multiplicity-adjusted inference and do not replace the primary 300-image comparison.

| Model | Arm | Lower-band paired n | ΔMAE | Other paired n | ΔMAE |
| --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | own_stats | 69 | -0.057 | 207 | 0.037 |
| GPT-6 Luna | swapped_stats | 69 | 0.103 | 207 | 0.096 |
| DeepSeek V4.1 Flash 1017 | own_stats | 69 | -0.078 | 207 | 0.032 |
| DeepSeek V4.1 Flash 1017 | swapped_stats | 69 | 0.129 | 207 | 0.176 |
| Luna + DeepSeek mean | own_stats | 69 | -0.069 | 207 | 0.043 |
| Luna + DeepSeek mean | swapped_stats | 69 | 0.119 | 207 | 0.142 |

## Supplemental blind review of controlled degradation

Supplemental descriptive technical-score diagnostics; all 100 controlled families remain in the original analysis. Blind categorical stimulus reviews select additional comparisons, never numeric labels. A comparison is verified only for clear_degradation. Parent comparisons, severe-versus-parent comparisons and fully verified adjacent ladders have separate denominators. Positive mean drop is comparison technical score minus derivative technical score. Native precision is retained; absolute differences <=1e-8 are ties. Exact (1,1) floor pairs are reported separately and can be policy-correct for art/nonphotographic content; neither ties nor every severe score at least as high as its parent is automatically an error. No aesthetic monotonicity is required. This is not PACE ground-truth-threshold Holistic Override Rate.

The audit covers all 100 families (400 images), 600 recorded entries and 500 unique image pairs. V1’s parent and previous comparisons name the same pair. All 100 families remain in the primary diagnostics; no label changed.

All three derivatives are verified against the parent in 73 families; 73 families have all three adjacent comparisons verified; 100 severe-versus-parent comparisons are verified. These are distinct subsets. Review classifications and per-pair reasons are preserved in `controlled-review/audit.json` and `controlled-analysis.json`.

Unique-pair review classifications: clear_degradation: 473; not_clearly_worse: 27; unjudgeable: 0.

### All derivatives versus parent

↓ / = / ↑ count strict decreases, ties and strict increases. Floor counts are included in ties. Drop uses all complete pairs, including ties.

| Model | Operator | Review subset | Complete/expected | Missing | ↓ / = / ↑ | Floor 1 | Mean technical drop |
|---|---|---|---:|---:|---:|---:|---:|
| luna | all | all | 300/300 | 0 | 255 / 31 / 14 | 7 | 0.8613 |
| luna | gaussian_blur | all | 102/102 | 0 | 96 / 5 / 1 | 5 | 1.4225 |
| luna | gaussian_noise | all | 99/99 | 0 | 88 / 8 / 3 | 2 | 0.8303 |
| luna | jpeg_compression | all | 99/99 | 0 | 71 / 18 / 10 | 0 | 0.3141 |
| deepseek | all | all | 300/300 | 0 | 227 / 43 / 30 | 7 | 0.8037 |
| deepseek | gaussian_blur | all | 102/102 | 0 | 93 / 7 / 2 | 6 | 1.6284 |
| deepseek | gaussian_noise | all | 99/99 | 0 | 69 / 18 / 12 | 1 | 0.3697 |
| deepseek | jpeg_compression | all | 99/99 | 0 | 65 / 18 / 16 | 0 | 0.3879 |
| jev | all | all | 300/300 | 0 | 260 / 1 / 39 | 0 | 0.9142 |
| jev | gaussian_blur | all | 102/102 | 0 | 102 / 0 / 0 | 0 | 1.8994 |
| jev | gaussian_noise | all | 99/99 | 0 | 92 / 0 / 7 | 0 | 0.5759 |
| jev | jpeg_compression | all | 99/99 | 0 | 66 / 1 / 32 | 0 | 0.2376 |
| liquid | all | all | 300/300 | 0 | 260 / 0 / 40 | 0 | 0.9161 |
| liquid | gaussian_blur | all | 102/102 | 0 | 102 / 0 / 0 | 0 | 2.0130 |
| liquid | gaussian_noise | all | 99/99 | 0 | 83 / 0 / 16 | 0 | 0.4255 |
| liquid | jpeg_compression | all | 99/99 | 0 | 75 / 0 / 24 | 0 | 0.2765 |
| qrealign_lite | all | all | 300/300 | 0 | 294 / 0 / 6 | 0 | 1.3054 |
| qrealign_lite | gaussian_blur | all | 102/102 | 0 | 102 / 0 / 0 | 0 | 2.1652 |
| qrealign_lite | gaussian_noise | all | 99/99 | 0 | 96 / 0 / 3 | 0 | 0.8650 |
| qrealign_lite | jpeg_compression | all | 99/99 | 0 | 96 / 0 / 3 | 0 | 0.8599 |
| luna_deepseek_mean | all | all | 300/300 | 0 | 260 / 20 / 20 | 6 | 0.8325 |
| luna_deepseek_mean | gaussian_blur | all | 102/102 | 0 | 96 / 5 / 1 | 5 | 1.5255 |
| luna_deepseek_mean | gaussian_noise | all | 99/99 | 0 | 88 / 5 / 6 | 1 | 0.6000 |
| luna_deepseek_mean | jpeg_compression | all | 99/99 | 0 | 76 / 10 / 13 | 0 | 0.3510 |
| luna | all | visually_verified | 273/273 | 0 | 240 / 26 / 7 | 7 | 0.9388 |
| luna | gaussian_blur | visually_verified | 102/102 | 0 | 96 / 5 / 1 | 5 | 1.4225 |
| luna | gaussian_noise | visually_verified | 99/99 | 0 | 88 / 8 / 3 | 2 | 0.8303 |
| luna | jpeg_compression | visually_verified | 72/72 | 0 | 56 / 13 / 3 | 0 | 0.4028 |
| deepseek | all | visually_verified | 273/273 | 0 | 214 / 38 / 21 | 7 | 0.8725 |
| deepseek | gaussian_blur | visually_verified | 102/102 | 0 | 93 / 7 / 2 | 6 | 1.6284 |
| deepseek | gaussian_noise | visually_verified | 99/99 | 0 | 69 / 18 / 12 | 1 | 0.3697 |
| deepseek | jpeg_compression | visually_verified | 72/72 | 0 | 52 / 13 / 7 | 0 | 0.4931 |
| jev | all | visually_verified | 273/273 | 0 | 249 / 1 / 23 | 0 | 1.0055 |
| jev | gaussian_blur | visually_verified | 102/102 | 0 | 102 / 0 / 0 | 0 | 1.8994 |
| jev | gaussian_noise | visually_verified | 99/99 | 0 | 92 / 0 / 7 | 0 | 0.5759 |
| jev | jpeg_compression | visually_verified | 72/72 | 0 | 55 / 1 / 16 | 0 | 0.3299 |
| liquid | all | visually_verified | 273/273 | 0 | 245 / 0 / 28 | 0 | 1.0045 |
| liquid | gaussian_blur | visually_verified | 102/102 | 0 | 102 / 0 / 0 | 0 | 2.0130 |
| liquid | gaussian_noise | visually_verified | 99/99 | 0 | 83 / 0 / 16 | 0 | 0.4255 |
| liquid | jpeg_compression | visually_verified | 72/72 | 0 | 60 / 0 / 12 | 0 | 0.3720 |
| qrealign_lite | all | visually_verified | 273/273 | 0 | 269 / 0 / 4 | 0 | 1.4224 |
| qrealign_lite | gaussian_blur | visually_verified | 102/102 | 0 | 102 / 0 / 0 | 0 | 2.1652 |
| qrealign_lite | gaussian_noise | visually_verified | 99/99 | 0 | 96 / 0 / 3 | 0 | 0.8650 |
| qrealign_lite | jpeg_compression | visually_verified | 72/72 | 0 | 71 / 0 / 1 | 0 | 1.1366 |
| luna_deepseek_mean | all | visually_verified | 273/273 | 0 | 241 / 18 / 14 | 6 | 0.9057 |
| luna_deepseek_mean | gaussian_blur | visually_verified | 102/102 | 0 | 96 / 5 / 1 | 5 | 1.5255 |
| luna_deepseek_mean | gaussian_noise | visually_verified | 99/99 | 0 | 88 / 5 / 6 | 1 | 0.6000 |
| luna_deepseek_mean | jpeg_compression | visually_verified | 72/72 | 0 | 57 / 8 / 7 | 0 | 0.4479 |

### Severe derivative versus parent

↓ / = / ↑ count strict decreases, ties and strict increases. Floor counts are included in ties. Drop uses all complete pairs, including ties.

| Model | Operator | Review subset | Complete/expected | Missing | ↓ / = / ↑ | Floor 1 | Mean technical drop |
|---|---|---|---:|---:|---:|---:|---:|
| luna | all | all | 100/100 | 0 | 94 / 5 / 1 | 2 | 1.2770 |
| luna | gaussian_blur | all | 34/34 | 0 | 32 / 1 / 1 | 1 | 1.7794 |
| luna | gaussian_noise | all | 33/33 | 0 | 31 / 2 / 0 | 1 | 1.3091 |
| luna | jpeg_compression | all | 33/33 | 0 | 31 / 2 / 0 | 0 | 0.7273 |
| deepseek | all | all | 100/100 | 0 | 95 / 4 / 1 | 2 | 1.4330 |
| deepseek | gaussian_blur | all | 34/34 | 0 | 32 / 2 / 0 | 2 | 2.4647 |
| deepseek | gaussian_noise | all | 33/33 | 0 | 31 / 1 / 1 | 0 | 0.7697 |
| deepseek | jpeg_compression | all | 33/33 | 0 | 32 / 1 / 0 | 0 | 1.0333 |
| jev | all | all | 100/100 | 0 | 100 / 0 / 0 | 0 | 1.4602 |
| jev | gaussian_blur | all | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.5712 |
| jev | gaussian_noise | all | 33/33 | 0 | 33 / 0 / 0 | 0 | 1.1109 |
| jev | jpeg_compression | all | 33/33 | 0 | 33 / 0 / 0 | 0 | 0.6648 |
| liquid | all | all | 100/100 | 0 | 98 / 0 / 2 | 0 | 1.3466 |
| liquid | gaussian_blur | all | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.3747 |
| liquid | gaussian_noise | all | 33/33 | 0 | 32 / 0 / 1 | 0 | 0.8996 |
| liquid | jpeg_compression | all | 33/33 | 0 | 32 / 0 / 1 | 0 | 0.7343 |
| qrealign_lite | all | all | 100/100 | 0 | 100 / 0 / 0 | 0 | 2.2624 |
| qrealign_lite | gaussian_blur | all | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.7556 |
| qrealign_lite | gaussian_noise | all | 33/33 | 0 | 33 / 0 / 0 | 0 | 1.8527 |
| qrealign_lite | jpeg_compression | all | 33/33 | 0 | 33 / 0 / 0 | 0 | 2.1640 |
| luna_deepseek_mean | all | all | 100/100 | 0 | 96 / 2 / 2 | 1 | 1.3550 |
| luna_deepseek_mean | gaussian_blur | all | 34/34 | 0 | 32 / 1 / 1 | 1 | 2.1221 |
| luna_deepseek_mean | gaussian_noise | all | 33/33 | 0 | 32 / 0 / 1 | 0 | 1.0394 |
| luna_deepseek_mean | jpeg_compression | all | 33/33 | 0 | 32 / 1 / 0 | 0 | 0.8803 |
| luna | all | visually_verified | 100/100 | 0 | 94 / 5 / 1 | 2 | 1.2770 |
| luna | gaussian_blur | visually_verified | 34/34 | 0 | 32 / 1 / 1 | 1 | 1.7794 |
| luna | gaussian_noise | visually_verified | 33/33 | 0 | 31 / 2 / 0 | 1 | 1.3091 |
| luna | jpeg_compression | visually_verified | 33/33 | 0 | 31 / 2 / 0 | 0 | 0.7273 |
| deepseek | all | visually_verified | 100/100 | 0 | 95 / 4 / 1 | 2 | 1.4330 |
| deepseek | gaussian_blur | visually_verified | 34/34 | 0 | 32 / 2 / 0 | 2 | 2.4647 |
| deepseek | gaussian_noise | visually_verified | 33/33 | 0 | 31 / 1 / 1 | 0 | 0.7697 |
| deepseek | jpeg_compression | visually_verified | 33/33 | 0 | 32 / 1 / 0 | 0 | 1.0333 |
| jev | all | visually_verified | 100/100 | 0 | 100 / 0 / 0 | 0 | 1.4602 |
| jev | gaussian_blur | visually_verified | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.5712 |
| jev | gaussian_noise | visually_verified | 33/33 | 0 | 33 / 0 / 0 | 0 | 1.1109 |
| jev | jpeg_compression | visually_verified | 33/33 | 0 | 33 / 0 / 0 | 0 | 0.6648 |
| liquid | all | visually_verified | 100/100 | 0 | 98 / 0 / 2 | 0 | 1.3466 |
| liquid | gaussian_blur | visually_verified | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.3747 |
| liquid | gaussian_noise | visually_verified | 33/33 | 0 | 32 / 0 / 1 | 0 | 0.8996 |
| liquid | jpeg_compression | visually_verified | 33/33 | 0 | 32 / 0 / 1 | 0 | 0.7343 |
| qrealign_lite | all | visually_verified | 100/100 | 0 | 100 / 0 / 0 | 0 | 2.2624 |
| qrealign_lite | gaussian_blur | visually_verified | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.7556 |
| qrealign_lite | gaussian_noise | visually_verified | 33/33 | 0 | 33 / 0 / 0 | 0 | 1.8527 |
| qrealign_lite | jpeg_compression | visually_verified | 33/33 | 0 | 33 / 0 / 0 | 0 | 2.1640 |
| luna_deepseek_mean | all | visually_verified | 100/100 | 0 | 96 / 2 / 2 | 1 | 1.3550 |
| luna_deepseek_mean | gaussian_blur | visually_verified | 34/34 | 0 | 32 / 1 / 1 | 1 | 2.1221 |
| luna_deepseek_mean | gaussian_noise | visually_verified | 33/33 | 0 | 32 / 0 / 1 | 0 | 1.0394 |
| luna_deepseek_mean | jpeg_compression | visually_verified | 33/33 | 0 | 32 / 1 / 0 | 0 | 0.8803 |

### Fully verified adjacent families

Ladder counts require all four technical scores. Pair tables above require only their two endpoints, so missing intermediate scores do not erase a valid parent–severe comparison.

| Model | Operator | Complete/expected families | Missing | Non-increasing | Strictly decreasing | Adjacent ↓ / = / ↑ | Adjacent floor 1 | Mean parent–severe drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| luna | all | 73/73 | 0 | 64 | 47 | 184 / 26 / 9 | 6 | 1.4863 |
| luna | gaussian_blur | 34/34 | 0 | 32 | 22 | 86 / 14 / 2 | 5 | 1.7794 |
| luna | gaussian_noise | 33/33 | 0 | 29 | 24 | 86 / 9 / 4 | 1 | 1.3091 |
| luna | jpeg_compression | 6/6 | 0 | 3 | 1 | 12 / 3 / 3 | 0 | 0.8000 |
| deepseek | all | 73/73 | 0 | 53 | 35 | 169 / 29 / 21 | 8 | 1.5945 |
| deepseek | gaussian_blur | 34/34 | 0 | 29 | 23 | 87 / 10 / 5 | 8 | 2.4647 |
| deepseek | gaussian_noise | 33/33 | 0 | 22 | 11 | 70 / 17 / 12 | 0 | 0.7697 |
| deepseek | jpeg_compression | 6/6 | 0 | 2 | 1 | 12 / 2 / 4 | 0 | 1.2000 |
| jev | all | 73/73 | 0 | 63 | 62 | 208 / 1 / 10 | 0 | 1.7704 |
| jev | gaussian_blur | 34/34 | 0 | 34 | 34 | 102 / 0 / 0 | 0 | 2.5712 |
| jev | gaussian_noise | 33/33 | 0 | 26 | 26 | 92 / 0 / 7 | 0 | 1.1109 |
| jev | jpeg_compression | 6/6 | 0 | 3 | 2 | 14 / 1 / 3 | 0 | 0.8600 |
| liquid | all | 73/73 | 0 | 54 | 54 | 200 / 0 / 19 | 0 | 1.5877 |
| liquid | gaussian_blur | 34/34 | 0 | 32 | 32 | 100 / 0 / 2 | 0 | 2.3747 |
| liquid | gaussian_noise | 33/33 | 0 | 19 | 19 | 85 / 0 / 14 | 0 | 0.8996 |
| liquid | jpeg_compression | 6/6 | 0 | 3 | 3 | 15 / 0 / 3 | 0 | 0.9123 |
| qrealign_lite | all | 73/73 | 0 | 70 | 70 | 216 / 0 / 3 | 0 | 2.3207 |
| qrealign_lite | gaussian_blur | 34/34 | 0 | 34 | 34 | 102 / 0 / 0 | 0 | 2.7556 |
| qrealign_lite | gaussian_noise | 33/33 | 0 | 30 | 30 | 96 / 0 / 3 | 0 | 1.8527 |
| qrealign_lite | jpeg_compression | 6/6 | 0 | 6 | 6 | 18 / 0 / 0 | 0 | 2.4308 |
| luna_deepseek_mean | all | 73/73 | 0 | 59 | 51 | 192 / 12 / 15 | 5 | 1.5404 |
| luna_deepseek_mean | gaussian_blur | 34/34 | 0 | 30 | 28 | 92 / 6 / 4 | 5 | 2.1221 |
| luna_deepseek_mean | gaussian_noise | 33/33 | 0 | 27 | 22 | 87 / 5 / 7 | 0 | 1.0394 |
| luna_deepseek_mean | jpeg_compression | 6/6 | 0 | 2 | 1 | 13 / 1 / 4 | 0 | 1.0000 |

The JSON preserves all-family ladders, adjacent-pair coverage, both scope-floor flags, missing endpoint IDs and native score differences. Review verification concerns visible degradation and does not establish an independently scored quality target.

## Protected style and scope adjudication

Visual review is complete for all 76 queued comparisons across 15 exact delivered images. This includes all 72 model/arm comparisons on the 12 prospectively selected style cases, plus newly flagged scope changes and descriptive uncertain-scope flips. The supplied score export, raw receipts, image bytes, reference freeze and reviewed input records were checked before joining the judgments. Primary references and model scores are unchanged.

These are nonblind diagnostic assistant opinions after opening the exact images. A visibly justified change has an image-based rationale; a regression has a visible quality or scope problem; ambiguous differences cannot be resolved confidently. Small score changes often remain ambiguous. The review neither provides human ground truth nor proves that a particular statistic caused a response difference. Derived-mean comparisons share their component responses and are not independent replications.

| Model | Arm | Visibly justified | Regression | Ambiguous | Not assessable |
|---|---|---:|---:|---:|---:|
| DeepSeek | Own stats | 3 | 3 | 7 | 0 |
| DeepSeek | Swapped stats | 1 | 1 | 10 | 0 |
| Luna | Own stats | 4 | 3 | 7 | 0 |
| Luna | Swapped stats | 2 | 3 | 8 | 0 |
| Luna + DeepSeek mean | Own stats | 6 | 2 | 4 | 0 |
| Luna + DeepSeek mean | Swapped stats | 1 | 1 | 10 | 0 |

The following comparisons have visually supported regressions. Both arms remain in the exports; no example or label was removed because it disagreed with the reference.

| Image | Model | Arm | Visual diagnosis |
|---|---|---|---|
| R0091 | DeepSeek | Own stats | Central petals, veins and leaf surfaces remain finely resolved. The 0.8 reduction overstates noise and low global contrast despite successful selective focus. |
| R0091 | Luna | Own stats | The -0.6 technical drop overstates frame-wide soft texture: focused central petals and leaves preserve detail, whereas most blur follows shallow depth of field. Visible fine grain does not adequately explain the larger penalty. |
| R0091 | Luna + DeepSeek mean | Own stats | The 0.7 mean reduction undervalues the crisp central flower and leaves because large intentional out-of-focus regions dominate the frame. |
| R0110 | DeepSeek | Own stats | Illuminated cloud edges and tonal layers remain distinct. Calling their natural soft forms and intentional dark silhouette a broad capture defect overstates the technical penalty. |
| R0110 | DeepSeek | Swapped stats | The cloud layers and horizon remain defined. The 0.5 deduction for haze and low contrast undervalues the deliberate low-key scene and its resolved edges. |
| R0110 | Luna | Own stats | The -1.0 technical drop treats the nearly featureless foreground silhouette and naturally soft cloud masses as substantial defects despite resolved illuminated cloud contours. That penalty is disproportionate to the visible scene. |
| R0110 | Luna + DeepSeek mean | Own stats | The mean falls 0.75 despite resolved illuminated textures; both component deductions over-penalize the deliberate dark masses and silhouette. |
| R0110 | Luna + DeepSeek mean | Swapped stats | The 0.5 technical reduction persists in the mean even though the scene has coherent edges and smooth tones rather than global capture blur. |
| R0180 | Luna | Swapped stats | The -0.4 technical reduction explicitly faults softer background branches even though selective focus leaves the main bark and lichen sharply resolved. This introduces an unsupported depth-of-field penalty. |
| R0181 | Luna | Own stats | The new declared eligibility floor is unsupported: carton edges and surrounding surface establish a physical-object photograph. A low technical score is warranted by darkness, but a categorical reproduction exclusion is a regression. Technical absolute error alone improves, so it must not conceal the scope error. The treatment forces aesthetics to 1 for an unsupported reproduction exclusion. Physical packaging and surrounding surface remain visible; poor photographic presentation should be judged within scope. |
| R0181 | Luna | Swapped stats | The new declared eligibility floor misclassifies the photographed carton despite visible boundaries and surrounding surface. Darkness can justify a low quality score, but does not justify this categorical exclusion. The 1-point aesthetic floor is caused by a mistaken reproduction exclusion rather than an assessment of the physical packaging photograph; this is a scope regression. |
| R0240 | Luna | Swapped stats | The -0.5 aesthetic drop overstates busy curtain folds and stage equipment: the curtains provide a subdued backdrop and the small devices stay at the bottom edge, while the relaxed figure remains clearly dominant. |
| R0331 | DeepSeek | Own stats | The exact view is a full-frame reproduction of printed text without surrounding scene; the declared document rule requires 1.00 despite its photographic capture. The same full-frame document eligibility rule requires 1.00; raising aesthetics to 1.50 is a scope regression rather than an artistic improvement. |

Reference-uncertain eligibility remains uncertain and its floor flips are descriptive, not confirmed scope errors. Review completion is separate from the numerical advancement tests and does not approve any labels for training. The numerical analysis artifact deliberately leaves its own protected-review status unassessed; this supplemental adjudication supplies the completed review.

The derived mean declares a scope floor only when both component models declare it. A new floor in just one component can still lower the mean without triggering a new declared mean-floor error. Component scope mistakes and their numerical effect therefore remain relevant; the absence of a new mean-floor flag does not establish scope safety.

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

## Operational segments and provider failures

Jev required an explicit provider-failure recovery. Its concurrency limit changed4→1→4 in separate documented segments: serial operation did not remove the observed503 failures and was substantially slower in the measured windows. DeepSeek retained billed no-answer token-limit failures and continued with unchanged concurrency8 and token/reasoning settings. A client-process interruption left21 request-only journals (Luna16,Liquid4,Jev1); they retain unknown original latency and cost. Subsequent execution uses durable user services. Single-request smokes are separate. These timings include upload, queueing and the complete response and are descriptive, not a controlled model-speed comparison. Failed labels remain null; known charges and unknown bills remain distinct. The original verified pre-serve capacity429 retry policy remains separate.

| Model | Phase | Segment | Concurrency limit | Attempts | Valid | Invalid | Failures | Known503 | Exhaustions | p50 all(s) | p95 all(s) | p50 scored(s) | p95 scored(s) | Unpriced |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek | benchmark | 2026-11-01T14-19-30.114456+00-00 | 8 | 1500 | 1500 | 0 | 0 | 0 | 0 | 8.42 | 23.32 | 8.42 | 23.32 | 0 |
| deepseek | rehearsal | 2026-11-01T14-13-30.838015+00-00 | 8 | 198 | 198 | 0 | 0 | 0 | 0 | 9.76 | 25.99 | 9.76 | 25.99 | 0 |
| deepseek | rehearsal | 2026-11-01T14-19-30.114456+00-00 | 8 | 2426 | 2426 | 0 | 0 | 0 | 0 | 9.72 | 21.82 | 9.72 | 21.82 | 0 |
| deepseek | rehearsal | initial_single_request_smoke | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 6.33 | 6.33 | 6.33 | 6.33 | 0 |
| deepseek | rehearsal | original_frozen_plan | 8 | 876 | 874 | 0 | 2 | 0 | 1 | 10.58 | 21.10 | 10.58 | 21.09 | 1 |
| jev | benchmark | 2026-11-01T14-49-02.948804+00-00 | 1 | 3 | 1 | 0 | 2 | 2 | 0 | 0.58 | 1.36 | 1.45 | 1.45 | 2 |
| jev | benchmark | 2026-11-01T14-51-21.762892+00-00 | 1 | 10 | 9 | 0 | 1 | 1 | 0 | 1.64 | 4.05 | 1.45 | 4.07 | 1 |
| jev | benchmark | 2026-11-01T14-52-32.220892+00-00 | 4 | 383 | 322 | 0 | 61 | 61 | 0 | 1.79 | 3.37 | 1.83 | 3.36 | 61 |
| jev | benchmark | 2026-11-01T14-58-03.329049+00-00 | 1 | 10 | 8 | 0 | 2 | 2 | 0 | 1.21 | 1.81 | 1.21 | 1.64 | 2 |
| jev | benchmark | 2026-11-01T15-00-53.007834+00-00 | 1 | 10 | 9 | 0 | 1 | 1 | 0 | 1.04 | 2.76 | 1.05 | 2.84 | 1 |
| jev | benchmark | 2026-11-01T15-01-58.391747+00-00 | 4 | 84 | 73 | 0 | 11 | 11 | 0 | 1.40 | 2.62 | 1.47 | 2.69 | 11 |
| jev | benchmark | 2026-11-01T15-57-54.813535+00-00 | 1 | 50 | 43 | 0 | 7 | 7 | 0 | 0.77 | 1.27 | 0.77 | 1.31 | 7 |
| jev | benchmark | 2026-11-01T15-58-48.296000+00-00 | 1 | 39 | 33 | 0 | 6 | 6 | 0 | 0.80 | 1.41 | 0.81 | 1.42 | 6 |
| jev | rehearsal | 2026-11-01T13-48-48.969461+00-00 | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 3.89 | 3.89 | 3.89 | 3.89 | 0 |
| jev | rehearsal | 2026-11-01T13-49-08.031529+00-00 | 4 | 34 | 32 | 0 | 2 | 2 | 0 | 4.80 | 7.30 | 4.80 | 7.32 | 2 |
| jev | rehearsal | 2026-11-01T13-50-16.101872+00-00 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1.68 | 1.68 | 1.68 | 1.68 | 0 |
| jev | rehearsal | 2026-11-01T13-50-59.949122+00-00 | 1 | 4 | 3 | 0 | 1 | 1 | 0 | 3.12 | 4.80 | 2.94 | 4.85 | 1 |
| jev | rehearsal | 2026-11-01T14-02-31.843566+00-00 | 1 | 121 | 109 | 0 | 12 | 11 | 0 | 3.19 | 5.10 | 3.18 | 5.01 | 12 |
| jev | rehearsal | 2026-11-01T14-20-46.842974+00-00 | 1 | 150 | 134 | 0 | 16 | 16 | 0 | 3.45 | 5.52 | 3.43 | 5.39 | 16 |
| jev | rehearsal | 2026-11-01T14-29-54.936464+00-00 | 4 | 144 | 121 | 0 | 23 | 23 | 0 | 4.81 | 7.02 | 4.84 | 6.86 | 23 |
| jev | rehearsal | 2026-11-01T15-01-58.391747+00-00 | 4 | 592 | 499 | 0 | 93 | 93 | 0 | 1.66 | 3.14 | 1.70 | 3.18 | 93 |
| jev | rehearsal | 2026-11-01T15-09-35.416943+00-00 | 1 | 10 | 9 | 0 | 1 | 1 | 0 | 1.83 | 3.39 | 1.70 | 3.45 | 1 |
| jev | rehearsal | 2026-11-01T15-10-47.529644+00-00 | 4 | 144 | 119 | 0 | 25 | 25 | 0 | 1.44 | 3.58 | 1.54 | 3.61 | 25 |
| jev | rehearsal | 2026-11-01T15-14-48.096374+00-00 | 1 | 10 | 10 | 0 | 0 | 0 | 0 | 0.87 | 1.86 | 0.87 | 1.86 | 0 |
| jev | rehearsal | 2026-11-01T15-16-05.259766+00-00 | 4 | 121 | 99 | 0 | 22 | 22 | 0 | 1.50 | 3.05 | 1.58 | 3.07 | 22 |
| jev | rehearsal | 2026-11-01T15-19-27.501232+00-00 | 1 | 10 | 9 | 0 | 1 | 1 | 0 | 0.72 | 1.16 | 0.72 | 1.17 | 1 |
| jev | rehearsal | 2026-11-01T15-20-34.340736+00-00 | 4 | 97 | 76 | 0 | 21 | 21 | 0 | 1.42 | 2.39 | 1.50 | 2.40 | 21 |
| jev | rehearsal | 2026-11-01T15-26-16.692882+00-00 | 4 | 2054 | 1721 | 0 | 333 | 333 | 0 | 1.12 | 1.99 | 1.15 | 2.01 | 333 |
| jev | rehearsal | 2026-11-01T15-58-48.296000+00-00 | 1 | 147 | 120 | 0 | 27 | 27 | 0 | 0.77 | 1.22 | 0.79 | 1.15 | 27 |
| jev | rehearsal | 2026-11-01T16-12-38.904888+00-00 | 1 | 500 | 421 | 0 | 79 | 79 | 0 | 0.80 | 1.20 | 0.82 | 1.23 | 79 |
| jev | rehearsal | initial_single_request_smoke | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0.91 | 0.91 | 0.91 | 0.91 | 0 |
| jev | rehearsal | original_frozen_plan | 4 | 6 | 4 | 0 | 2 | 2 | 0 | 2.49 | 2.64 | 2.49 | 2.64 | 2 |
| liquid | benchmark | 2026-11-01T14-19-49.364344+00-00 | 4 | 500 | 500 | 0 | 0 | 0 | 0 | 1.75 | 3.07 | 1.75 | 3.07 | 0 |
| liquid | rehearsal | 2026-11-01T14-19-49.364344+00-00 | 4 | 1805 | 1805 | 0 | 0 | 0 | 0 | 3.25 | 5.74 | 3.25 | 5.74 | 0 |
| liquid | rehearsal | 2026-11-01T15-57-43.953510+00-00 | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 1.63 | 1.71 | 1.63 | 1.71 | 0 |
| liquid | rehearsal | initial_single_request_smoke | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1.78 | 1.78 | 1.78 | 1.78 | 0 |
| liquid | rehearsal | original_frozen_plan | 4 | 1694 | 1690 | 0 | 4 | 0 | 0 | 3.32 | 5.41 | 3.32 | 5.41 | 4 |
| luna | benchmark | 2026-11-01T14-19-32.029167+00-00 | 16 | 1500 | 1500 | 0 | 0 | 0 | 0 | 9.03 | 12.35 | 9.03 | 12.35 | 0 |
| luna | rehearsal | 2026-11-01T14-19-32.029167+00-00 | 16 | 896 | 896 | 0 | 0 | 0 | 0 | 9.08 | 12.44 | 9.08 | 12.44 | 0 |
| luna | rehearsal | 2026-11-01T15-55-59.697997+00-00 | 1 | 16 | 16 | 0 | 0 | 0 | 0 | 6.07 | 7.25 | 6.07 | 7.25 | 0 |
| luna | rehearsal | initial_single_request_smoke | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 6.08 | 6.08 | 6.08 | 6.08 | 0 |
| luna | rehearsal | original_frozen_plan | 16 | 2603 | 2587 | 0 | 16 | 0 | 0 | 8.58 | 11.98 | 8.58 | 11.98 | 16 |

The exact known503 class was quarantined under a prospective bounded policy: five consecutive failures or20 in the last100 requests stops further dispatch. Other error classes stop immediately. All unknown costs keep their conservative budget reservations; known subtotals are not complete billed totals. [Detailed operational audit](operations-summary.json).

Later Jev circuit resets are explicit operational amendments. Each requires at least120seconds of cooldown and a separate ten-request serial half-open batch using never-attempted jobs; a second exact503 stops that batch. Continuation requires all ten responses and at most one exact503. The original five-consecutive/20-of100 thresholds then apply to the documented new window, including the probes. Historical failures, labels and unknown-cost reservations remain intact. The frozen benchmark phase was prioritized for coverage during provider instability, before completing the rehearsal; score magnitudes did not determine this order.

| Stage | Status | Concurrency | New attempts | Exact503 | Prior observed responses retained |
|---|---|---:|---:|---:|---:|
| probe | probe_failed | 1 | 3 | 2 | 461 |
| probe | probe_succeeded | 1 | 10 | 1 | 464 |
| continue | stopped_requires_audit | 4 | 383 | 61 | 464 |
| probe | probe_failed | 1 | 10 | 2 | 857 |
| probe | probe_succeeded | 1 | 10 | 1 | 867 |
| continue | stopped_requires_audit | 4 | 676 | 104 | 867 |
| probe | probe_succeeded | 1 | 10 | 1 | 1553 |
| continue | stopped_requires_audit | 4 | 144 | 25 | 1553 |
| probe | probe_succeeded | 1 | 10 | 0 | 1707 |
| continue | stopped_requires_audit | 4 | 121 | 22 | 1707 |
| probe | probe_succeeded | 1 | 10 | 1 | 1838 |
| continue | stopped_requires_audit | 4 | 97 | 21 | 1838 |

The remaining Jev first pass used a separately approved availability amendment after1,945 observed responses showed296 exact503s(15.22%), a longest failure run of4 and repeated20-of100 rolling trips despite prompt successful responses. It retained the five-consecutive stop, raised the rolling stop to30of100 and added a cumulative≥20% stop after100 observed responses using the complete initial history. Only availability tolerance changed: original request payloads, grading acceptance, budgets, unknown-cost reservations and the ban on replaying first-pass jobs remained intact. This later amendment does not alter the separately bounded postpass policy.

Serial recovery later reached the original20-of100 availability stop after236 additional Jev attempts produced196 fully scored recoveries and40 exact503s(16.95%). A separately pinned operational override retained serial dispatch, the original immutable allowance/job caps, protected-output rules and$15/full-reserve limits; it kept the five-consecutive stop, used30of100, and applied cumulative≥20% after100 across the entire observed postpass history. Five already exhausted cases remained missing. Each subsequent admission has separate amendment/source hashes, chronological guard evidence and a verified budget reservation; original admission receipts and the loader remain unchanged.

After every initial API job had been attempted, a separate procedural amendment allowed bounded recovery of the exact Jev503 class (up to two additional attempts) and the21 interrupted requests (one additional attempt). Existing valid, partial or invalid scored responses were never replayed. Original unknown bills remain unknown, including possible charges for interrupted requests. DeepSeek no-answer token-limit failures remain missing. This recovery was an explicit later amendment, not part of the preregistered initial protocol.

| Model | Eligible jobs | Completed responses obtained | Fully scored recoveries | Partial/invalid recoveries | Persistent without response | All jobs missing either score |
|---|---:|---:|---:|---:|---:|---:|
| deepseek | 0 | 0 | 0 | 0 | 0 | 3 |
| luna | 16 | 16 | 16 | 0 | 0 | 0 |
| jev | 630 | 617 | 617 | 0 | 13 | 13 |
| liquid | 4 | 4 | 4 | 0 | 0 | 0 |

Source, partition, category, operator and content-cluster missingness remain in the detailed recovery report. Recovered-response counts do not imply complete two-task labels; final accuracy uses task-level validity and reports coverage.

## Q-ReAlign Lite infrastructure

All 4,000 Lite requests completed with valid technical and aesthetic scores. Their reported model price is **$0.00 per image**, as requested. This is an accounting override; Runpod compute and storage are accounted separately below.

The evaluation pod, `gm21jcse3yohgp` in EU-RO-1, ran from **13:46:11.489 to 15:16:43 UTC on 1 November 2026**. It used a secure RTX 4090 at a quoted **$0.89/hour**, with the unchanged scorer and checkpoint. The pod was stopped through the Runpod plugin after all responses were downloaded and their image, request and response hashes were verified.

An earlier replacement, `1e4rvgw2q4hkci`, incurred startup overhead from 13:42:39.928 to 13:44:24 UTC and received no image requests. Its posted charge was **$0.02587008**. The original pod could not resume because its host had no available GPU; that failed start created no new compute session.

At the latest **16:12:27.976 UTC** post-stop billing query:

- **Posted infrastructure subtotal: $1.29615666** across both replacements: **$1.27028659** for the evaluation pod, including reported storage, and **$0.02587008** for the startup-only pod. The evaluation pod's 13:00, 14:00 and 15:00 UTC billing buckets are now present. The provider did not explicitly confirm settlement, so this is a posted subtotal rather than a final invoice amount.
- **Elapsed-time compute estimate: $1.36851913**, calculated from both recorded lifecycle intervals at the quoted hourly rate. This estimate excludes storage and is not a settled invoice amount.
- **Settled infrastructure total: unavailable**, stored as null pending billing settlement. Neither the provisional subtotal nor the estimate is added to the user-assigned $0.00 model-price rows.

The earlier **15:17:49 UTC** posted subtotal of **$0.47511232** is retained with its original receipts in the infrastructure history. The [16:12:27.976 UTC refresh receipt](lite-billing-refresh-2026-11-01T16-12-27-976Z.json) records the new pod-specific billing response and confirms the original pod and both trial replacements remain **EXITED**; no evaluation GPU remains running. No pod was restarted or created during this check. Detailed provider receipts, audit timestamps and the distinction between posted charges and estimates are retained in [lite-infrastructure.json](lite-infrastructure.json). The complete request audit is in [lite-completion-validation.json](lite-completion-validation.json).

## Total cost per model

These totals count every recorded attempt once across all arms, including retries and errors when a charge is reported. Unknown receipts remain unknown; the known subtotal is not a complete total when an attempt is unpriced. Derived means add no charge. Q-ReAlign Lite and Reference prices are explicitly assigned $0.00 as requested. Runpod infrastructure is accounted separately; consult the infrastructure supplement for posted charges and estimates.

| Model | All attempts | Known subtotal | Unknown-price attempts | Total | Basis |
| --- | --- | --- | --- | --- | --- |
| GPT-6 Luna | 5016 | $3.933034375 | 16 | Unknown | Provider receipts |
| DeepSeek V4.1 Flash 1017 | 5001 | $2.66338736 | 1 | Unknown | Provider receipts |
| Jev 1.14 | 4736 | $0.716425836 | 749 | Unknown | Provider receipts |
| Liquid d1.1 | 4004 | $0.19634044 | 4 | Unknown | Provider receipts |
| Q-ReAlign Lite | 4000 | $0.00 | 0 | $0.00 | Assigned zero |
| Reference | 0 | $0.00 | 0 | $0.00 | Assigned zero |
| All reported model rows | 22757 | $7.509188011 | 770 | Unknown | Receipts and assigned-zero rows |
