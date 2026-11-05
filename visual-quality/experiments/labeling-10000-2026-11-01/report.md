# 10,000-image labeling rehearsal: agreement report

Generated 2026-11-02T03:20:14.600195+00:00. **Inference finished; coverage exceptions retained**.

This measures agreement with 1,500 independent assistant assessments, not accuracy against human ratings. The original 500 references were reused; 500 fresh examples and 500 confirmation references were assessed blind and frozen before admission to inference. The confirmation cohort was not used here to fit weights, prompts or score corrections.

The canonical score remains 37.5% Luna + 37.5% DeepSeek + 25% Liquid, separately for technical quality and aesthetics. All three valid numeric components are required. Calculations use native precision; only presentation is rounded. There is no replacement score for a failed or unscorable observation.

On the same 500 confirmation images, the fixed ensemble has technical MAE 0.363 and aesthetic MAE 0.275; Luna alone has 0.358 and 0.337. Small differences are descriptive: no confidence intervals or human-rating validation were run. Lite is compared on its own shared coverage below.

**Coverage across the full corpus**

| Assessor | Expected | Numeric T/A pairs | Observation statuses |
| --- | --- | --- | --- |
| Luna | 10000 | 10000 | valid: 10000 |
| DeepSeek | 10000 | 9999 | valid: 10000 |
| Liquid d1.1 | 10000 | 10000 | valid: 10000 |
| Jev 1.14 | 10000 | 9997 | valid: 9997, failed: 3 |
| Q-ReAlign Lite | 10000 | 10000 | valid: 10000 |
| Ensemble | 10000 | 9999 | valid: 9999, missing_or_invalid_components: 1 |
| Reference | 1500 | 1500 | valid: 1500 |

MAE is absolute error on the 1–5 scale; lower is better. Bias is prediction minus reference. Spearman ρ compares ranking, with average ranks for ties; a constant score series has no defined correlation. Coverage is reported separately for each outcome. Different coverage can change apparent rankings.

## Original references (500 images)

| Assessor | T n | T MAE | T bias | T ρ | A n | A MAE | A bias | A ρ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 500 | 0.330 | +0.16 | 0.91 | 500 | 0.295 | +0.21 | 0.88 |
| DeepSeek | 500 | 0.427 | +0.26 | 0.83 | 500 | 0.289 | +0.14 | 0.85 |
| Liquid d1.1 | 500 | 0.356 | -0.04 | 0.88 | 500 | 0.268 | -0.12 | 0.88 |
| Jev 1.14 | 500 | 0.335 | +0.08 | 0.88 | 500 | 0.278 | -0.17 | 0.90 |
| Q-ReAlign Lite | 500 | 0.625 | +0.55 | 0.82 | 500 | 1.079 | -0.86 | 0.63 |
| Ensemble | 500 | 0.346 | +0.15 | 0.92 | 500 | 0.226 | +0.10 | 0.92 |

**Same-image comparisons.** Each Δ compares the ensemble with that assessor on their shared numeric outcomes. Negative Δ means lower ensemble MAE; different assessor rows can have different shared samples.

| Compared with | Shared T n | Δ T MAE | Shared A n | Δ A MAE |
| --- | --- | --- | --- | --- |
| Luna | 500 | +0.016 | 500 | -0.069 |
| DeepSeek | 500 | -0.081 | 500 | -0.063 |
| Liquid d1.1 | 500 | -0.010 | 500 | -0.042 |
| Jev 1.14 | 500 | +0.011 | 500 | -0.051 |
| Q-ReAlign Lite | 500 | -0.279 | 500 | -0.853 |

All six assessors, common images only: technical n=500; aesthetic n=500.

| Assessor | Common T MAE | Common A MAE |
| --- | --- | --- |
| Luna | 0.330 | 0.295 |
| DeepSeek | 0.427 | 0.289 |
| Liquid d1.1 | 0.356 | 0.268 |
| Jev 1.14 | 0.335 | 0.278 |
| Q-ReAlign Lite | 0.625 | 1.079 |
| Ensemble | 0.346 | 0.226 |

Four API models and ensemble, common images only: technical n=500; aesthetic n=500.

| Assessor | Common T MAE | Common A MAE |
| --- | --- | --- |
| Luna | 0.330 | 0.295 |
| DeepSeek | 0.427 | 0.289 |
| Liquid d1.1 | 0.356 | 0.268 |
| Jev 1.14 | 0.335 | 0.278 |
| Ensemble | 0.346 | 0.226 |

**Technical tail errors.** Denominators are reference-poor (≤2) or reference-strong (≥4) images with a numeric model outcome.

| Assessor | Poor rated ≥3 | Strong rated <3 |
| --- | --- | --- |
| Luna | 4/92 | 0/195 |
| DeepSeek | 13/92 | 0/195 |
| Liquid d1.1 | 4/92 | 2/195 |
| Jev 1.14 | 4/92 | 0/195 |
| Q-ReAlign Lite | 36/92 | 0/195 |
| Ensemble | 3/92 | 0/195 |

**Photography eligibility decisions.** Only explicit recorded API floor decisions are compared. False-floor and missed-exclusion fractions use known decisions within their reference class. Unknown or failed decisions are counted separately, not treated as correct. Uncertain reference eligibility is excluded from both class denominators. No gate decision is inferred for Lite or the ensemble from a low score.

| Model | Eligible refs | False floors / known | Eligible unknown | Excluded refs | Misses / known | Excluded unknown | Uncertain refs | All unknown decisions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 451 | 0/451 | 0 | 39 | 4/39 | 0 | 10 | 0 |
| DeepSeek | 451 | 1/451 | 0 | 39 | 3/39 | 0 | 10 | 0 |
| Liquid d1.1 | 451 | 0/451 | 0 | 39 | 5/39 | 0 | 10 | 0 |
| Jev 1.14 | 451 | 0/451 | 0 | 39 | 7/39 | 0 | 10 | 0 |

**Sharpness diagnostic.** Among 451 reference-eligible photographs, the within-cohort 25th percentile of `sharpness_peak` is 735.32. This is a descriptive review cue, not an absolute focus threshold or score correction.

| Assessor | Soft n | Soft T bias | Other n | Other T bias |
| --- | --- | --- | --- | --- |
| Luna | 113 | +0.41 | 338 | +0.07 |
| DeepSeek | 113 | +0.64 | 338 | +0.15 |
| Liquid d1.1 | 113 | +0.22 | 338 | -0.18 |
| Jev 1.14 | 113 | +0.31 | 338 | -0.02 |
| Q-ReAlign Lite | 113 | +0.37 | 338 | +0.39 |
| Ensemble | 113 | +0.45 | 338 | +0.04 |

## Fresh reference examples (500 images)

| Assessor | T n | T MAE | T bias | T ρ | A n | A MAE | A bias | A ρ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 500 | 0.363 | +0.14 | 0.88 | 500 | 0.349 | +0.27 | 0.88 |
| DeepSeek | 500 | 0.454 | +0.26 | 0.80 | 500 | 0.333 | +0.19 | 0.84 |
| Liquid d1.1 | 500 | 0.385 | -0.09 | 0.86 | 500 | 0.292 | -0.09 | 0.88 |
| Jev 1.14 | 500 | 0.372 | +0.08 | 0.86 | 500 | 0.304 | -0.11 | 0.87 |
| Q-ReAlign Lite | 500 | 0.502 | +0.41 | 0.83 | 500 | 0.993 | -0.94 | 0.81 |
| Ensemble | 500 | 0.365 | +0.13 | 0.90 | 500 | 0.262 | +0.15 | 0.92 |

**Same-image comparisons.** Each Δ compares the ensemble with that assessor on their shared numeric outcomes. Negative Δ means lower ensemble MAE; different assessor rows can have different shared samples.

| Compared with | Shared T n | Δ T MAE | Shared A n | Δ A MAE |
| --- | --- | --- | --- | --- |
| Luna | 500 | +0.002 | 500 | -0.087 |
| DeepSeek | 500 | -0.089 | 500 | -0.071 |
| Liquid d1.1 | 500 | -0.020 | 500 | -0.030 |
| Jev 1.14 | 500 | -0.006 | 500 | -0.042 |
| Q-ReAlign Lite | 500 | -0.137 | 500 | -0.731 |

All six assessors, common images only: technical n=500; aesthetic n=500.

| Assessor | Common T MAE | Common A MAE |
| --- | --- | --- |
| Luna | 0.363 | 0.349 |
| DeepSeek | 0.454 | 0.333 |
| Liquid d1.1 | 0.385 | 0.292 |
| Jev 1.14 | 0.372 | 0.304 |
| Q-ReAlign Lite | 0.502 | 0.993 |
| Ensemble | 0.365 | 0.262 |

Four API models and ensemble, common images only: technical n=500; aesthetic n=500.

| Assessor | Common T MAE | Common A MAE |
| --- | --- | --- |
| Luna | 0.363 | 0.349 |
| DeepSeek | 0.454 | 0.333 |
| Liquid d1.1 | 0.385 | 0.292 |
| Jev 1.14 | 0.372 | 0.304 |
| Ensemble | 0.365 | 0.262 |

**Technical tail errors.** Denominators are reference-poor (≤2) or reference-strong (≥4) images with a numeric model outcome.

| Assessor | Poor rated ≥3 | Strong rated <3 |
| --- | --- | --- |
| Luna | 7/70 | 1/206 |
| DeepSeek | 14/70 | 0/206 |
| Liquid d1.1 | 5/70 | 0/206 |
| Jev 1.14 | 9/70 | 0/206 |
| Q-ReAlign Lite | 8/70 | 0/206 |
| Ensemble | 4/70 | 0/206 |

**Photography eligibility decisions.** Only explicit recorded API floor decisions are compared. False-floor and missed-exclusion fractions use known decisions within their reference class. Unknown or failed decisions are counted separately, not treated as correct. Uncertain reference eligibility is excluded from both class denominators. No gate decision is inferred for Lite or the ensemble from a low score.

| Model | Eligible refs | False floors / known | Eligible unknown | Excluded refs | Misses / known | Excluded unknown | Uncertain refs | All unknown decisions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 481 | 1/481 | 0 | 15 | 6/15 | 0 | 4 | 0 |
| DeepSeek | 481 | 0/481 | 0 | 15 | 6/15 | 0 | 4 | 0 |
| Liquid d1.1 | 481 | 2/481 | 0 | 15 | 6/15 | 0 | 4 | 0 |
| Jev 1.14 | 481 | 0/481 | 0 | 15 | 12/15 | 0 | 4 | 0 |

**Sharpness diagnostic.** Among 481 reference-eligible photographs, the within-cohort 25th percentile of `sharpness_peak` is 744.73. This is a descriptive review cue, not an absolute focus threshold or score correction.

| Assessor | Soft n | Soft T bias | Other n | Other T bias |
| --- | --- | --- | --- | --- |
| Luna | 121 | +0.41 | 360 | +0.02 |
| DeepSeek | 121 | +0.62 | 360 | +0.11 |
| Liquid d1.1 | 121 | +0.17 | 360 | -0.23 |
| Jev 1.14 | 121 | +0.30 | 360 | -0.07 |
| Q-ReAlign Lite | 121 | +0.33 | 360 | +0.36 |
| Ensemble | 121 | +0.43 | 360 | -0.01 |

## Confirmation references (500 images)

| Assessor | T n | T MAE | T bias | T ρ | A n | A MAE | A bias | A ρ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 500 | 0.358 | -0.03 | 0.89 | 500 | 0.337 | +0.18 | 0.88 |
| DeepSeek | 500 | 0.405 | +0.08 | 0.79 | 500 | 0.323 | +0.12 | 0.84 |
| Liquid d1.1 | 500 | 0.457 | -0.28 | 0.85 | 500 | 0.320 | -0.18 | 0.89 |
| Jev 1.14 | 500 | 0.382 | -0.11 | 0.85 | 500 | 0.328 | -0.23 | 0.91 |
| Q-ReAlign Lite | 500 | 0.395 | +0.21 | 0.82 | 500 | 1.064 | -1.04 | 0.80 |
| Ensemble | 500 | 0.363 | -0.05 | 0.89 | 500 | 0.275 | +0.07 | 0.91 |

**Same-image comparisons.** Each Δ compares the ensemble with that assessor on their shared numeric outcomes. Negative Δ means lower ensemble MAE; different assessor rows can have different shared samples.

| Compared with | Shared T n | Δ T MAE | Shared A n | Δ A MAE |
| --- | --- | --- | --- | --- |
| Luna | 500 | +0.005 | 500 | -0.062 |
| DeepSeek | 500 | -0.041 | 500 | -0.048 |
| Liquid d1.1 | 500 | -0.094 | 500 | -0.045 |
| Jev 1.14 | 500 | -0.018 | 500 | -0.053 |
| Q-ReAlign Lite | 500 | -0.032 | 500 | -0.789 |

All six assessors, common images only: technical n=500; aesthetic n=500.

| Assessor | Common T MAE | Common A MAE |
| --- | --- | --- |
| Luna | 0.358 | 0.337 |
| DeepSeek | 0.405 | 0.323 |
| Liquid d1.1 | 0.457 | 0.320 |
| Jev 1.14 | 0.382 | 0.328 |
| Q-ReAlign Lite | 0.395 | 1.064 |
| Ensemble | 0.363 | 0.275 |

Four API models and ensemble, common images only: technical n=500; aesthetic n=500.

| Assessor | Common T MAE | Common A MAE |
| --- | --- | --- |
| Luna | 0.358 | 0.337 |
| DeepSeek | 0.405 | 0.323 |
| Liquid d1.1 | 0.457 | 0.320 |
| Jev 1.14 | 0.382 | 0.328 |
| Ensemble | 0.363 | 0.275 |

**Technical tail errors.** Denominators are reference-poor (≤2) or reference-strong (≥4) images with a numeric model outcome.

| Assessor | Poor rated ≥3 | Strong rated <3 |
| --- | --- | --- |
| Luna | 4/57 | 0/261 |
| DeepSeek | 7/57 | 0/261 |
| Liquid d1.1 | 4/57 | 3/261 |
| Jev 1.14 | 4/57 | 4/261 |
| Q-ReAlign Lite | 6/57 | 0/261 |
| Ensemble | 3/57 | 0/261 |

**Photography eligibility decisions.** Only explicit recorded API floor decisions are compared. False-floor and missed-exclusion fractions use known decisions within their reference class. Unknown or failed decisions are counted separately, not treated as correct. Uncertain reference eligibility is excluded from both class denominators. No gate decision is inferred for Lite or the ensemble from a low score.

| Model | Eligible refs | False floors / known | Eligible unknown | Excluded refs | Misses / known | Excluded unknown | Uncertain refs | All unknown decisions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 489 | 3/489 | 0 | 9 | 6/9 | 0 | 2 | 0 |
| DeepSeek | 489 | 0/489 | 0 | 9 | 5/9 | 0 | 2 | 0 |
| Liquid d1.1 | 489 | 1/489 | 0 | 9 | 6/9 | 0 | 2 | 0 |
| Jev 1.14 | 489 | 0/489 | 0 | 9 | 7/9 | 0 | 2 | 0 |

**Sharpness diagnostic.** Among 489 reference-eligible photographs, the within-cohort 25th percentile of `sharpness_peak` is 771.47. This is a descriptive review cue, not an absolute focus threshold or score correction.

| Assessor | Soft n | Soft T bias | Other n | Other T bias |
| --- | --- | --- | --- | --- |
| Luna | 123 | +0.19 | 366 | -0.14 |
| DeepSeek | 123 | +0.38 | 366 | -0.05 |
| Liquid d1.1 | 123 | -0.05 | 366 | -0.40 |
| Jev 1.14 | 123 | +0.10 | 366 | -0.23 |
| Q-ReAlign Lite | 123 | +0.07 | 366 | +0.21 |
| Ensemble | 123 | +0.20 | 366 | -0.17 |

## Interpretation and next steps

Keep the accepted ensemble fixed through this rehearsal. Use the separate confirmation results to judge whether its agreement generalizes; a smaller assistant-reference error alone does not establish human preference or label suitability. Review poor-image overrating, high-quality underrating and missing outcomes before enabling training masks. Any subsequent weight or prompt change needs a new untouched confirmation set.

Image statistics remain separate debugging metadata. The sharpness split describes where errors concentrate and does not justify overwriting labels. `focus_area` is relative to each image’s own peak detail; histogram entropy is the `luma_entropy` alias. No statistics were passed to the graders in this run.

## Request latency

These are selected-attempt end-to-end HTTP durations through complete response receipt, including completed HTTP errors. Transport failures and unmeasured requests are excluded from these quantiles. Retries are retained in cost accounting. Reference reviews and the derived ensemble have no HTTP latency. Lite reuses 4,000 compatible historical observations; its fresh-request columns avoid mixing those historical timings with this run. Luna and DeepSeek used 16 workers; Liquid and Jev increased from four to eight. Lite increased from two to three, with a separate 100-image trial at six overlapping requests. These observed network and workload timings are not an isolated model-speed benchmark.

| Model | Measured n | Median s | p95 s | Reused n | Fresh n | Fresh median s | Fresh p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Luna | 10000 | 11.02 | 15.71 | 0 | 10000 | 11.02 | 15.71 |
| DeepSeek | 10000 | 15.97 | 31.10 | 0 | 10000 | 15.97 | 31.10 |
| Liquid d1.1 | 10000 | 5.32 | 9.23 | 0 | 10000 | 5.32 | 9.23 |
| Jev 1.14 | 10000 | 4.02 | 8.04 | 0 | 10000 | 4.02 | 8.04 |
| Q-ReAlign Lite | 10000 | 5.46 | 9.58 | 4000 | 6000 | 7.16 | 10.74 |

## Accounting

Selected receipts describe the latest retained observation per image. All-attempt totals include retries and the bounded missing-response recovery pass. An unpriced attempt has an unknown charge; it is not assumed free. The known subtotal is therefore not a complete bill where unpriced attempts remain. Lite and Reference are $0.00 under the requested scoring-cost convention; paid Runpod infrastructure is accounted for separately. The ensemble adds no request or charge and is not double-counted.

This report and its companion JSON contain aggregate results only. Input file hashes are recorded in the JSON for reproducibility.

| Infrastructure | Posted USD | Latest bucket end | Receipt scope |
| --- | --- | --- | --- |
| EmbeddingGemma 2 | $0.675457 | 2026-11-01T22:00:00Z | Posted completed embedding interval; pod verified EXITED with no retained volume mounts |
| Q-ReAlign Lite | $3.638507 | 2026-11-02T03:00:00Z | Partial posted billing; bucket ends do not establish complete billing through that time. Pod verified EXITED with no retained volume mounts. Creation-to-stop-confirmation GPU-rate wall estimate $4.5117, excluding disk; this estimate is not added to the posted subtotal. Later billing may differ. |

Posted infrastructure amounts are separate from inference-row prices and are not estimates of a complete invoice unless the receipt says so.

**Total cost per model (USD)**

| Model | Selected known subtotal | Selected unpriced | All attempts | All-attempt known subtotal | Unpriced attempts | Total |
| --- | --- | --- | --- | --- | --- | --- |
| Luna | $8.093621 | 0 | 10000 | $8.093621 | 0 | $8.093621 |
| DeepSeek | $6.532398 | 0 | 10083 | $6.552428 | 81 | unknown |
| Liquid d1.1 | $0.502199 | 0 | 10000 | $0.502199 | 0 | $0.502199 |
| Jev 1.14 | $1.813157 | 3 | 11983 | $1.813157 | 1986 | unknown |
| Q-ReAlign Lite | $0.000000 | 0 | — | $0.000000 | 0 | $0.000000 |
| Reference | $0.000000 | 0 | — | $0.000000 | 0 | $0.000000 |
