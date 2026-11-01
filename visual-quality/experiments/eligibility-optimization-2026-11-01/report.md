# Liquid and Jev: eligibility prompt optimisation

Ten candidate prompts were tested on a 40-image development set, followed by the frozen model-specific candidates on a separate 40-image fresh holdout and a 20-image historical natural-image challenge. There were 20 excluded and 20 eligible cases in each fresh partition. These are assistant-reviewed labels, not human ground truth.

The fresh cases are drawn from source families absent from the previous 4,000-image run, three separately sourced public-domain Commons works, and 19 synthetic browser screenshots paired with fresh photographs within the same split. The original views are hash-checked; exact hashes and source families differ from the prior run, but this does not prove that no visual near-duplicate exists. The screenshot controls share one UI template across splits. The held-out excluded class includes only two natural flat artwork/document cases; the rest are synthetic screenshots, simple charts or obvious renders. The sample is balanced by design and is not representative of deployment prevalence. A Wikimedia rate limit restricted acquisition of additional provenance-complete public-domain posters.

The 20-image natural challenge (10 excluded, 10 eligible) was previously model-scored and includes two already identified disagreements. It supplies more natural poster/document cases, but is a selected historical stress set rather than fresh unbiased evidence. No challenge or held-out model result was used to choose prompts.

[Fresh and challenge image/label manifest](reference-manifest.json) · [Raw case comparisons and all round statistics](results.json) · [Ten full prompt texts: rounds 1–5](rounds-first.json) · [rounds 6–10](rounds-second.json) · [protocol](protocol.md) · [comparison gallery](gallery/eligibility-trial.html)

## Ten development rounds

Original baseline (missed exclusions / false floors, valid): Liquid **10 / 0 (40/40)**; Jev **10 / 0 (40/40)**. Each class has 20 expected cases. Each revision changes only the eligibility-question text, retaining the previous common instruction, both quality questions and 0.50 threshold. Positive and negative changes below are on the same paired cases; they are exploratory development results, not independent confirmations. A missing case reduces coverage and is never counted as a correct decision.

| Round | Liquid misses / false floors (valid) | Liquid paired impact: Δmiss / Δfalse | Jev misses / false floors (valid) | Jev paired impact: Δmiss / Δfalse |
| --- | --- | --- | --- | --- |
| r01 | 0 / 0 (40/40) | Improved: -10 / +0 (40 paired) | 0 / 0 (40/40) | Improved: -10 / +0 (40 paired) |
| r02 | 5 / 1 (40/40) | Mixed: -5 / +1 (40 paired) | 2 / 0 (40/40) | Improved: -8 / +0 (40 paired) |
| r03 | 0 / 0 (40/40) | Improved: -10 / +0 (40 paired) | 0 / 1 (39/40) | Mixed: -10 / +1 (39 paired) |
| r04 | 6 / 0 (40/40) | Improved: -4 / +0 (40 paired) | 3 / 0 (40/40) | Improved: -7 / +0 (40 paired) |
| r05 | 1 / 0 (40/40) | Improved: -9 / +0 (40 paired) | 0 / 0 (40/40) | Improved: -10 / +0 (40 paired) |
| r06 | 1 / 0 (40/40) | Improved: -9 / +0 (40 paired) | 0 / 0 (39/40) | Improved: -10 / +0 (39 paired) |
| r07 | 2 / 0 (40/40) | Improved: -8 / +0 (40 paired) | 0 / 0 (40/40) | Improved: -10 / +0 (40 paired) |
| r08 | 0 / 0 (40/40) | Improved: -10 / +0 (40 paired) | 0 / 1 (40/40) | Mixed: -10 / +1 (40 paired) |
| r09 | 1 / 0 (40/40) | Improved: -9 / +0 (40 paired) | 7 / 0 (40/40) | Improved: -3 / +0 (40 paired) |
| r10 | 1 / 0 (40/40) | Improved: -9 / +0 (40 paired) | 0 / 0 (40/40) | Improved: -10 / +0 (40 paired) |

Rounds 6–10 were written after reviewing only rounds 1–5 development results. Their motivation is in [rounds-second-rationale.md](rounds-second-rationale.md). Prompt versions combine several edits, so the trial does not isolate the causal contribution of individual sentences.

## Fresh held-out evaluation

| Model | Prompt | Missed exclusions / valid of expected | Wrongly floored photos / valid of expected | All valid |
| --- | --- | --- | --- | --- |
| Liquid | Original | 11 / 20 of 20 | 0 / 20 of 20 | 40 / 40 |
| Liquid | r03 | 0 / 20 of 20 | 1 / 20 of 20 | 40 / 40 |
| Jev | Original | 9 / 19 of 20 | 0 / 20 of 20 | 39 / 40 |
| Jev | r06 | 1 / 19 of 20 | 0 / 20 of 20 | 39 / 40 |

**Liquid (r03):** paired 40/40; missed exclusions 11 → 0; false floors 0 → 1. **Jev (r06):** paired 39/40; missed exclusions 9 → 1; false floors 0 → 0. The 20 examples per class and shared screenshot template are too small and narrow to establish production safety or population-level rates. Binomial Wilson intervals are recorded in results.json as descriptive calculations; this purposive, family-paired sample is not an independent random deployment sample.

| Model | Reference category | Baseline errors / valid | Candidate errors / valid | Expected |
| --- | --- | --- | --- | --- |
| Liquid | art | 1/1 | 0/1 | 1 |
| Liquid | context | 0/10 | 1/10 | 10 |
| Liquid | plots | 0/4 | 0/4 | 4 |
| Liquid | poster | 0/1 | 0/1 | 1 |
| Liquid | renders | 0/4 | 0/4 | 4 |
| Liquid | scene | 0/10 | 0/10 | 10 |
| Liquid | screenshot | 10/10 | 0/10 | 10 |
| Jev | art | 1/1 | 1/1 | 1 |
| Jev | context | 0/10 | 0/10 | 10 |
| Jev | plots | 0/4 | 0/4 | 4 |
| Jev | poster | 0/0 | 0/0 | 1 |
| Jev | renders | 0/4 | 0/4 | 4 |
| Jev | scene | 0/10 | 0/10 | 10 |
| Jev | screenshot | 8/10 | 0/10 | 10 |

## Previously scored natural challenge

| Model | Prompt | Missed exclusions / valid of expected | Wrongly floored photos / valid of expected | All valid |
| --- | --- | --- | --- | --- |
| Liquid | Original | 4 / 10 of 10 | 0 / 10 of 10 | 20 / 20 |
| Liquid | r03 | 1 / 10 of 10 | 0 / 10 of 10 | 20 / 20 |
| Jev | Original | 5 / 10 of 10 | 0 / 10 of 10 | 20 / 20 |
| Jev | r06 | 4 / 10 of 10 | 0 / 10 of 10 | 20 / 20 |

| Model | Reference category | Baseline errors / valid | Candidate errors / valid | Expected |
| --- | --- | --- | --- | --- |
| Liquid | natural_flat | 4/10 | 1/10 | 10 |
| Liquid | physical_context | 0/10 | 0/10 | 10 |
| Jev | natural_flat | 5/10 | 4/10 | 10 |
| Jev | physical_context | 0/10 | 0/10 | 10 |

## Post-hoc W0002 image-size diagnostic

Both original Jev requests for W0002 returned HTTP 400, 'Invalid image dimensions', so they remain missing in the frozen holdout. After completing the holdout, a separate copy U0002 was enlarged from 482 × 236 to 572 × 280 pixels using Lanczos interpolation without a crop or padding. Both models accepted that copy under both unchanged prompts. This single post-hoc test suggests a size-related compatibility issue; it is not an independent held-out image and does not replace the original missing results. [Diagnostic freeze](resolution-retest-freeze.json) · [diagnostic results](resolution-retest-results.json).

| Model | Prompt | Status | Exclude probability | Decision |
| --- | --- | --- | --- | --- |
| Liquid | baseline | valid | 0.7814 | Floor |
| Liquid | candidate | valid | 0.9582 | Floor |
| Jev | baseline | valid | 0.6900 | Floor |
| Jev | candidate | valid | 0.7900 | Floor |

Liquid's candidate wrongly floored C4864, an eligible photograph of a physical printed object with visible surroundings. Jev's candidate still missed C1546, a full-frame drawing. These image judgments were reviewed after results were available, so are explanatory rather than blind adjudication.

## Photographic quality score drift

For eligible photos, these compare the native 1–5 scores before any scope floor. No independent technical or aesthetic reference scores were collected for this trial, so differences measure drift, not an improvement or degradation in quality accuracy. Provider stochasticity and the changed question may both contribute.

| Model | Panel | Task | Paired photos | Mean signed change | Mean absolute change | Changes ≥0.50 |
| --- | --- | --- | --- | --- | --- | --- |
| Liquid | heldout | Technical | 20 | -0.004 | 0.065 | 0 |
| Liquid | heldout | Aesthetic | 20 | -0.031 | 0.058 | 0 |
| Liquid | natural challenge | Technical | 10 | 0.059 | 0.071 | 0 |
| Liquid | natural challenge | Aesthetic | 10 | 0.012 | 0.033 | 0 |
| Jev | heldout | Technical | 20 | 0.021 | 0.026 | 0 |
| Jev | heldout | Aesthetic | 20 | -0.002 | 0.025 | 0 |
| Jev | natural challenge | Technical | 10 | 0.012 | 0.018 | 0 |
| Jev | natural challenge | Aesthetic | 10 | 0.014 | 0.026 | 0 |

## Prompt selection and recommendation

The candidate for each model was frozen before any held-out request. It was chosen from the ten development variants by the coverage and false-floor guard, then fewest paired errors; ties favored fewer false floors, shorter wording and lower round ID. A development candidate must reduce missed exclusions without increasing total paired errors to replace the original. The exact rule and all counts are in [winners.json](winners.json).

**Liquid:** candidate **r03**, development advancement passed; fresh paired misses 11 → 0, false floors 0 → 1. Recommendation: **do not adopt as a default on this evidence**. Even a zero count among 20 photos cannot rule out an appreciable false-floor rate.

**Jev:** candidate **r06**, development advancement passed; fresh paired misses 9 → 1, false floors 0 → 0. Recommendation: **promising for a wider independent trial**. Even a zero count among 20 photos cannot rule out an appreciable false-floor rate.

The prompts were evaluated at the existing uncalibrated 0.50 cutoff. Do not lower or retune it using this holdout. Before a larger labeling run, validate the selected candidates on a new human-reviewed, family-held-out set with diverse real poster scans, screenshots, photographed physical packaging and art in scenes; include different UI designs, photographs with text, and random cases. Preserve the original baseline and retain the underlying native scores when the gate applies.

## Latency and costs

End-to-end latency runs from HTTP request through the complete response. The median below is for valid requests across all rounds; concurrency was four combined workers and the routes are not a controlled production latency benchmark. Known charges include all recorded attempts. An unpriced error or interruption remains unknown rather than being assigned $0. Raw parsed provider response bodies and their file hashes are retained; hashes of received wire text were recorded, but the exact wire text was not retained and cannot be independently reconstructed from the parsed object.

| Model | Attempts | Valid responses | Other outcomes | Median valid s | Known subtotal | Unpriced attempts | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Liquid | 562 | 562 | 0 | 1.17 | $0.02409936 | 0 | $0.02409936 |
| Jev | 662 | 558 | 104 | 0.90 | $0.086886324 | 104 | Unknown |

Observed attempt statuses: error: 2, retryable_503: 102, valid: 1120. No score, eligibility reference or original trial output was overwritten.
