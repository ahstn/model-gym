# Two models at high reasoning: 2,000-image photography trial

Completed 2,000 images per model, with 4,000 selected responses from 4,005 HTTP attempts, using DeepSeek V4.1 Flash 1017 through RunInfra and GPT-6 Luna through OpenRouter. Both explicitly requested **high reasoning**, with the same photography prompt and schema-constrained technical and aesthetic scores as the previous trial.

The 50/50 average is closest to the assistant references across the two dimensions on this fresh subset. This supports continuing to evaluate the average.

[72-image grid](collage/originals-grid.png) · [Page 1](collage/originals-grid-page-1.png) · [Page 2](collage/originals-grid-page-2.png) · [Interactive gallery](collage/originals-grid.html) · [All 2,000 rows](per-image.csv)

## Distance from the independent reference assessments

The primary comparison uses **71 fresh reference cases**, excluding the known poster regression check. Each model and the average use the same valid cases for each dimension. Lower mean absolute error (MAE) is better; signed bias above zero means higher scores than the reference.

| Assessor | Technical MAE | Aesthetic MAE | Mean of both MAEs | Technical bias | Aesthetic bias |
|---|---:|---:|---:|---:|---:|
| DeepSeek V4.1 Flash 1017 | 0.490 | 0.289 | 0.389 | +0.293 | +0.089 |
| GPT-6 Luna | 0.392 | 0.334 | 0.363 | +0.315 | +0.292 |
| 50/50 average | 0.415 | 0.269 | 0.342 | +0.304 | +0.190 |

GPT-6 Luna is closest on technical quality, while 50/50 average is closest on aesthetics. The combined result therefore does not mean the same option wins both dimensions. The mean's signed technical bias is +0.304: averaging does not remove the tendency to score technical quality above these references.

| Paired comparison | Average minus individual MAE | Descriptive 95% interval |
|---|---:|---:|
| Average versus DeepSeek V4.1 Flash 1017 | -0.047 | [-0.087, -0.008] |
| Average versus GPT-6 Luna | -0.020 | [-0.065, +0.026] |

Negative differences favor the average. These intervals use 10,000 paired bootstrap resamples of the 71 cases valid across both tasks and all predictors, with a fixed recorded seed. An interval spanning zero does not clearly separate the options on this sample. The intervals are conditional on these assistant references and this deliberately selected sample; they do not quantify population accuracy or human judgment uncertainty.

Common valid counts: technical 71/71; aesthetic 71/71. The average is calculated separately for each dimension, without rounding before analysis. It requires both valid model scores; missing scores are never treated as zero or replaced with the other model's score.

**Scope-boundary sensitivity:** T0632 shows a printed page with visible tabletop and perspective. Its frozen reference excludes it under the near-full-frame reproduction rule; a later visual audit considered that boundary ambiguous. Without changing the reference, a secondary comparison removes T0632 as well as the known poster: mean of the two MAEs is DeepSeek V4.1 Flash 1017 0.373; GPT-6 Luna 0.368; 50/50 average 0.336. This sensitivity was added after reference freeze, before inspecting that case's student outputs, and was not preregistered. It does not replace the primary result.

These are **assistant reference judgments, not human ground truth**. Two fresh-context reviewers inspected 36 images each, without student outputs or source quality labels. The 72 images were selected before inference and all reference scores were frozen before inference. The subset is deliberately stratified, has 72 distinct families, and is not population-weighted. There are no assistant reference scores for the other 1,928 images; whole-corpus agreement is not an accuracy estimate.

## Sample and controls

The corpus contains 1,999 cases not used in the recorded earlier trials, plus the exact known poster as T2000. Selection excluded known source identities, related families and near duplicates from earlier work. Fresh here means new to these trials, not proven absent from the models' training data.

| Stratum | Images |
|---|---:|
| Natural standalone images | 1,279 |
| Real capture problems: blur, brightness, darkness, framing, obstruction, rotation | 240 |
| Controlled families: 120 parents, each with three defect severities | 480 |
| Known poster regression check | 1 |
| **Total** | **2,000** |

The reference subset contains 41 standalone images, 18 real capture problems, 12 controlled cases from separate families, and the poster. No student scores were used to select it. Two non-reference source files failed the unchanged ICC conversion policy; deterministic replacements are documented in [source-format-resolution.json](source-format-resolution.json). The fixed 72-image selection was preserved.

The prompt was not tuned against this sample. Clearly non-photographic content receives 1.00 for both dimensions with `unsupported_domain`. Photographs of physical labels, signs or art in context remain eligible; uncertain provenance does not itself trigger a floor. Experimental outputs remain unapproved for training.

## Validity and reasoning evidence

| Model | Selected responses | Numeric technical scores | Numeric aesthetic scores | Calls reporting positive reasoning tokens | Reasoning tokens |
|---|---:|---:|---:|---:|---:|
| DeepSeek V4.1 Flash 1017 | 2,000 | 2000 | 2000 | 2000 | 932,462 |
| GPT-6 Luna | 2,000 | 2000 | 2000 | 2000 | 691,351 |

Both routes requested high reasoning and a 16,384-token output allowance. Reported reasoning-token use is evidence that reasoning ran; a request setting alone does not prove the provider's exact internal budget. DeepSeek did not return an effort-confirmation header in this run. Raw responses, usage and headers are retained. Invalid task outputs remain invalid, while a valid sibling score is preserved.

RunInfra initially refused two requests because its shared model pool reached capacity. A separate, bounded continuation retained the 17 completed responses, lowered concurrency and retried only the explicit pre-serve capacity rejection. It used identical frozen requests and image bytes. No valid or invalid scoring response was repeated, and no model fallback was used. Every HTTP attempt, including rejections, remains in the billing ledger. See [continuation-protocol.md](continuation-protocol.md) and [provider recovery evidence](provider-recovery-notes.md).

## Eligibility checks

The 72 references contain 68 eligible images, two exclusions and two uncertain cases. The following counts compare model decisions with that reference subset; flags elsewhere in the 2,000 images have not all received independent scope review.

| Model | Missed or incomplete reference exclusions | Exclusions on reference-eligible images | Exclusions on uncertain references | Scope flags across all 2,000 |
|---|---:|---:|---:|---:|
| DeepSeek V4.1 Flash 1017 | 1 | 0 | 0 | 17 |
| GPT-6 Luna | 0 | 0 | 0 | 21 |

Averaging does not resolve photography-eligibility disagreements: a 1.00 exclusion from one model and an ordinary score from the other produce a mean above the policy floor. Scope flags remain separate in the saved rows and should be reviewed before using the mean as a dataset acceptance score. The scope-boundary case T0632 is explicitly sensitivity-tested above.

## Controlled technical degradation

This checks whether technical scores are non-increasing from parent through three stronger defect levels. It is a consistency diagnostic, not a human accuracy measure. Flat sequences count as non-increasing, so the detailed analysis also reports strict decreases, flat families and parent-to-severe score changes. Aesthetic changes are reported separately without requiring monotonic decline.

| Defect | Common complete families | DeepSeek non-increasing | Luna non-increasing | Average non-increasing |
|---|---:|---:|---:|---:|
| all operators | 120 | 78 | 94 | 85 |
| gaussian blur | 40 | 37 | 39 | 39 |
| gaussian noise | 40 | 22 | 34 | 30 |
| jpeg compression | 40 | 19 | 21 | 16 |

Luna has the most non-increasing families on this technical-consistency diagnostic. Averaging does not improve every ordering check: for JPEG compression, its non-increasing count is lower than either individual model's. These strict counts include small score reversals; the detailed results retain their magnitudes.

[analysis.json](analysis.json) includes results by defect/source stratum, eligible-only reference comparisons, all 72 reference comparisons, distributions, student disagreement and a sensitivity view excluding families flagged outside photography scope by either model. Small strata are descriptive; this run does not establish a universal winner or validate new prompt changes.

## Agreement across all 2,000 images

This measures how similarly the students score the corpus, without treating agreement as correctness. Signed difference is DeepSeek minus Luna; a positive value means DeepSeek scores higher on average.

| Dimension | Common valid images | Mean absolute difference | Signed difference | Pearson correlation | Images differing by at least 1 point |
|---|---:|---:|---:|---:|---:|
| Technical | 2000 | 0.303 | +0.003 | 0.856 | 88 |
| Aesthetic | 2000 | 0.323 | -0.195 | 0.821 | 77 |

## Saved artifacts and reproducibility

The gallery includes the exact 72 delivered PNGs under `collage/images`, with relative links that work after checkout. Its three rows show Ref, Luna and DeepSeek; [per-image.csv](per-image.csv) and this report also include the arithmetic mean. The full 2,000-image local input cache remains ignored. The manifest preserves source identities, transformations and hashes. Saved results include request journals and unchanged raw response strings, validated against the frozen runs.

See [protocol.md](protocol.md), [manifest.json](manifest.json), [references-freeze.json](references-freeze.json), [publication-manifest.json](publication-manifest.json), [render validation](collage/render-validation.json), [delivery validation](delivery-validation.json) and [cost-summary.json](cost-summary.json). [Rendering instructions](collage/README.md) distinguish rendering the published gallery from rebuilding analysis with the local input cache. The gallery uses originals scaled without cropping; no image-generation model altered the photographs or score text.

## Total cost per model

USD, for this 2,000-image trial only. Costs are reconciled with reported usage, including confirmed BYOK upstream charges where applicable, without double-counting ordinary provider charges. Earlier trials are excluded. All HTTP attempts, including probes and rejected requests, are accounted for. Mean reported cost divides by 2,000 selected images per model.

**Billing completeness:** 5 capacity rejections have no usage receipt, so their exact charges remain unknown. Zero charge is expected from the documented pre-serve behavior, but is an inference, not a returned billing amount. The runner retained $0.8355840 as a conservative budget reserve; that reserve is not a charge. The table reports known spend, not a verified final invoice. Each route stayed within its $10 admission cap, including reserves.

| Model | Reasoning | HTTP attempts | Selected images | Reported cost (USD) | Unpriced rejections | Mean reported cost per image (USD) |
|---|---|---:|---:|---:|---:|---:|
| DeepSeek V4.1 Flash 1017 | High | 2,005 | 2,000 | $1.02420423 | 5 | $0.00051210 |
| GPT-6 Luna | High | 2,000 | 2,000 | $1.56724988 | 0 | $0.00078362 |
| **Total** | | **4,005** | **4,000** | **$2.59145410** | **5** | |
