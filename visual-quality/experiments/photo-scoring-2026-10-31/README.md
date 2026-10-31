# Photo scoring: consolidated 31 October results

This is the working space for the 2,000-image Luna/DeepSeek trial and the follow-up decision-model comparison. The working comparison retains GPT-6 Luna, DeepSeek V4.1 Flash, their 50/50 mean, Claude Haiku 5.5, Jev 1.14 and Liquid d1.1. Reference scores are independent assistant judgments; none of these labels is approved for training.

[Comparison page 1](comparison-page-1.png) · [Page 2](comparison-page-2.png) · [Page 3](comparison-page-3.png) · [Page 4](comparison-page-4.png) · [Interactive gallery](gallery.html)

Follow-up: [Q-ReAlign Lite on the same 72 images](../qrealign-lite-72-2026-10-31/README.md), with [comparison gallery](../qrealign-lite-72-2026-10-31/gallery.html), native outputs and per-image latency. The model assesses visual quality without the photography eligibility floor.

The four image pages show all 72 original delivered views, sorted by reference technical score, with a `TUNING` or `HELD OUT` label. The known poster is identified separately. [scores-72.csv](scores-72.csv) contains every displayed score to two decimals, the original image SHA-256, scope probabilities and native pre-floor decision scores. [Selected responses](selected-responses-72.jsonl) preserves the actual outputs, usage, distributions and normalized scores for Jev, Liquid and Haiku in one file. The 72 PNGs under `images/` are the exact delivered views used for the evaluations.

## Decision-model comparison

Three prompt variants were evaluated on 24 development cases, one of which was the known poster. The chosen variants minimize the combined technical and aesthetic error on the other 23 development images: **brief** for Jev and **anchors** for Liquid. The remaining 48 cases were held out from prompt selection. The exact questions, anchors, selected settings and partition IDs are in [decision-prompts.json](decision-prompts.json).

| Model | Technical MAE | Aesthetic MAE | Mean MAE |
|---|---:|---:|---:|
| Liquid d1.1 | 0.370 | 0.265 | 0.317 |
| Jev 1.14 | 0.389 | 0.247 | 0.318 |
| Haiku 5.5, low requested | 0.394 | 0.306 | 0.350 |
| Luna + DeepSeek mean | 0.469 | 0.278 | 0.373 |
| Luna, high requested | 0.421 | 0.331 | 0.376 |
| DeepSeek, high requested | 0.533 | 0.308 | 0.421 |

These are paired comparisons against the same 48 assistant references on a 1–5 scale; lower MAE is closer. The bootstrap 95% interval for the combined-error difference from the Luna/DeepSeek mean is [-0.127, +0.005] for Liquid and [-0.119, +0.006] for Jev. Neither establishes superiority. This subset is small and deliberately varied, and reference labels are not human gold.

**Scope sensitivity:** the nearly black T0571 has uncertain medium. Liquid assigned an exclusion probability of 0.592, causing the 1.00 floor. If uncertain cases instead use native decision scores, Liquid's combined MAE becomes 0.322 and Jev remains 0.318. The tiny ordering reverses; treat them as tied. The held-out set has 46 reference-eligible images, two uncertain images and no confirmed excluded images, so it does not establish exclusion recall. All selected models floor the known development poster T2000 at 1.00/1.00. Do not apply a floor solely because a photo is blurry, dark or contains a sign or artwork.

Decision models emit typed five-level distributions rather than generated JSON evidence sentences. Their native expected score is 0–4; add one to compare with the 1–5 rubric. When `excluded_probability >= 0.50`, both displayed values are floored at 1.00. The threshold is a fixed study rule, not a validated probability calibration. Raw distributions and pre-floor scores are retained. The Jev probability buckets are rounded to hundredths; an initial consistency failure was revalidated with a 0.055 rounding tolerance without resending the image. Haiku used the strict schema in [teacher-prompt.json](teacher-prompt.json); low reasoning was requested, and four of 72 responses reported positive reasoning tokens.

### Dropped models

Per the current decision, **Drex Max** and **Decider 1.2** are excluded from the active comparison and per-image working data. Their 48-case aggregate results are retained in [results.json](results.json) for context:

| Dropped model | Technical MAE | Aesthetic MAE | Mean MAE | Technical bias |
|---|---:|---:|---:|---:|
| Drex Max | 0.486 | 0.225 | 0.355 | +0.437 |
| Decider 1.2 | 0.555 | 0.240 | 0.397 | +0.476 |

Both were notably less accurate on technical scoring and had a stronger positive technical bias than Liquid and Jev. Drex's aesthetic MAE was the best in this sample, and its overall mean was below the Luna/DeepSeek mean, so the decision to drop it is not evidence of uniform underperformance. Differences on 48 images are uncertain; this is a prioritization decision rather than a population-wide conclusion.

## 2,000-image Luna/DeepSeek run

[scores-2000.csv](scores-2000.csv) retains per-image technical and aesthetic results, eligibility flags, the 50/50 means and request-cost fields. [manifest-2000.json](manifest-2000.json) retains the source identity, family grouping and delivered-view hashes. Source paths in the historical manifest and earlier CSVs describe the original machine and are not portable file links; the 72 comparison views are present in `images/`. [references-72.jsonl](references-72.jsonl) retains the frozen assistant scores and evidence. Two images outside the 72-image review needed deterministic source-format replacements; see [source-format-resolution.json](source-format-resolution.json).

The sample has 1,999 cases fresh to the recorded trials and the known poster, spanning 1,640 source families. It includes 1,279 standalone images, 240 authentic capture-failure images and 480 controlled images from 120 parent families. Both models were requested at high reasoning with the same image views, photography policy and schema-constrained JSON. All 2,000 selected responses per model and all 8,000 task scores were valid. DeepSeek reported positive reasoning tokens on every response but did not return an effort-confirmation header.

On the 71 fresh reference cases, the Luna/DeepSeek mean had combined MAE 0.342, Luna 0.363 and DeepSeek 0.389. The average's technical bias was +0.304. The mean was closest on aesthetics; Luna was closest on technical quality. In the 120 controlled distortion families, technical scores were non-increasing with severity in 94 families for Luna, 78 for DeepSeek and 85 for their mean. These results do not make the mean an independent reference or prove it is best on every dimension.

The known reported 2,000-image API cost was $2.591454105 for both models; five explicit capacity refusals were unpriced. The decision trial added $0.100853056 of known cost across all four tested decision models and Haiku; two model-busy refusals were unpriced. Those are reported usage totals, not reconciled invoices.

## Earlier 72-image comparison

The earlier 72-image corpus used different images from the reference subset above. Its four-model photo-scope comparison and DeepSeek high-reasoning follow-up are preserved as image pages and score tables:

- Photo scope: [page 1](earlier-comparison/photo-scope-page-1.png), [page 2](earlier-comparison/photo-scope-page-2.png), [scores](earlier-comparison/photo-scope-scores.csv).
- DeepSeek high versus none: [page 1](earlier-comparison/deepseek-high-page-1.png), [page 2](earlier-comparison/deepseek-high-page-2.png), [scores](earlier-comparison/deepseek-high-scores.csv).

The older figures are historical comparisons and include Gemini, which is not a current candidate. DeepSeek high reasoning on those 72 cases improved aesthetic MAE from 0.2958 to 0.2611 and worsened technical MAE from 0.3583 to 0.3931. The 1.00 eligibility floor correctly caught the known poster but its scope recall was not independently established.

## Rebuilding image pages

Run `python gallery.py` and `python render_pages.py` from this directory; the latter requires Pillow. Both read only the retained 72-image CSV and local PNGs; no model calls or API keys are needed. [image-pages-validation.json](image-pages-validation.json) records the source CSV hash and image-page hashes. The consolidated workspace keeps selected native responses and summaries; repetitive per-request audit snapshots and unused candidate responses are omitted. The earlier committed raw trial bundles remain in repository history at commit `030da29`.
