# DeepSeek high reasoning on the same 72 images

Completed all 72 requests with 144 valid scores. **High reasoning improved aesthetic agreement but worsened technical agreement**, with no change in their equally weighted mean absolute error. It cost $0.03646735, versus $0.01690555 for the preceding no-reasoning run (about 2.16× as much).

The preceding DeepSeek V4.1 Flash 1017 run explicitly requested `reasoning_effort: "none"`; all 72 responses reported zero reasoning tokens. This repeat requests `high` through the same RunInfra endpoint and keeps the exact image bytes, photography eligibility policy, quality prompts, strict JSON schema, temperature and reference judgments.

The output allowance increases from 2,048 to 16,384 tokens so reasoning has room before the final JSON. These are the only two changed request fields, checked separately for every image before sending. The previous four-model run remains available for comparison. See the [protocol](protocol.md), [route settings](routes.json) and [provider verification](provider-notes.md).

P048 was sent first and counts toward the 72. It returned valid structured scores of 1.00/1.00 and reported 334 reasoning tokens. The provider did not return an applied-effort header for this request: `high` is the requested setting, and nonzero usage confirms reasoning occurred without independently measuring the provider's internal effort allocation.

Across the completed run, all 72 responses reported positive reasoning usage: **32,329 tokens total**, ranging from 74 to 1,615 per image. Every completion finished with `stop`; none exhausted its output allowance. No applied-effort header was returned on any response. P048 remained the only explicit scope exclusion, with no incorrectly excluded reference photographs. All requested scores passed the existing task contracts.

## Effect on agreement with the references

The following comparisons use the same 72 cases with valid scores in both conditions. A positive error change means worse agreement.

| Scoring method | Task | None MAE | High MAE | Change |
|---|---|---:|---:|---:|
| DeepSeek | Technical | 0.3583 | 0.3931 | +0.0347 |
| DeepSeek | Aesthetic | 0.2958 | 0.2611 | −0.0347 |
| Average of DeepSeek and Luna | Technical | 0.3139 | 0.3465 | +0.0326 |
| Average of DeepSeek and Luna | Aesthetic | 0.2160 | 0.2056 | −0.0104 |

DeepSeek's equally weighted mean across both tasks is 0.3271 in either condition. The DeepSeek/Luna average worsens overall from 0.2649 to 0.2760. This does not support switching both dimensions to high reasoning on the present evidence. High reasoning is a candidate for further aesthetic evaluation; choosing different settings per task now would be a post-hoc choice that needs a fresh sample.

The direction is the same on the 70 eligible photographs, the 49 retained eligible photographs and the 22 additions: technical error rises while aesthetic error falls. On the 22 additions, DeepSeek technical MAE changes 0.3500 → 0.3955 and aesthetic MAE changes 0.3136 → 0.2864. The technical regressions extend beyond the familiar examples.

There are useful gains and clear regressions. P023 technical improves 3.20 → 3.90 against reference 4.00; P045 aesthetic improves 3.20 → 4.00 against reference 3.90. But P051 technical rises 1.60 → 2.80 against reference 1.60, and P057 aesthetic rises 2.00 → 2.70 against reference 1.70. The collage retains these disagreements. Neither the references nor student outputs were adjusted to make high reasoning look better.

## Reading the comparison

The [72-image comparison grid](collage/originals-grid.html) and [PNG](collage/originals-grid.png) show Ref, DeepSeek none, DeepSeek high, Luna high, Gemini medium and Haiku low. For easier inspection, use [images 1–36](collage/originals-grid-page-1.png) and [images 37–72](collage/originals-grid-page-2.png). The original delivered images are fitted without cropping or generative redrawing. The [CSV](collage/scores.csv) keeps exact two-decimal scores and validation information.

Only DeepSeek was rerun. Luna, Gemini and Haiku scores and costs are reused from the preceding 72-image trial. Invalid task responses remain invalid, while independently valid sibling scores remain visible. No responses are manually repaired, and no image requests are retried.

Mean absolute error (MAE) measures distance from the frozen assistant assessments on the 1–5 scale; lower is closer. References are not human gold. The sample contains 54 image families, reused development examples, and only one confirmed non-photographic case. A single response per setting is descriptive evidence, not a causal or statistically established accuracy result. The larger output cap is also a changed setting.

The [complete analysis](analysis.json) reports all 72 images, the 70 eligible photographs, the 49 retained eligible images and the 22 additions separately. It compares valid scores on matching cases, reports missing/invalid coverage, and includes both the DeepSeek-only comparison and the DeepSeek/Luna averages. The [cost ledger](cost-summary.json) reconciles individual raw usage records.

## Reproduction and validation

The durable runner freezes request identities and source hashes, journals before sending, resumes only unattempted cases and limits this new run to $2. Four offline tests cover the permitted setting changes, resume without resends, uncertain attempts and input integrity. Paid responses are stored under `visual-quality/data/deepseek-high-trial-2026-10-31/deepseek/images/`.

These commands regenerate derived artifacts without additional inference:

```sh
visual-quality/.venv/bin/python visual-quality/experiments/deepseek-high-trial-2026-10-31/runner.py verify
visual-quality/.venv/bin/python visual-quality/experiments/deepseek-high-trial-2026-10-31/analyze.py
visual-quality/.venv/bin/python visual-quality/experiments/deepseek-high-trial-2026-10-31/build_collage.py
node visual-quality/experiments/deepseek-high-trial-2026-10-31/collage/render.mjs
```

The [independent integrity review](final-integrity-review.json) and [render validation](collage/render-validation.json) record completed checks and artifact hashes.

## Total cost per model

Costs cover the preceding 72-image four-model trial plus this 72-image DeepSeek rerun; earlier 50/200-image experiments are excluded. Each call scores both tasks. Only DeepSeek incurred new charges in this repeat. These are exact API-reported costs, including confirmed Luna BYOK upstream usage, reconciled against saved responses; they are not invoice verification. No calls are unpriced. Cache discounts are included where reported, so the cost ratio describes these runs. See the [machine-readable ledger](cost-summary.json).

| Model | Original 72-image run (USD) | High-reasoning rerun (USD) | Total calls | Total cost (USD) |
|---|---:|---:|---:|---:|
| DeepSeek V4.1 Flash 1017 | 0.016905550 | 0.036467350 | 144 | 0.053372900 |
| GPT-6 Luna, high | 0.055711975 | 0 | 72 | 0.055711975 |
| Gemini 3.8 Flash, medium | 0.321563250 | 0 | 72 | 0.321563250 |
| Claude Haiku 5.5, low | 0.058444000 | 0 | 72 | 0.058444000 |
| **Total** | **0.452624775** | **0.036467350** | **360** | **0.489092125** |
