# Four-model scoring prompt trial — 30 October 2026

Keep the baseline prompts and the separate DeepSeek–Luna averages for now. Testing the four additions individually and together did not establish a reliable improvement over the baseline. The combined prompt increased the pair's technical mean absolute error (MAE) from **0.328 to 0.344** and aesthetic MAE from **0.179 to 0.190** over all 50 images. These small increases are uncertain, but they provide no reason to replace the baseline. No prompt change or trial label has been promoted into the production/training workflow.

This evaluates agreement with frozen, blind assistant assessments, **not human accuracy**. Lower MAE means closer to those assessments on the 1–5 scale. Technical quality and aesthetics remain independent throughout.

The requested [36-image collage](collage/originals-grid.png) is complete. Its [HTML version](collage/originals-grid.html) opens each original image and exposes evidence on hover. It shows the fixed **combined** prompt for all four models, plus the assistant reference. That display arm was chosen before outcomes; it is not the winning prompt. Images are the original scored PNGs, scaled to fit without cropping or generative redrawing. [Exact scores and evidence](collage/scores.csv) retain two-decimal score strings; the grid displays one decimal for readability.

## What was tested

The earlier 50-image and 200-image experiments informed the proposed changes, so this trial used a fresh, similar corpus rather than reusing those development images. The 50 views represent 32 source families: six controlled families with a parent and three distorted variants each (two blur, two noise, two JPEG), fourteen VizWiz images, and twelve standalone Megalith images. Previously exposed families, exact hashes and conservative near duplicates were excluded. Source parents are not necessarily pristine.

Each model scored every image under six conditions: baseline, A alone, B alone, C alone, D alone, and A+B+C+D. That is **1,200 completed image requests**, producing 2,400 potential task ratings. There were no failed, unsent or unreconciled image requests. Each condition was sampled once. Arm order was deterministically interleaved; all routes received identical full-frame image bytes, with a maximum edge of 1,536 pixels and a shared viewing scale within each controlled family.

| Model | Route | Requested reasoning |
|---|---|---|
| DeepSeek V4.1 Flash | RunInfra `deepseek-ai/DeepSeek-V4.1-Flash-1017`, API alias `deepseek-v4-1-flash-1017` | None |
| GPT-6 Luna | OpenRouter `openai/gpt-6-luna`, OpenAI provider | High |
| Gemini 3.8 Flash | OpenRouter `google/gemini-3.8-flash`, Google AI Studio provider | Medium |
| Claude Haiku 5.5 | OpenRouter `anthropic/claude-haiku-5.5`, Anthropic provider | Low |

All returned identities matched the configured routes. Haiku received the requested low setting; its usage records reported zero reasoning tokens, so this trial cannot establish how much internal thinking occurred. Exact settings and provider metadata are saved in [routes.json](routes.json), the catalog snapshots and each request record.

Three fresh-context assistant reviewers inspected anonymous original images individually using the baseline rubric, without seeing student predictions, proposed additions or source metadata. Each image had one reviewer. Point scores and subjective uncertainty intervals were frozen before inference, together with the corpus, prompts, analysis code and display selection. The [protocol](protocol.md), [reference freeze](references-freeze.json), [references](references.jsonl) and [exact additions](arms.json) preserve the method.

## Effect of each addition

Each cell below is **technical / aesthetic ΔMAE**, addition minus that model's baseline on identical valid cases. Negative is an observed positive impact; positive is an observed negative impact. A numerical change alone is not evidence that it will generalize. The [complete tables](analysis-tables.md) give every interval, coverage count, effect classification and regression check.

| Addition | DeepSeek | Luna | Gemini | Haiku |
|---|---:|---:|---:|---:|
| A — Recognizability versus fidelity | +0.040 / −0.026 | −0.002 / +0.064 | −0.024 / −0.002 | +0.012 / −0.023 |
| B — Main subject and focus | +0.052 / +0.018 | −0.008 / +0.014 | −0.038 / −0.026 | +0.046 / −0.013 |
| C — Whole-image aesthetic contribution | +0.014 / −0.004 | 0.000 / +0.024 | 0.000 / +0.022 | +0.012 / −0.007 |
| D — Score anchors and consistent evidence | +0.010 / +0.008 | +0.012 / +0.070 | −0.008 / −0.024 | +0.008 / −0.005 |
| All four combined | +0.024 / −0.026 | +0.020 / +0.020 | −0.046 / +0.030 | +0.002 / −0.019 |

**A has a useful local effect but an unfavorable tradeoff.** DeepSeek's technical error on targeted controlled noise/JPEG variants decreased by 0.050, while error on the other 38 images increased by 0.068. On the 17 high-reference technical images it increased by 0.188. Luna's aesthetic error increased by 0.064 overall. Haiku's aesthetic aggregate improved, but the targeted subset worsened by 0.100 while the complement improved by 0.055. The overall result would hide that reversal. Do not adopt A across the board.

**B did not reliably improve focus judgments.** DeepSeek worsened on both dimensions. Haiku's technical error increased by 0.046 overall and 0.052 outside the targeted blur cases; its targeted blur error was essentially unchanged. Gemini improved overall, but its technical error on the six targeted blur views increased by 0.050. Luna's small technical improvement accompanied an aesthetic regression. This does not support a general focus-prompt change.

**C was mostly neutral or mixed.** It produced no clear aggregate benefit, and aesthetic error increased for Luna and Gemini. The prespecified VizWiz proxy showed aesthetic improvements for Luna and Haiku but deterioration for DeepSeek and Gemini. VizWiz is an imperfect proxy for difficult composition, so this cannot establish a universal effect on clutter or subject hierarchy.

**D did not improve calibration consistently.** DeepSeek and Luna worsened on both dimensions; Luna's aesthetic error increased by 0.070. Gemini improved slightly, while Haiku's technical error increased and aesthetic validity deteriorated. Haiku's valid aesthetic coverage fell from 49/50 at baseline to 44/50 with D. Better-sounding anchor instructions did not translate into a dependable scoring gain.

Across the 32 exploratory individual-addition comparisons, nominal 95% intervals excluded zero for Luna's aesthetic regressions with A and D, Haiku's technical regression with B, and Gemini's aesthetic improvement with B. These were not corrected for those 32 comparisons and should not be treated as confirmed discoveries. All other individual intervals crossed zero.

For the eight primary combined-versus-baseline comparisons, **every Bonferroni-adjusted 99.375% interval crossed zero**. Gemini's technical improvement had an ordinary 95% interval below zero, but did not pass the prespecified correction. The DeepSeek–Luna combined-minus-baseline changes also remained uncertain: technical +0.016, nominal 95% interval [−0.026, +0.060]; aesthetic +0.011, [−0.021, +0.045].

## Which model was closest?

The following comparisons use the **same valid images across all four models within each task and arm**: 50 technical images in both arms, 49 aesthetic images at baseline and 48 in the combined arm. Thus comparisons across models within a column are fair; differences between columns are not the paired prompt-effect estimates above.

| Model or average | Baseline technical MAE | Baseline aesthetic MAE | Combined technical MAE | Combined aesthetic MAE |
|---|---:|---:|---:|---:|
| DeepSeek | 0.326 | 0.235 | 0.350 | 0.215 |
| Luna | 0.366 | 0.245 | 0.386 | 0.273 |
| Gemini | 0.530 | 0.441 | 0.484 | 0.465 |
| Haiku | 0.390 | 0.253 | 0.392 | 0.231 |
| DeepSeek + Luna, equal weights | 0.328 | 0.181 | 0.344 | 0.198 |
| All four, equal weights | 0.343 | 0.204 | 0.342 | 0.209 |

DeepSeek had the lowest observed individual error on both dimensions in both arms. Luna remains useful in the pair: on the full baseline sample, averaging it with DeepSeek reduced aesthetic MAE to 0.179 and was effectively tied with DeepSeek alone technically (0.328 versus 0.326). The pair also had a baseline technical rank correlation of 0.955. These findings support retaining the existing pair rather than fitting new weights to this small sample.

Haiku was competitive on aesthetics, especially under the combined prompt, but did not establish an advantage over the pair and had worse technical agreement and output reliability. Gemini remained furthest from the reference scores on both dimensions. At baseline it scored about 0.30 lower technically and 0.36 lower aesthetically on average; its ordering was considerably better than those absolute errors suggest. This supports leaving it out of the current uncalibrated average, rather than claiming it cannot rank image quality.

Adding all four models did not establish a useful gain. The all-four combined technical average was slightly better than the pair (0.342 versus 0.344), but the 0.002 difference was inconclusive. Its combined aesthetic error was higher on the same 48 cases. [Common-case metrics and exploratory comparison intervals](common-case-comparison.json) retain these qualifications; no ensemble weights were fitted.

## Regressions and explanation quality

The regression checks covered source strata, distortion operators/severity, source parents, variants, low/high reference scores, issue tags and targeted versus other images. There were no new critical contradictions under the prespecified extreme-score definition. That is a coarse safeguard, not proof of no degradation. The [1,464 subgroup rows](guardrail-details.csv) and per-case output document smaller regressions and cases where the prespecified 0.10 noninferiority margin could not be established.

In the combined arm, DeepSeek technical error increased by 0.107 on the fourteen VizWiz cases, and Haiku technical error increased by 0.133 on the six source parents. Small overlapping subgroup results remain exploratory. Several groups were too sparse to establish absence of harm.

Post-outcome visual inspection supplies concrete examples, not additional independent validation:

- **P024, JPEG-degraded church:** DeepSeek moved from 3.5 to 3.0, closer to the technical reference of 2.4. Visible large blocks remained under-penalized. Haiku stayed at 3.6, and A alone moved it to 3.8.
- **P047, purple-lit drummer:** DeepSeek moved from 2.6 to 3.5 against a technical reference of 2.5. Coarse grain, soft facial detail and clipped highlights remained visible. Its aesthetic score also moved further from the reference, from 3.6 to 4.0 against 2.7.
- **P042, outdoor gown photo session:** DeepSeek dropped from 4.5 to 4.0 against a technical reference of 4.6, penalizing minor softness despite substantial visible detail. This is a high-quality-image regression outside the intended severe-defect correction.
- **P027, defocused barcode:** DeepSeek rose from 1.8 to 2.3 against a reference of 1.8 while continuing to describe blur. Naming the defect did not ensure a suitable score.

Original paths and fuller observations are saved in [visual-check-notes.md](visual-check-notes.md). The controlled parent-to-severe technical gap increased for DeepSeek from 1.45 to 1.65 with the combined prompt, yet aggregate accuracy worsened. More separation is therefore not sufficient evidence of better discrimination.

A separate blind audit assessed **192 explanations** on twelve images selected before inference: baseline and combined, four models, two tasks. Reviewers saw original images and anonymous evidence text, without model names, arms, scores or references. Each explanation received one judgment; this was not an inter-rater study.

| Model | Material evidence issues, baseline | Material evidence issues, combined |
|---|---:|---:|
| DeepSeek | 2/24 | 4/24 |
| Luna | 2/24 | 0/24 |
| Gemini | 3/24 | 3/24 |
| Haiku | 4/24 | 4/24 |
| Total | 11/96 | 11/96 |

Material issues combine unsupported specific claims and contradicted claims. Examples included naming objects that severe blur did not establish and asserting compression without a distinct visible signature. Luna's evidence improved in this small audit even though its numeric error worsened. Across models the total issue count was unchanged. The [audit results](explanation-audit/results.json), anonymous packets and individual judgments preserve all findings. This audit covers baseline and combined only; it does not attribute explanation effects to individual additions.

## Structured output, missing ratings and costs

The schema retained exactly two tasks, each with task/status/score/evidence/limitations. Scores are strings from `"1.00"` through `"5.00"` in `"0.10"` increments, or null for an unscorable task. Two-decimal serialization does not imply hundredth-point measurement precision. Evidence targets 12–20 words with a hard 30-word maximum. Invalid tasks are excluded independently; a valid sibling task remains usable. None were silently repaired.

DeepSeek, Luna and Gemini each returned 300/300 fully valid image responses. **Haiku returned 284/300 fully valid responses**, with 17 invalid task ratings across sixteen replies: nine evidence-length violations and eight `status:"ok"` / `score:null` mismatches. Technical validity was 299/300 and aesthetic validity 284/300. This is a semantic contract failure despite structured JSON: shape and enum constraints do not by themselves enforce word counts or consistency between fields.

Haiku's aesthetic valid counts for baseline/A/B/C/D/combined were 49/49/49/45/44/48. Its paired combined aesthetic ΔMAE is −0.01875 on 48 common valid cases; subtracting its raw available-case means (0.253061 and 0.231250) would give the wrong paired estimate. A separately labeled [score-only sensitivity analysis](score_only_sensitivity.json) recovers the nine schema-valid scalar scores from length-only failures, while leaving null/status mismatches missing. It gives a combined aesthetic change of −0.014, with a nominal 95% interval [−0.052, +0.024], and does not reveal a clear improvement. It does not alter the main metrics or collage.

Before image inference, Haiku's first text probe rejected the nullable-enum schema encoding. All sixteen initial probe attempts and their method inputs were archived. The schema was changed for **every** route and arm to an equivalent `anyOf` string-enum/null representation, with identical accepted score values, baseline wording and validation semantics. All twenty replacement probes passed. The compatibility amendment is recorded in the protocol and [preflight archive](preflight-v1/archive.json), consistent with [Anthropic's structured-output documentation](https://platform.claude.com/docs/en/build-with-claude/structured-outputs). No image was scored under the rejected encoding.

Known API costs, including both probe rounds and image inference, were **$1.710105745**: DeepSeek $0.068230620; Luna $0.211880600; Gemini $1.228387425; Haiku $0.201607100. Luna's confirmed BYOK upstream charges are included; gateway and overlapping upstream costs are not blindly added. The original rejected Haiku probe returned no usage and remains unpriced. Its $0.54096 worst-case admission reserve is **not a measured charge**; known cost plus that reserve is $2.251065745, within the $20 total admission cap. [Accounting details](accounting-summary.json) retain every attempt and reconciliation, so the known subtotal should not be described as an exact final invoice.

## Limits and decision

The fresh corpus reduces direct overfitting to the images that motivated the additions, but it does not establish broad generalization or absence from model pretraining. Fifty views contain only 32 independent families, with 24 controlled views and two families per distortion operator. There are only two reference aesthetic scores at or above 4.0, so upper-end aesthetic discrimination remains weakly tested. Reference scores came from one assistant reviewer per image; inter-reviewer reliability and human agreement were not measured. Their intervals express subjective judgment uncertainty, not statistical confidence intervals.

Confidence intervals resample whole families within source strata using 10,000 draws. They describe uncertainty conditional on this corpus and these references; there were no repeated calls per condition to estimate output variability separately. Provider-internal image processing can differ despite identical bytes. Absolute errors should not be compared directly with the previous experiment as evidence that a model improved, because the images and references changed.

Retain the baseline DeepSeek–Luna pair, retain separate technical and aesthetic outputs, and do not promote A, B, C, D or their combination from this trial. In particular, do not select C for technical scores or B for aesthetics solely because those pair averages happened to be slightly better here; doing so would turn this holdout into another tuning set. Any further revision should be evaluated on another frozen sample with stronger high-aesthetic coverage and human-reviewed calibration examples. That is a next-step recommendation, not work performed in this trial.

## Artifacts and reproduction

The independent [final integrity review](final-integrity-review.json) verifies all 48 available MAEs, all 40 paired deltas, 83 pinned inputs per route, frozen-reference timing, model identities, cost accounting, all 360 collage cells and all 192 audit labels. The focused runner/schema/report suite passed 47 tests before inference. The renderer verified 36 decoded originals in six columns at 1,920 × 3,355 pixels; the completed PNG was also visually inspected.

Primary artifacts are [results.json](results.json), [all 2,400 task comparisons](scores-comparisons.csv), [readable tables](analysis-tables.md), [subgroup checks](guardrail-details.csv), [common-case comparisons](common-case-comparison.json), [accounting](accounting-summary.json), and the [collage validation](collage/render-validation.json). Durable requests, raw responses and parsed observations are under `visual-quality/data/prompt-ablation-2026-10-30/<route>/{probes,images}/`; original views are under that data directory's `views/`. They contain no API keys. This README and the supplemental summaries were written after outcomes; the frozen protocol and analysis were not retuned.

From the repository root, regenerate derived artifacts without additional API calls:

```sh
visual-quality/.venv/bin/python visual-quality/experiments/prompt-ablation-2026-10-30/report.py
visual-quality/.venv/bin/python visual-quality/experiments/prompt-ablation-2026-10-30/summarize_results.py
visual-quality/.venv/bin/python visual-quality/experiments/prompt-ablation-2026-10-30/common_case_comparison.py
visual-quality/.venv/bin/python visual-quality/experiments/prompt-ablation-2026-10-30/accounting.py
visual-quality/.venv/bin/python visual-quality/experiments/prompt-ablation-2026-10-30/score_only_sensitivity.py
visual-quality/.venv/bin/python visual-quality/experiments/prompt-ablation-2026-10-30/summarize_explanation_audit.py
visual-quality/.venv/bin/python visual-quality/experiments/prompt-ablation-2026-10-30/build_collage.py
node visual-quality/experiments/prompt-ablation-2026-10-30/collage/render.mjs
```
