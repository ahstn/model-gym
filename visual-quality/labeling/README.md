# Photo labeling

The canonical ensemble label is **37.5% GPT-6 Luna, 37.5% DeepSeek V4.1 Flash and 25% Liquid d1.1**, independently for technical quality and aesthetics. This accepted configuration replaces the previous Luna/DeepSeek mean for current and historical result views. Each model inspects the same prepared image independently. Scores are judgments on a 1–5 scale, not human opinion scores. Keep all component responses and their provenance; a missing or invalid component makes that task's ensemble score missing, with no two-model fallback or silent reweighting.

[`visual_quality.ensemble`](../src/visual_quality/ensemble.py) owns the implementation; [ensemble.json](ensemble.json) records its configuration. Arithmetic uses native component precision, with two decimal places only for display. A derived ensemble has no independent HTTP latency or charge: these fields are null, while component observations retain measured latency and price. Component pricing is also summarized separately and must not be added again to model spending. The scorer does not invent an eligibility vote: it uses the already-applied component scores and flags scope disagreements for review.

The current material requires review before training. The pre-expansion [records](records/) describe the earlier 4,000 views and 500 assistant references; their raw observations remain provenance, not a fresh run with the revised prompts. The 10,000-image expansion has a 1,500-image reference benchmark. Only the first 1,000 benchmark images' scores and labels are eligible for new git publication; remaining per-image results, responses and galleries stay in ignored local data. The publication allowlist, rather than a row count, enforces that image-level limit. The [manifest](records/manifest.json) describes image families, splits and delivered-view hashes. Images and raw responses remain in the local data directory recorded in those files; the absolute paths are host-specific. Reference HTTP latency is null, and visual-review duration is stored separately. Q-ReAlign Lite and reference prices are $0.00; infrastructure costs are separate. None of the rows is approved for training.

On the shared host, the [interactive corpus gallery](http://127.0.0.1:8765/gallery/index.html) and [eligibility comparison](http://127.0.0.1:8765/gallery/expanded-eligibility.html) display the delivered views and recorded scores.

The earlier Liquid and Jev observations used their earlier eligibility question. The expansion reruns these models with the revised decision prompts. Historical canonical ensemble overlays can be recalculated from existing observations, but that recalculation does not make old component responses newly rescored results. Reused scores keep their image, request, prompt and response provenance.

In the earlier 4,000-image run, Luna and Liquid have 4,000 numerical pairs, DeepSeek has 3,997, and the canonical ensemble therefore has 3,997 complete pairs. Q-ReAlign Lite has 4,000 pairs and Jev has 3,987. All 500 earlier reference cases have both scores. Check score values and validity fields: a structurally valid response can contain a null assessment.

## Current request configuration

| Model | Role | Endpoint | Reasoning |
|---|---|---|---|
| `openai/gpt-6-luna` | Teacher | OpenRouter chat completions | High requested |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | Teacher | RunInfra chat completions | High requested |
| `liquid/d1.1` | Ensemble component and eligibility review | OpenRouter Decisions | Provider's decision interface |
| `typesafe/jev-1.14` | Comparison and eligibility review | OpenRouter Decisions | Provider's decision interface |

[routes.json](routes.json) records the exact requested route settings and credential environment-variable names. Provider/model availability, prices, and parameters may change; check them and set a budget before any new run. The specified provider price ceiling is a request constraint, not a current price quote. Never commit credentials.

[teacher.json](prompts/teacher.json) contains the shared two-task teacher prompt and strict JSON schema. It requests each score as a two-decimal string in 0.10 steps, plus task-specific evidence and limitations. [decision.json](prompts/decision.json) contains the current Liquid and Jev questions: two typed quality scores and one explicit yes/no exclusion decision. Their native 0–4 expected scores are converted to the 1–5 scale by adding one. Preserve the native response and its precision; round only for display.

`visual_quality.labeling_profile` loads this bundle and constructs the corresponding payload from prepared PNG bytes. It does not send requests, prepare views, validate provider responses, handle budgets or publish labels. The existing `visual-pilot label` command is a separate technical-only pilot and does not implement this current two-teacher workflow.

From `visual-quality/`, run `uv run python labeling/verify.py` to check the profile, record hashes, score bounds and retained row counts without network access. Add `--images` on the machine holding the views to hash all 4,000 image files as well.

## Photography eligibility

The image itself is excluded when it is clearly a screenshot or interface, a flat poster, document, chart, illustration or artwork reproduction, or an obvious computer render. For clearly excluded content, the teacher prompt floors **both** dimensions at 1.00. In the decision interface, the existing research rule applies that floor when the excluded-answer probability is at least 0.50. That threshold is a study rule, not a calibrated probability or proof of eligibility.

A photograph of a physical object or scene remains eligible when visible depth, perspective or surroundings establish the context, even if it contains text, packaging, a sign, screen or artwork. Keep genuinely uncertain cases eligible for scoring and flag them for review; do not infer a screenshot or computer generation from blur, unusual content or low resolution alone. Unreadable images and failed requests are missing, not 1.00 ratings. Keep native scores, scope evidence and each model's result so disagreements and ambiguous cases can be reviewed before a floor or label is adopted.

## Review and next work

The accepted three-model weights balance Liquid's complementary errors against its tendency to underrate some high-quality images. They are the current configuration, not a claim of optimal weights or human-validated accuracy. The tested image-statistics text did not improve overall technical accuracy and caused regressions, so statistics are separate debugging and review metadata rather than prompt inputs or automatic corrections. The more explicit decision questions catch many missed exclusions, but ambiguous photographed artwork still needs adjudication. Some alternative yes/no wording reversed decisions in development; keep the polarity explicit and validate changes before use.

## Historical result migration

Create a canonical export without changing original observations:

```sh
PYTHONPATH=src python -m visual_quality.ensemble \
  --input experiments/rehearsal-4000-2026-11-01/scores-long.jsonl \
  --output data/rehearsal-4000-2026-11-01/canonical/scores-long.jsonl
```

The command removes retired `luna_deepseek_mean` rows, preserves every model/reference observation, and derives `photo_ensemble` rows. Optional `--published-case-ids` takes a JSON array of at most 1,000 distinct case IDs for a commit-eligible export. Historical stats-arm ensembles explicitly retain baseline Liquid while using the corresponding Luna/DeepSeek arm. Frozen historical reports describe the original analysis; canonical score exports and regenerated galleries supersede the retired ensemble display without rewriting provider evidence.

The held-out eligibility check contained 60 eligible and 60 excluded views, including synthetic screenshots and previously scored natural works. Its labels are assistant judgments, and three physical-object boundaries have plausible alternative interpretations. Further human-reviewed, family-separated examples should cover real screenshots, artwork, packaging, selective focus, low light and sharp-background/soft-subject images. Continue random audits alongside disagreement and image-statistics review. Recheck both quality dimensions, false floors, missed exclusions, score drift, latency and cost before a larger labeling run. Embedding similarity helps organize the corpus; it is not a quality label.
