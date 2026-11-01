# Photo labeling

The working teacher label is the **50/50 mean of GPT-6 Luna and DeepSeek V4.1 Flash** for each of technical quality and aesthetics. Each teacher inspects the same prepared image independently. Scores are judgments on a 1–5 scale, not human opinion scores. Keep both teacher responses and the mean; a missing or invalid task stays missing and must not be replaced with an invented score.

The current material is a rehearsal and requires review before training. The [records](records/) contain 4,000 image views from 3,700 source families, including 500 assistant-assessed reference images. [scores.csv](records/scores.csv) retains the baseline results, the two-teacher mean, the reference and the comparison-model observations, with scores, eligibility fields, latency, cost and provenance. The [manifest](records/manifest.json) describes the image families, splits and delivered-view hashes. Images and raw responses remain in the local data directory recorded in those files; the absolute paths are host-specific. Reference latency is visual-review duration, not an HTTP response time. The Q-ReAlign Lite and reference prices were recorded as $0.00; infrastructure costs are separate. None of the rows is approved for training.

On the shared host, the [interactive corpus gallery](http://127.0.0.1:8765/gallery/index.html) and [eligibility comparison](http://127.0.0.1:8765/gallery/expanded-eligibility.html) display the delivered views and recorded scores.

The retained Liquid and Jev scores were collected with their earlier eligibility question. The revised decision prompts below are the current candidates for future calls; they have **not** been applied retroactively to the 4,000 records. Do not treat the old scope decisions as newly rescored results.

Both numerical scores are present for 4,000 Luna and 3,997 DeepSeek views, so the retained mean has 3,997 complete pairs. Liquid and Q-ReAlign Lite have 4,000 pairs each; Jev has 3,987. All 500 reference cases have both scores. Check the score values themselves as well as the validity fields: two DeepSeek responses were valid but contain null assessments.

## Current request configuration

| Model | Role | Endpoint | Reasoning |
|---|---|---|---|
| `openai/gpt-6-luna` | Teacher | OpenRouter chat completions | High requested |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | Teacher | RunInfra chat completions | High requested |
| `liquid/d1.1` | Comparison and eligibility review | OpenRouter Decisions | Provider's decision interface |
| `typesafe/jev-1.14` | Comparison and eligibility review | OpenRouter Decisions | Provider's decision interface |

[routes.json](routes.json) records the exact requested route settings and credential environment-variable names. Provider/model availability, prices, and parameters may change; check them and set a budget before any new run. The specified provider price ceiling is a request constraint, not a current price quote. Never commit credentials.

[teacher.json](prompts/teacher.json) contains the shared two-task teacher prompt and strict JSON schema. It requests each score as a two-decimal string in 0.10 steps, plus task-specific evidence and limitations. [decision.json](prompts/decision.json) contains the current Liquid and Jev questions: two typed quality scores and one explicit yes/no exclusion decision. Their native 0–4 expected scores are converted to the 1–5 scale by adding one. Preserve the native response and its precision; round only for display.

`visual_quality.labeling_profile` loads this bundle and constructs the corresponding payload from prepared PNG bytes. It does not send requests, prepare views, validate provider responses, handle budgets or publish labels. The existing `visual-pilot label` command is a separate technical-only pilot and does not implement this current two-teacher workflow.

From `visual-quality/`, run `uv run python labeling/verify.py` to check the profile, record hashes, score bounds and retained row counts without network access. Add `--images` on the machine holding the views to hash all 4,000 image files as well.

## Photography eligibility

The image itself is excluded when it is clearly a screenshot or interface, a flat poster, document, chart, illustration or artwork reproduction, or an obvious computer render. For clearly excluded content, the teacher prompt floors **both** dimensions at 1.00. In the decision interface, the existing research rule applies that floor when the excluded-answer probability is at least 0.50. That threshold is a study rule, not a calibrated probability or proof of eligibility.

A photograph of a physical object or scene remains eligible when visible depth, perspective or surroundings establish the context, even if it contains text, packaging, a sign, screen or artwork. Keep genuinely uncertain cases eligible for scoring and flag them for review; do not infer a screenshot or computer generation from blur, unusual content or low resolution alone. Unreadable images and failed requests are missing, not 1.00 ratings. Keep native scores, scope evidence and each model's result so disagreements and ambiguous cases can be reviewed before a floor or label is adopted.

## Review and next work

The two-teacher average is the working choice because it was closest to the assistant references on the principal 300-image comparison. The tested image-statistics text did not improve overall technical accuracy and caused regressions, so statistics are review cues rather than prompt inputs or automatic corrections. The more explicit decision questions catch many missed exclusions, but ambiguous photographed artwork still needs adjudication. Some alternative yes/no wording reversed decisions in development; keep the polarity explicit and validate changes before use.

The held-out eligibility check contained 60 eligible and 60 excluded views, including synthetic screenshots and previously scored natural works. Its labels are assistant judgments, and three physical-object boundaries have plausible alternative interpretations. Further human-reviewed, family-separated examples should cover real screenshots, artwork, packaging, selective focus, low light and sharp-background/soft-subject images. Continue random audits alongside disagreement and image-statistics review. Recheck both quality dimensions, false floors, missed exclusions, score drift, latency and cost before a larger labeling run. Embedding similarity helps organize the corpus; it is not a quality label.
