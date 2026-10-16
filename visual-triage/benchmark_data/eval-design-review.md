# Evaluation design review

Date: 2026-10-11. This is a proposed protocol, written before the expanded benchmark calls. The run record must state which choices were adopted and any changes made before execution.

Use the original 30 images for development. Keep two new panels of 60 images each for final evaluation: one for technical quality and aesthetics, and one for AI-origin detection. All compared models must receive the same delivered bytes for each panel. The panels must not share source images, parent scenes, or known prompt groups with development. Do not claim that this prevents overlap with the models' unknown pretraining data.

## Assessment and split controls

Store an assistant assessment for every original image before new model calls. Inspect the delivered image, not just a contact-sheet thumbnail. Record technical quality and aesthetics on the fixed 1–5 scale, a short visible-evidence note, uncertainty, visual suspicion of AI origin, and any known source label in separate fields. These are model-reference judgments, not human gold. Earlier predictions and source metadata were already visible in this conversation, so the assessment is not blind and can contain anchoring bias. Do not change it after seeing trial results. If a factual correction is needed, preserve both versions and report its effect.

The original panel has only three distortion parents. Its six AI images also share the prompt “statue of a man.” A split that separates these images creates a weak impression of independence. This proposed split keeps each distortion parent and the shared AI prompt together:

| Purpose | Sample suffixes | Count |
| --- | --- | ---: |
| Tune | 000–007, 012, 013, 015, 016, 018, 020, 022, 024–029 | 21 |
| Development check | 008–011, 014, 017, 019, 021, 023 | 9 |

The nine-image check is already exposed historical material. It is a development check, not a sealed test. It contains no AI positives; it can reveal false positives but cannot validate AI recall. Keep this limitation explicit. The new AI panel provides the first independent comparison in this exercise, subject to its own source limits.

## Four prompt rounds

Keep the output schema, delivered view, endpoint, provider policy, scale, and scoring code fixed. For each round, save the full prompt, hash, motivation, allowed evidence, and results. Run all 21 tune images for each round. Use the tune results to choose only general changes. Do not insert sample names, dataset names, generator names, subjects, expected labels, or per-image examples.

| Round | Change to investigate |
| --- | --- |
| 1 | Baseline with separate quality, aesthetics, and AI-suspicion fields. Preserve the original anchors where possible. Treat this schema change as a new baseline; do not compare its costs as if it used the old six-question payload. |
| 2 | Improve technical-quality guidance: score visible defects at the supplied scale; permit a valid low score for a very damaged image; do not infer hidden full-resolution damage. |
| 3 | Improve aesthetics guidance: distinguish composition, visual order, light, and expressive effect from sharpness, subject appeal, and photographic medium. Clean charts and renders can have high technical quality. |
| 4 | Improve origin guidance: require specific visible inconsistencies; separate conventional CGI from generative AI; preserve uncertainty for plausible images and avoid treating blur, polish, stylization, or low quality as proof. |

After round 4, run all four frozen prompts once on the nine-image development check. Do not edit any prompt after inspecting this check. Select the quality/aesthetics candidate with the lowest group-balanced mean absolute error across the two axes. For each axis, average errors within a source-parent group, then average groups. Give the two axes equal weight. Count a missing or invalid scalar as an error of 4 in this selection metric; also report the valid-only MAE and coverage separately. If candidates differ by less than 0.05 points, use the shorter prompt, then lower measured cost as a second tie-break. These cutoffs are design choices, not statistical guarantees.

Report the AI behavior of every round, but do not select an AI threshold from the six correlated positives. Freeze the general origin rubric before opening the new AI results. If a separate AI prompt is selected from development, record the reason and the lack of positive validation evidence. Use the same final rubric for the comparator models. Keep model-specific transport instructions separate from task meaning.

## Final metrics and uncertainty

For quality and aesthetics, report MAE, rank correlation, valid coverage, and the per-image differences from the assistant reference. Report human dataset ratings separately and preserve their original construct and scale. An AGIQA perception MOS is not automatically a technical-quality or aesthetics label. Provide group-level results when scenes are related. If uncertainty intervals are used for reference scores, show interval error as a secondary result; do not alter the primary selection metric after viewing scores.

For AI detection, use verified dataset provenance for the binary target where available. Keep visual suspicion and provenance distinct. Report confusion counts, AI recall, false-positive rate on real images, balanced accuracy, and abstention coverage. Report CGI and diagrams as a separate non-generative control slice. Compare Brier scores only when every model returns the same defined probability of generative origin; a multiclass medium probability is not interchangeable. Treat probability calibration from 60 selected images as exploratory. Do not use these data to authorize automatic rejection.

Freeze manifests, hashes, references, prompt winner, reasoning effort, metrics, and selection rules before final calls. Evaluate the selected Decider prompt, Gemini, and Luna once on both new panels. Do not rerun only inconvenient cases. A repair must use a documented protocol and retain the failed call and its cost. If labels or prompts change after final results are visible, call that panel development and acquire a new evaluation panel.

## Gemini reasoning and measured cost

[Google documents](https://ai.google.dev/gemini-api/docs/generate-content/thinking) `low`, `medium`, and `high` thinking levels for Gemini 3.8 Flash; `medium` is the default and `minimal` returns an error. [OpenRouter maps](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) `reasoning.effort` directly to Google's thinking level. Effort names do not imply equal compute across Gemini and Luna.

Use a small paired sensitivity run before choosing effort. A fixed 12-image tune subset is 000, 003, 004, 007, 012, 015, 018, 020, 024, 025, 027, and 029. Compare low and medium effort with identical task prompts and images. Alternate request order to reduce time-of-day effects. Choose low unless medium reduces the quality/aesthetics MAE by at least 0.10 points without worse valid coverage or more false AI flags on known non-generative controls. Report both settings even if their results disagree. This is a screening rule, not proof of the best effort setting. Do not choose effort from the final panels.

Use a completion cap that leaves room for thinking and the final JSON; 4,096 tokens is a reasonable initial bounded choice for this compact task. If preflight shows truncation, revise it before the paired comparison and use the same cap for both efforts. [Google's token-limit guidance](https://ai.google.dev/gemini-api/docs/thinking) states that thought tokens share the output cap. A small cap can produce empty output while still incurring charges. Retain finish reason, token counts, reasoning tokens where available, and schema failures.

The [current OpenRouter price](https://openrouter.ai/google/gemini-3.8-flash) is $0.75 per million input tokens and $3.75 per million output tokens. Thinking is billed as output. Google's [release announcement](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/) states that introductory pricing ends on 2026-12-31, with $1.50/$7.50 rates from 2027-01-01. Fetch the live catalog before requests; base experiment totals on reported usage, not this price estimate.

Report total cost, cost per attempted image, cost per valid result, median latency, and nearest-rank P95 for each model, effort, task, and provider. Latency is end-to-end request time from this machine. Keep retries and development costs separate from final evaluation, but include both in the experiment total. A provider change creates a separate stratum.

Research used three Exa searches with 15 requested result slots and four direct primary-source fetches. Result slots are not independently verified sources. No paid model calls were made for this design review.
