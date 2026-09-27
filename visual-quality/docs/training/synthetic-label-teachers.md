# Teachers for synthetic image-quality labels

Checked: 2026-09-27. Prices are USD, before tax. Benchmark results are externally measured; costs are planning estimates. No scoring API calls or training runs were made.

Start with **Gemini 3.8 Flash for bulk labels**, **Claude Opus 5.5 for a second opinion**, and **GPT-6 Luna as the low-cost control**. Include GPT-6 Astra in the pilot as an accuracy reference. Select the final teacher on image-quality validation results, not MMMU-Pro rank. This is a proposed experiment, not evidence that these models match Q-Align.

## 1. Benchmark shortlist

The table uses one benchmark and one evaluator: [Artificial Analysis MMMU-Pro](https://artificialanalysis.ai/evaluations/mmmu-pro). Rank distinct model releases by their best published configuration. Do not count reasoning settings as separate models. Include older releases in the ranking; the evaluator's `deprecated` field is not proof that a provider has closed an endpoint.

The extracted model comparison data contains 279 configurations with a non-null MMMU-Pro result. The smaller default evaluation page contains only 23 configurations, so it cannot establish the full ranking by itself. The [saved evidence](exa-results/visual-teachers-2026-09-27.json) records the selected rows, source URLs, exact scores, and cost assumptions.

| Rank | Model / API ID | Evaluated setting | MMMU-Pro accuracy | Proposed role |
|---|---|---|---:|---|
| 1 | Claude Opus 5.5 / `claude-opus-5-5` | Adaptive, max effort, default fallback | 87.69% | Second teacher; test against Astra |
| 2 | GPT-6 Astra / `gpt-6-astra` | max | 86.88% | Expensive accuracy reference |
| 3 | Gemini 3.8 Flash / `gemini-3.8-flash` | high | 85.61% | First bulk-label candidate |
| 4 | Gemini 3.7 Flash / `gemini-3.7-flash` | high | 85.49% | Older-generation control |
| 5 | Claude Opus 5 / `claude-opus-5` | Adaptive, max effort | 84.74% | Older-generation control |
| Extra | GPT-6 Luna / `gpt-6-luna` | max | 75.55% | Requested low-cost control; outside top five |

Source: live page data from [Artificial Analysis's Gemini 3.7 comparison page](https://artificialanalysis.ai/models/gemini-3-7-flash), checked against the [evaluation page](https://artificialanalysis.ai/evaluations/mmmu-pro). These are benchmark accuracy percentages, not IQA correlations. Small differences do not establish a statistically significant lead. Opus 5.5's fallback configuration must remain part of its result label. A low-effort label run does not inherit a max-effort benchmark score.

Artificial Analysis marks Gemini 3.7 Flash and Opus 5 as deprecated in its data. Google still lists [no announced shutdown date for Gemini 3.7 Flash](https://ai.google.dev/gemini-api/docs/deprecations). Confirm endpoint access before scheduling a run. Prefer the newer releases for a new dataset.

### Benchmarks and their limits

| Benchmark | Evidence found | Use for this project |
|---|---|---|
| MMMU / MMMU-Pro | Current cross-provider MMMU-Pro scores above; [official HF dataset](https://huggingface.co/datasets/MMMU/MMMU_Pro) | Shortlist general vision models. Keep standard and vision-only variants separate. Do not combine original MMMU scores with MMMU-Pro. |
| RealWorldQA | [Official HF dataset](https://huggingface.co/datasets/xai-org/RealworldQA); recent author results in [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B) | Useful real-world perception check, but no matched current table for Astra, Luna, and Gemini 3.8 was verified. |
| Q-Bench / Q-Bench2 | [Primary repository](https://github.com/Q-Future/Q-Bench), [HF single-image set](https://huggingface.co/datasets/q-future/Q-Bench-HF) | Closer to the task: distortion perception, description, and quality assessment. Published closed-model rows are older; run current candidates ourselves. |
| BLINK | [Official HF dataset](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) | Add visual similarity and forensic tasks as diagnostics. They do not replace MOS correlation. |

The Qwen card reports RealWorldQA scores of 85.9 for Qwen3.8-27B, 84.1 for Qwen3.6-27B, 86.9 for Qwen3.7-Plus, and 73.9 for Opus 4.6 Max. These are author-reported under the card's protocol, not the independent MMMU-Pro ranking. Qwen3.8-27B is a useful open-weight teacher control if direct label logits become necessary. Its self-hosting cost needs measured throughput and a selected GPU; no API price is inferred from its parameter count.

Hugging Face repository inspection worked for these datasets and Qwen3.8-27B. The plugin's search endpoints returned `Tool not found`; discovery used Exa, followed by direct Hugging Face repository checks. No dataset was downloaded.

## 2. Fit to our data plan

The [existing label evidence](data-mixture-evidence.md#71-teacher-synthetic-labels-on-unlabeled-images) supports testing a frozen teacher plus human-score calibration. It does not establish the IQA accuracy of the six models above. Old GPT-4o aesthetics results also do not prove that every newer teacher will fail. They justify a separate aesthetics gate.

The supplied planning note and [dataset inventory](../datasets/iqa.md) report **259,647 Q-Align training images**, comprising 24,049 technical-quality images and 235,598 AVA aesthetics images, plus 28,056 videos. Thus 250,000 technical-quality labels would have similar image count but a very different task mix. The estimates below cover still images only. Multiply by 1.038588 for the exact reported image count, or 1.2 for the note's proposed 300,000-photo pool. Neither adjustment covers video.

Use a broad photo pool with authentic defects and controlled distortions. Keep technical quality and aesthetics as separate targets. Keep all crops and distortions of a source image in one split, and deduplicate against the locked tests before labeling.

## 3. Verified token prices

Rates per million tokens, direct provider APIs, short requests, no cache discount. Output includes billed reasoning or thinking tokens.

| Model | Standard input | Standard output | Batch input | Batch output | Source |
|---|---:|---:|---:|---:|---|
| Claude Opus 5.5 | $4.00 | $20.00 | $2.00 | $10.00 | [Model overview](https://platform.claude.com/docs/en/models/opus-5-5/overview) |
| GPT-6 Astra | $10.00 | $50.00 | $5.00 | $25.00 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing) |
| Gemini 3.8 Flash | $0.75 | $3.75 | $0.375 | $1.875 | [Google pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| Gemini 3.7 Flash | $0.75 | $3.75 | $0.375 | $1.875 | [Google pricing](https://ai.google.dev/gemini-api/docs/pricing) |
| Claude Opus 5 | $5.00 | $25.00 | $2.50 | $12.50 | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| GPT-6 Luna | $0.10 | $0.50 | $0.05 | $0.25 | [OpenAI pricing](https://developers.openai.com/api/docs/pricing) |

Gemini 3.7 and 3.8 rates double on **1 January 2027**. The general Anthropic price page retrieved during this check omitted Opus 5.5; its specific model overview supplied the current rates and 50% batch discount. These source differences are why model-specific verification matters.

## 4. Cost assumptions

The later [prompt pack](synthetic-label-prompts.md) uses fuller rubrics and recommends two separate task calls for the pilot. Recalculate costs from measured usage for those prompts; the single-call estimates below remain the original planning scenario.

The [project efficiency protocol](../evaluation/protocol.md#9-efficiency-measurement) already uses a 1,000-image sample. Use that size for the first teacher screen, then 10,000 images for a label-generation trial. These are proposed sample sizes, not existing teacher datasets.

Each request scores one image, with 300 text input tokens and 100 final output tokens. The central budget adds **1,000 billed reasoning tokens**, for 1,100 total output tokens. This is a planning allowance, not measured usage or a promise that max effort fits this budget. Opus 5.5 cannot disable thinking. Do not impose a 100-token total response limit and assume that the score will still fit.

Use a single image up to 1024 × 1024 for the base estimate, with high detail where applicable. Preserve aspect ratio. Square dimensions give a simple, conservative patch estimate within that size bound. Resizing can hide defects; test a full view plus native crop before fixing the production policy. The teacher can receive more detail than the compact student.

| Model family | Image input tokens per request | Basis |
|---|---:|---|
| GPT-6 Astra | 1,229 | `ceil(32 × 32 × 1.2)` at 1024², [OpenAI vision guide](https://developers.openai.com/api/docs/guides/images-vision#calculating-costs) |
| Gemini 3.7 / 3.8 | 1,120 | Gemini 3 high image resolution, [media-resolution guide](https://ai.google.dev/gemini-api/docs/media-resolution) |
| Claude Opus 5 / 5.5 | 1,369 | `ceil(1024 / 28)²`, [Claude vision guide](https://platform.claude.com/docs/en/build-with-claude/vision) |
| GPT-6 Luna | 1,229 assumed | Image input is [supported](https://developers.openai.com/api/docs/models/gpt-6-luna), but the retrieved vision multiplier table did not list Luna. Uses Astra's count as an explicit placeholder. Replace with observed usage before scale-up. |

Formula: `cost = image_count × ((image_tokens + 300) × input_rate + (100 + reasoning_tokens) × output_rate) / 1,000,000`. Batch uses the batch rates above. No discounts for cached text are assumed.

### Central cost estimate

| Model | 1,000 images, standard | 10,000 images, standard | 250,000 images, standard | 250,000 images, batch |
|---|---:|---:|---:|---:|
| Claude Opus 5.5 | $28.68 | $286.76 | $7,169.00 | $3,584.50 |
| GPT-6 Astra | $70.29 | $702.90 | $17,572.50 | $8,786.25 |
| Gemini 3.8 Flash | $5.19 | $51.90 | $1,297.50 | $648.75 |
| Gemini 3.7 Flash | $5.19 | $51.90 | $1,297.50 | $648.75 |
| Claude Opus 5 | $35.85 | $358.45 | $8,961.25 | $4,480.63 |
| GPT-6 Luna, assumed image count | $0.70 | $7.03 | $175.73 | $87.86 |

Batch halves the 1,000- and 10,000-image figures too. Estimates exclude retries, additional crops, second teachers, storage, human ratings, data preparation, student training, and tax. They do not include free-tier credits. Round money only after calculating the full workload.

### Reasoning and resolution sensitivity

| Model | 250K batch: 0 reasoning tokens, arithmetic floor | 250K batch: 4,000 reasoning tokens |
|---|---:|---:|
| Claude Opus 5.5 | $1,084.50 | $11,084.50 |
| GPT-6 Astra | $2,536.25 | $27,536.25 |
| Gemini 3.8 / 3.7 Flash | $180.00 | $2,055.00 |
| Claude Opus 5 | $1,355.63 | $13,855.63 |
| GPT-6 Luna, assumed image count | $25.36 | $275.36 |

The zero-reasoning column is a cost floor, not an available operating mode for every model. The 4,000-token column is a scenario, not an upper bound. Measure actual billed usage and valid-score rate in the pilot.

A second image view in the same request adds image input cost before any extra reasoning: about $1,536.25 for Astra, $105.00 for Gemini, $684.50 for Opus 5.5, or $855.63 for Opus 5 over 250K batch requests, if its token count matches the first view. Native crops can differ. A separate second call repeats the text prompt and output cost as well.

## 5. Pilot and promotion decision

1. Build a 1,000-image development manifest with 250 authentic photos, 250 controlled-distortion images, 250 generated or enhanced images, and 250 aesthetics examples. Use human-rated development data where available. Split calibration and teacher-selection subsets by source; keep final test sets untouched. The 250-per-group allocation is a proposed screen, not a powered final evaluation.
2. Score all six candidates using the same task definitions and image policy. Report IQA and aesthetics separately. Compare SRCC, raw PLCC, pair-order errors, score spread, repeated-run stability, parse failures, and cost per valid label. Fit scale calibration only on the calibration subset.
3. Return the five levels `bad`, `poor`, `fair`, `good`, `excellent`, plus a normalized distribution if requested. Compute `score = sum(p[level] × level_index)` for indices 1–5. A JSON distribution is a model's stated judgment; it is **not** the label-token probability distribution used by Q-Align. Do not describe it as logits or calibrated uncertainty. Where full label logits are unavailable, compare stated distributions with repeated votes and hard-label targets.
4. On a fixed subset, compare low and high reasoning budgets, then one view versus full view plus crop. Preserve image bytes across teachers. Do not add JPEG artifacts during preprocessing. Record model ID, provider, date, prompt, effort, dimensions, crop policy, token usage, and raw response.
5. Select teachers on development evidence. Run a matched 10K-image student experiment before 250K. Compare synthetic labels against distortion-ranking supervision and a human-calibrated variant at fixed training exposure. Use the [existing protocol](../evaluation/protocol.md) for held-out evaluation and source-level bootstrap intervals.

Do not use a pooled score to hide a weak domain. Do not merge aesthetics labels into technical-quality targets. A teacher that sees subject matter well can still miss noise, oversharpening, or compression artifacts.

### Proposed budget

The six-model 1K screen costs about **$145.89 standard**, or **$72.95 batch**, under the central assumptions. Reserve 20% for repeated or failed calls: about **$175 standard**. This does not cover a large resolution sweep or human ratings.

For production, Gemini 3.8 on all 250K images plus Opus 5.5 on a 10% audit sample costs **$1,007.20 batch**, or **$1,208.64 with a 20% allowance**. This buys a second opinion on 25K images, not consensus labels for the whole set. Start with a random stratified audit; add disagreement-driven review only after measuring how to detect bad labels. Full two-teacher coverage costs **$4,233.25 batch** before retries.

After the Gemini promotion ends, the same bulk-plus-10%-audit plan costs **$1,655.95 batch**, or **$1,987.14 with 20% allowance**. Luna could reduce bulk cost further, but its IQA accuracy and image token usage must pass the pilot first.

The recommendation remains conditional: Gemini 3.8 is the strongest cost candidate from this shortlist; Opus 5.5 supplies a different model family for checks; Astra tests whether extra spend improves our actual labels. No evidence here establishes Q-Align-level quality from any of them.
