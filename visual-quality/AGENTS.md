# Visual-quality labeling rules

These are the user's accepted project decisions, updated 2026-11-01. Apply them to the active labeling workspace and future runs. Historical raw observations retain their provenance; they are not evidence that an old configuration is still current.

## Canonical ensemble

- Use **0.375 Luna + 0.375 DeepSeek + 0.25 Liquid** independently for technical quality and aesthetics. The shared ensemble implementation is authoritative.
- The previous 50/50 Luna/DeepSeek scorer is superseded, including in current presentations of prior results. Recompute derived observations from the original component scores. Retain raw responses and provenance; do not relabel old calls as new calls.
- Require all three valid component scores for each outcome. Missing or invalid components produce a missing ensemble outcome, not silently renormalized weights or an invented score.
- Calculate from native precision and round only for presentation. Keep both scores on the 1–5 scale. Do not combine technical and aesthetic into one training target.
- Derived ensembles have no independent HTTP latency or billable API call. Keep component timing/cost provenance and avoid counting their costs twice.

## Models and prompts

- GPT-6 Luna: `openai/gpt-6-luna`, OpenRouter, high reasoning.
- DeepSeek: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, RunInfra, high reasoning; gateway model spelling may be `deepseek-v4-1-flash-1017`.
- Liquid: `liquid/d1.1`; Jev: `typesafe/jev-1.14`; use OpenRouter's typed Decisions interface and the revised model-specific exclusion questions introduced in the work consolidated by `9a2f7ce`.
- Use the `visual-rating-v2` technical rubric in a versioned, two-outcome profile with the existing aesthetic rubric and photography eligibility rule. Keep schema-constrained JSON and validate outcome status, score bounds, decimal increments, evidence and limitations. Preserve native Decisions precision separately from displayed scores.
- The photography eligibility rule overrides any generic rubric wording permitting artwork to score highly. Clearly excluded flat reproductions, posters, screenshots/interfaces, illustrations and obvious renders receive both scores 1.00. Photographed packaging, signage, artwork or objects with meaningful physical context remain eligible. Do not infer excluded origin from blur, darkness, compression or unusual content alone.
- Preserve each model's eligibility evidence and the evaluated Decisions threshold; do not invent a new ensemble voting or floor rule. Uncertain scope requires review. Missing images and failed requests are missing assessments, not low-quality scores.
- Q-ReAlign Lite uses the unchanged model/checkpoint, two task prompts and score mapping from `0a156b4` / the 4,000-image rehearsal. Its native scoring interface is not a chat JSON-schema interface. Record that distinction; do not pretend it follows a prompt it never received.
- Reuse an inference result only with an explicit view/model/prompt/settings provenance check. A different prompt is a different observation. The 10,000-image expansion must rerun Liquid and Jev with the revised questions; Luna and DeepSeek also need fresh calls when adopting the different v2 rubric.
- Do not apply experimental prompt additions merely because they were suggested in a discussion. The proposed poor/fair wording remains an experiment until explicitly adopted.

## Corpus, references and publication

- The current expansion targets **10,000 total images**, including a **1,500-image reference benchmark**. Reusing prior images and compatible observations is permitted.
- Use actual EmbeddingGemma 2 image embeddings for content clustering, overrepresentation checks, varied sampling, related-image retrieval and duplicate candidates. Record checkpoint, processor, inference configuration and any reuse. Similar content does not establish equal image quality.
- Preserve source licenses, byte hashes, exact delivered views, transformations, source families and split identity. Review likely duplicates and keep a family within one evaluation partition.
- References are independent assistant visual assessments, not human opinion scores. Reuse the original 500 with provenance and assess the additional benchmark images without viewing model scores. Freeze reference records before admitting those images to inference.
- Define a stable publication allowlist: the **first 1,000 benchmark images by explicit publication order** are the committed reference/example set. Keep per-image scores and labels for every other image outside git, including the remaining 500 benchmark images. Do not rely on lexicographic case IDs or row counts; one image has multiple model rows.
- Full manifests, observations, raw responses, images, embeddings, review ledgers and detailed reports live under ignored `visual-quality/data/`. Tracked code, configuration and aggregate reports may describe the full run, but must not embed per-image scores beyond the allowlist in CSV, JSON, HTML, collages, logs or archives.
- Do not rewrite git history to erase previously committed historical data. Apply the publication rule to new exports and active records going forward. Clearly distinguish superseded historical observations from current derived results.
- Keep training masks disabled until labels are reviewed and explicitly approved for training. A benchmark used to choose prompts or weights is no longer untouched confirmation data.

## Image statistics

- Compute statistics using this repository's local `image-stats/` package on the exact delivered image. Store them in separate image metadata, outside the inference response, and join by case ID plus delivered-view hash.
- Retain `sharpness_peak`, `luma_entropy`, `focus_area`, histogram entropy, `channel_clip`, and Hasler-Suesstrunk `colourfulness`, with formula descriptions, units, package/source version and analysis dimensions.
- The package's histogram entropy is `luma_entropy`: a 256-bin luma Shannon entropy in bits. An exported `histogram_entropy` field is a documented alias, not a separate measured signal.
- `focus_area` is relative to the image's own peak sharpness; an evenly blurry image can have a large value. It must not be presented as the percentage of the frame that is objectively in focus.
- Statistics are debugging/review metadata. Do not feed them to graders, overwrite scores, impose gates or include them in training features by default. Prior own/swapped-stats prompt trials did not justify general adoption.

## Timing, accounting and operations

- Every model/reference record contains technical score, aesthetic score, latency fields and price fields, even when an assessment or measurement is unavailable.
- For inference, measure end-to-end time immediately around HTTP invocation through complete response receipt. Keep payload preparation and server-only timing separate. Reused rows retain their original measured latency and reuse provenance.
- References have no HTTP request latency: record null with an explicit latency kind and separately measured visual-review duration. Never fabricate a timing measurement.
- Q-ReAlign Lite and Reference rows have `price_usd=0.00` as the user's accounting rule. Track actual Runpod infrastructure cost separately. Unknown API charges are null, never zero. Retain retries and failed-call cost accounting.
- The user approved Runpod and OpenRouter spend for all models and necessary requests in this 10,000-image session; RunInfra credential use was previously authorized. Do not ask again for routine authorized calls. Credentials may be sourced quietly from `~/.config/zsh/.zsh-private`; never print or commit them.
- Prefer the Runpod plugin for provisioning and lifecycle operations. Prepare workloads before starting paid compute, retain receipts, and stop owned trial pods when work completes or fails. Do not modify unrelated pods.
- Use resumable, hashed requests and raw response records; retry bounded transient failures without retrying merely because a valid score is surprising. Account for ambiguous in-flight calls honestly.
- Validate schema, exact image joins, benchmark separation, score completeness, publication allowlist and actual cleanup before reporting completion. Report partial coverage and unresolved errors explicitly.
