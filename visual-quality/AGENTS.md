# Visual-quality labeling rules

These are the user's accepted project decisions, updated 2026-11-17. Apply them to the active labeling workspace and future runs. Historical raw observations retain their provenance; they are not evidence that an old configuration is still current.

## Canonical ensemble

- Use **0.375 Luna + 0.375 DeepSeek + 0.25 Liquid** independently for technical quality and aesthetics. The shared ensemble implementation is authoritative.
- The previous 50/50 Luna/DeepSeek scorer is superseded, including in current presentations of prior results. Recompute derived observations from the original component scores. Retain raw responses and provenance; do not relabel old calls as new calls.
- Require all three valid component scores for each outcome. Missing or invalid components produce a missing ensemble outcome, not silently renormalized weights or an invented score.
- Calculate from native precision and round only for presentation. Keep both scores on the 1–5 scale. Do not combine technical and aesthetic into one training target.
- Derived ensembles have no independent HTTP latency or billable API call. Keep component timing/cost provenance and avoid counting their costs twice.

The 2026-11-13 500-image diagnostic comparison did not justify replacing this ensemble with Perplexity Decider or Gemini Decision. The user reaffirmed the existing ensemble on 2026-11-16; do not add either new model to routine labeling.

## Models and prompts

- GPT-6 Luna: `openai/gpt-6-luna`, OpenRouter, high reasoning.
- DeepSeek: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, RunInfra, high reasoning; gateway model spelling may be `deepseek-v4-1-flash-1017`.
- Liquid: `liquid/d1.1`; Jev: `typesafe/jev-1.14`; use OpenRouter's typed Decisions interface and the revised model-specific exclusion questions introduced in the work consolidated by `9a2f7ce`.
- Use the `visual-rating-v2` technical rubric in a versioned, two-outcome profile with the existing aesthetic rubric and photography eligibility rule. Keep schema-constrained JSON and validate outcome status, score bounds, decimal increments, evidence and limitations. Preserve native Decisions precision separately from displayed scores.
- The photography eligibility rule overrides any generic rubric wording permitting artwork to score highly. Clearly excluded flat reproductions, posters, screenshots/interfaces, illustrations and obvious renders receive both scores 1.00. Photographed packaging, signage, artwork or objects with meaningful physical context remain eligible. Do not infer excluded origin from blur, darkness, compression or unusual content alone.
- Preserve each model's eligibility evidence and the evaluated Decisions threshold; do not invent a new ensemble voting or floor rule. Uncertain scope requires review. Missing images and failed requests are missing assessments, not low-quality scores.
- No new Q-ReAlign Lite inference or comparisons for the 20K expansion or subsequent labeling. Retain the initial 10K observations as historical comparison data. Historically, Q-ReAlign Lite uses the unchanged model/checkpoint, two task prompts and score mapping from `0a156b4` / the 4,000-image rehearsal. Its native scoring interface is not a chat JSON-schema interface. Record that distinction; do not pretend it follows a prompt it never received.
- Reuse an inference result only with an explicit view/model/prompt/settings provenance check. A different prompt is a different observation. The 10,000-image expansion must rerun Liquid and Jev with the revised questions; Luna and DeepSeek also need fresh calls when adopting the different v2 rubric.
- Do not apply experimental prompt additions merely because they were suggested in a discussion. The proposed poor/fair wording remains an experiment until explicitly adopted.

## Corpus, references and publication

- The current expansion adds **20,000 new images**, bringing the dataset from **20,000 to 40,000 total images**. Preserve the completed 20K exports byte-for-byte and the initial **1,500-image reference benchmark** unchanged; do not automatically expand assistant references. New images receive Luna, DeepSeek, Liquid and Jev observations plus the canonical ensemble. Exclude already labeled source identities, exact duplicates and protected evaluation families before selection.
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

- Keep the completed 10,000-image evaluation as the frozen comparison baseline. Keep current expansion orchestration and aggregate reports under `labeling/runs/`, and the accepted 500-image Decisions comparison under `labeling/trials/`. Historical code, media and receipts are recoverable through `archives/experiments.md`; do not recommit restored trees.
- Reusable execution and rendering helpers belong in `src/visual_quality/scoring/`, not dated experiment folders. Historical data paths resolve separately through `scoring.history`; the pinned Lite server and health identity live in `labeling/runtime/`.
- Preserve immutable old source hashes and observations when moving code. New executions use new source provenance; exact historical replay uses the pinned historical checkout.

- Every model/reference record contains technical score, aesthetic score, latency fields and price fields, even when an assessment or measurement is unavailable.
- For inference, measure end-to-end time immediately around HTTP invocation through complete response receipt. Keep payload preparation and server-only timing separate. Reused rows retain their original measured latency and reuse provenance.
- References have no HTTP request latency: record null with an explicit latency kind and separately measured visual-review duration. Never fabricate a timing measurement.
- Q-ReAlign Lite and Reference rows have `price_usd=0.00` as the user's accounting rule. Track actual Runpod infrastructure cost separately. Unknown API charges are null, never zero. Retain retries and failed-call cost accounting.
- The user approved the 40,000-image expansion on 2026-11-17 and OpenRouter spend for all required grading requests, continuing the existing EmbeddingGemma 2 pipeline and prior Runpod authorization for necessary embedding compute; RunInfra credential use was previously authorized. Do not ask again for routine authorized calls. Credentials may be sourced quietly from `~/.config/zsh/.zsh-private`; never print or commit them.
- Prefer the Runpod plugin for provisioning and lifecycle operations. Prepare workloads before starting paid compute, retain receipts, and stop owned trial pods when work completes or fails. Do not modify unrelated pods.
- A provider content-filter denial is a terminal missing assessment for that image. Do not retry or bypass it; continue independent allowed images while retaining the failure and unknown charges. Distinguish it from credential/billing errors, which stop the run. Keep the aggregate failure circuit breaker enabled.
- Session recovery (2026-11-16): the user confirmed that they temporarily disabled their secret-redaction guardrail. The 2,855 denials per Liquid/Jev model comprise 2,757 `SECRET:bitcoin-wif-compressed`, 85 `SECRET:bitcoin-wif-uncompressed`, and 13 `SECRET:google-api-key` flags. The user explicitly approved resending these flagged payloads after the automatic-review clarification. The approval is recorded locally; do not request it again for this batch. Preserve prior receipts and observations in the pre-recovery snapshot; keep image bytes, prompts and scores unchanged for all other completed cases. Stop if a new 403 recurs. This does not authorize changing provider safety controls or retrying other denials.
- Use resumable, hashed requests and raw response records; retry bounded transient failures without retrying merely because a valid score is surprising. Account for ambiguous in-flight calls honestly.
- 40K run operational recovery (2026-11-17): `labeling/runs/expansion-40000-2026-11-17/recover_empty.py` may append exactly one identical request for a verified HTTP-200 `length` cutoff with no visible answer and all 16,384 completion tokens reported as reasoning. First require the model's original `expansion` baseline to finish completely, then freeze the eligible cases, original request/response hashes, per-case attempt limits and protected observations in the separate `empty-length-once` phase. Keep image bytes, profile, high reasoning and token cap unchanged. Exclude any prior valid score, invalid numeric assessment, unscorable/refusal answer, content denial or nonempty/truncated visible answer. Preserve every original receipt, cost and latency; even a recorded transport failure consumes the single recovery attempt. An admitted request without a response receipt requires reconciliation and must never be resent automatically. Coordinate phase ordering before freezing if another recovery phase is needed. This is run-specific handling of missing output within the requested labeling work, not a new default grader policy or a separate human-approval claim; the baseline retry policy remains unchanged.
- Validate schema, exact image joins, benchmark separation, score completeness, publication allowlist and actual cleanup before reporting completion. Report partial coverage and unresolved errors explicitly.

## Dataset backup

- The 40K snapshot was verified on 2026-11-18 at `s3://model-gym-backups/visual-quality/labeling-40000-2026-11-17/`. It contains all 40,000 exact scored views and the canonical labels. Full object readback and an independent restore passed. Use the [restore guide](labeling/backups/README.md#restore-the-40k-dataset) and [aggregate completion receipt](labeling/backups/receipt-40000-2026-11-18.json). The 20K snapshot has its own prefix and receipt.

- On 2026-11-16 the user approved a private Cloudflare R2 backup of the complete 20K labeled dataset, including all exact scored images. The destination is `model-gym-backups/visual-quality/labeling-20000-2026-11-16/`; procedures and restore instructions live in `labeling/backups/`.
- Follow the combined manifest across historical view directories. Preserve byte hashes, canonical labels, raw evidence, image statistics, clustering and source/configuration provenance. Do not re-encode images or silently fill missing scores.
- Keep public bucket access disabled. Exclude credentials, temporary operations, unselected candidates and caches. Private backup authorization does not expand the first-1,000 Git publication allowlist.
- Use an explicit upload inventory, resumable verification and full remote readback checksums. Publish a completion receipt only after all planned objects pass. Revoke temporary transfer credentials after use. Document that exact scored views are included, while unscored original source files are not.

## Training decision recorded 2026-11-17

- The user accepted storing the research decision in `docs/training/qwen-adaptation-review-2026-11-17/`: supervised BF16 LoRA on Qwen3.5-9B is the first proposed training route, with a frozen-feature control and one score-aware/detail/parameter adaptation ablation. Full supervised tuning remains a challenger; defer RL until supervised baselines and rewards justify it. This is a research decision, not an instruction to launch training during labeling.
- The user reported $50 Runpod credit for experimentation. Keep necessary embedding infrastructure costs separate from future training costs and preserve budget headroom; do not launch a training run as part of this expansion.
- Current acquisition `split=train` is not a training authorization or safe evaluation split. Build future family-disjoint training manifests that respect `benchmark_partition`, reviewed duplicate groups and exposure history. Preserve all-false training masks until explicit review/approval.
- Do not reuse the November 16 batch-specific secret-scanner resend authorization for new denials. New provider content denials remain terminal; retain evidence and report missing coverage.
- The 40K expansion keeps all new per-image observations, manifests, vectors, statistics and review artifacts under ignored `data/labeling-40000-2026-11-17/`. Commit code, aggregate reports and the accepted research decision only. Use no new Q-ReAlign calls or assistant references.
