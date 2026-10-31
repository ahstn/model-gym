# RunInfra comparison on the frozen stage-A views

This diagnostic uses the 50 stage-A image identities from the 18 October prompt experiment. It compares RunInfra with the saved OpenRouter/Relace results. Both use the unchanged visual-rating-v1 prompt, response schema, temperature zero, high image detail, and 1024 output-token limit. RunInfra uses `reasoning_effort: none`, which is its documented control for disabling reasoning. The original OpenRouter request used `reasoning.enabled: false`.

The requested model is `deepseek-ai/DeepSeek-V4.1-Flash-1017`. RunInfra publishes that name on its model page, but the live catalog exposes API ID `deepseek-v4-1-flash-1017`, as in the user's example. A catalog lookup with the full name returned HTTP 404. Send the catalog ID and preserve both names. The live catalog, model revision, prompt, inputs, settings, request hashes, response bodies, and returned identities are retained. The provider route and date differ from the historical baseline. The experiment cannot prove that both services use identical weights or image processing.

## Input and cost controls

Send the original full-frame PNG bytes. Do not resize, crop, recompress, or publish images. Preflight all 50 requests against the live catalog limit of 3,670,016 bytes. Oversized requests remain missing with a reason; they are not silently replaced with smaller images. Any later user-approved changed views need a separate versioned run and separate analysis.

Use a $1 admission budget. Before each request, reserve the full 1,048,576-token context at $0.15 per million input tokens plus 1024 output tokens at $0.60 per million. This gives a conservative reserve of $0.1579008 per request. Actual cost comes from the returned `usage.cost`, checked against `usage.runinfra.cost_microcents`. Unknown costs stop the run. These rates were checked on the official model page and live catalog during this task.

Make at most one attempt per eligible view. Stop after a failed, invalid, incomplete, unpriced, or uncertain attempt. Do not retry, change models, repair responses, or use another provider automatically. A complete saved run is verified and reused without calls. Preserve original manifests, prompts, observations, raw replies, and migration records.

## Analysis fixed before requests

Use the saved stage-A assistant point judgments and plausible intervals unchanged. They are model judgments, not human ground truth. Report mean distance outside those intervals, point-judgment MAE, interval hits, errors at least one point outside an interval, and critical contradictions. A critical contradiction is a score of at least 4 with reference upper bound at most 2, or a score at most 2 with reference lower bound at least 4.

Pair the candidate and baseline on matching original views. For a partial cohort, report both methods on exactly the same retained cases. Preserve the full-50 historical baseline as context only. Report exclusions by source stratum and family, and report coverage. Resample whole retained families within each source stratum with 10,000 bootstrap draws and seed 20261018. Use observed retained stratum weights and ratio means for unequal retained family sizes. Also report a sensitivity check restricted to complete original families.

The full-cohort diagnostic score rule requires all 50 valid scorable pairs, mean interval-distance improvement at least 0.15, a paired 95% family-bootstrap lower bound above zero, no new critical contradictions, and no stratum worsening above 0.10. An incomplete cohort cannot pass that rule. A positive partial result is a diagnostic lead only. Explanation support requires a separate image audit; do not infer it from score agreement.

All observations remain pending human review. Technical quality stays on the 1–5 scale in 0.1 steps; tenths do not establish perceptual precision. Aesthetics remains null. Photo eligibility retains its original true, false, or null value. All task masks stay false and `training_ready=false`. Stage A is exposed development evidence. Stage B remains unchanged and is not fresh evidence for another prompt revision.

## Primary sources checked

- [RunInfra model page](https://runinfra.ai/inference-api/deepseek-v4-1-flash-1017)
- [Chat Completions contract](https://runinfra.ai/docs/api-reference/chat-completions)
- [Model discovery contract](https://runinfra.ai/docs/api-reference/models)

The API key comes from `RUN_INFRA_API_KEY` in the user-requested shell loader. It is never written into the run records. Images and raw replies stay in the ignored local data directory, apart from the authorized inference requests to RunInfra.
