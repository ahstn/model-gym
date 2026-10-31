# DeepSeek high reasoning: provider verification

Verified 31 October 2026 using Exa and a live authenticated, read-only `GET /v1/models` request. The [catalog snapshot](runinfra-catalog.json) identifies `deepseek-v4-1-flash-1017`, the same revision used in the preceding trial. Its supported efforts include `none` and `high`, with `high` listed as the default.

The preceding run explicitly overrode that default with `reasoning_effort: "none"`. The new request explicitly sets `reasoning_effort: "high"` and raises `max_tokens` from 2,048 to 16,384. Prompt text, image bytes, strict JSON schema, temperature and endpoint remain unchanged.

RunInfra documents that `reasoning_effort` is the supported control on Chat Completions, and reasoning consumes the output-token allowance. Its compatibility notes also describe restrictions on combining reasoning with schema enforcement for some models. Those older-model restrictions should not be assumed to apply to this revision; the first actual corpus request must establish compatibility before the remaining requests run. See [Chat Completions](https://runinfra.ai/docs/api-reference/chat-completions) and [structured output](https://runinfra.ai/docs/cookbook/structured-output).

The [1017 model page](https://runinfra.ai/inference-api/deepseek-v4-1-flash-1017) lists image input and JSON support and uses a 16,384-token example. The live catalog confirms the configured input/output prices and limits. Analysis uses returned usage costs rather than a ceiling-based estimate, and distinguishes requested effort, the applied-effort response header, and reported reasoning-token usage.

This comparison changes both reasoning effort and the output allowance needed to accommodate it. It is not an isolated test of the effort field under an identical token cap.
