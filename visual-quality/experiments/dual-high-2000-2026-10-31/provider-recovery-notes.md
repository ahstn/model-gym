# RunInfra shared-capacity refusals

Research date: 31 October 2026. Sources were discovered and read with Exa; only official RunInfra documentation was used. No new inference or authenticated financial request was made during this investigation.

Two original DeepSeek requests returned HTTP 429 with `hosted_shared_model_concurrency_limit`, limit 60, `Retry-After: 1` and `Retry-After-Ms: 1000`. Both replies contain an error envelope and no `usage` object. Their preserved request IDs are:

| Case | Request ID |
|---|---|
| T1423 | `req_1dcdfde4-8ccc-4cd8-87fc-9c9e727d1f56` |
| T1977 | `req_02231970-2392-458f-942c-efd41afb8366` |

RunInfra classifies this exact error as a shared-model admission refusal. Capacity on the model must free up; this is not evidence that our own concurrency exceeded 60. The documented recovery is to respect the response's retry interval and retry the same request. Lower client concurrency and paced retries reduce repeated contention but cannot guarantee shared capacity. See [rate limits](https://runinfra.ai/docs/api-reference/rate-limits) and [errors](https://runinfra.ai/docs/api-reference/errors).

**Billing treatment: expected zero by inference; reported cost remains unknown.** These requests were refused at admission and returned no generated output. RunInfra's general chat billing description says responses without billable output settle at zero. That supports an expectation of no inference charge. However, neither saved refusal includes a cost receipt, and the exact shared-concurrency error entry does not explicitly promise a zero charge. The docs do explicitly give that assurance for the different `hosted_model_at_capacity` error; it should not be silently substituted for the observed code. Preserve both original error records and distinguish inferred nonbilling from API-reported zero. See [chat billing](https://runinfra.ai/docs/api-reference/chat-completions).

No documented request-ID reconciliation endpoint was found. The [published OpenAPI specification](https://runinfra.ai/docs/api-reference/openapi.json) lists financial reads for `/v1/credits` and `/v1/usage`, alongside model discovery reads, but no GET request-history or per-request settlement route. [Plan usage](https://runinfra.ai/docs/api-reference/usage) reports current workspace policy and counters rather than a request receipt. [Credits](https://runinfra.ai/docs/api-reference/credits) reports workspace balances, holds and aggregate settled spending in cents; concurrent requests and fractional-cent carry prevent attributing a difference to either refusal. Reading those endpoints would not establish a per-request charge, so they were not called.

RunInfra retains request IDs with cost and settlement metadata for support and disputes, according to its [data-retention documentation](https://runinfra.ai/docs/security/data-retention). Those IDs permit later provider reconciliation, but no support message was sent. Until a receipt or provider confirmation is available, retain the unknown amounts separately from known costs and continue reserving the original worst-case amount against the spending cap. Retries must have their own journals and costs; they must not replace the original failures or claim those original calls returned scores.

[Machine-readable notes](provider-recovery-notes.json) preserve the two local result hashes, reviewed endpoint list, evidence links and the distinction between expected and confirmed cost.
