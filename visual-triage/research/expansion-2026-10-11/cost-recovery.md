# Cost guard recovery

The quality search stopped during proposal 10 on 11 October 2026. A connection error returned no response, generation ID or usage record. The attempt for `koniq-v2-316` lasted 0.152 seconds. Its original ledger remains unchanged: cost and prediction are unknown, and its reward is zero. Resuming the search must not retry this request.

OpenRouter's [generation lookup](https://openrouter.ai/docs/api/api-reference/generations/get-generation) requires a generation ID. Its [analytics API](https://openrouter.ai/blog/announcements/activity-dashboard/) requires a management key. The existing inference key returned HTTP 403 for the read-only metadata endpoint. Concurrent requests from the AI track also prevent attribution from a change in the account balance.

The cost guard therefore uses a separate, explicit **$0.10 reservation** for this attempt. This is a budget allowance, not a billed amount or a claim that the call was free. The [saved pricing response](../../ARCHIVE.md) lists 262,144 context tokens at $0.00000002 per input token, with no output charge. A full context would cost $0.00524288 at that price. The reservation provides more than 19 times that amount.

The reservation must match the ledger location, sample, request hash and original record hash. It does not resolve any other unknown charge. Further unknown costs still stop the run. Reports separate known provider costs, unresolved attempts and budget reservations. Total billed cost remains unknown until the provider resolves the missing charge.

This operational change does not alter images, labels, prompts, search rewards, selection rules or final test access. It permits the frozen 20-proposal search to resume within its original budget while retaining the failed attempt.

Archived source snapshots and raw records require [archive restoration](../../ARCHIVE.md). The guide identifies the original paths and hashes.
