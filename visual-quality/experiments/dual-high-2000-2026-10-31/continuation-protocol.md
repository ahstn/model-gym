# DeepSeek continuation after shared-capacity rejection

The original DeepSeek run stopped after 19 HTTP requests: 17 valid responses
and two HTTP 429 rejections for `hosted_shared_model_concurrency_limit`, each
with a provider request ID and `Retry-After: 1`. These are explicit provider
capacity rejections. The original runner, inputs, run metadata, reports, raw
attempts, and reference judgments remain frozen and unchanged.

This separately frozen operational continuation completes the same authorized
2,000-image trial. It retains all 17 original responses and continues the 1,983
remaining cases. Every request uses identical original image bytes, prompt,
schema, high-reasoning settings, payload hash, and maximum token allowance.
The only repeatable outcome is a fully received HTTP 429 envelope with the
exact shared-capacity error code and type, matching JSON/header request IDs,
and a valid Retry-After interval. An unknown transport outcome, unrelated
error, incomplete response, unexpected model/provider/reasoning identity, or
unpriced successful response still stops new admissions. Valid or invalid
rating responses are terminal and never retried or repaired.

Each continuation attempt receives a new ID containing its original image ID
and monotonically increasing per-image attempt sequence. Request metadata is
fsynced before submission, and each raw reply is fsynced on receipt. The new
journal lives under `deepseek/continuation-01`; original requests and replies
are never overwritten. Both original and continuation process locks are held
during calls. An existing attempt directory is never sent again. An interrupted
continuation call blocks further work rather than being presumed rejected.

Concurrency starts at 8, halves after each exact capacity rejection to a minimum
of 2, and increases by 1 toward a maximum of 8 after 64 priced terminal responses
without an intervening rejection. No new calls are admitted during a rejection
cooldown. The cooldown starts at 2 seconds, doubles after repeated rejections
up to 60 seconds, and always honors a longer provider Retry-After interval.
Sixteen priced responses reset the exponential backoff. Existing admitted
calls are drained and retained; they cannot be assumed cancelled. Each image
may have at most five retries after explicit rejection. New admissions stop
once 40 total capacity rejections have been observed, including the original
two; already active requests may raise the final rejection count above 40.

The original **$10 DeepSeek cap** covers all original and continuation requests.
Known API charges from every attempt are added exactly. A rejection without
reported cost retains its full-context **$0.1671168** liability reserve; its
cost stays null and is never presented as zero. Every active request also has
a full-context reserve. Admission uses known charges plus accumulated unknown
liability reserves plus active request reserves. Provider capacity rejection
documentation does not establish zero billing, so no fee assumption releases
these reserves. A later documented billing reconciliation, if available, must
be recorded separately rather than changing the original evidence.

The logical combined route can be complete when every original image has one
terminal response, while the original run remains blocked and some explicit
rejection costs remain unknown. This distinction is explicit in the report.
`report.attempts` contains one selected response per completed image;
`report.all_attempts` and `all-attempts.jsonl` retain every HTTP attempt, including
unselected rejections. Ledger entries identify original run and continuation,
actual request/result paths, hashes, request IDs, selection, status, exact known
cost or null, and any retained liability reserve. `http_attempts` counts sends;
`selected_response_calls` counts completed unique image responses. Known
spending, unpriced rejections, held reserves, and admission debit are separate.

The adapter `load_verified_route(route)` returns `(run, report, observations,
artifact_pins)` for both DeepSeek and Luna. It verifies raw evidence and derived
exports. DeepSeek uses the combined continuation ledger; Luna uses its original
run with the same ledger shape. Model score comparisons use one terminal
response per image; costs and HTTP reliability include every attempt. No
continuation policy changes the photographic reference judgments.

```sh
visual-quality/.venv/bin/python visual-quality/experiments/dual-high-2000-2026-10-31/continuation.py freeze
visual-quality/.venv/bin/python visual-quality/experiments/dual-high-2000-2026-10-31/continuation.py probe --case-id T1423 --case-id T1977
visual-quality/.venv/bin/python visual-quality/experiments/dual-high-2000-2026-10-31/continuation.py run
visual-quality/.venv/bin/python visual-quality/experiments/dual-high-2000-2026-10-31/continuation.py verify
```

The probe counts within the same 2,000-image total. Export checkpoints occur
every 32 new completions, after capacity rejections, and at exit; raw evidence
is persisted individually. Resuming reconstructs the journal before admitting
any new calls. No additional requests occur during freeze or verification.
