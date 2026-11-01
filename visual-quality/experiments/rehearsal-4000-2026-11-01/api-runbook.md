# API rehearsal runner

Implementation was independently reviewed before any paid request. `api-readiness.json` records the implementation hashes and checks. The four routes retain the previous accepted model/provider settings and selected prompts. The Luna and DeepSeek benchmark treatments append only the frozen technical-statistics block; the image and both score fields remain unchanged.

## Readiness and input gates

Run the following with the existing visual-quality Python environment:

```sh
PY=/home/ahstn/git/model-gym/visual-quality/.venv/bin/python
E=/tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01
$PY -m pytest -q "$E/test_api.py"
$PY -m ruff check "$E/api_runner.py" "$E/api_contract.py" "$E/test_api.py"
```

A phase cannot freeze without the full 4,000-case `manifest.json`, including the agreed 3,500 rehearsal / 100 development / 300 representative / 100 challenge partition counts. All delivered view hashes must be unique, and the 500 benchmark cases must have distinct source families. The runner hashes every rendered request and rejects changed image bytes.

Benchmark freezes additionally require `references-freeze.json`, with the exact 500 case IDs, reference file path/hash, full manifest hash, rubric path/hash, teacher prompt file hash, and a past timezone-aware freeze timestamp. Luna/DeepSeek also require `stats-prompts.jsonl` and `stats-donors.json`. Donor IDs stay outside the prompts; no donor can share the receiving image's family. No reference scores are sent to a grader.

The implementation, prompts, provider snapshot, protocol, manifest, dependencies and runtime versions are pinned into each phase plan. Finish preparation and reference review before freezing the applicable phase. A changed pinned artifact blocks resuming; investigate changes explicitly instead of editing plan hashes.

## Freeze, smoke, run and verify

Use `deepseek`, `luna`, `jev` or `liquid` as the route and `rehearsal` or `benchmark` as the phase. Freeze once after its inputs are final:

```sh
$PY "$E/api_runner.py" freeze --route deepseek --phase rehearsal
```

Load keys silently in the shell used for inference. Do not enable tracing or print the environment:

```sh
set +x
source ~/.config/zsh/.zsh-private >/dev/null 2>&1
$PY "$E/api_runner.py" run --route deepseek --phase rehearsal --limit 1
```

After inspecting the durable smoke response and receipt, continue the same frozen plan:

```sh
$PY "$E/api_runner.py" run --route deepseek --phase rehearsal
$PY "$E/api_runner.py" verify --route deepseek --phase rehearsal
```

`--limit` limits newly selected jobs for a smoke, not completed responses or cost. Repeatable `--case-id` restricts a smoke to explicitly chosen cases without changing the frozen request. A normal resume skips every valid or invalid completed response. A route lock prevents concurrent phases from bypassing its shared budget. Separate routes may run concurrently. Benchmark jobs use a deterministic interleaved order across baseline, own-stats and swapped-stats arms.

## Evidence and accounting

All mutable execution data live under `/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/results/api/<route>/<phase>/`. Immutable `plan.json` stores the complete request contract. Per-attempt `request.json` is fsynced before dispatch; `result.json` retains the full response, safe provider headers and actual timing. `attempts.jsonl`, `observations.jsonl` and `summary.json` are rebuilt from those journals. Attempt IDs include phase, job and sequence.

`latency_seconds` measures HTTP invocation through the complete nonstreaming response, including network, upload and provider queue time. `scheduler_wait_seconds` covers executor waiting; `preparation_seconds` covers local image/payload/journal preparation. No timing is invented for an interrupted request. Actual model/provider, requested reasoning, returned reasoning effort when provided, reasoning-token usage and original task-level validation are retained.

Provider receipts determine prices. OpenRouter BYOK adds its gateway charge and upstream inference cost exactly once; ordinary OpenRouter cost is not augmented. RunInfra requires agreement between its cost and integer microcent fields. Missing, incomplete or inconsistent receipts remain null, and every attempt counts once. Unknown attempts retain a conservative full-context reservation for budget admission. Prices retain enough precision to avoid displaying tiny calls as zero. Scores retain native precision; report display should use two decimals.

| Route | Jobs including benchmark treatments | Fixed concurrency | Shared cap | Conservative per-attempt reservation |
|---|---:|---:|---:|---:|
| DeepSeek | 5,000 | 8 | $30.00 | $0.1671168 |
| Luna | 5,000 | 16 | $30.00 | $0.222288 |
| Jev | 4,000 | 4 | $15.00 | $0.011010048 |
| Liquid | 4,000 | 4 | $15.00 | $0.00131072 |

Reservations are admission bounds, not claimed actual charges. All phases share the route cap. Decisions routes preserve their existing typed score/probability contract, map native scores by adding one, and apply only the existing exclusion-probability rule. Raw/native scores remain available. Chat routes use the existing strict two-task JSON schema, preserving a valid sibling task when the other fails. No image-stats gate overwrites a score.

## Recovery boundaries

The only automatic retry is the exact previously verified RunInfra pre-serve capacity rejection: HTTP 429, `rate_limit_error` / `hosted_shared_model_concurrency_limit`, matching request IDs and a valid retry delay, with no scored choices or usage receipt. Its unreported price remains unknown. Retrying is bounded to five rejections per job and 60 per route, including resumed runs. Generic HTTP errors, timeouts, interrupted journals, unsupported provider/model responses and unknown billing stop the route for investigation. Invalid completed labels are retained and never retried to seek another answer. A successfully scored request is never resent.

Use `verify` to audit immutable inputs, every request hash and all journals before analysis. Do not delete attempt journals, coalesce missing prices to zero, or relax validation to continue an ambiguous billable call. Route failures may be investigated while unaffected routes proceed. Published labels remain experimental (`training_ready=false`).

## Execution and explicit provider-failure recovery

`api-launch.json` records all eight frozen plans, four successful smoke receipts, and the initial execution sessions. `api_route_driver.py` resumes the rehearsal, verifies its complete payload/journal set, then runs and verifies the benchmark. Three synthetic checks ensure blocked or unverified phases cannot advance. Each route has a durable `driver-status.json` and `driver.log` outside its phase journals.

Jev returned two HTTP503 responses with `api_usage_error: Image processing is unavailable` after five successful requests. The initial route stopped as designed. `api-incident-jev-503-01.json` preserves the evidence. The error replies contain no usage receipt or provider generation identifier; actual cost remains unknown and each retains its full conservative reservation.

`api_recovery.py` is a separate operational driver; it does not modify the frozen runner, request plans, prompts or journals. `api-recovery-jev-approval-01.json` pins the two reviewed response/request receipts. The driver skips every previously attempted job, including failed ones, and allows only these exact reviewed failures in its admission audit. A smoke on the next unattempted image succeeded before the remaining jobs resumed at the original concurrency4. Any new failure stops further dispatch and requires another explicit receipt review. Six synthetic checks cover replay prevention and receipt/request/model/status tampering.

Recovery segment records under `results/api/jev/recovery-segments/` retain effective concurrency, approval and driver hashes, timestamps, and attempt IDs. Original phase summaries retain their blocked/missing status because the historical failed responses are preserved. A recovery segment can finish with all jobs attempted while still reporting those missing labels and unknown bills. Any later concurrency change must be recorded as a distinct latency segment; do not edit original plan concurrency or merge latency distributions without noting the operational difference.

The first recovery produced32 additional valid responses and two more identical HTTP503 failures, both dispatched together. `api-recovery-jev-approval-02.json` records the four exact reviewed failures and authorizes serial continuation as an explicit operational amendment. The next unattempted serial smoke succeeded. Subsequent serial attempts remain in separate recovery segments with `effective_concurrency=1`; this is an operational diagnostic, not evidence that concurrency caused the errors. No prior attempt was replayed.

A fifth identical503 occurred serially. The property/content audit in `api-incident-jev-503-diagnosis.json` found valid still RGB PNGs within documented dimensions and overlapping successful file sizes; the cause remains unproven. Official sources were retrieved through Exa. No matching error-specific billing assurance was found.

`api-recovery-jev-approval-03.json` is the prospective operational policy for continuing only unattempted jobs. It permits quarantine of exactly the reviewed503 body, with no usage/answers and an authenticated local receipt hash; all other errors halt immediately. A circuit breaker stops at five consecutive known failures, or20 known failures in the most recent100 requests, including historical requests. The original shared$15 budget and full reservations for unknown costs remain in force. No request is retried and no missing score is filled. The driver stays serial and records each quarantined attempt in its recovery segment. Ten recovery tests plus three phase-driver tests pass. The earlier driver source is preserved as `pipeline-snapshots/api_recovery-v1.py` for prior segment provenance.

DeepSeek later returned a billed HTTP200 with `finish_reason=length`, no answer content, and all16,384 completion tokens reported as reasoning. The receipt and outcome are retained in `api-incident-deepseek-exhaustion-01.json`; no score is derived from reasoning. `deepseek_recovery.py` and `api-recovery-deepseek-approval-01.json` admit only this exact reviewed no-answer class while using the original scheduler, concurrency8, max_tokens16384, highreasoning, budget and pre-serve429 handling. The separate operational adapter temporarily narrows fatal admission in memory, restores the original function before final summaries, and changes no frozen source or request. Completed failures are skipped, never replayed. Ten observed exhaustion failures stop new dispatch; any other fatal result stops immediately. Nine synthetic checks cover exact output/billing classification, receipt binding and the threshold. Operational reporting separately counts billed no-answer exhaustions.
