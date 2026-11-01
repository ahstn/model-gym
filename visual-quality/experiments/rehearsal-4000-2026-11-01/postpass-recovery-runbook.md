# Post-initial-pass API recovery

This is a user-authorized operational amendment made after the initial pass began. It is not a preregistered retry policy. The policy exists before recovery, but the receipt-bound allowlist must be frozen only after all 18,000 initial API jobs have an attempt and all four initial API services have stopped. Lite execution is independent of this gate.

The original exact Jev image-processing HTTP 503 error may receive up to two additional attempts. Each of the 21 original audited client interruptions (Luna 16, Liquid 4, Jev 1) may receive one additional attempt. An interruption retry returning the Jev 503 consumes that single allowance; it does not acquire another allowance. DeepSeek's billed 16,384-token no-answer failures remain missing and are never retried. No valid, partial, invalid or otherwise scored response is replayed.

Every recovery request uses the original frozen payload, image, exact model, route, prompt and runtime. Recovery runs serially with concurrency 1, in case-ID, phase, job-ID and sequence order within each route. The global postpass lock and all four original route locks prevent concurrent recovery segments and original route dispatch. Route invocation order does not affect eligibility or per-route circuit breakers.

## Prepare only after the initial API pass

Use the same pinned Python environment as the initial runner:

```sh
PY=/home/ahstn/git/model-gym/visual-quality/.venv/bin/python
E=/tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01
D=/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/results/api
"$PY" "$E/postpass_recovery.py" prepare --output "$D/postpass/allowlist.json"
```

`prepare` checks the four `modelgym-rehearsal-<route>-20261101.service` user units using only `ActiveState`, `MainPID` and `ControlGroup`. Each unit must be inactive or failed with no main or surviving cgroup processes. It uses current process state, not old PIDs or invocation IDs. It checks service state before loading any initial responses, acquires all route locks, validates every original input pin and payload, requires all planned job IDs, rejects a terminal capacity-retry state, and rejects any unreviewed request-only journal or error beyond the pinned audit and reviewed error classes. It requires exactly the original 21 audited interruptions to remain intact.

The new allowlist is written exclusively, made read-only and never updated. Its contents bind all initial plan IDs, plan file hashes, job IDs, request and response receipts, original attempt histories, permitted sequence limits, the initial service/status snapshot, the policy hash and current helper hashes. Helper hashes are captured at preparation time so bounded operational fixes during the initial pass do not invalidate this prospective policy. A second preparation cannot overwrite an existing allowlist.

Record the `allowlist_sha256` printed by `prepare` outside the mutable shell command. The following examples use `APPROVAL_SHA` for that exact recorded value. Do not derive a fresh hash from a potentially changed allowlist to bypass a failed check.

## Run and resume the bounded attempts

Ensure the original environment already supplies `OPENROUTER_API_KEY` without printing it or embedding it in command arguments. Run the three eligible routes sequentially:

```sh
"$PY" "$E/postpass_recovery.py" run --route jev --allowlist "$D/postpass/allowlist.json" --allowlist-sha256 "$APPROVAL_SHA"
"$PY" "$E/postpass_recovery.py" run --route liquid --allowlist "$D/postpass/allowlist.json" --allowlist-sha256 "$APPROVAL_SHA"
"$PY" "$E/postpass_recovery.py" run --route luna --allowlist "$D/postpass/allowlist.json" --allowlist-sha256 "$APPROVAL_SHA"
```

`--limit 1` optionally limits a segment to one additional attempt. Re-running the identical command after a normally limited segment resumes only remaining explicitly allowed sequences. A completed response ends that job immediately. No source journal, original summary or old operational metadata is rewritten. New request/result journals append under the original phase/job with the next allowed sequence; the original frozen runner cannot reconstruct those newly authorized sequences and must not be used for postpass reporting.

Admission charges all attempts across both original phases to the same route cap: $15 for Jev and Liquid, $30 for Luna and DeepSeek. Every known bill retains its actual per-attempt amount. Every unknown bill, including the original interruptions and every unpriced 503, retains the original full context/output reserve. A request is admitted only when existing liability plus its full reserve is within the cap. Unknown costs are never replaced with zero.

A new transport/provider error, unpriced scored response or bill exceeding its reserve stops all recovery dispatch. Partial and invalid completed responses are retained once and never replayed; independent jobs can continue when their billing is known and the original fatal check passes. The exact approved Jev 503 may continue within the original job's allowance, subject to stopping at five consecutive such failures or 20 in the last 100 observed HTTP responses. Each route's complete observed history, including the initial pass and earlier recovery segments, determines its breaker. Request-only journals have no observed HTTP outcome and do not count as successful responses. A new error or open breaker cannot be bypassed by restarting another route.

A crash after a request journal consumes that explicit sequence and preserves full liability. A new interrupted recovery attempt blocks subsequent dispatch and requires audit. An admission metadata file without a request journal also requires audit; it is never overwritten or silently reused. No command in this adapter increases allowances, resets a breaker or approves a new error class.

## Reports and integration

```sh
"$PY" "$E/postpass_recovery.py" report --allowlist "$D/postpass/allowlist.json" --allowlist-sha256 "$APPROVAL_SHA"
```

Reports live only under `$D/postpass/`. `report.json` aggregates all four routes; each `<route>/report.json` includes recovered and persistent eligible jobs, remaining missingness by source, benchmark partition, stratum/category, operator and content cluster, attempt status counts, known bills, unknown bill counts and full-reserve liabilities. `recovered` means a completed response was received; `fully_scored_recovered` requires both validated numeric tasks, while `partial_or_invalid_recovered` reports the remaining completed responses. Partial or invalid returned responses can still appear in missingness. No score magnitude affects selection or retry order.

Each `<route>/attempts.jsonl` exports the complete original-plus-recovery attempt history. Reports and full exports checkpoint every 25 attempts and at segment exit; immutable per-attempt evidence is fsynced on every request. `<route>/segments/` records start/end times, effective concurrency, immutable allowlist and policy hashes, helper source hashes and attempt membership. `<route>/attempt-metadata/<phase>/<job>/<sequence>/admission.json` binds the source receipt, explicit allowance, request hash, admission time and full reserve before dispatch; `outcome.json` retains result receipt hashes and the measured HTTP timestamps and latency. Original operational segments remain separate for latency comparisons.

The integration API is:

```python
from pathlib import Path
import postpass_recovery

approval = postpass_recovery.read_allowlist(Path(allowlist_path), recorded_allowlist_sha256)
records = postpass_recovery.reconstruct(original_plan, approval)
# Or, for validated plans grouped by route and phase:
records_by_route = postpass_recovery.all_records(plans_by_route_and_phase, approval)
```

Validate original plans with `api_runner.validate_plan` before calling the loader; `load_plans()` performs the full production coverage, budget, reserve and payload checks. The authorization loader validates the externally recorded allowlist hash, its internal ID, the policy and helper source hashes. `reconstruct` validates unchanged original receipts and admits only new sequences with bound metadata. Pass every retained attempt to `analyze.build_long`: failed or unknown originals plus one completed response retain all known and unknown cost accounting. More than one completed/scored response is rejected, including when one is invalid. Final analysis must use this loader after recovery and must retain the amendment and missingness limitations.

## Local verification

```sh
cd "$E"
"$PY" -m pytest -q test_postpass_recovery.py test_analysis.py test_api.py test_interruption_resume.py test_deepseek_recovery.py
```

The tests use synthetic journals and block live HTTP. They cover exact error classification, original 21-interruption binding, all-route preparation barriers, original receipt and helper drift, sequence caps, interrupted retry returning 503, completed-response protection, partial/invalid outcomes, request-only recovery stops, serial resume, circuit breakers, full unknown reserves, cross-phase budgets and failed-plus-completed analysis accounting. Creating these four implementation/policy/test/runbook files does not prepare an allowlist or dispatch recovery.

## Serial orchestration and immutable initial snapshot

The completed initial pass is retained in `results/api/postpass/initial-snapshot/`: `attempts.jsonl` contains22,001 validated API/Lite attempts and `scores-long.jsonl` contains27,500 expected model/arm/reference/derived rows. `snapshot.json` pins both exports and their validated inputs. These files were made read-only before the first recovery call. Protected Luna/DeepSeek benchmark review can use this stable table while recovery proceeds.

`postpass_pilot_driver.py` invokes the unchanged serial dispatcher for Luna, Liquid, a50-attempt Jev pilot, then remaining Jev allowances. It retains the original global/all-route locks and validates all frozen plans once for the sequence; each paid invocation still repeats the original allowance, source, input and payload checks. The pilot checkpoint `jev-serial-pilot.json` records the exact first segment and its report before continuing under the same serial policy. Any blocker stops the entire sequence. This orchestration does not change concurrency, eligibility, per-job allowances, budget or the original postpass breaker. Three synthetic checks cover order, one shared validation pass, stopping on an earlier route failure and refusing an accidental orchestrator restart.

## Later serial availability override

The original serial policy stopped after236 Jev recovery attempts:196 produced fully scored responses and40 returned the same exact503 error (16.95%). The last100 contained20 failures, the longest recovery failure run was3, and conservative route liability was$8.016984948 of$15. Five cases had exhausted their two additional attempts; all429 still-eligible cases were untouched rehearsal jobs. `postpass-availability-assessment-01.json` preserves the evidence. The final500-image benchmark therefore became immutable before the remaining rehearsal recovery; its6,500-row table and4,589 attempts are retained under `results/api/postpass/final-benchmark-snapshot/`.

The parent explicitly approved a separate availability override for remaining serial Jev recovery. `postpass_availability.py` preserves the original dispatcher, metadata writer, loader, frozen allowlist, protected-output rules, per-job attempt caps and full-reserve$15 limit. It retains the five-consecutive stop, uses30 failures in the last100 observed postpass responses, and stops at cumulative≥20% after100 across the complete observed postpass history. Other errors and unknown new bills still stop. Existing exhausted jobs stay missing. This is a later operational amendment, not a change to score acceptance or a retroactive edit of the original policy.

`postpass-availability-amendment-01.json` binds the adapter/source hashes, original allowlist/policy hashes, stopped segment, evidence and all4,236 prior Jev attempt receipts. The service is `modelgym-postpass-availability-20261101.service`; its log is `results/api/postpass/availability-service.log`. Every new admission has an extra immutable record under `availability-amendments/<amendment_id>/admissions/` with guard counts, liability, reservation, source/request/amendment hashes and pre-HTTP time. The original admission remains untouched. `api_operations_summary.py` independently recomputes those guards and liabilities in chronological serial order, checks cooldown and all timestamps, and includes the hashes in the final operational audit. Final `analyze.py` arguments do not change because original reconstruction/eligibility rules remain intact.
