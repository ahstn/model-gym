# API scoring for the 10,000-image expansion

The active profile is `labeling/prompts/teacher.json`, `decision.json` and `labeling/routes.json`, version `photo-labeling-v2-2026-11-01`. `api_profile.py` records the adaptation of the repository's v2 technical rubric to the two-task photography contract. The aesthetic rubric and two-decimal string schema are retained; clear excluded media still receive 1.00 on both tasks. The revised Liquid/Jev questions from `9a2f7ce` are preserved. No image statistics enter requests.

All four API routes are freshly scored. The old Luna/DeepSeek prompt differs, and the user explicitly requested fresh Liquid/Jev decisions. Q-ReAlign Lite is managed separately. Raw observations from earlier experiments are never relabeled as fresh calls.

## Preflight and smoke

`api_preflight.py` records public model/endpoint metadata plus the authenticated RunInfra model catalog under ignored `data/labeling-10000-2026-11-01/api-provider-snapshot.json`. Four nonbenchmark images per model passed the smoke test with both scores valid and provider cost receipts present. Luna and DeepSeek requested high reasoning and returned nonzero reasoning token counts; the responses did not separately echo reasoning effort. Smoke calls are reused by the resumed batch.

Initial execution uses 16 Luna workers, 16 DeepSeek workers, four Liquid workers and four Jev workers. The user authorized the session's API spend. Initial conservative per-route guards total $50: $25 Luna, $15 DeepSeek, $8 Jev and $2 Liquid. They include worst-case reservations for active or unpriced calls, not just known charges. A guard stop calls for an accounting review; it is not evidence of a user-imposed spending limit.

## Storage and resumption

All full outputs are outside git:

```text
/home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01/results/api/<model>/
  plans/<phase>.json
  attempts/<case_id>/<attempt>/request.json
  attempts/<case_id>/<attempt>/result.json
  observations/<case_id>.json
  <phase>-status.json
```

Plans pin the model/profile, image hashes, execution code and any reference freeze. Request journals store payload hashes and metadata, never credentials or inline image payloads. Responses and every attempt remain available for audit. The observations directory contains the latest terminal observation for each image/model under the pinned profile.

The first phase is `existing-rehearsal`, covering the prior 3,500 nonbenchmark images. The original 500 benchmark cases remain benchmark cases; no rehearsal image is promoted into the new blind reference set. New phases may use another manifest and overlap earlier cases only when their exact constructed request hashes match; completed responses are reused without another request.

Example freeze and launch from the worktree:

```sh
/home/ahstn/git/model-gym/visual-quality/.venv/bin/python \
  visual-quality/experiments/labeling-10000-2026-11-01/api_runner.py freeze \
  --model luna --phase expanded-rehearsal --selection rehearsal \
  --manifest /absolute/path/to/manifest.json

/usr/bin/zsh visual-quality/experiments/labeling-10000-2026-11-01/api_service_entry.zsh \
  run --model luna --phase expanded-rehearsal
```

The service entry sources `~/.config/zsh/.zsh-private` quietly. A route lock prevents simultaneous workers from different phases racing on the same image. Do not start a second phase for a model until its current phase finishes.

For benchmark phases, use `--selection benchmark --references /absolute/path/to/freeze.json`. `reference_pins()` checks the reference file hash, exact case/view matches and a past timezone-aware freeze timestamp. A freeze may cover the original 500 references or a new completed batch; it need not cover all 1,500 before earlier frozen cases run. Optional `--case-ids` accepts a JSON list and limits admission to those IDs. Reference labels and statistics are never included in payloads.

## Failures, timing and costs

Retry only transport failures or HTTP 429/502/503/504, bounded at three attempts for Luna and five for the other routes. Backoff respects `Retry-After` up to 60 seconds. A valid or invalid score response is not retried merely to obtain a different score. An interrupted attempt has an unknown charge and may be retried within the same fixed attempt allowance; its original request journal remains and its conservative cost reservation is retained.

Authentication/payment errors stop new admission. A circuit stops admission when more than 25% of the latest 20–100 completed cases remain failed after their bounded retries. In-flight requests finish and are journaled. Resume requires unchanged pinned code and requests; do not edit the runner in place while a plan is active.

`latency_seconds` measures the last attempt's HTTP invocation through receipt of the complete response. `total_attempt_latency_seconds` is separate. `price_usd` is the selected attempt's provider receipt, while `total_attempt_known_price_usd`, `total_attempt_price_usd` and `unpriced_attempts` describe all attempts. Missing charges remain null. Returned model/provider identities and schema are validated; inconsistent chat scope floors invalidate both outcomes. Native Decisions scores and scope probabilities remain available separately from the applied eligibility floor.

Phase `complete` means all scheduled cases reached their bounded terminal state. It does not guarantee that every outcome is numeric: inspect per-task validity and nulls in observations, plus terminal status counts. Root publication checks independently count usable outcomes and enforce the explicit first-1,000-image allowlist. Do not commit full observations, raw results, score-bearing logs or benchmark records outside that allowlist.

## Checks

```sh
PYTHONPATH=visual-quality/src \
  /home/ahstn/git/model-gym/visual-quality/.venv/bin/python -m pytest -q \
  visual-quality/experiments/labeling-10000-2026-11-01/test_api_runner.py \
  visual-quality/tests/test_labeling_profile.py
```

The 18 initial tests cover v2 adaptation, strict floor consistency, exact reference-image admission, retry classification, valid-response reuse, changed-image rejection, unknown-charge reservation, native Decisions precision and absence of statistical prompt inputs.

## Expanded phases and operational amendments

The frozen 10,000-image manifest is at `data/labeling-10000-2026-11-01/manifest.json`, SHA-256 `65f4df78e3e9d7aea62a0695f386462e3363957773eace00848c61bb5a75e859`. `new-rehearsal-manifest.json` contains only the 5,000 additional nonbenchmark views. Their `expanded-rehearsal` phases run after each route's original benchmark phase.

Operational amendment 01 raises future DeepSeek/Jev conservative accounting guards to $40 each and Liquid concurrency from four to eight. Liquid had no transient failures in its first 3,450 outcomes; Jev initially remained at four workers because its intermittent HTTP503 failures persisted. Prompts, model settings, validation, retry ceilings and unknown-charge reserves are unchanged. Amendment 02 replaces Jev's still-waiting original benchmark phase with `existing-benchmark-amended`, using the $40 guard; the unused original frozen plan remains recorded. Amendment 03 resumes Jev's original rehearsal under `existing-rehearsal-guard-recovery` after its initial $8 guard stopped admission. The recovery reuses valid responses and retains the original five-attempt allowance, every cost receipt and all unknown-charge reserves. The replacement phase completed all 3,500 original rehearsal cases successfully; subsequent Jev work uses the amended serial chain. These guards are internal accounting controls, not new user spending limits. All amendment receipts live in the ignored data directory.

Amendment 04 increases Jev to eight workers after a graceful SIGINT/drain left zero pending requests. The immutable `expanded-rehearsal-workers8` plan and replacement queued reference plans preserve every request setting and reuse valid outputs. The first 200 newly started cases completed at 73.74 images/minute versus 31.86 in the previous 200-case window. HTTP503 attempt share was 13.04% versus 17.01%; terminal failure rate was zero versus 0.50%. The approved rollback triggers—an increase exceeding ten percentage points in HTTP503 share or terminal failure rate above 2%—were not reached, so eight workers were retained. This is an operational comparison across different images and times, not a causal quality result. `api_jev_concurrency_audit.py` selects cases by initial request time and waits for every terminal outcome, retaining slower failures. The initial completion-ordered audit is preserved as superseded; the corrected frozen comparison and attempt hashes remain under the ignored data directory.

`api_queue_benchmarks.py --watch` validates completed reference packet/freezes and creates one serial phase per packet/model. It never prints benchmark score values. `api-benchmark-queue.json` records admitted IDs, frozen plans and dynamic unit names; `api-admission/` retains each admission receipt before services start. Each route waits for its previous phase, so the queue does not multiply model concurrency. Duplicate IDs, changed freeze files, changed packet/label hashes and unfrozen references are rejected. The watcher exits when all 1,000 new benchmark cases are admitted.

`api_audit.py` writes coverage, join checks, failure categories and full attempt costs to `api-audit.json` without displaying individual score values. Check the separate `pending_or_interrupted_requests` count while workers are active.

After all 10,000 main-phase cases reach terminal observations, `api_missing_postpass.py freeze --model MODEL` can freeze the root-authorized one-shot recovery allowance. Eligibility requires no numeric outcome or native score, and a failed/unknown/invalid status; valid numeric and valid-unscorable assessments are excluded. `run --model MODEL` sends exactly the same payload once more, with unchanged token limit, preserving original observation snapshots, all attempts and costs. This is a separate operational amendment for missing outputs, not score-based retry selection. It will not start before main-phase coverage is complete.

The `modelgym-10000-api-completion.service` runs `api_missing_postpass.py watch` to perform that final allowance automatically as each route completes. It requires both exact observation coverage and completed main-phase admissions; a running final phase cannot trigger recovery early. The watcher holds a process lock, retains frozen postpass plans on restart and writes aggregate completion state to `api-completion-state.json`. Empty recovery selections produce a recorded no-op. Main-phase prompt and runner source files remain pinned throughout.

## Completed API observations and aggregate reporting

All four routes reached 10,000 terminal observations. The one-shot missing-output pass recovered three DeepSeek responses and two Jev responses; Luna and Liquid needed no recovery calls. Final numeric technical/aesthetic pairs are 10,000 Luna, 9,999 DeepSeek, 10,000 Liquid and 9,997 Jev. One DeepSeek valid-unscorable response retains null outcomes: reconstructing its frozen request verified the exact delivered image bytes and an image-bearing request, and the provider reported image tokens. Three Jev responses remain HTTP 400 image-dimension errors after the bounded allowance. Their raw attempts, unknown charges and missing assessments remain in the local dataset.

The aggregate accounting audit contains 10,000 Luna attempts, 10,083 DeepSeek attempts, 10,000 Liquid attempts and 11,983 Jev attempts. Known subtotals are $8.093620750, $6.55242845, $0.50219912 and $1.813157262 respectively. DeepSeek has 81 unpriced attempts and Jev has 1,986; these are unknown charges, not assumed zero. The audit found no pending requests or image-join problems. `api-completion-state.json`, `api-audit.json` and the immutable recovery plans retain the detailed receipts outside git.

`report_run.py` reads the full local publication exports and writes only aggregates. It separates the original 500, fresh example 500 and confirmation 500 reference images; reports task-specific coverage, MAE, bias, rank agreement, matched comparisons, technical tail errors and scope confusion; and keeps missing classifications separate from errors. Its sharpness analysis is descriptive, with no score correction. Selected-request latency and selected-versus-all-attempt cost accounting are separate. Optional Runpod billing snapshots appear as infrastructure receipts; Lite and Reference inference-row prices remain zero. The total-cost-per-model table ends the report.

For a preview, use `--run-state in-progress` and a destination under ignored data. After all inference is terminal and publication exports are regenerated, run:

```sh
/home/ahstn/git/model-gym/visual-quality/.venv/bin/python \
  visual-quality/experiments/labeling-10000-2026-11-01/report_run.py \
  --exports /home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01/exports \
  --stats /home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01/stats.jsonl \
  --output visual-quality/experiments/labeling-10000-2026-11-01/report.md \
  --run-state finished \
  --embedding-billing /absolute/path/to/embedding-billing-receipt.json \
  --lite-billing /absolute/path/to/lite-billing-receipt.json
```

The companion JSON records aggregate metrics and input hashes. Input changes during reading, inconsistent summary coverage or unfinished observations in a report marked finished are rejected. `test_report_run.py` checks missing outcomes, exact threshold boundaries, native precision, tied ranks, matched coverage, scope uncertainty, view joins, publication privacy and accounting semantics. No reference labels, weights or prompts are fitted by this report.
