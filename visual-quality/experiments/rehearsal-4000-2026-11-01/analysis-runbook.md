# Rehearsal analysis and export

**Historical study archive:** the 50/50 Luna/DeepSeek mean below is superseded. Its metrics and study decisions retain their original provenance; they are not the current scorer. Current displays recompute 37.5% Luna + 37.5% DeepSeek + 25% Liquid independently per outcome from the original observations. See the [current canonical gallery](../../data/rehearsal-4000-2026-11-01/canonical/gallery/index.html) and [current labeling rules](../../AGENTS.md).

`analyze.py` reconstructs API journals and independently reparses Lite raw responses after the 500-reference freeze. It validates view/request identities, duplicate attempts, frozen runtime/configuration, reference timing and coverage. It does not open live results during implementation; `test_analysis.py` uses synthetic fixtures.

```sh
PY=/home/ahstn/git/model-gym/visual-quality/.venv/bin/python
E=/tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01
$PY -m pytest -q "$E/test_analysis.py" "$E/test_api.py"
$PY "$E/analyze.py" --qc-thresholds "$E/qc-reference.json"
```

The default bootstrap is 10,000 paired source-family resamples with seed 20261101. The primary comparison is the Luna/DeepSeek mean with own-image stats versus its unchanged baseline on the 300 representative images. Each task must retain at least 95% paired coverage of the full panel and have no increase in missing/invalid numerical scores relative to baseline before a comparison is eligible for a decision. Expected panel, numerical-reference and paired counts are reported separately. Lower-coverage estimates remain descriptive; numerical targets alone cannot authorize advancement. The 100 challenge and 100 development images have separate outputs. The pooled 400 evaluation cases are exploratory. Primary paired coverage does not depend on unrelated challenger validity. Additional fair-comparison tables restrict to cases valid for both tasks in all five baselines.

Outputs are `scores-long.csv`, `scores-long.jsonl`, `analysis.json` and `cost-per-model.csv` under the experiment directory. The CSV has all 22,000 expected model/image/arm slots, 500 references and 5,000 derived mean slots: 27,500 rows even when some responses are missing. Missing scores, costs and latencies are empty cells; explicit Lite/Reference zero-price overrides have a distinguishing basis and do not imply a request occurred. `technical_score` and `aesthetic_score` retain full precision; the corresponding `_display` fields have exactly two decimals. Mean rows use `model=luna_deepseek_mean` and the original arm; their price and HTTP latency remain empty. References use `model=reference,arm=reference`.

A model row's `price_usd` is the selected attempt receipt. `total_attempt_known_price_usd`, `unpriced_attempts` and `total_attempt_price_usd` describe its complete attempt history. Aggregate cost tables count every attempt exactly once, including charged errors, with unknown costs remaining unknown. Derived means contribute no second charge. Lite infrastructure costs remain separate from the requested $0.00 display override. Latency includes all attempts and a separate completed-response distribution; unmeasured attempts have an explicit count.

Accuracy includes separate technical/aesthetic MAE, signed bias and Spearman agreement, task-specific coverage, common-two-task combined MAE, native-score agreement on reference-eligible images and scope errors. Full-corpus descriptive and pairwise agreement tables make no claim of correctness for the 3,500 images without references. Family bootstrap uncertainty is conditional on assistant labels; it does not represent human MOS uncertainty. The technical target requires at least 0.05 MAE improvement and an upper two-sided 95% bound below zero. Aesthetic noninferiority uses a one-sided 95% upper bound at most +0.03. These checks do not replace review of protected scope and intentional-effects slices.

QC uses the exact image-stats percentile records and hashes generated from the frozen nonbenchmark calibration families. It checks the uniform-frame gate, sharpness percentile <=25 with a technical score >=2, and noise percentile >=90 with a technical score >=2. Labels remain unchanged. A fixed 10% panel review budget reserves 20% of its slots, rounded up, for deterministic random unflagged audits. The remaining slots prioritize gate, soft and noisy cases. Slots move between pools only when a pool cannot fill its allocation; reserved and actual audit counts and case IDs are retained. Precision/recall use absolute technical error >=0.75; a 1,000-draw random-review comparison uses the same budget. Omit `--qc-thresholds` only for an explicitly incomplete analysis; the result then records QC as not computed.

Controlled diagnostics retain all 100 parent-plus-three-variant families, count non-increasing technical ladders, adjacent ties, strict inversions and severe-variant scores at least as high as the parent. They report parent-minus-severity-3 differences and completeness by operator. No aesthetic monotonicity is required, no family is removed because a proxy metric fails to move, and this is not PACE's ground-truth-threshold Holistic Override Rate. An independently verified subset is not claimed unless a later documented inspection supplies it.

The analysis does not approve labels for training. It refuses multiple completed responses for the same model/image/arm rather than selecting whichever score looks preferable. Any known excluded diagnostic smoke run should live outside the corpus result tree and be accounted separately; successful corpus smoke attempts should be resumed within their original frozen phase plan.

Render the report after the audited analysis export:

```sh
$PY "$E/report.py"
```

The renderer verifies that its manifest matches the analysis input hash, preserves missing coverage and unknown bills, and ends with the total cost-per-model table. Numerical/coverage targets remain separate from the pending protected-slice review; `advancement_ready` stays false until that independent review is documented.

Preflight safeguards: inclusive technical-improvement, aesthetic-noninferiority and 0.75 QC-error boundaries use an absolute 1e-12 tolerance solely for floating-point arithmetic noise; the CI upper-bound requirement remains strictly below zero. API execution artifacts without a frozen phase plan cause analysis to stop rather than appearing as zero-cost unstarted work. Task-specific all-five-common MAE, bias, Spearman and counts retain a valid technical score when aesthetics is missing, and vice versa; the combined common-two-task comparison remains separate.

The secondary `baseline_comparisons` output compares each individual baseline with the fixed Luna/DeepSeek mean on representative and challenge panels. It reports individual-minus-ensemble ΔMAE and paired source-family 95% intervals separately for technical, aesthetic and common-both combined error, with task-specific paired counts. Default resamples and seed are 10,000 and 20261101. These contrasts are exploratory, uncorrected and conditional on assistant labels; they do not alter the primary stats comparison or select favorable cases. A concise table is included in report.md.

Protected review uses the explicitly rebound selection after the objective R0045 benchmark replacement. The original `protected-slices.json` remains unchanged; `protected-slices-v2.json` pins it, the declaration, and the primary reference freeze. Style and excluded/uncertain memberships are unchanged. Run after the audited score export:

```sh
$PY "$E/protected_review.py" --protected "$E/protected-slices-v2.json" --benchmark "$E/benchmark-manifest-v2.json"
```

The helper validates the exact revision receipt and refuses silent rebinding, altered original reviews, or an undeclared membership change. The rebound timestamp must precede every benchmark request.

After every API driver/recovery segment stops, run `$PY "$E/api_operations_summary.py"`. It writes `operations-summary.json`, an `operations-attempt-segments.jsonl` sidecar, and `operations-report.md`. The audit validates original phase plans and recovery approval/source hashes, rejects overlapping or impossible segment membership, and separates single-request smokes, original execution, and each Jev recovery segment. Per-segment tables retain effective concurrency, valid/invalid/failure and exact503 counts, all-attempt and scored-response p50/p95 HTTP latency, known cost subtotals and unknown-bill counts. Insert the report fragment before the final total-cost table; do not interpret combined or segmented timing as a controlled speed ranking. The helper refuses execution while any recorded driver/segment remains running. Five synthetic reporting checks plus13 recovery/phase-driver checks pass.

After the separately authorized post-initial-pass recovery, both final loaders require explicit allowlist provenance. Pass `--postpass-allowlist <frozen-allowlist-path> --postpass-allowlist-sha256 <sha256-returned-by-prepare>` to `analyze.py` and `api_operations_summary.py`. There is no automatic relaxed replay fallback. The new loader validates every original receipt, bounded added sequences, unchanged payload hashes, immutable admission metadata and at most one completed/scored response per job. Historical failed/unknown attempts and costs remain in the final analysis. The operational fragment adds initial/recovery missingness and distinguishes completed recovery responses from fully scored two-task recoveries. Postpass recovery is described as a later procedural amendment, not preregistered.
