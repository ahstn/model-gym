## Operational segments and provider failures

Jev required an explicit provider-failure recovery. Its concurrency limit changed4→1→4 in separate documented segments: serial operation did not remove the observed503 failures and was substantially slower in the measured windows. DeepSeek retained billed no-answer token-limit failures and continued with unchanged concurrency8 and token/reasoning settings. A client-process interruption left21 request-only journals (Luna16,Liquid4,Jev1); they retain unknown original latency and cost. Subsequent execution uses durable user services. Single-request smokes are separate. These timings include upload, queueing and the complete response and are descriptive, not a controlled model-speed comparison. Failed labels remain null; known charges and unknown bills remain distinct. The original verified pre-serve capacity429 retry policy remains separate.

| Model | Phase | Segment | Concurrency limit | Attempts | Valid | Invalid | Failures | Known503 | Exhaustions | p50 all(s) | p95 all(s) | p50 scored(s) | p95 scored(s) | Unpriced |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepseek | benchmark | 2026-11-01T14-19-30.114456+00-00 | 8 | 1500 | 1500 | 0 | 0 | 0 | 0 | 8.42 | 23.32 | 8.42 | 23.32 | 0 |
| deepseek | rehearsal | 2026-11-01T14-13-30.838015+00-00 | 8 | 198 | 198 | 0 | 0 | 0 | 0 | 9.76 | 25.99 | 9.76 | 25.99 | 0 |
| deepseek | rehearsal | 2026-11-01T14-19-30.114456+00-00 | 8 | 2426 | 2426 | 0 | 0 | 0 | 0 | 9.72 | 21.82 | 9.72 | 21.82 | 0 |
| deepseek | rehearsal | initial_single_request_smoke | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 6.33 | 6.33 | 6.33 | 6.33 | 0 |
| deepseek | rehearsal | original_frozen_plan | 8 | 876 | 874 | 0 | 2 | 0 | 1 | 10.58 | 21.10 | 10.58 | 21.09 | 1 |
| jev | benchmark | 2026-11-01T14-49-02.948804+00-00 | 1 | 3 | 1 | 0 | 2 | 2 | 0 | 0.58 | 1.36 | 1.45 | 1.45 | 2 |
| jev | benchmark | 2026-11-01T14-51-21.762892+00-00 | 1 | 10 | 9 | 0 | 1 | 1 | 0 | 1.64 | 4.05 | 1.45 | 4.07 | 1 |
| jev | benchmark | 2026-11-01T14-52-32.220892+00-00 | 4 | 383 | 322 | 0 | 61 | 61 | 0 | 1.79 | 3.37 | 1.83 | 3.36 | 61 |
| jev | benchmark | 2026-11-01T14-58-03.329049+00-00 | 1 | 10 | 8 | 0 | 2 | 2 | 0 | 1.21 | 1.81 | 1.21 | 1.64 | 2 |
| jev | benchmark | 2026-11-01T15-00-53.007834+00-00 | 1 | 10 | 9 | 0 | 1 | 1 | 0 | 1.04 | 2.76 | 1.05 | 2.84 | 1 |
| jev | benchmark | 2026-11-01T15-01-58.391747+00-00 | 4 | 84 | 73 | 0 | 11 | 11 | 0 | 1.40 | 2.62 | 1.47 | 2.69 | 11 |
| jev | benchmark | 2026-11-01T15-57-54.813535+00-00 | 1 | 50 | 43 | 0 | 7 | 7 | 0 | 0.77 | 1.27 | 0.77 | 1.31 | 7 |
| jev | benchmark | 2026-11-01T15-58-48.296000+00-00 | 1 | 39 | 33 | 0 | 6 | 6 | 0 | 0.80 | 1.41 | 0.81 | 1.42 | 6 |
| jev | rehearsal | 2026-11-01T13-48-48.969461+00-00 | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 3.89 | 3.89 | 3.89 | 3.89 | 0 |
| jev | rehearsal | 2026-11-01T13-49-08.031529+00-00 | 4 | 34 | 32 | 0 | 2 | 2 | 0 | 4.80 | 7.30 | 4.80 | 7.32 | 2 |
| jev | rehearsal | 2026-11-01T13-50-16.101872+00-00 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1.68 | 1.68 | 1.68 | 1.68 | 0 |
| jev | rehearsal | 2026-11-01T13-50-59.949122+00-00 | 1 | 4 | 3 | 0 | 1 | 1 | 0 | 3.12 | 4.80 | 2.94 | 4.85 | 1 |
| jev | rehearsal | 2026-11-01T14-02-31.843566+00-00 | 1 | 121 | 109 | 0 | 12 | 11 | 0 | 3.19 | 5.10 | 3.18 | 5.01 | 12 |
| jev | rehearsal | 2026-11-01T14-20-46.842974+00-00 | 1 | 150 | 134 | 0 | 16 | 16 | 0 | 3.45 | 5.52 | 3.43 | 5.39 | 16 |
| jev | rehearsal | 2026-11-01T14-29-54.936464+00-00 | 4 | 144 | 121 | 0 | 23 | 23 | 0 | 4.81 | 7.02 | 4.84 | 6.86 | 23 |
| jev | rehearsal | 2026-11-01T15-01-58.391747+00-00 | 4 | 592 | 499 | 0 | 93 | 93 | 0 | 1.66 | 3.14 | 1.70 | 3.18 | 93 |
| jev | rehearsal | 2026-11-01T15-09-35.416943+00-00 | 1 | 10 | 9 | 0 | 1 | 1 | 0 | 1.83 | 3.39 | 1.70 | 3.45 | 1 |
| jev | rehearsal | 2026-11-01T15-10-47.529644+00-00 | 4 | 144 | 119 | 0 | 25 | 25 | 0 | 1.44 | 3.58 | 1.54 | 3.61 | 25 |
| jev | rehearsal | 2026-11-01T15-14-48.096374+00-00 | 1 | 10 | 10 | 0 | 0 | 0 | 0 | 0.87 | 1.86 | 0.87 | 1.86 | 0 |
| jev | rehearsal | 2026-11-01T15-16-05.259766+00-00 | 4 | 121 | 99 | 0 | 22 | 22 | 0 | 1.50 | 3.05 | 1.58 | 3.07 | 22 |
| jev | rehearsal | 2026-11-01T15-19-27.501232+00-00 | 1 | 10 | 9 | 0 | 1 | 1 | 0 | 0.72 | 1.16 | 0.72 | 1.17 | 1 |
| jev | rehearsal | 2026-11-01T15-20-34.340736+00-00 | 4 | 97 | 76 | 0 | 21 | 21 | 0 | 1.42 | 2.39 | 1.50 | 2.40 | 21 |
| jev | rehearsal | 2026-11-01T15-26-16.692882+00-00 | 4 | 2054 | 1721 | 0 | 333 | 333 | 0 | 1.12 | 1.99 | 1.15 | 2.01 | 333 |
| jev | rehearsal | 2026-11-01T15-58-48.296000+00-00 | 1 | 147 | 120 | 0 | 27 | 27 | 0 | 0.77 | 1.22 | 0.79 | 1.15 | 27 |
| jev | rehearsal | 2026-11-01T16-12-38.904888+00-00 | 1 | 500 | 421 | 0 | 79 | 79 | 0 | 0.80 | 1.20 | 0.82 | 1.23 | 79 |
| jev | rehearsal | initial_single_request_smoke | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0.91 | 0.91 | 0.91 | 0.91 | 0 |
| jev | rehearsal | original_frozen_plan | 4 | 6 | 4 | 0 | 2 | 2 | 0 | 2.49 | 2.64 | 2.49 | 2.64 | 2 |
| liquid | benchmark | 2026-11-01T14-19-49.364344+00-00 | 4 | 500 | 500 | 0 | 0 | 0 | 0 | 1.75 | 3.07 | 1.75 | 3.07 | 0 |
| liquid | rehearsal | 2026-11-01T14-19-49.364344+00-00 | 4 | 1805 | 1805 | 0 | 0 | 0 | 0 | 3.25 | 5.74 | 3.25 | 5.74 | 0 |
| liquid | rehearsal | 2026-11-01T15-57-43.953510+00-00 | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 1.63 | 1.71 | 1.63 | 1.71 | 0 |
| liquid | rehearsal | initial_single_request_smoke | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1.78 | 1.78 | 1.78 | 1.78 | 0 |
| liquid | rehearsal | original_frozen_plan | 4 | 1694 | 1690 | 0 | 4 | 0 | 0 | 3.32 | 5.41 | 3.32 | 5.41 | 4 |
| luna | benchmark | 2026-11-01T14-19-32.029167+00-00 | 16 | 1500 | 1500 | 0 | 0 | 0 | 0 | 9.03 | 12.35 | 9.03 | 12.35 | 0 |
| luna | rehearsal | 2026-11-01T14-19-32.029167+00-00 | 16 | 896 | 896 | 0 | 0 | 0 | 0 | 9.08 | 12.44 | 9.08 | 12.44 | 0 |
| luna | rehearsal | 2026-11-01T15-55-59.697997+00-00 | 1 | 16 | 16 | 0 | 0 | 0 | 0 | 6.07 | 7.25 | 6.07 | 7.25 | 0 |
| luna | rehearsal | initial_single_request_smoke | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 6.08 | 6.08 | 6.08 | 6.08 | 0 |
| luna | rehearsal | original_frozen_plan | 16 | 2603 | 2587 | 0 | 16 | 0 | 0 | 8.58 | 11.98 | 8.58 | 11.98 | 16 |

The exact known503 class was quarantined under a prospective bounded policy: five consecutive failures or20 in the last100 requests stops further dispatch. Other error classes stop immediately. All unknown costs keep their conservative budget reservations; known subtotals are not complete billed totals. [Detailed operational audit](operations-summary.json).

Later Jev circuit resets are explicit operational amendments. Each requires at least120seconds of cooldown and a separate ten-request serial half-open batch using never-attempted jobs; a second exact503 stops that batch. Continuation requires all ten responses and at most one exact503. The original five-consecutive/20-of100 thresholds then apply to the documented new window, including the probes. Historical failures, labels and unknown-cost reservations remain intact. The frozen benchmark phase was prioritized for coverage during provider instability, before completing the rehearsal; score magnitudes did not determine this order.

| Stage | Status | Concurrency | New attempts | Exact503 | Prior observed responses retained |
|---|---|---:|---:|---:|---:|
| probe | probe_failed | 1 | 3 | 2 | 461 |
| probe | probe_succeeded | 1 | 10 | 1 | 464 |
| continue | stopped_requires_audit | 4 | 383 | 61 | 464 |
| probe | probe_failed | 1 | 10 | 2 | 857 |
| probe | probe_succeeded | 1 | 10 | 1 | 867 |
| continue | stopped_requires_audit | 4 | 676 | 104 | 867 |
| probe | probe_succeeded | 1 | 10 | 1 | 1553 |
| continue | stopped_requires_audit | 4 | 144 | 25 | 1553 |
| probe | probe_succeeded | 1 | 10 | 0 | 1707 |
| continue | stopped_requires_audit | 4 | 121 | 22 | 1707 |
| probe | probe_succeeded | 1 | 10 | 1 | 1838 |
| continue | stopped_requires_audit | 4 | 97 | 21 | 1838 |

The remaining Jev first pass used a separately approved availability amendment after1,945 observed responses showed296 exact503s(15.22%), a longest failure run of4 and repeated20-of100 rolling trips despite prompt successful responses. It retained the five-consecutive stop, raised the rolling stop to30of100 and added a cumulative≥20% stop after100 observed responses using the complete initial history. Only availability tolerance changed: original request payloads, grading acceptance, budgets, unknown-cost reservations and the ban on replaying first-pass jobs remained intact. This later amendment does not alter the separately bounded postpass policy.

Serial recovery later reached the original20-of100 availability stop after236 additional Jev attempts produced196 fully scored recoveries and40 exact503s(16.95%). A separately pinned operational override retained serial dispatch, the original immutable allowance/job caps, protected-output rules and$15/full-reserve limits; it kept the five-consecutive stop, used30of100, and applied cumulative≥20% after100 across the entire observed postpass history. Five already exhausted cases remained missing. Each subsequent admission has separate amendment/source hashes, chronological guard evidence and a verified budget reservation; original admission receipts and the loader remain unchanged.

After every initial API job had been attempted, a separate procedural amendment allowed bounded recovery of the exact Jev503 class (up to two additional attempts) and the21 interrupted requests (one additional attempt). Existing valid, partial or invalid scored responses were never replayed. Original unknown bills remain unknown, including possible charges for interrupted requests. DeepSeek no-answer token-limit failures remain missing. This recovery was an explicit later amendment, not part of the preregistered initial protocol.

| Model | Eligible jobs | Completed responses obtained | Fully scored recoveries | Partial/invalid recoveries | Persistent without response | All jobs missing either score |
|---|---:|---:|---:|---:|---:|---:|
| deepseek | 0 | 0 | 0 | 0 | 0 | 3 |
| luna | 16 | 16 | 16 | 0 | 0 | 0 |
| jev | 630 | 617 | 617 | 0 | 13 | 13 |
| liquid | 4 | 4 | 4 | 0 | 0 | 0 |

Source, partition, category, operator and content-cluster missingness remain in the detailed recovery report. Recovered-response counts do not imply complete two-task labels; final accuracy uses task-level validity and reports coverage.
