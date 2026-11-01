# Independent artifact-readiness review

Reviewed 2026-11-01 while API inference was still active. This is a read-only review of corpus, reference, analysis and packaging contracts, not a final completeness certificate. No live model scores were reanalysed and no frozen inputs were changed.

The frozen corpus audit's artifact hashes all match. The manifest contains 4,000 images across 3,700 recorded families: 100 development, 300 representative, 100 challenge and 3,500 rehearsal cases. The 500 benchmark families are unique. All 500 reference identities match the benchmark, and the reference JSONL matches its freeze hash. Reference scope counts are 451 eligible, 39 excluded and ten uncertain.

The focused analysis, postpass-loader integration, packaging, protected-review and reference-review suite passed: **102 tests and three subtests**. The corpus unit-test module could not be collected in the scoring environment because that environment lacks scikit-learn; corpus tests belong in the separately recorded embedding environment. This dependency issue is not a failed corpus-integrity check.

No blocking score-selection, double-counted billing, benchmark-family, or paired-comparison defect was found. The exporters retain absent model/image/arm slots; all attempt prices remain distinct from the selected response's price; uncertain bills remain unknown. Postpass recovery requires an explicit hash-pinned allowlist and refuses replay of a completed/scored response. Reference and derived-mean HTTP latency remain inapplicable. Paired uncertainty resamples families and retains task-specific valid cases. These checks do not establish the truth of assistant quality judgments.

Before final publication:

- Reconcile the report's protected-review wording with the completed adjudication supplement. `report.py` currently describes review as pending even when a supplement is present; completing review must not imply approval of training labels.
- Show the already computed eligible-photograph/native-score comparison beside whole-domain results when interpreting Q-ReAlign Lite. Its unchanged native model has no photography-scope floor; total errors therefore combine scope and photographic-quality differences. Do not retrofit a gate using these benchmark labels.
- Run the final analysis only with the completed recovery allowlist, then regenerate the protected review, supplements and gallery from that exact export. API jobs were active during this audit, so final coverage and prices were deliberately not certified here.
- Package only after all services and recovery processes stop. Retain both the selected-attempt and all-attempt price semantics in the gallery/report, plus the separate stopped-pod billing receipt or explicitly unsettled infrastructure estimate.
- Preserve a durable copy of the final publication artifacts. The manifest and raw-response contracts intentionally contain absolute paths; exact-view gallery links require this machine's local image data and will not become portable merely by pushing the report to Git.

One minor exporter robustness issue was also identified: `normalize_attempt` assigns `latency_kind=http_request_through_complete_response` whenever `response_received_at` is set, while the frozen API runner also sets that timestamp after a transport exception. A recorded transport failure can therefore receive an overly strong latency-kind label despite its error status. No actual affected completed score was identified in this read-only review; adjust derived export labeling if such attempts occur, without changing frozen request execution.
