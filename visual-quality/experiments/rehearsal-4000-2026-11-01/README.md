# Four-thousand-image scoring rehearsal

This experiment builds a varied 4,000-image corpus with EmbeddingGemma 2 and a nested, fresh 500-image assistant-reference benchmark. It compares Luna and DeepSeek at high reasoning, Liquid d1.1, Jev 1.14 and Q-ReAlign Lite. Luna and DeepSeek also receive own-image and swapped-image statistics on the benchmark.

**Status: runs and bounded recovery complete.** All 22,000 model/image/arm slots were attempted; 21,984 have both numerical scores. Thirteen Jev cases remain missing after their allowed attempts. DeepSeek has one billed no-answer response and two nullable assessments. All 500 reference cases are complete, and all inference services and trial GPU pods are stopped. None of the labels is approved for training.

[Evaluation report](report.md) · [Full score CSV](scores-long.csv) · [Interactive 4,000-image gallery](gallery/index.html) · [Costs](cost-per-model.csv)

The 72-image collage is split into four readable pages: [1](gallery/comparison-page-1.png) · [2](gallery/comparison-page-2.png) · [3](gallery/comparison-page-3.png) · [4](gallery/comparison-page-4.png).

The baseline Luna/DeepSeek average has the lowest combined-error point estimate on the common representative cases, although its advantage over Liquid and Jev remains uncertain. The tested statistics text does not meet the predefined technical-improvement target and introduces visible regressions. Statistics remain useful for targeted review, with limited recall. See the report for paired intervals, intentional-effects checks, scope ambiguity and next-trial recommendations.

The [corpus audit](corpus-audit.json) verifies 3,700 recorded source families, 40 content clusters and 500 distinct benchmark families. [Content coverage](content-coverage.md) documents the sampling balance; the gallery exposes five related images per view. One confirmed historical near-duplicate was replaced before inference under the declared freshness rule, preserving the original manifest and all superseded review evidence in [the revision record](benchmark-revisions.json).

A separate [50-image blind repeat](reference-reliability/report.md) leaves primary labels unchanged. All [100 controlled families](controlled-review/audit.json) have also received blind visual comparison: all severe variants are visibly worse than their parents, while 27 mild JPEG comparisons are not clearly worse. Full-family results and independently verified subsets will both be retained.

The [protocol](protocol.md) defines selection, blind review, treatment comparisons, latency and cost accounting before inference. The [corpus method](corpus-method.md) records image preparation and sampling. The [API runbook](api-runbook.md), [Lite protocol](lite-protocol.md) and [analysis runbook](analysis-runbook.md) document execution and reconstruction.

Saved artifacts:

- `manifest.json`: all 4,000 delivered views, hashes, source families, partitions and preprocessing.
- `references.jsonl` and `references-freeze.json`: 500 independently inspected images, two scores each, visible evidence and measured review durations; frozen before benchmark calls.
- `scores-long.csv` and `scores-long.jsonl`: 22,000 model/image/arm slots, 500 reference rows and 5,000 derived Luna/DeepSeek mean rows. Missing observations remain explicit.
- `analysis.json`, `cost-per-model.csv` and the final report: coverage, assistant-reference agreement, paired stats comparisons, latency, cost and diagnostic results.
- `gallery/index.html`: all 4,000 images with their recorded scores, latency and price. Four PNG pages show a fixed selection of 72 benchmark images spanning reference technical scores.

Reported per-image prices for Reference and Q-ReAlign Lite are $0.00 by request. Actual Runpod infrastructure usage is accounted separately. Reference assessment has no HTTP latency; measured visual-review duration is a separate field. The Luna/DeepSeek mean has no independent request or charge.

Bulk image views and raw execution journals live under `/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/`. The interactive gallery links to those exact local views. Seven compressed [execution archives](execution-records/index.json) preserve the API, recovery and Lite receipts. [Reproduction notes](reproduction.md) explain local paths and the durable viewing copy. File hashes, configuration snapshots and raw receipts support reproducibility; credentials are never included in experiment artifacts.
