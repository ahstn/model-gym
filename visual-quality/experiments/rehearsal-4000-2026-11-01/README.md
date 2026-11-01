# Four-thousand-image scoring rehearsal

This experiment builds a varied 4,000-image corpus with EmbeddingGemma 2 and a nested, fresh 500-image assistant-reference benchmark. It compares Luna and DeepSeek at high reasoning, Liquid d1.1, Jev 1.14 and Q-ReAlign Lite. Luna and DeepSeek also receive own-image and swapped-image statistics on the benchmark.

**Status: preparation and independent reference review.** Results are not yet available. This file will be updated after the frozen requests have completed and their journals have been audited. None of the labels is approved for training.

The [protocol](protocol.md) defines selection, blind review, treatment comparisons, latency and cost accounting before inference. The [corpus method](corpus-method.md) records image preparation and sampling. The [API runbook](api-runbook.md), [Lite protocol](lite-protocol.md) and [analysis runbook](analysis-runbook.md) document execution and reconstruction.

Expected artifacts after completion:

- `manifest.json`: all 4,000 delivered views, hashes, source families, partitions and preprocessing.
- `references.jsonl` and `references-freeze.json`: 500 independently inspected images, two scores each, visible evidence and measured review durations; frozen before benchmark calls.
- `scores-long.csv` and `scores-long.jsonl`: 22,000 model/image/arm slots, 500 reference rows and 5,000 derived Luna/DeepSeek mean rows. Missing observations remain explicit.
- `analysis.json`, `cost-per-model.csv` and the final report: coverage, assistant-reference agreement, paired stats comparisons, latency, cost and diagnostic results.
- `gallery/index.html`: all 4,000 images with their recorded scores, latency and price. Four PNG pages show a fixed selection of 72 benchmark images spanning reference technical scores.

Reported per-image prices for Reference and Q-ReAlign Lite are $0.00 by request. Actual Runpod infrastructure usage is accounted separately. Reference assessment has no HTTP latency; measured visual-review duration is a separate field. The Luna/DeepSeek mean has no independent request or charge.

Bulk image views and raw execution journals live under `/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/`. The interactive gallery links to those exact local views. File hashes, configuration snapshots and raw receipts support reproducibility; credentials are never included in experiment artifacts.
