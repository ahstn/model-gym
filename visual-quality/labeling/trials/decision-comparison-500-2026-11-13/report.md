# 500-image Decisions model comparison

Frozen before new inference: 500 family-distinct images spanning all 64 EmbeddingGemma 2 clusters, comprising 384 Megalith, 97 VizWiz and 19 controls. Selection uses the first 1,000 reference-labelled examples: 375 favour high disagreement across the four existing API graders; 125 are sampled from the remaining images within each cluster. Reference values and new model outcomes were not used for selection. The final 500 confirmation references remain untouched.

This is a disagreement-enriched diagnostic sample, not an unbiased population accuracy benchmark. MAE measures agreement with assistant references, not human ground truth. Both new models receive the frozen Jev questions unchanged, including independent technical/aesthetic outcomes and the 0.50 photographic-exclusion floor. No prompt tuning or ensemble weight change was made.

| Assessor | Pairs | Mean tech | Mean aest | Tech MAE | Aest MAE | Median seconds | Recorded cost* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reference | 500 | 3.26 | 3.05 | 0.000 | 0.000 | — | $0.000000 |
| Luna (high) | 500 | 3.44 | 3.33 | 0.357 | 0.363 | 11.48 | $0.403838 |
| DeepSeek (high) | 500 | 3.58 | 3.27 | 0.482 | 0.360 | 17.82 | $0.341836 |
| Liquid d1.1 | 500 | 3.17 | 2.90 | 0.375 | 0.299 | 5.74 | $0.024922 |
| Jev 1.14 | 500 | 3.32 | 2.89 | 0.365 | 0.318 | 4.97 | $0.090035 |
| Q-ReAlign Lite | 500 | 3.73 | 2.17 | 0.593 | 1.048 | 5.07 | $0.000000 |
| Ensemble (37.5/37.5/25) | 500 | 3.43 | 3.20 | 0.364 | 0.269 | — | — |
| Perplexity Decider v1.3 (free) | 500 | 3.59 | 3.26 | 0.420 | 0.319 | 1.69 | $0.000000 |
| Gemini Decision 1 | 500 | 3.37 | 2.67 | 0.360 | 0.485 | 2.02 | $0.079106 |

*Recorded costs include known retries. Only the two new models incur new spend. Historical observations retain their old request latency and cost; these are not simultaneous hardware/throughput comparisons. Lite and Reference rows are $0.00 by the accepted accounting convention. The ensemble has no independent request or charge.

## Common coverage

All nine rows share 500 images with both scores. Missing outcomes remain null; no model substitution or score-driven retry is used.

| Assessor | T MAE | T bias | T rho | A MAE | A bias | A rho |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| Luna (high) | 0.357 | 0.184 | 0.911 | 0.363 | 0.279 | 0.871 |
| DeepSeek (high) | 0.482 | 0.321 | 0.817 | 0.360 | 0.226 | 0.836 |
| Liquid d1.1 | 0.375 | -0.087 | 0.873 | 0.299 | -0.145 | 0.880 |
| Jev 1.14 | 0.365 | 0.060 | 0.864 | 0.318 | -0.159 | 0.875 |
| Q-ReAlign Lite | 0.593 | 0.475 | 0.821 | 1.048 | -0.876 | 0.676 |
| Ensemble (37.5/37.5/25) | 0.364 | 0.168 | 0.925 | 0.269 | 0.153 | 0.920 |
| Perplexity Decider v1.3 (free) | 0.420 | 0.329 | 0.874 | 0.319 | 0.209 | 0.869 |
| Gemini Decision 1 | 0.360 | 0.108 | 0.879 | 0.485 | -0.381 | 0.871 |

## High Disagreement

| Assessor | T MAE | A MAE |
| --- | --- | --- |
| Reference | 0.000 | 0.000 |
| Luna (high) | 0.373 | 0.392 |
| DeepSeek (high) | 0.527 | 0.398 |
| Liquid d1.1 | 0.390 | 0.309 |
| Jev 1.14 | 0.389 | 0.337 |
| Q-ReAlign Lite | 0.592 | 1.055 |
| Ensemble (37.5/37.5/25) | 0.386 | 0.283 |
| Perplexity Decider v1.3 (free) | 0.444 | 0.321 |
| Gemini Decision 1 | 0.378 | 0.486 |

## Diversity Sample

| Assessor | T MAE | A MAE |
| --- | --- | --- |
| Reference | 0.000 | 0.000 |
| Luna (high) | 0.308 | 0.275 |
| DeepSeek (high) | 0.348 | 0.246 |
| Liquid d1.1 | 0.328 | 0.269 |
| Jev 1.14 | 0.293 | 0.261 |
| Q-ReAlign Lite | 0.593 | 1.026 |
| Ensemble (37.5/37.5/25) | 0.296 | 0.229 |
| Perplexity Decider v1.3 (free) | 0.346 | 0.314 |
| Gemini Decision 1 | 0.307 | 0.483 |

## Photography eligibility

| Assessor | False floors / eligible | Missed exclusions / excluded |
| --- | --- | --- |
| Luna (high) | 1/461 | 4/34 |
| DeepSeek (high) | 1/461 | 2/34 |
| Liquid d1.1 | 2/461 | 4/34 |
| Jev 1.14 | 0/461 | 11/34 |
| Perplexity Decider v1.3 (free) | 0/461 | 10/34 |
| Gemini Decision 1 | 0/461 | 7/34 |

[Per-image tables and exact images](../../../data/decision-comparison-500-2026-11-13/gallery/index.html) · [Compact score CSV](../../../data/decision-comparison-500-2026-11-13/exports/scores.csv) · [Image-level dataset](../../../data/decision-comparison-500-2026-11-13/exports/labelled-images.jsonl). Raw responses and every request attempt remain under `results/`. Image statistics remain separate metadata and were not sent to the models.

## Total costs per model

| Assessor | Known USD | Unpriced attempts | New requests | Basis |
| --- | --- | --- | --- | --- |
| Reference | 0.000000 | 0 | 0 | retained historical receipts; no new spend |
| Luna (high) | 0.403838 | 0 | 0 | retained historical receipts; no new spend |
| DeepSeek (high) | 0.341836 | 4 | 0 | retained historical receipts; no new spend |
| Liquid d1.1 | 0.024922 | 0 | 0 | retained historical receipts; no new spend |
| Jev 1.14 | 0.090035 | 84 | 0 | retained historical receipts; no new spend |
| Q-ReAlign Lite | 0.000000 | 0 | 0 | retained historical receipts; no new spend |
| Ensemble (37.5/37.5/25) | — | 0 | 0 | derived; no independent charge |
| Perplexity Decider v1.3 (free) | 0.000000 | 0 | 500 | new run; every attempt |
| Gemini Decision 1 | 0.079106 | 0 | 500 | new run; every attempt |
