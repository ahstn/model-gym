# Two new Decisions models on 500 varied images

Requested 13 November 2026: `perplexity/pplx-decider-v1.3:free` and `google/gemini-decision-1`, compared with the existing Luna, DeepSeek, Liquid, Jev, Q-ReAlign Lite, accepted ensemble and assistant reference observations. The canonical scoring configuration is unchanged.

## Completed results

Both new models returned 500/500 valid technical/aesthetic pairs, with no retries. All 500 image hashes and 1,000 new responses passed the independent audit; all 3,500 prior model, ensemble and reference observations are unchanged. The targeted offline checks passed 87 tests.

The [comparison report](report.md) contains the full table and separate sampling-arm results; [summary.json](summary.json) retains aggregate metrics. The [local gallery](../../../data/decision-comparison-500-2026-11-13/gallery/index.html) shows all 500 images with nine Tech/Aest/Seconds/Price rows each. Full labelled data and request receipts remain local and ignored.

- Perplexity: technical/aesthetic MAE **0.420 / 0.319**, median HTTP latency **1.69 seconds**, new cost **$0.00**. Technical scores average **0.329** above the reference. It is a fast, free comparison, but does not improve on the accepted ensemble in either outcome here.
- Gemini Decision 1: MAE **0.360 / 0.485**, median **2.02 seconds**, new cost **$0.079106**. Technical agreement is close to Luna (**0.357**) and the ensemble (**0.364**), without evidence here that these small differences establish a population ranking. Aesthetic scores average **0.381** below the reference, with worse agreement in both sampling arms.
- The accepted ensemble retains the lowest aesthetic MAE (**0.269**). Keep its weights unchanged on this evidence. Neither candidate falsely floors an eligible reference in this subset, but Perplexity misses 10/34 exclusions and Gemini misses 7/34; five uncertain references are outside those class denominators.

Only the two new models were called. Historical cost totals contain four unpriced DeepSeek attempts and 84 unpriced Jev attempts, and historical latency reflects the original run conditions. These are not new charges or directly controlled speed comparisons. Assistant-reference agreement and the selection bias limit conclusions about general accuracy.

## Frozen selection and requests

The sampling pool is the first 1,000 published reference-labelled images from the 10K run. This supplies existing independent references and leaves the separate 500 confirmation cases untouched. Sampling uses the existing EmbeddingGemma 2 content clusters; no new embedding inference is needed.

Allocate 500 slots across the 64 clusters in proportion to the square root of cluster size, capped by available images. Within cluster quotas, allocate exactly 375 slots to the highest mean technical/aesthetic population variance across Luna, DeepSeek, Liquid and Jev. Sample the remaining 125 deterministically from the rest of each cluster. Lite is excluded from this selection statistic because its systematic aesthetic offset would dominate disagreement. Reference values and new-model results are excluded from selection.

The frozen selection contains 384 Megalith, 97 VizWiz and 19 control images, all from distinct source families. Existing model variance averages 0.103853, compared with 0.071354 in the pool, a 45.54% increase. This is a diagnostic sample enriched for disagreement, not an unbiased population benchmark or a fresh blinded reference study.

Both new models receive the exact common text and Jev questions from `labeling/prompts/decision.json`: two five-level scores and the yes/no photographic-exclusion question. Native 0–4 expected values map to 1–5 by adding one. The accepted exclusion threshold of 0.50 floors both outcomes to 1.00. Statistics, references and other models' labels are never sent. Native precision is retained; tables display two decimals.

There is one selected image in each preflight call; successful preflight observations are reused in the 500 total. The runner pins the plan, prompt source, source files, exact delivered bytes and request hashes. It retains every attempt, final response, returned model identity, HTTP end-to-end latency and reported cost. At most three attempts are allowed for transport errors or HTTP 429/500/502/503/504. Valid or schema-invalid assessments are not retried for different scores. The runner stops admission for authentication, budget and unavailable-route errors. Each model has a $5 local accounting guard with $0.03 reserved per in-flight or unpriced request; this is not a provider-enforced cap.

Provider discovery on 13 November:

- [Perplexity model page](https://openrouter.ai/perplexity/pplx-decider-v1.3%3Afree): image-capable Decisions model, free route; catalogue canonical ID `perplexity/pplx-decider-v1.3-20261112`.
- [Gemini model page](https://openrouter.ai/google/gemini-decision-1): image-capable Decisions model, $0.10/M input tokens and $0/M output tokens; catalogue canonical ID `google/gemini-decision-1-20261111`.
- [Multimodal Decisions contract](https://openrouter.ai/docs/guides/community/multimodal-decisions): use `POST /api/alpha/decisions`; enumerate these models using `GET /api/v1/models?output_modalities=decisions&input_modalities=image`.

The two successful preflights returned those exact canonical IDs from Perplexity and Google AI Studio respectively. Account discovery showed 1,000 free-model requests remaining before the trial. Prices and limits are dated evidence, not promises for future runs. Actual receipts govern the report.

## Local artifacts and reproduction

Everything containing full per-image output stays in ignored `visual-quality/data/decision-comparison-500-2026-11-13/`:

- `plan.json`, `subset.json`, `selection-summary.json`: frozen protocol, original labels/statistics, sampling decisions and hashes.
- `images/`: 500 exact delivered PNGs, locally hard-linked to verified existing corpus files. Do not edit them in place.
- `results/{perplexity,gemini}/`: request journals, full responses and terminal observations.
- `exports/labelled-images.jsonl`, `scores-long.jsonl`, `scores.csv`: all nine assessors and separate image statistics.
- `report.md`, `report.json`, `gallery/index.html`: summary, metrics and per-image Tech/Aest/Seconds/Price tables.

From the repository root, using the visual-quality environment with its `corpus` dependency group:

```sh
PYTHONPATH=visual-quality/src visual-quality/.venv/bin/python -m pytest -q visual-quality/labeling/trials/decision-comparison-500-2026-11-13/test_trial.py
PYTHONPATH=visual-quality/src visual-quality/.venv/bin/python visual-quality/labeling/trials/decision-comparison-500-2026-11-13/report.py
```

`prepare.py` refuses to replace an existing frozen plan. For explicitly authorized inference, source credentials quietly and use `run.py perplexity` or `run.py gemini`; a route lock and request identity checks prevent duplicate valid calls. Do not alter frozen source during an active run. Existing raw results remain authoritative for offline reports; rerendering needs no model calls. Keep the 10K prompts, labels, reference scores and 37.5/37.5/25 weights unchanged.
