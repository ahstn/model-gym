# Jev 1.14 comparison, 18 October 2026

**Complete: 378/378 valid results.** Jev used the same frozen images and prompts as the prior runs. Reported inference cost is $0.030727; 0 attempts have unknown cost. Missing and invalid results remain in the planned denominator.

The shared panel compares all three requested models. The latest panel gives stronger human quality evidence, but has no Luna baseline. These panels use different images and prompts, so their scores must remain separate.

**Shared quality panel: 58 images**

| Model | Valid | Human MOS MAE | Human rank correlation | Assistant quality / aesthetics MAE | Cost USD | Latency p50 / p95 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.14, 18 Oct | 58/58 | 0.325 | 0.467 | 0.492 / 0.218 | 0.003008 | 0.479 / 0.730 |
| Perplexity Decider v1.1 | 58/58 | 0.294 | 0.687 | 0.209 / 0.204 | 0.001162 | 0.608 / 1.015 |
| GPT-6 Luna, low | 58/58 | 0.354 | 0.349 | 0.316 / 0.240 | 0.007224 | 1.834 / 3.180 |

Lower mean absolute error (MAE) is better. Quality uses a 1–5 scale. Human MOS is the mean opinion score. UHD-IQA MOS is mapped from 0.01–1 to 1–5 with 1 + 4 × (MOS − 0.01) / 0.99; this scale conversion does not calibrate it to our technical-quality rubric. The assistant assessments are weak references, not human labels. This UHD-IQA panel lacks examples below 2.6/5, and its human ratings used larger image views. Human aesthetic labels are absent. Two narrow panoramas are excluded because the original Jev image limits reject them.

**Shared AI-origin panel: 60 images**

| Model | Valid | Accuracy | AI detected | Real falsely flagged | Brier | AUROC | Cost USD | Latency p50 / p95 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.14, 18 Oct | 60/60 | 95.0% | 30/30 | 3/30 | 0.101 | 0.989 | 0.003171 | 0.542 / 0.754 |
| Perplexity Decider v1.1 | 60/60 | 76.7% | 16/30 | 0/30 | 0.159 | 0.982 | 0.001484 | 0.648 / 0.998 |
| GPT-6 Luna, low | 60/60 | 83.3% | 20/30 | 0/30 | 0.130 | 0.934 | 0.009377 | 3.014 / 4.747 |

The fixed threshold is 0.5. Labels come from source provenance. The set includes 30 real photographs and 30 generated images; 12 generated images are graphics or renders. Source and style differences limit general claims about photo detection.

**Latest quality panel: 100 KonIQ photographs**

| Model | Valid | Human MOS MAE | Rank correlation | Low quality detected | False flags | Cost USD | Latency p50 / p95 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.14, 18 Oct | 100/100 | 0.238 | 0.943 | 42/50 | 2/50 | 0.013460 | 0.780 / 1.168 |
| Perplexity Decider v1.1 | 100/100 | 0.443 | 0.948 | 34/50 | 0/50 | 0.005706 | 1.054 / 1.391 |

Low quality means a score below 2.6. The set contains 50 low-quality and 50 other photographs. This is the direct human-label check for a quality scorer; it does not validate aesthetic scoring.

**Latest AI-origin panel: 160 images**

| Model | Valid | Accuracy | AI detected | Real falsely flagged | Brier | AUROC | Cost USD | Latency p50 / p95 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.14, 18 Oct | 160/160 | 88.1% | 64/80 | 3/80 | 0.124 | 0.948 | 0.011088 | 0.586 / 0.888 |
| Perplexity Decider v1.1 | 160/160 | 76.9% | 43/80 | 0/80 | 0.160 | 0.947 | 0.006410 | 0.738 / 1.136 |

This panel contains 80 VISION camera photographs and 20 images each from DALL-E 2, DALL-E 3, Firefly and Midjourney v5. Only image pixels were supplied. Accuracy counts missing or invalid outputs as wrong; score metrics use valid outputs.

**Fixed review gates**

| Panel | Model | Accept real / AI | Review real / AI | Reject real / AI | Invalid real / AI |
| --- | --- | ---: | ---: | ---: | ---: |
| shared | Jev 1.14, 18 Oct | 0 / 0 | 30 / 14 | 0 / 16 | 0 / 0 |
| shared | Perplexity Decider v1.1 | 29 / 8 | 1 / 10 | 0 / 12 | 0 / 0 |
| shared | GPT-6 Luna, low | 30 / 6 | 0 / 12 | 0 / 12 | 0 / 0 |
| latest | Jev 1.14, 18 Oct | 5 / 0 | 75 / 58 | 0 / 22 | 0 / 0 |
| latest | Perplexity Decider v1.1 | 76 / 14 | 4 / 43 | 0 / 23 | 0 / 0 |

The existing gates are p ≤ 0.2 for accept, 0.2 < p < 0.8 for review, and p ≥ 0.8 for reject. These are descriptive benchmark gates. The run does not authorize deletion or automatic rejection.

**Uncertainty and comparison limits**

- Latest quality reward (1 − absolute error / 4): Jev minus Decider = 0.0511; paired source-group bootstrap 95% interval [0.0388, 0.0631], 2000 replicates. Higher reward is better.
- Latest AI reward (1 − squared error): Jev minus Decider = 0.0357; paired source-group bootstrap 95% interval [-0.0262, 0.1026], 2000 replicates. Higher reward is better.

The JSON summary retains Wilson intervals for shared AI rates and group bootstrap intervals for the latest panels. These intervals do not cover source bias. The shared comparator calls ran on 11 October 2026; latest Decider calls ran on 11–12 October. Jev ran on 18 October. Latency includes network time and is a historical comparison, not a simultaneous serving test. Prompts were selected for Decider and transferred without Jev tuning.

Historical Luna used `openai/gpt-6-luna` through chat completions, low reasoning effort and a 4,096-token output cap. It is not a test of a current Luna Decisions endpoint. Decider resolved to `perplexity/pplx-decider-v1.1-27b-20261006`. Luna reported no revision suffix. Its nonzero BYOK upstream cost is included even where the OpenRouter charge was zero. Costs are provider-reported inference costs, not invoice reconciliation.

[JSON summary](summary.json) contains hashes, coverage, failures, model settings and detailed metrics. [Per-image comparison](cases.csv) retains all planned cases. Raw responses remain in the ignored data directory. Image bytes, prompts and prior evidence were preserved.
