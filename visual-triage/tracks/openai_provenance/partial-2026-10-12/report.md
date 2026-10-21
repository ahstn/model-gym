# OpenAI provenance: native benchmark check

**Partial: valid results for 15/160 images.** Requests: 17, including 2 failed attempts and 0 retries. Recovered cases: 0; unresolved failures: 2; unattempted: 143.

Request failures: HTTP 429: 1, HTTP 502: 1. The server's error message asked for a 2932.782-second wait (48.9 minutes). This records its response, not a live countdown.

OpenAI signals were detected in 0/4 successfully checked older DALL-E images (40 planned). Missing signals remain unresolved. They do not establish camera origin or provide general AI-detection accuracy.

| Source | Valid / tried / planned | Failed cases | Untried | OpenAI signals | C2PA states |
| --- | ---: | ---: | ---: | ---: | --- |
| real | 7 / 8 / 80 | 1 | 72 | 0 | not_present: 7 |
| dalle2 | 2 / 2 / 20 | 0 | 18 | 0 | not_present: 2 |
| dalle3 | 2 / 3 / 20 | 1 | 17 | 0 | not_present: 2 |
| firefly | 2 / 2 / 20 | 0 | 18 | 0 | valid: 2 |
| midjourney-v5 | 2 / 2 / 20 | 0 | 18 | 0 | not_present: 2 |

False OpenAI attributions among successful checks: 0/7 real photos and 0/4 images from other generators. C2PA validity and OpenAI attribution are separate: a valid Adobe manifest can return `not_detected` for an OpenAI signal.

Successful request latency: median 2.347 s, P95 7.171 s. API cost is unknown; 17 attempts have no charge receipt. Unknown does not mean free.

## Comparison with Decider

These are the same image identities. The API received native files; Decider saw 512-pixel JPEG crops and, for metadata arms, extracted text. Decider predicts broad AI origin. The API checks supported OpenAI provenance signals, so its unresolved results cannot be ranked as classifier errors.

| Decider prompt | Accuracy | AI detected | Real false flags | Brier | Cost | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| image_only | 76.9% | 43/80 | 0/80 | 0.1595 | $0.006410 | 0.738 s |
| provenance | 76.9% | 43/80 | 0/80 | 0.1502 | $0.008099 | 0.735 s |
| all_context | 78.1% | 45/80 | 0/80 | 0.1418 | $0.012371 | 0.738 s |

The table uses the full 160-image Decider reference. The API has valid results on 15 of those same images.

This small coverage sample cannot establish a ranking. The subset below contains only images with valid API results; failed and untried files remain outside it.

| Decider on valid API cases only | Correct | AI detected | Real false flags |
| --- | ---: | ---: | ---: |
| image_only | 13/15 | 6/8 | 0/7 |
| provenance | 12/15 | 5/8 | 0/7 |
| all_context | 14/15 | 7/8 | 0/7 |

### Earlier models: separate images

The initial 60-image AI panel has no image-hash overlap with this trial. It provides context, not a ranking against native provenance checks.

| Model | AI detected | Real false flags | Cost / 60 | Median |
| --- | ---: | ---: | ---: | ---: |
| Decider v1.1 | 16/30 | 0/30 | $0.001484 | 0.648 s |
| GPT-6 Luna | 20/30 | 0/30 | $0.009377 | 3.014 s |
| Gemini 3.8 Flash | 21/30 | 0/30 | $0.094588 | 3.263 s |

The AI files come from the 2023 Synthbuster release. This cohort has no current GPT-image positive controls, so it cannot measure recall on current supported output. Use provenance as an early check and retain visual review when the result is unresolved.

[API guide](https://developers.openai.com/api/docs/guides/content-provenance) · [Per-image results](results/cases.csv) · [Compact metrics](results/summary.json). Raw responses and image files remain outside Git.
