# Jev 1.14: latest quality and AI tests

Jev's quality error was 47% lower. It detected 67 generated images versus Decider's 43, with 4 real photos falsely flagged versus 0. Quality false flags were 2 versus 0. Ranking scores remained close: quality Spearman 0.944 / 0.948; AI AUROC 0.952 / 0.947 (Jev / Decider).

**Complete: 260/260 valid results.** This is the main comparison: 100 KonIQ quality images and 160 AI-origin images. Jev and Decider used the same frozen pixels and selected prompts. No Jev prompt tuning was performed.

## Image quality

| Model | Valid / planned | Human MOS MAE | Rank correlation | Low quality detected | False flags | Cost USD | Latency p50 / p95 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.14 | 100/100 | 0.236 | 0.944 | 42/50 | 2/50 | 0.013460 | 0.792 / 1.181 |
| Decider v1.1, selected prompt | 100/100 | 0.443 | 0.948 | 34/50 | 0/50 | 0.005706 | 1.054 / 1.391 |

MAE compares technical quality with human ratings on a 1–5 scale; lower is better. Low quality means a score below 2.6. The panel contains 50 low-quality and 50 other photographs. Aesthetic scores are saved per image, but no human aesthetic labels are available.

## AI detection

| Model | Valid / planned | Accuracy | AI detected | Real falsely flagged | Brier | AUROC | Cost USD | Latency p50 / p95 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.14 | 160/160 | 89.4% | 67/80 | 4/80 | 0.123 | 0.952 | 0.011088 | 0.563 / 0.796 |
| Decider v1.1, selected prompt | 160/160 | 76.9% | 43/80 | 0/80 | 0.160 | 0.947 | 0.006410 | 0.738 / 1.136 |

The fixed AI threshold is 0.5. Labels come from source provenance: 80 VISION camera photos and 20 images each from DALL-E 2, DALL-E 3, Firefly and Midjourney v5. Both models receive image pixels only, with no EXIF or C2PA text. Accuracy counts missing or invalid results as wrong; the other score metrics use valid results.

AI detections by source (Jev vs Decider): DALL-E 2 14/20 vs 5/20; DALL-E 3 20/20 vs 20/20; Firefly 15/20 vs 5/20; Midjourney v5 18/20 vs 13/20.

| Existing probability gate | Jev: real / AI | Decider: real / AI |
| --- | ---: | ---: |
| Accept, p ≤ 0.2 | 7 / 0 | 76 / 14 |
| Review, 0.2 < p < 0.8 | 73 / 57 | 4 / 43 |
| Reject, p ≥ 0.8 | 0 / 23 | 0 / 23 |

Jev sends more images to review under the existing gates. Its higher accuracy at 0.5 does not provide greater automatic coverage. These results do not validate automatic rejection.

## Image size and context

| Panel | Image size | Input tokens: min / median / max | Maximum context used |
| --- | --- | ---: | ---: |
| quality | 1024×768 JPEG | 3200 / 3205 / 3208 | 1.22% |
| ai | 512×512 JPEG | 1644 / 1650 / 1654 | 0.63% |

The catalogue lists a 262,144-token context window. Token counts include text, questions and image processing. The panels use different prompts, so their token difference does not isolate image size. The [separate size probes](../shared-results-2026-10-15/README.md) test the same two development images at 320, 512 and 1,024 pixels. Upscaling adds no source detail.

Latest-trial cost: $0.024548 known; 0 unpriced attempts. The earlier compatibility failure has an unknown charge and is excluded from this trial total. Its budget reservation is not an actual charge.

The fixed parser allowed bounded two-decimal rounding from the start. The original strict parser would reject 10 responses. The CSV marks those recovered cases with `parser_review_applied`; `strict_parser_valid` records the original result. Native scores and probabilities are unchanged. Both experimental headers were sent; OpenRouter forwarding was not independently confirmed.

Decider was tested on 11–12 October 2026; Jev was tested later. Latency is historical, not a simultaneous serving test. Source bias and the limited generator set restrict wider conclusions. The [older Luna/Gemini panel](../shared-results-2026-10-15/README.md) uses different images and is a separate comparison.

[Per-image results](cases.csv) and [JSON summary](summary.json) contain failure counts, source-group intervals, paired comparisons and input hashes. Raw images and responses remain outside Git. See [OpenRouter multimodal Decisions](https://openrouter.ai/docs/guides/community/multimodal-decisions), [TypeSafe image limits](https://docs.typesafe.ai/images) and [Jev pricing](https://openrouter.ai/typesafe/jev-1.14).
