# Jev 1.14: quality and AI detection

**Partial: 118/118 primary results are valid.** The primary comparison uses the same image bytes and prompts as the earlier benchmark. Two panoramas require separate compatibility views; their results are excluded from the primary scores.

The original parser rejected 7 responses. A fixed rounding rule recovered 7 of them and was applied to all 125 responses. It preserves native scores and probabilities; no values were renormalised. The raw results and original parser errors remain intact.

Invalid results were retained: 1 × ValueError: HTTP or model validation failed. Failed or missing results are never treated as negative AI detections.

## Quality and aesthetics

| Model | Valid / planned | Quality MAE | Aesthetics MAE | Human MOS rank correlation | Cost USD | Latency p50 / p95 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.14 | 58/58 | 0.493 | 0.224 | 0.487 | 0.003007 | 0.514 / 0.701 |
| Decider v1.1 | 58/58 | 0.209 | 0.204 | 0.687 | 0.001162 | 0.608 / 1.015 |
| GPT-6 Luna | 58/58 | 0.316 | 0.240 | 0.349 | 0.007224 | 1.834 / 3.180 |
| Gemini 3.8 Flash | 58/58 | 0.584 | 0.447 | 0.612 | 0.089482 | 3.227 / 38.955 |

Lower MAE is better. Score metrics use valid outputs. These errors measure agreement with the saved assistant assessments, not human ground truth. Rank correlation uses human quality MOS. Human aesthetic ratings are absent. UHD-IQA has no low-quality examples below 2.6/5 in this panel, and its human ratings used 3,840-pixel views. This test does not establish recall for rejecting poor images.

## AI detection

| Model | Valid / planned | Accuracy | AI detected | Real falsely flagged | Brier | AUROC | Cost USD | Latency p50 / p95 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.14 | 60/60 | 93.3% | 29/30 | 3/30 | 0.100 | 0.990 | 0.003171 | 0.527 / 0.718 |
| Decider v1.1 | 60/60 | 76.7% | 16/30 | 0/30 | 0.159 | 0.982 | 0.001484 | 0.648 / 0.998 |
| GPT-6 Luna | 60/60 | 83.3% | 20/30 | 0/30 | 0.130 | 0.934 | 0.009377 | 3.014 / 4.747 |
| Gemini 3.8 Flash | 60/60 | 85.0% | 21/30 | 0/30 | 0.135 | 0.902 | 0.094588 | 3.263 / 9.328 |

AI labels come from source provenance. The fixed threshold is 0.5. Accuracy counts invalid or missing outputs as wrong; detection counts, Brier and AUROC use valid outputs. The panel has 30 real photos and 30 generated images, including 12 graphics or renders. This source and style mix limits claims about general photo detection.

The JSON report also compares each baseline on the exact subset with valid Jev results. No prompt, threshold or image allocation was tuned on these results.

## Separate compatibility and size checks

| Phase | Image | Size | Quality / aesthetics | Input tokens | Cost USD | Latency s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| size_probe | sample-024 | 320×320 | 2.350 / 1.500 | 1035 | 0.000043 | 1.001 |
| size_probe | sample-024 | 512×512 | 2.490 / 1.580 | 1469 | 0.000062 | 1.022 |
| size_probe | sample-024 | 1024×1024 | 2.530 / 1.640 | 3483 | 0.000146 | 1.033 |
| size_probe | sample-025 | 320×320 | 3.320 / 3.290 | 1036 | 0.000044 | 0.635 |
| size_probe | sample-025 | 512×512 | 3.440 / 3.440 | 1466 | 0.000062 | 0.739 |
| size_probe | sample-025 | 1024×1024 | 3.370 / 3.410 | 3483 | 0.000146 | 1.441 |
| compatibility | quality-000 | 512×256 | — / — | — | — | 0.421 |
| compatibility | quality-019 | 512×256 | — / — | — | — | — |

Compatibility views add lossless padding to meet the minimum image dimension. They are not the original benchmark views. Size probes use two development images at three resolutions; they show observed token, cost and latency changes, not an accuracy result or an optimal resolution. The 1,024-pixel probes upscale the decoded 512-pixel source and add no source detail. Reported input tokens include text, questions and image processing; request bytes are retained separately.

All 125 attempts have a known cost subtotal of $0.006681. Unknown-cost attempts: 1. Unknown does not mean free. Costs include failed calls when charges are available. BYOK upstream inference is included.

Comparator calls were made on 11 October 2026; Jev was tested later. Latency includes network time and is not a simultaneous serving comparison. Image-token accounting may differ by provider. The prompts were selected for Decider and reused without Jev optimisation.

The [OpenRouter model catalogue](https://openrouter.ai/typesafe/jev-1.14) lists a 262,144-token context, $0.042 per million input tokens and no output-token charge. Measured costs above use response usage. Both experimental headers were sent; forwarding by OpenRouter was not independently confirmed. See [multimodal Decisions](https://openrouter.ai/docs/guides/community/multimodal-decisions) and [TypeSafe image constraints](https://docs.typesafe.ai/images).
The largest primary request used 1,420 reported input tokens, or 0.54% of the listed context window.


[Per-image results](cases.csv) and [machine-readable summary](summary.json) retain the exact joins, coverage, error counts, hashes and paired-subset comparisons. Raw images and responses remain outside Git.
