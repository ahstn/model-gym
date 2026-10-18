# Jev 1.14 confidence audit

Date: 2026-10-18. Model: `typesafe/jev-1.14-20261014`. This is a descriptive analysis of the completed run. It adds no model calls and changes no frozen results.

Jev returned signals that can send uncertain cases to review. All three false flags in the larger panel had AI probabilities close to 0.5 and an origin-uncertainty label of `high`. The existing 0.8 probability threshold already sends these cases to review. However, this threshold also reduces the number of generated images flagged from 64 to 22 out of 80.

## What the response contains

All 220 AI-origin responses contain the following fields:

| Field | Returned value | Separate native confidence |
| --- | --- | --- |
| `ai_probability` | A `noul` value from 0 to 1 | No |
| `medium` | A choice and its option probabilities | Yes |
| `uncertainty` | `low`, `medium`, or `high`, with option probabilities | Yes |

The native confidence belongs to its own question. For example, a high confidence in `uncertainty=high` means strong support for an uncertain origin. It does not mean strong confidence that the image is AI-generated. Choice confidence is derived from the option distribution; it is not a separate estimate of correctness. [TypeSafe confidence documentation](https://docs.typesafe.ai/confidence).

The `ai_probability` answer has no separate confidence rating. The optional transform `abs(2*p - 1)` measures how far its probability is from 0.5. This value was not returned in the run and would add no independent information. [TypeSafe Noul documentation](https://docs.typesafe.ai/primitives/noul).

## Three false flags in the larger panel

These are the real photos counted as AI-generated at the binary threshold of `ai_probability >= 0.5`. All three enter the existing review band of `0.2 < p < 0.8`.

| Real-photo ID | AI probability | Origin uncertainty | P(uncertainty = high) | Native confidence in that uncertainty choice |
| --- | ---: | --- | ---: | ---: |
| `vision-D21_I_nat_0041` | 0.59 | high | 0.54 | 0.31 |
| `vision-D12_I_nat_0141` | 0.58 | high | 0.68 | 0.52 |
| `vision-D04_I_nat_0005` | 0.56 | high | 0.53 | 0.30 |

The machine-readable [analysis](confidence-analysis.json) preserves the complete raw answers for these cases, including the medium probabilities and all uncertainty probabilities.

## Observed filtering tradeoffs

Each row below defines a candidate flag for review. A case removed from a flag set remains unresolved; this analysis does not reclassify it as a real photo. The 0.8 probability threshold was already in the benchmark report. The combinations with uncertainty are post-hoc checks, not validated operating rules.

| Panel | Candidate-flag rule | Generated images flagged | Real photos falsely flagged |
| --- | --- | ---: | ---: |
| Larger panel | Probability >= 0.5 | 64/80 | 3/80 |
| Larger panel | Probability >= 0.8 | 22/80 | 0/80 |
| Larger panel | Probability >= 0.5 and uncertainty is not high | 28/80 | 0/80 |
| Larger panel | Probability >= 0.5 and uncertainty is low | 21/80 | 0/80 |
| Shared panel | Probability >= 0.5 | 30/30 | 3/30 |
| Shared panel | Probability >= 0.8 | 16/30 | 0/30 |
| Shared panel | Probability >= 0.5 and uncertainty is not high | 3/30 | 0/30 |
| Shared panel | Probability >= 0.5 and uncertainty is low | 0/30 | 0/30 |

Removing high-uncertainty cases eliminates the three observed false flags in each panel, but also removes 36 of the 64 generated-image flags in the larger panel and 27 of the 30 in the shared panel. Jev reports high uncertainty for 132/160 and 57/60 images, respectively. The difference between panels shows why this filter needs a separate calibration set.

In the larger panel, the existing three-way probability gates place 22 generated images in the reject band, 5 real photos in the accept band, and 133 images in review. These are descriptive gate names from the benchmark, not permission to delete or automatically reject an image. No real photo enters the reject band in this sample. Zero false flags in 80 real photos does not establish a zero population false-positive rate.

For the pilot, preserve the probability, uncertainty label, full option distributions, and native confidence fields. Route ambiguous origin judgments to review. Validate any operating threshold on a separate set with reliable origin labels before using it for an automated decision.

## Evidence

- Raw responses: `visual-quality/data/jev14-2026-10-18/attempts.jsonl`.
- Raw-response SHA-256: `821f5deef725af4910b5ec50276a2b4b90addd2b455cb71df4a368a4430bc5d5`.
- Larger-panel labels: the frozen [manifest](manifest.json), sourced from `visual-triage/tracks/metadata_v3/dataset.json` in the recorded source worktree.
- Shared-panel labels: `visual-triage/evidence/benchmarks/ai.json` in the recorded source worktree, pinned by [protocol](protocol.json).
- Counts, complete false-positive answers, field coverage, and additional rule combinations: [confidence-analysis.json](confidence-analysis.json).

The panels are separate evaluations. Their counts must not be pooled as independent evidence. This audit does not fit thresholds, measure new calibration, or change the benchmark results.
