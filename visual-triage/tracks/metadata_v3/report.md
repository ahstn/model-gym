# Decider: native metadata and controls

On 160 held-out images, selected image-only Decider reached 76.9% accuracy. All context reached 78.1%. AI detections were 43/80 for image-only and 45/80 for all context.

C2PA coverage is a separate limit: 20/80 AI images had an AI claim with a valid signature and image binding; 60/80 lacked such a claim. Missing credentials do not prove camera origin. Native metadata can also identify source datasets, which limits how well these results can transfer.

See the [methods](README.md#frozen-experiment) and [paper review](../../research/metadata-2026-10-12/paper-and-design.md). Source notes cover [camera images](../../research/metadata-2026-10-12/real-sources.md), [generated images](../../research/metadata-2026-10-12/ai-sources.md) and [content credentials](../../research/metadata-2026-10-12/c2pa-sources.md).

<details>
<summary>Source coverage and label shortcuts</summary>

| Split | Class | Images | EXIF present | Numeric camera fields | Groups |
| --- | --- | ---: | ---: | ---: | ---: |
| dev | real | 40 | 40 | 40 | 5 |
| dev | generated | 40 | 10 | 0 | 5 |
| test | real | 80 | 80 | 80 | 10 |
| test | generated | 80 | 20 | 0 | 10 |
| train | real | 40 | 40 | 40 | 5 |
| train | generated | 40 | 10 | 0 | 5 |

On development data, a rule based only on EXIF presence scored 87.5% accuracy. A rule based on numeric camera settings scored 100.0%. These scores show how metadata availability can identify the source dataset.

Across all 160 native test images, 20 had manifests and 20 had explicit AI claims; 20 claims passed signature and binding checks. The fixed trust check marked 0 trusted and 20 untrusted. Signature validity and signer trust are separate checks; neither guarantees true content.

</details>

## Independently selected GEPA pipelines

| Arm | Accuracy (%) | AI recall (%) | Real FPR (%) | Brier (0–1) | Cost USD | p50 s | p95 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| image_only | 76.9% | 53.8% | 0.0% | 0.1595 | 0.006410 | 0.738 | 1.136 |
| resolution | 69.4% | 38.8% | 0.0% | 0.1898 | 0.008283 | 0.755 | 1.071 |
| camera | 72.5% | 45.0% | 0.0% | 0.1807 | 0.007938 | 0.728 | 0.987 |
| tone | 75.0% | 50.0% | 0.0% | 0.1597 | 0.010490 | 0.753 | 1.113 |
| texture | 73.1% | 46.2% | 0.0% | 0.1823 | 0.008643 | 0.754 | 1.062 |
| provenance | 76.9% | 53.8% | 0.0% | 0.1502 | 0.008099 | 0.735 | 1.069 |
| all_context | 78.1% | 56.2% | 0.0% | 0.1418 | 0.012371 | 0.738 | 1.070 |

Each arm returned 160/160 valid outputs. Review or failure rates range from 23.1% to 29.4%; per-arm values are in JSON.

<details>
<summary>Common prompt and paired comparisons</summary>

## Common prompt: fixed feature effect

| Arm | Accuracy (%) | AI recall (%) | Real FPR (%) | Brier (0–1) | Cost USD | p50 s | p95 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| image_only | 65.6% | 31.2% | 0.0% | 0.2391 | 0.005315 | 0.744 | 0.998 |
| resolution | 64.4% | 28.7% | 0.0% | 0.2441 | 0.006939 | 0.732 | 1.010 |
| camera | 65.6% | 31.2% | 0.0% | 0.2408 | 0.006863 | 0.728 | 1.002 |
| tone | 64.4% | 28.7% | 0.0% | 0.2494 | 0.008358 | 0.732 | 1.049 |
| texture | 64.4% | 28.7% | 0.0% | 0.2629 | 0.007530 | 0.746 | 1.100 |
| provenance | 68.1% | 36.2% | 0.0% | 0.2043 | 0.006928 | 0.728 | 1.141 |
| all_context | 74.4% | 48.8% | 0.0% | 0.1716 | 0.012391 | 0.723 | 1.037 |

Each arm returned 160/160 valid outputs. Review or failure rates range from 10.6% to 25.0%; per-arm values are in JSON.

Accuracy and AI recall count invalid or missing outputs as failures. Brier assigns these outputs loss 1; lower scores are better. Real FPR uses valid outputs only. Cost cells show the known subtotal and the number of unknown charges. Latency uses recorded request durations. JSON retains balanced accuracy, AUROC and failure rates.

## Paired reward changes

| Arm | Fixed feature effect vs image only | Selected pipeline vs image only | GEPA effect within arm |
| --- | --- | --- | --- |
| image_only | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0796 [0.0658, 0.0940] |
| resolution | -0.0049 [-0.0089, -0.0013] | -0.0303 [-0.0360, -0.0244] | 0.0543 [0.0441, 0.0671] |
| camera | -0.0017 [-0.0090, 0.0051] | -0.0212 [-0.0291, -0.0144] | 0.0601 [0.0475, 0.0770] |
| tone | -0.0103 [-0.0160, -0.0049] | -0.0002 [-0.0048, 0.0038] | 0.0897 [0.0780, 0.1035] |
| texture | -0.0237 [-0.0333, -0.0174] | -0.0228 [-0.0308, -0.0166] | 0.0806 [0.0687, 0.0948] |
| provenance | 0.0348 [0.0267, 0.0420] | 0.0093 [-0.0047, 0.0223] | 0.0541 [0.0435, 0.0692] |
| all_context | 0.0675 [0.0588, 0.0744] | 0.0177 [0.0067, 0.0272] | 0.0299 [0.0175, 0.0431] |

Values are changes in 1−Brier with 95% intervals from source-group resampling, with equal class weights. Selected-pipeline effects include prompt search. Intervals are descriptive, with no correction for multiple comparisons. Proposal counts are in the restored JSON; a positive estimate alone does not establish an improvement.

</details>

## Separate controls

Removing C2PA changed AI detections on the same images (provenance: 15/20 to 3/20; all_context: 10/20 to 4/20). These controls show how the model used credentials. They contain only generated images, so they cannot measure false positives on real photos.

| Condition | Arm | Native / control Brier | Paired ΔBrier (95% CI) | Valid / intended | Integrity passed / asserted |
| --- | --- | --- | --- | --- | --- |
| all_stripped | all_context | 0.1418 / 0.1652 | 0.0234 [0.0164, 0.0307] | 160 / 160 | 0 / 0 |
| c2pa_invalid_signature | provenance | 0.1680 / 0.5617 | 0.3937 [0.3217, 0.4536] | 20 / 20 | 20 / 20 |
| c2pa_invalid_signature | all_context | 0.2685 / 0.5126 | 0.2440 [0.1852, 0.2959] | 20 / 20 | 20 / 20 |
| c2pa_stripped | provenance | 0.1680 / 0.5690 | 0.4010 [0.3241, 0.4648] | 20 / 20 | 20 / 20 |
| c2pa_stripped | all_context | 0.2685 / 0.5312 | 0.2626 [0.2094, 0.3104] | 20 / 20 | 20 / 20 |
| c2pa_swapped | provenance | 0.1680 / 0.5750 | 0.4070 [0.3322, 0.4707] | 20 / 20 | 20 / 20 |
| c2pa_swapped | all_context | 0.2685 / 0.5169 | 0.2484 [0.1922, 0.3009] | 20 / 20 | 20 / 20 |
| camera_stripped | camera | 0.1807 / 0.1835 | 0.0028 [0.0003, 0.0054] | 160 / 160 | 0 / 0 |
| camera_stripped | all_context | 0.1418 / 0.1436 | 0.0018 [-0.0022, 0.0054] | 160 / 160 | 0 / 0 |
| camera_swapped | camera | 0.1807 / 0.1951 | 0.0144 [0.0106, 0.0189] | 160 / 160 | 0 / 0 |
| camera_swapped | all_context | 0.1418 / 0.1492 | 0.0074 [0.0043, 0.0102] | 160 / 160 | 0 / 0 |
| resolution_stripped | resolution | 0.1898 / 0.1678 | -0.0220 [-0.0260, -0.0177] | 160 / 160 | 0 / 0 |
| resolution_stripped | all_context | 0.1418 / 0.1402 | -0.0015 [-0.0052, 0.0026] | 160 / 160 | 0 / 0 |

Controls use the same native images, including failed or missing predictions. Positive ΔBrier means worse performance; represented classes and source groups have equal weight. The restored JSON retains accuracy, probability shifts, decision flips and class counts. Integrity checks test the validator, not origin detection; missing or trusted claims do not guarantee an origin label.

Actual total cost across all raw trials, reflection, final requests and controls: **unknown**. Known reported cost: $1.157380; unknown-cost attempts: 26. Baseline reuse adds no calls. Do not sum costs across repeated reference rows.

Native image preparation took 32.226 seconds in total; 0 images have no recorded time. Extra preparation time for the controls was not measured.

[Per-image results](results/cases.csv) and [selected prompts](prompts/) remain in Git. [Restore the local evidence](README.md#run-and-verify) for `results/summary.json`: source slices, class counts, intervals, extraction times, selection records and hashes. Raw responses and detailed evidence remain outside Git.

<details>
<summary>Spending guard and unknown charges</summary>

Unverified allowances reserve **$1.300000**; the guard-accounted total is **$2.457380**. Allowances are not charged amounts or proven billing limits. Missing costs remain unknown, with no retries. [Restored amendment and receipt evidence](README.md#run-and-verify) records each allowance.

The current guard caps known spending at $0.65 per arm. All unknown allowances share the $5.00 global cap, which includes known costs and in-flight reservations. The global guard accounts for $2.457380, including $0.000000 in flight. The [restored amendment chain](README.md#run-and-verify) preserves earlier limits; requests and retry rules are unchanged.

</details>
