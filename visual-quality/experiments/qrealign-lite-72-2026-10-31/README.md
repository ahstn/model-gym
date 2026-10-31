# Q-ReAlign Lite: 72-image Runpod trial

[All per-image scores](scores.csv) · [Analysis](analysis.json) · [Prior 72-image gallery](../photo-scoring-2026-10-31/gallery.html)

Q-ReAlign Lite 4B was evaluated on the same 72 original delivered views on a Runpod 24 GB RTX 4090. Checkpoint `q-future/Q-ReAlign-Lite-4B` revision `06add46a6bc4a7dfcf0a59a2fb1ed15a6e3d30b7`; raw responses, level probabilities, input token counts, latency and image SHA-256 are retained under `results/`. Image processor uses model defaults; no external resizing was applied. The GPU model's native five-word weighted 0–1 score is mapped to 1–5 by `1+4*score`. Scores are raw: no eligibility classifier or floor is applied.

The table compares the 48 previously held-out images on common valid cases, against prior assistant references. Error is mean absolute error on the 1–5 scale; positive bias means higher than the reference. These are not human gold and the sample is small. Additional results for all 71 fresh images and 68 reference-eligible images are in `analysis.json`.

| Model | Tech MAE | Aesthetic MAE | Mean MAE | Tech bias |
|---|---:|---:|---:|---:|
| liquid | 0.370 | 0.265 | 0.318 | +0.091 |
| jev | 0.389 | 0.247 | 0.318 | +0.238 |
| haiku | 0.394 | 0.306 | 0.350 | +0.077 |
| average | 0.469 | 0.278 | 0.373 | +0.329 |
| luna | 0.421 | 0.331 | 0.376 | +0.321 |
| deepseek | 0.533 | 0.308 | 0.421 | +0.338 |
| lite | 0.556 | 1.029 | 0.792 | +0.496 |

On 48 paired held-out cases, Lite's combined error minus the Luna/DeepSeek mean is +0.419; a 5,000-resample paired bootstrap interval is [+0.306, +0.528]. This measures sampling variability conditional on the fixed assistant references and selected images; it is not human accuracy uncertainty.

Known poster T2000 receives native technical 4.01 and aesthetics 2.92, while the agreed photography-eligibility reference is 1.00 / 1.00. The model has no eligibility gate; a separate validated classifier is required for the policy. This control is excluded from the fresh and held-out metrics, and the separate uncertain boundary cases remain in the held-out results as in the prior experiment.

On the six photos previously assessed by the hosted Mini model, Lite's technical/aesthetic MAE is 0.423 / 1.002; Mini's is 0.898 / 0.480 using the previously published two-decimal scores. The Mini Space applied a 768-pixel cap, while Lite used its default processor on the supplied views. This tiny targeted subset and differing preprocessing cannot establish a general model difference. [Mini comparison](../qalign-inference-2026-10-31/comparison.md).

Metrics use unrounded Lite outputs; the earlier models and reference values are read from the consolidated two-decimal CSV. The gallery and CSV display two decimals. The earlier Luna/DeepSeek/Haiku/Jev/Liquid assessments and references were not rerun. Their prompts, eligibility handling, reasoning and input processors differ from this scalar Q-ReAlign model; this is a useful task comparison, not a controlled architecture-only ablation. Assistant references remain subjective and no labels are approved for training.

The two question phrasings and their answer stems are present in the author's [task prompt library](https://github.com/Q-Future/Q-ReAlign/blob/main/qalign/prompts.py). The scorer follows the model card's direct Transformers example; this was not a separate run of the author's ms-swift evaluation pipeline. The five space-prefixed level labels were each verified as a distinct single token before the server started.

## Latency

Each request processed one image and both dimensions sequentially. End-to-end time includes upload and proxy/network overhead. Server processing starts after image decode and includes two independent preprocessing and forward passes. Startup and checkpoint download are excluded. The first request is included; the run used one request at a time. These are experimental single-stream timings, not a production throughput benchmark.

The initial cold startup took approximately 56.7 seconds from container creation to the model-ready log, including dependency installation and checkpoint download; it is recorded separately in `startup.json` and was not part of the per-image timing.

| Timing | Median | P95 (nearest rank) | Maximum |
|---|---:|---:|---:|
| End to end | 5.85 s | 12.18 s | 31.55 s |
| Server processing | 2.79 s | 3.06 s | 4.28 s |

Peak reported allocated GPU memory was 10.88 GiB. The model reported falling back to PyTorch linear attention because optional fast kernels were not installed; do not compare these timings with optimized kernels or batched serving. Per-image times and token counts are in `analysis.json` and the individual response files.

## Deployment and cost

The authenticated Runpod pod `j8qvat8tmzyhgr` was created at a reported GPU price of $0.89/hour. Approximate GPU cost through the final saved result is $0.136, based on elapsed time, not a reconciled invoice. Storage and subsequent runtime are additional. [Deployment details](deployment.json) and [cost estimate](cost-estimate.json) contain the configuration and assumptions.

The pod is left running as the requested inference server and continues to incur charges until stopped. Its HTTP endpoint requires a bearer token; the local trial client reads that token from `/tmp/qrealign-lite-token`, which is outside the repository and may be removed by system cleanup. Treat that file as a credential. Pod stop/restart behavior and pricing should be checked before depending on it as a long-lived service.
