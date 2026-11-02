# Q-ReAlign Lite inference and reuse

Lite is a comparison model with an unchanged native interface. It receives the same exact delivered images as the API graders, returns separate quality and aesthetic score distributions, and retains the historical mapping to the 1–5 scale. It does not receive the chat JSON schema, the revised photography gate, or image-statistics text. Its per-row reported price is $0.00 by the user's accounting rule; paid Runpod infrastructure is separate.

## Pinned identity

The checkpoint is `q-future/Q-ReAlign-Lite-4B`, revision `06add46a6bc4a7dfcf0a59a2fb1ed15a6e3d30b7`. The server implementation is the unchanged experiment from `0a156b4`, SHA256 `aaab73e53225093d30e1fbe800425c3b9754d7a5547013f735de8f358a408dbb`. The container is `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, with Transformers 5.2.0. Each task is scored serially on the GPU using the original level tokens; returned health identity must exactly match the previously validated server.

The 4,000 historical results are reused only after checking case identity, exact delivered-view hash, family, pinned checkpoint/revision, original raw response, and recomputed native score. Reused rows retain the original measured HTTP latency, source attempt ID and raw-response provenance. No fresh request is claimed for those rows.

The new server passed two fixed operational controls with exactly matching native scores before the 6,000 new images were admitted. Controls are kept in a separate operations directory and do not inflate corpus coverage. A startup authentication mismatch caused by copying a redacted environment placeholder was corrected before scoring began; the real token was transferred without printing it and is never retained in tracked files.

## New requests

The `lite_run.py` adapter checks the live identity, manifest and selected image hashes. Its configuration records client/adapter hashes, concurrency, endpoint and reference-freeze provenance. Payload preparation happens before the HTTP timer. The timer starts immediately before sending and ends after receiving the complete response; server-only inference duration is stored separately. All new attempts, including failures, remain under ignored `results/lite/`.

Example invocation from the repository root, with an already running authorized endpoint:

```sh
python visual-quality/experiments/labeling-10000-2026-11-01/lite_run.py \
  --manifest /absolute/path/to/manifest.json \
  --output /absolute/path/to/new-unique-run-directory \
  --partition benchmark \
  --references-freeze /absolute/path/to/references-freeze.json \
  --case-ids /absolute/path/to/selected-case-ids.json \
  --concurrency 3 \
  --url https://POD_ID-8888.proxy.runpod.net \
  --token-file /absolute/path/to/private-token
```

New benchmark requests require matching frozen independent references. The original 4,000 images are excluded from new selections. Case lists are explicit, unique subsets of the selected partition. A continuation uses a new output directory and the unchanged image/model/payload; it never overwrites a previous attempt.

The first new rehearsal phase used two concurrent HTTP clients. It stopped on a transient HTTP520 after 1,469 valid results. A disjoint continuation preserved those results and retried the failed transport plus unattempted images with three clients. Both phases use the same serialized GPU scorer; concurrency is an operational throughput setting, not a model change. The exact executed client sources and request configuration hashes are retained in `lite-operations/`, including the adapter snapshot from before formatting changes.

`lite_queue.py` waits for the active continuation's durable summary and handles remaining requests with a maximum of three total attempts per image. Only transport/parse failures and the listed transient HTTP statuses qualify. Valid, partial and invalid assessments are not repeated to obtain more favourable scores. Every repeated request must retain its original request hash. The exporter rejects multiple scored assessments for one image instead of choosing the better result.

An operator must coordinate any disjoint parallel phases with the waiting queue so two clients cannot admit the same image. Each phase keeps its own settings and measured latency. Observed end-to-end timings depend on concurrency, image size, network and server load; they are not isolated hardware benchmarks.

A disjoint 100-image benchmark phase tested three additional clients while the three-client rehearsal continued. The waiting queue was paused to prevent duplicate admission. All 100 benchmark requests returned valid assessments, but combined throughput was 0.4084 requests/s against the preceding 0.4318 baseline, a 5.43% decrease. HTTP p95 rose to 17.70 seconds. The predeclared 15% throughput-improvement criterion was not met, so the remaining benchmark images were left to the sequential queue, which was restored after the trial drained. This comparison was not randomized and payload sizes differed between windows; it supports the operational choice to end overlap, not a causal claim about the model. The necessary trial assessments are retained, with no extra rescore or score selection.

## Finish and account

This run completed on 2026-11-02: the 6,002 new attempts contain 6,000 valid assessments and two retained HTTP520 failures, each recovered with the same payload. Together with 4,000 verified reused assessments, Lite covers all 10,000 images. The queue finished successfully and the owned pod was confirmed `EXITED` at 03:16:42 UTC with no retained volume mounts. The final report separates partial posted billing from the creation-to-stop GPU-rate estimate; neither is included in the $0.00 inference-row prices.

The exporter combines the verified historical reuse and every executed new phase. It validates request/view identity across retries, preserves all attempt receipts and costs, and uses the sole scored assessment if present. A terminal error remains a missing score. Latency for a selected response is distinct from the sum of its failed and successful attempts.

After the final phase, verify 10,000 unique Lite image slots and compare numerical coverage with the run summaries. Stop the owned Runpod worker through the plugin and retain the returned state. Report provider billing when available; otherwise report the GPU-rate estimate and state that the actual bill is unposted. Do not treat the assigned $0.00 label price as zero infrastructure spending. Leave unrelated pods and existing user gallery services untouched.
