# Q-ReAlign Lite rehearsal protocol

Prepared 1 November 2026, before the new corpus was scored.

Reuse the existing stopped Runpod pod `j8qvat8tmzyhgr`, the unmodified scorer from commit `0a156b4`, and checkpoint `q-future/Q-ReAlign-Lite-4B` at revision `06add46a6bc4a7dfcf0a59a2fb1ed15a6e3d30b7`. The live Runpod catalogue and pod record both report $0.89/hour for this secure RTX 4090 setup. The catalogue reports 24 GB GPU memory; the API's pod `gpu.memory` field reports 31 and is not treated as a VRAM measurement. Runtime identity and measured peak allocated VRAM will be retained.

Each request contains one unchanged delivered PNG and asks both of the author's prior task questions. Technical quality and aesthetics are computed as two independent, sequential forward passes through the unchanged model-default processor. Five single-token levels (`excellent`, `good`, `fair`, `poor`, `bad`) have weights 1, .75, .5, .25, 0. The native 0–1 expected score maps to the project scale as `1 + 4 * score`. The runner retains both native distributions and unrounded normalized scores. It applies no scope or image-stats floor.

The client starts with concurrency **2**. The existing server remains single-threaded, so this only overlaps network/proxy/upload work; it changes neither GPU batching nor attention kernels. Every row's end-to-end latency includes its actual queue wait. It is not directly comparable to the previous single-stream latency as a service benchmark. Image loading, base64 encoding and request preparation occur before the latency timer; timing begins immediately before HTTP send and ends after the complete response body arrives. Server processing time is stored separately. Startup, dependency installation, checkpoint download and health checks are also recorded separately.

The user-requested reported per-image price is **$0.00**, with `price_basis=user_assigned_zero`. This is an accounting override, not a claim that Runpod is free. Pod lifecycle, elapsed billable time, hourly quote and any available billing receipts are kept separately from scorer prices.

## Freezing, execution and recovery

- Pin the corpus manifest, delivered hashes, client hash, checkpoint, server source and selected case order before inference. Verify the live health identity before each invocation.
- The initial run can process the 3,500 nonbenchmark images while reference assessments are being prepared. Any benchmark request, including a one-image subset, requires the complete 500-case reference freeze against the full 4,000-image manifest. Admission verifies the exact benchmark case set, 500 distinct families, declared partition counts, every reference image hash, and the manifest and reference-file hashes. Benchmark families cannot overlap the nonbenchmark partition.
- Preserve an fsynced reservation before sending, then an immutable raw HTTP response and an updated attempt record. Fsync newly created directories and their parent entries, and fsync the containing directory after each exclusive file creation or atomic replacement. A durability failure prevents dispatch or leaves the attempt unresolved; it never authorizes a resend. Retain actual HTTP status, timings and native response. Never log the bearer token or full request image payload.
- Resume only completed responses whose configuration and image hashes match. A prior reservation, in-flight request, HTTP error or transport/parse failure requires inspection; there are no automatic retries. A valid sibling task is preserved when the other task fails validation.
- The Runpod supervisor starts the pod only when a runnable partition is ready. Stop the resumed pod through the Runpod plugin after all responses are downloaded and checked, or at the eight-hour lifecycle cap. `--max-seconds` additionally stops scheduling requests for an individual invocation; it is not itself a cloud shutdown mechanism.
- Do not deploy a replacement pod, modify the scorer, change model processing, or alter the reported scale merely to avoid a failure. Diagnose and record any necessary change as a separate run configuration.

Before launch, run the local validation/resume tests and a separate smoke check on existing historical controls. Report startup and control calls outside the 4,000-image measurement table.

The previous trial's median server processing was 2.79 seconds per image and median end-to-end time was 5.85 seconds. These imply approximately 3.1 GPU-processing hours and 6.5 hours of strictly sequential end-to-end time for 4,000 images, before startup or variability. Two outstanding requests may reduce idle network time; the actual throughput must be measured.
