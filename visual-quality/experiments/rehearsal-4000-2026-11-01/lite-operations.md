# Q-ReAlign Lite execution

The 4,000-image run uses the unchanged scorer and checkpoint from the earlier 72-image trial. Each image receives separate technical-quality and aesthetic forward passes. Native probabilities and scores are retained, and reported scores use `1 + 4 × native_score`. There is no eligibility floor or image-stat correction in this model.

## Deployment and startup

The original stopped pod, `j8qvat8tmzyhgr`, could not resume because its host had no free GPU. A deployment-only amendment, recorded before scoring in `lite-deployment-amendment.json`, allowed another secure RTX 4090 at the same $0.89/hour rate. The scorer source, model revision, container image, Transformers version, task prompts and client concurrency remain unchanged.

The evaluation pod was `gm21jcse3yohgp`, assigned to EU-RO-1 and created at 2026-11-01 13:46:11.489 UTC. It was stopped through the Runpod plugin at 15:16:43 UTC and is confirmed `EXITED`. Its host driver reported CUDA 13.2; its model runtime reported PyTorch 2.8.0+cu128. The HTTP health identity exactly matched the previous trial, including the checkpoint revision and all five level-token IDs.

An earlier replacement, `1e4rvgw2q4hkci`, was stopped before any image request. Runpod's pod response redacted the scorer credential, and copying that response copied the placeholder instead of the credential. The authenticated health check rejected it. The credential was then restored through an opaque local-file transfer to the plugin, without printing it or saving it in the repository. Restarting that stopped host failed because its GPU had already been reassigned. A subsequent correctly credentialed deployment produced the active pod. The original and earlier replacement remain stopped; their startup overhead is accounted separately.

The two historical startup controls, T0231 and T0984, reproduced both previous scores exactly. Their probability distributions passed validation, and their health identity matched exactly. These controls are excluded from the 4,000-image results and from benchmark comparisons. Startup control peak allocated VRAM was 11,282,718,720 bytes.

## Run contract

- Full manifest SHA256: `57d6c46923b9ae20151bb8334373b0872f954c8af8cd0152150c1a7b76b8257a`.
- Reference freeze SHA256: `458c370109100e45be0a3adb4b0a674fa0267296b2f2d9d9757f1690b903d24e`.
- Frozen reference file SHA256: `4d93e9fcda5b20ba8c1e297c9c474cb1d6919b7e5cb6fa74f6a8fcb3bfabfc86`.
- Unchanged server source SHA256: `aaab73e53225093d30e1fbe800425c3b9754d7a5547013f735de8f358a408dbb`.
- Checkpoint: `q-future/Q-ReAlign-Lite-4B`, revision `06add46a6bc4a7dfcf0a59a2fb1ed15a6e3d30b7`.
- Client concurrency: two outstanding HTTP requests; the unchanged server processes images serially.
- Benchmark admission verified the complete 500-case frozen reference set, distinct families and exact image identities before the run.

The immutable run configuration, attempt records and raw responses are under `/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/results/lite/all/`. Operations, startup controls, deployment incidents and billing receipts are kept separately under the adjacent `lite-operations/` directory. The client never automatically retries an ambiguous or failed request.

Each row records HTTP invocation through complete response-body receipt, including upload, network, proxy, queue and server time. Local payload preparation precedes the timer. Server processing seconds and peak allocated GPU memory are separate fields. Reported per-image price is the user's requested **$0.00** override; actual Runpod infrastructure spend is separate. An empty, not-yet-posted billing response is not evidence of zero cost.

The job completed within its 28,000-second scheduling limit and well before the common eight-hour lifecycle deadline. All **4,000 responses are valid**: 3,500 rehearsal, 100 development, 300 representative and 100 challenge cases. The final audit recomputed every delivered-image hash and exact request-payload hash, reparsed every raw response, checked native probabilities and scale mapping, and revalidated the complete 500-case reference admission. The receipt hash index and checks are in `lite-completion-validation.json`.

Requests ran from 13:47:40.565093 to 15:14:40.388941 UTC, a window of 86.997 minutes. Median end-to-end latency was **2.0403 seconds**, mean 2.5756 seconds; median server processing was 0.650 seconds. Peak allocated VRAM was 11,686,957,056 bytes. The client exited successfully, and the pod was stopped after local receipt verification. No request was retried.

`lite-infrastructure.json` separates the requested $0.00 model-price override from cloud spend. The two billed lifecycle intervals imply approximately **$1.37 in compute at the quoted hourly rate**, excluding storage. At the post-stop billing query, provider receipts had posted only **$0.475112321553751826** across both pods; the active pod's final hour was absent and earlier buckets were still updating. This is a provisional subtotal, not a settled total. The infrastructure total therefore remains null until billing settles. The original pod and both trial replacements are confirmed stopped.
