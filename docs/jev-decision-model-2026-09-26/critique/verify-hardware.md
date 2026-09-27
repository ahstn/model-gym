# Hardware verification: rental prices, conflicts, and per-job costs

Prices were fetched live on 2026-09-26, around 12:30–12:50 UTC. They are spot quotes: marketplace listings change hourly, so re-check a listing before you order. `[INFERENCE]` marks my own arithmetic or judgement.

## 1. Live price table ($/GPU-hour)

| GPU | Lium (live range, stock) | RunPod Community / Secure | Vast p10 (min / median, avail) | Daytona Preempt / On-demand | Lambda | Modal (GPU only) | Prime (front page) |
|---|---|---|---|---|---|---|---|
| RTX PRO 6000 96GB | Server $1.19–1.29 (10 pods, 11 GPUs); Workstation $1.29–1.35 (2 pods) | **$1.69 / $2.09** | Server ("S") **$1.20** (1.19 / 1.53, 169); WS $1.14 (1.12 / 1.41, 149); Max-Q $0.93 (44) | $1.74 / **$3.03** | — | $3.03 | $3.14 (see note) |
| A100 80 PCIe | $0.45 reference price, **no pods** | $1.19 / $1.59 | $0.43 (0.27 / 0.92, 72) | — | — | $2.50 (A100-80) | — |
| A100 80 SXM | $1.12–1.23 (6 pods, 48 GPUs) | $1.39 / $1.59 | $0.53 (0.31 / 0.80, 144) | — | $2.79 | — | — |
| H100 SXM/HBM3 | $1.69–2.75 (4 pods, 32 GPUs) | $2.69 / $3.49 | $1.73 (1.73 / 2.27, 80) | $2.27 / $3.95 | $3.99 (1x–8x, per GPU) | $3.95 | $2.43; spot $0.94 |
| H100 PCIe | $1.30 reference price, no pods | $1.99 / $2.89 | $1.87 (16) | — | — | — | — |
| L40S | $0.38 (4 pods, 8 GPUs) | $0.79 / $1.09 | $0.47 (113) | — | — | $1.95 | — |
| RTX 4090 | $0.58 (1 pod) | $0.34 / $0.74 | $0.40 (572) | $0.57 / $0.99 | — | — | — |
| RTX 5090 | $0.65 (1 pod) | $0.69 / $0.99 | $0.45 (948) | $0.74 / $1.29 | — | — | — |

Sources:
- Lium: [lium.io/pricing](https://lium.io/pricing), generated 2026-09-26T12:46Z.
- RunPod: the JSON-LD on [runpod.io/pricing](https://www.runpod.io/pricing) gives pairs in the form "Community Cloud $X, Secure Cloud $Y". The static render of that page shows only the Secure column. [runpod.io/gpu-models](https://www.runpod.io/gpu-models) shows only Community.
- Vast: [public feed](https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json), `updated 2026-09-26T12:30:18Z`.
- Daytona: [daytona.io/pricing](https://www.daytona.io/pricing). The on-demand panel is hidden in the HTML (`gpu-pricing-panel-ondemand`), so a plain reader shows only the preemptible prices.
- Lambda: [lambda.ai/pricing](https://lambda.ai/pricing) and [lambda.ai/instances](https://lambda.ai/instances). Both list 8x/4x/2x/1x instances at the same $/GPU/hr.
- Modal: [modal.com/pricing](https://modal.com/pricing), billed per second:
  - PRO 6000: $0.000842/s = $3.03/h
  - H100: $0.001097/s = $3.95/h
  - A100-80: $0.000694/s = $2.50/h
  - L40S: $0.000542/s = $1.95/h
  - CPU: $0.0000131/core-s
  - RAM: $0.00000222/GiB-s
- Prime Intellect: [primeintellect.ai](https://www.primeintellect.ai/).

## 2. Claim verdicts and conflict resolution

| # | Claim (source note) | Verdict | Evidence / corrected value |
|---|---|---|---|
| 1 | RunPod PRO 6000 $1.69 (sol) vs $2.09 Secure (astra) | **CONFIRMED (both)**: not a real conflict | $1.69 is the Community price and $2.09 is Secure. Sol quoted gpu-models (Community only) without naming the tier; astra quoted Secure. [runpod.io/pricing](https://www.runpod.io/pricing) JSON-LD. Plans should name the tier. |
| 2 | RunPod A100 PCIe $1.19 / SXM $1.39, H100 PCIe $1.99 / SXM $2.69, L40S $0.79, 4090 $0.34, 5090 $0.69 (sol) | **CONFIRMED (Community only)** | Secure prices: A100 $1.59/$1.59, H100 $2.89/$3.49, L40S $1.09, 4090 $0.74, 5090 $0.99. Astra's LoRA rows ("Community $.79 L40S", "$1.19 A100") are correct. |
| 3 | "PRO 6000 not present in Vast snapshot" (hw-rental-sol line 17) | **REFUTED** | The feed has `rtx pro 6000 s`: p10 $1.2007, min $1.1877, median $1.5347, 169 available. `ws`: p10 $1.1403, 149 available. `max-q`: p10 $0.934, 44 available. Vast is a real third source of PRO 6000 capacity. |
| 4 | Lium PRO 6000 Server $1.19–1.29 with 9 pods; Workstation $1.29–1.35 with 2 pods | **CONFIRMED** (stock drifted) | Prices unchanged. Stock is now 10 pods / 11 GPUs. |
| 5 | Lium A100 SXM $1.12–1.23 with 6 pods; A100 PCIe $0.45 reference only | **CONFIRMED** | Same on [lium.io/pricing](https://lium.io/pricing). |
| 6 | Lium H100 HBM3 $1.30–2.75 with 5 pods (sol) | **PARTLY** | Now $1.69–2.75 with 4 pods. $1.30 is the *H100 PCIe* reference price, which has no pods. |
| 7 | Lium L40S $0.38 "reference only, no pod" (both notes) | **REFUTED (today)** | 4 pods / 8 GPUs are rentable at $0.38. |
| 8 | Lium 5090 $0.40 reference, zero pods (both notes) | **REFUTED** | $0.65 with 1 pod / 2 GPUs. |
| 9 | Daytona preemptible prices: 4090 $0.57, 5090 $0.74, PRO 6000 $1.74, H100 $2.27, H200 $2.61 | **CONFIRMED** | [daytona.io/pricing](https://www.daytona.io/pricing). Neither note had the **on-demand** prices, which are 4090 $0.99, 5090 $1.29, PRO 6000 $3.03, H100 $3.95, H200 $4.54. On-demand Daytona costs the same as Modal and is roughly 1.5–2.5x RunPod or Lium. |
| 10 | Lambda H100 SXM $3.99 and A100 SXM $2.79 | **CONFIRMED** | Same $/GPU/hr for 1x through 8x, so 8x H100 = $31.92/h. The page says "No egress fees" ([lambda.ai/pricing](https://lambda.ai/pricing)). |
| 11 | Modal per-second prices (sol line 19) | **CONFIRMED** | Exact match on [modal.com/pricing](https://modal.com/pricing). |
| 12 | Prime front page: H100 $2.43, H200 $3.14, PRO 6000 $3.14, H100 spot $0.94 | **PARTLY / UNRELIABLE** | The strings are present, but the same widget prints $3.14 for GH200, PRO 6000, A100 *and* A40, and prints "H200" beside 80 GB cards at $0.47–1.99. It looks like marketing or placeholder data, so do not use it for planning. Get a real quote from the `prime` availability CLI ([docs](https://docs.primeintellect.ai/cli-reference/check-gpu-availability)). |
| 13 | Vast p10 figures in sol (4090 .400, 5090 .454, L40S .467, A100 .429/.534, H100 1.868/1.734, H200 2.633) | **CONFIRMED** | Identical in the 12:30Z feed. |
| 14 | RunPod network volume $0.07/GB-month below 1 TB (astra) | **CONFIRMED (consistent)** | The pricing page says storage is "starting at $0.05/GB/mo" ($0.05 is the tier above 1 TB, per [docs.runpod.io/pods/pricing](https://docs.runpod.io/pods/pricing), cited in hw-rental-astra line 90). |

Summary of the conflicts: sol's table used RunPod Community prices without saying so, and astra used Secure. Both sets of figures are correct. Sol's claim that Vast has no PRO 6000 capacity is wrong. Both notes' "Lium reference-only" warnings for the L40S and 5090 are stale; the A100 PCIe and H100 PCIe warnings still hold.

## 3. Throughput assumptions (sources and hedges)

- **PRO 6000 Server.** Memory bandwidth is 1,597 GB/s and FP8 is 2 PFLOPS (sparse). ([NVIDIA](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)) NVIDIA does not publish a dense BF16 figure. `[INFERENCE]` 2 PF sparse FP8 → 1 PF dense FP8 → **~0.5 PF dense BF16**. I use this as the upper bound.
- **H100 SXM.** 989 TF dense BF16 and 3.35 TB/s ([NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/)). H100 PCIe is about 756 TF.
- **A100-80.** 312 TF BF16 and ~2.0 TB/s ([NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/)).
- **FLOP counts.** A forward pass costs ≈2N FLOPs per token (Kaplan et al., [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)). A LoRA training step costs ≈4N (forward plus the activation-gradient backward; the frozen weights need no weight gradients), or **≈6N** with gradient checkpointing recompute. `[INFERENCE]`
- **Utilisation (MFU) guesses.** `[INFERENCE]` Large-batch prefill: ~45% on PRO 6000 (GDDR7, PCIe card) and ~50% on H100. LoRA training: ~35% on PRO 6000 and ~40% on A100.
- **Empirical eval timings.** Published per-decision mean latencies on the 1x PRO 6000 board, 120,340 decisions per model (../research/leaderboard.md lines 205–218):
  - Decider 4B: 59.7 ms → 2.0 h serial-equivalent
  - AutoJev-27B: 428.6 ms → 14.3 h
  - Decider chat Qwen3.6-27B: 2,227 ms → 74.5 h

## 4. Per-job cost estimates

### (a) Full decision-index suite eval, 1x PRO 6000 (121K decisions, 43 benchmarks)

Wall time = serial-equivalent hours from §3 + ~0.5 h (4B) or ~1 h (27B) for setup, download and warmup. The 27B BF16 weights are ~54 GB, and the suite corpus must also be downloaded.

| Model | Wall h | Lium $1.24 (mid) | Vast S p10 $1.20 | RunPod Comm $1.69 | RunPod Secure $2.09 |
|---|---|---|---|---|---|
| 4B (Decider-4B-like, 59.7 ms) | 2.5 | $3.10 | $3.00 | $4.23 | $5.23 |
| 27B, single-pass readout (AutoJev-like, 428.6 ms) | 15.3 | $18.97 | $18.36 | $25.86 | $31.98 |
| 27B, generative/chat readout (2,227 ms) | 75.5 | $93.62 | $90.60 | $127.60 | $157.80 |

`[INFERENCE]` With continuous batching the 27B single-pass run could plausibly take 3–6 h, but budget for the serial figure. On leaderboard parity: the board uses the **Server Edition** ([leaderboard note](../research/leaderboard.md)). Vast's cheaper Max-Q (a lower-power variant) and the Workstation cards are not latency-comparable.

**Recommendation:** Lium Server Edition, or Vast "RTX PRO 6000 S" as a backup (host with reliability >99% and at least 200 GB disk). Use RunPod Secure only for the final run that gets published.

### (b) Teacher labeling: 150K prompts × 600 tokens, 27B BF16, vLLM prefill-only logprob scoring

- Tokens: 150,000 × 600 = 9.0e7.
- FLOPs: 2 × 27e9 × 9.0e7 = **4.86e18**.
- Memory: the weights are ~54 GB. On an 80 GB H100 that leaves ~20 GB for the KV cache, which is ample for 600-token prefill with no decoding. On the PRO 6000 it leaves ~36 GB.

| GPU | Effective TF | Compute h | +1 h overhead | Price options → cost |
|---|---|---|---|---|
| PRO 6000 | 0.45 × 500 = 225 | 4.86e18 / 2.25e14 = 21,600 s = **6.0 h** | 7.0 h | Lium $1.24 → **$8.68**; Vast S $1.20 → $8.40; RunPod Comm $1.69 → $11.83; RunPod Secure → $14.63 |
| H100 SXM | 0.50 × 989 = 495 | 4.86e18 / 4.95e14 = 9,820 s = **2.7 h** | 3.7 h | Vast p10 $1.73 → **$6.42**; Lium $1.69–2.75 → $6.25–10.18; RunPod Comm $2.69 → $9.95; Lambda $3.99 → $14.76 |

`[INFERENCE]` Pitfall: vLLM `prompt_logprobs` returns top-k logprobs for *every* prompt position. That leaves the FLOP count unchanged but can make the job CPU- and serialization-bound, taking several times longer than the compute estimate. Request only the candidate-answer positions (for example with `prompt_logprobs=0` or a custom logits processor), or the PRO 6000 figure could double. FP8 weights would roughly halve compute time on both GPUs but change teacher fidelity.

**Recommendation:** H100 SXM on Vast or Lium; it is about 2.2x faster at a similar total cost. If H100 stock is thin, use the PRO 6000 on Lium. Either way the job costs under $15.

### (c) Gemma-4-12B BF16 LoRA on 40M tokens

- FLOPs: 6 × 12e9 × 4.0e7 = **2.88e18** (including checkpoint recompute).
- Memory: ~24 GB of weights plus activations fits on both GPUs.

| GPU | Effective TF | Compute h | +1.5 h (setup, eval, checkpoints) | Cost |
|---|---|---|---|---|
| PRO 6000 | 0.35 × 500 = 175 | 2.88e18 / 1.75e14 = 16,460 s = **4.6 h** | 6.1 h | Lium $1.24 → **$7.56**; Vast S $1.20 → $7.32; RunPod Comm → $10.31; RunPod Secure → $12.75 |
| A100-80 | 0.40 × 312 = 125 | 2.88e18 / 1.25e14 = 23,040 s = **6.4 h** | 7.9 h | Vast SXM p10 $0.53 → **$4.22**; Lium SXM $1.12 → $8.85; RunPod PCIe Comm $1.19 → $9.40; RunPod Secure $1.59 → $12.56; Lambda $2.79 → $22.04 |

`[INFERENCE]` Without gradient checkpointing (4N per token), both times drop by about a third. The PRO 6000's 96 GB of memory allows larger micro-batches and longer contexts, which is how its effective utilisation could beat the 35% guess.

**Recommendation:** Vast A100 SXM for the cheapest run if a reliable host is available. Otherwise use Lium's PRO 6000 at about the same cost as the RunPod A100 with ~1.3x the speed.

## 5. Storage and egress gotchas

- **RunPod.** No ingress or egress fees. Container disk ($0.10/GB-month) is wiped on stop. A pod volume costs $0.10/GB-month while running and $0.20 while stopped. A network volume costs $0.07/GB-month below 1 TB ($0.05 above) and is billed hourly while it exists. Network volumes are region-locked, so match the GPU's region. ([docs](https://docs.runpod.io/pods/pricing), via hw-rental-astra line 90.)
- **Vast.** Storage is priced per host and billed *even while the instance is stopped*. Upload and download bandwidth is priced per host and can cost more than the GPU on a download of 54 GB or more. A stopped instance may not restart if the host has rented out its GPU. ([docs.vast.ai pricing](https://docs.vast.ai/guides/instances/pricing), via hw-rental-astra line 91.) Filter offers on `inet_down_cost`.
- **Lium.** Rebooting recreates the container, and everything outside the explicit volume is lost. Volume and egress tariffs are unpublished (UNVERIFIED). Set `HF_HOME` to the volume. ([llms-full.txt](https://lium.io/llms-full.txt), via hw-rental-astra line 92.)
- **Daytona.** GPU sandboxes are ephemeral and capped at 512 GB disk. Preemptible sandboxes can be killed without warning. Storage costs $0.000108/GiB-h (about $0.078/GiB-month) after the first 5 GiB. ([pricing](https://www.daytona.io/pricing), [docs](https://www.daytona.io/docs/sandboxes.md).) Only use Daytona for jobs that checkpoint to external storage.
- **Modal.** CPU and RAM are billed on top of the GPU: 4 cores + 32 GiB adds ~$0.44/h. Volumes cost $0.09/GiB-month with 1 TiB/month free. ([pricing](https://modal.com/pricing).)
- **Lambda.** No egress fees. The persistent filesystem is billed separately while it exists. ([pricing](https://lambda.ai/pricing).)
- **Hugging Face downloads.** A 27B BF16 model plus an FP8 copy is roughly 80 GB. On a host limited to 1 Gbps that takes about 11 minutes just to download, and it is billed on hourly GPU providers. Pre-stage the weights on a network volume if you run the job more than once.

## 6. Recommended provider per job

| Job | First choice | Fallback | Est. cost |
|---|---|---|---|
| (a) 4B suite eval | Lium PRO 6000 Server | Vast PRO 6000 S | ~$3 |
| (a) 27B suite eval (single-pass) | Lium PRO 6000 Server | RunPod Secure PRO 6000 (for the published run) | $19–32 |
| (b) 27B teacher labeling | Vast/Lium H100 SXM | Lium PRO 6000 | $6–15 |
| (c) 12B LoRA, 40M tokens | Vast A100 SXM (p10) | Lium PRO 6000 or RunPod A100 Community | $4–10 |
| Anything preemptible or orchestrated | Daytona/Modal only if the code checkpoints externally | — | 1.5–2.5x VM price when on-demand |
