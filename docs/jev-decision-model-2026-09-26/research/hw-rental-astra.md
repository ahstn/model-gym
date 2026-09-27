# GPU rental decision — Astra adversarial review

**Observed 2026-09-26. USD unless noted.** Primary-source research only; no GPU was rented, no model weights/datasets downloaded, no repo files changed. Rates are observed public asks, not capacity reservations. `[INFERENCE]` marks our calculations, recommendations, or unmeasured performance assumptions; `[UNVERIFIED]` marks information not established by loaded sources.

## Executive decision

1. **[INFERENCE] Rent one full RTX PRO 6000 96GB for final leaderboard reproduction, not every development experiment.** Start with Lium's currently rentable server-edition $1.19–1.29/h or workstation $1.29–1.35/h; use Runpod Secure $2.09/h when persistent storage/network predictability matters. Match the leaderboard's *edition, power limit, software, quantization, batch size and suite revision*, not just the name. Vast's cheap Max-Q is not an automatically valid latency comparator. Prices: [Lium](https://lium.io/pricing), [Runpod](https://www.runpod.io/pricing), [Vast live feed](https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json). Leaderboard source: [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index).
2. **[INFERENCE] For 9–12B LoRA, prefer 48GB L40S or 80GB A100, not automatically H100.** A 4090/5090 is sensible for QLoRA and short contexts, but ordinary BF16 12B weights already consume about 24GB before activations/adapters. Use cheap A100 capacity to remove memory-driven experimental compromises. Memory basis: [HF memory anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy); rates below.
3. **[INFERENCE] For genuine full 27B/26B-A4B FT, rent an interconnected node: 8×H100 80GB, or 4×H200 141GB / 4×B200 180GB only where that actual topology is purchasable.** MoE active parameters reduce compute, not all optimizer state. Do not buy a single 96GB GPU expecting conventional Adam full FT. Budget pilot first; large corpus duration cannot be inferred from model size alone. Memory basis: [HF](https://huggingface.co/docs/transformers/model_memory_anatomy).

## Price matrix: dedicated/marketplace compute

Every row's citation supports that row; `—` = no current price verified, **not** proof GPU is unavailable. Prices are per GPU-hour; some require a larger node. GPU RAM generally: 4090 24GB, 5090 32GB, PRO 6000 96GB, L40S 48GB, A100/H100 here 80GB, H200 141GB; B200 usually advertised 180GB usable, although marketplace descriptions sometimes say 192GB. [Runpod](https://www.runpod.io/pricing), [Nebius hardware specifications](https://docs.nebius.com/compute/virtual-machines/types).

| Provider / tariff | 4090 | 5090 | PRO 6000 96GB | L40S | A100 80GB | H100 80GB | H200 | B200 | Evidence / caveat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Runpod Community | .34 | .69 | 1.69 | .79 | 1.19 PCIe / 1.39 SXM | 1.99 PCIe / 2.69 SXM | 3.59 | 5.98 | [Pricing + embedded JSON-LD](https://www.runpod.io/pricing); modified Sept 13, raw page published Sept 25 |
| Runpod Secure | .74 | .99 | 2.09 | 1.09 | 1.59 PCIe/SXM | 2.89 PCIe / 3.49 SXM | 4.59 | 6.79 | [Pricing](https://www.runpod.io/pricing); reader's default visible tier is Secure, NOT Community |
| Vast live median | .4944 | .6281 | 1.5347 server / 1.4082 workstation / 1.4224 Max-Q | .8002 | .9227 PCIe / .8012 SXM4 | 2.53 PCIe / 2.2671 SXM | 4.6059 | 9.3766 | [Public feed](https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json), 2026-09-26 12:30:18Z; host/quantity/type qualification still required |
| Lium currently rentable | .58 | — | 1.19–1.29 server / 1.29–1.35 workstation | — | 1.12–1.23 SXM4 | 1.30–2.75 HBM3 | 3.00–5.50 | 5.60 | [Live pricing](https://lium.io/pricing), generated 12:30:55Z Sept 26; pod GPU counts vary |
| Lambda advertised | — | — | — | — | 2.79 SXM | 3.99 SXM | — | 6.69 | [Instances](https://lambda.ai/instances); listed CPU/RAM/storage look like node configurations; do not assume these rates apply to every 1-GPU SKU |
| TensorDock historical advertised | .35 | — | — | — | 1.50 PCIe / 1.80 SXM | 2.25 SXM5 | — | — | [GPU cloud](https://marketplace.tensordock.com/cloud-gpus.html) explicitly **last updated July 24, 2024**; not a verified Sept-2026 bookable quote; CPU/RAM/storage extra |
| Prime Intellect advertised | — | — | 3.14* | — | 3.14* | 2.43 | 3.14 | 3.49 | [Homepage](https://www.primeintellect.ai/); *several implausibly repeated cards and malformed H200 80GB cards: obtain availability quote before using any card in a budget |
| Hyperbolic on demand | — | — | — | — | — | 3.19 SXM | 3.99 | 5.99 | [Marketplace](https://www.hyperbolic.ai/marketplace), refreshed weekly; single **node** advertised, not proof every SKU is single-GPU |
| Nebius on demand | — | — | 1.80 | from 1.55 Intel / 1.82 AMD | — | 3.85 | 4.50 | 7.15 | [Pricing](https://nebius.com/prices); actual 1-GPU presets verified [here](https://docs.nebius.com/compute/virtual-machines/types) |
| CoreWeave single-GPU inference rate, NOT generic training rental | — | — | 2.50 high-memory | 2.25 | 2.70 | 6.16 | 6.31 | 8.60 | [North America pricing](https://www.coreweave.com/pricing); generic corresponding nodes are 8-GPU |
| Together GPU clusters, NOT established single-GPU rental | — | — | — | — | — | 3.99 | 5.99 | 8.19 | [Pricing](https://www.together.ai/pricing); cluster documentation uses minimum 8 GPUs [API guide](https://docs.together.ai/docs/gpu-clusters-api) |

**Important corrections to simplistic shopping lists:** Lium lists $0.40 5090, $0.38 L40S and $0.45 A100 PCIe as *reference prices with no currently rentable pod*. They are not actionable live offers. [Lium](https://lium.io/pricing). Prime's older H100-multinode page appears in search results with $1.49 H100/.79 A100/.32 4090 but direct fetch returned **404**; reject those stale search snippets as current prices. [Old URL](https://www.primeintellect.ai/compute/nvidia-h100s-multinode). TensorDock's headline is explicitly 2024 pricing, so a 2026 comparison repeating it without caveat is misleading. [TensorDock](https://marketplace.tensordock.com/cloud-gpus.html).

### Vast live distribution, not just a misleading minimum

Snapshot from the primary JSON feed (timestamp above). These are model aggregates, not a guarantee a *single GPU* of that type at the minimum satisfies our reliability/network/disk needs. [Feed](https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json), [feed meaning](https://vast.ai/pricing).

| GPU | Minimum | p10 | Median | Available offers |
|---|---:|---:|---:|---:|
| 4090 | .2014 | .4003 | .4944 | 572 |
| 5090 | .3340 | .4540 | .6281 | 948 |
| PRO 6000 server | 1.1877 | 1.2007 | 1.5347 | 169 |
| PRO 6000 workstation | 1.1161 | 1.1403 | 1.4082 | 149 |
| PRO 6000 Max-Q | .7223 | .9340 | 1.4224 | 44 |
| L40S | .4668 | .4668 | .8002 | 113 |
| A100 PCIe 80GB | .2671 | .4294 | .9227 | 72 |
| A100 SXM4 80GB | .3113 | .5337 | .8012 | 144 |
| H100 PCIe 80GB | 1.8681 | 1.8681 | 2.5300 | 16 |
| H100 SXM 80GB | 1.7337 | 1.7340 | 2.2671 | 80 |
| H200 | 2.6325 | 2.6329 | 4.6059 | 74 |
| B200 | 6.2507 | 6.2507 | 9.3766 | 29 |

The feed labels H100 NVL as 80GB, while Runpod lists its NVL as 94GB; [INFERENCE] this is another reason to inspect the actual card with `nvidia-smi` rather than trusting marketplace normalization. [Vast feed](https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json), [Runpod](https://www.runpod.io/pricing).

## Daytona: yes, it now offers real GPU sandboxes

Do **not** dismiss Daytona as CPU-only. The loaded pricing page, published Sept 24, lists these **preemptible** rates; documentation explicitly supports GPU fine-tuning/CUDA, on-demand and spot capacity, up to 8 GPUs. [Pricing](https://www.daytona.io/pricing), [sandbox documentation](https://www.daytona.io/docs/sandboxes.md).

| GPU | Preemptible $/GPU-hour |
|---|---:|
| RTX 4090 | .57 |
| RTX 5090 | .74 |
| RTX PRO 6000 | 1.74 |
| H100 | 2.27 |
| H200 | 2.61 |
| B200 | 3.59 |

A100/L40S prices were not listed. **On-demand GPU tariff [UNVERIFIED]**: page exposes an on-demand toggle but the successfully loaded text exposes only the preemptible tariff. CPU $.0504/vCPU-hour, memory $.0162/GiB-hour, disk $.000108/GiB-hour (first 5GiB free) are listed separately; all billing per second. Whether a particular GPU offer includes those resources must be established in the actual quote—do not silently compare GPU-only Daytona to bundled Runpod. [Daytona pricing](https://www.daytona.io/pricing).

GPU sandboxes are **ephemeral**, cannot auto-pause, and each GPU permits up to 16 vCPU /192GB RAM/512GB disk. Spot can die **immediately without preemption warning or dedicated webhook**; the 24h retrievable record is not a promise of recoverable filesystem contents. Shared-region GPU placement is the aggregate `earth` region, with requested target ignored. [Sandbox docs](https://www.daytona.io/docs/sandboxes.md). Docker image-based sandboxes are documented, but GPU-specific image/SSH/volume compatibility and egress prices were not fully established here **[UNVERIFIED]**. **[INFERENCE] Use Daytona for disposable smoke/pilot jobs with explicit remote checkpoint export; not first choice for an uninterrupted long full-FT run.** If CPU/RAM are additive, an 8-vCPU/64GiB allocation adds $.4032+$1.0368=$1.44/hour before disk—nearly doubling a $1.74 GPU headline. Arithmetic is conditional, not an observed invoice.

## Serverless: different product, often wrong for a long experiment

| GPU | Runpod serverless $/h equivalent | Modal GPU tasks $/h equivalent |
|---|---:|---:|
| 4090 | 1.10 | — |
| 5090 | 1.58 | — |
| PRO 6000 | 3.49 | 3.0312 |
| L40S | 1.75 grouped with L40/6000 Ada/MIG48 | 1.9512 |
| A100 80 | 2.72 | 2.4984 |
| H100 | 4.79 | 3.9492 |
| H200 | 5.93 | 4.5396 |
| B200 | 8.64 | 6.2496 |

Sources: [Runpod pricing](https://www.runpod.io/pricing), [Modal pricing](https://modal.com/pricing); Modal converted from the displayed per-second rates ×3600. Runpod bills startup/model load, execution, and idle timeout; Flex scales to zero, Active workers run continuously and currently require sales inquiry for discounts—do not reuse obsolete universal active/flex discounts. [Serverless billing](https://docs.runpod.io/serverless/pricing). **[INFERENCE] A single long 121K-decision eval gains little from scale-to-zero; use a Pod unless parallelism/operations justify the premium.** Never use a grouped SKU when the acceptance criterion is exact GPU matching.

Modal adds CPU and RAM: physical core $.04716/hour, RAM $.007992/GiB-hour for GPU Tasks; Sandbox/Notebook CPU and RAM are about 3× those rates, while GPU prices are the same. Volumes $.09/GiB-month with 1TiB/month free per pricing page; $30/month Starter compute credit is advertised but eligibility/applicability must be checked. [Modal pricing](https://modal.com/pricing). **[INFERENCE] Four physical cores+32GiB adds $.444384/hour**; a 12h PRO6000 run becomes $41.71 rather than $36.37 GPU-only. Volumes need commit/reload semantics; final explicit commit is advisable before considering checkpoint export complete. [Modal volume docs](https://modal.com/docs/guide/volumes).

## Operational comparison: billed time is not useful training time

| Provider | Billing/storage/network and workflow evidence | Trial implication |
|---|---|---|
| Runpod | Compute per second; no ingress/egress fees. Container disk $.10/GB-month erased on stop; volume disk $.10 running/$.20 stopped retained until Pod deletion; network volume $.07/GB-month below 1TB, $.05 above, portable between Pods; network volume billed hourly. [Billing](https://docs.runpod.io/pods/pricing) | [INFERENCE] Best uncomplicated default: Docker Pod+SSH workflow, persistent HF cache/checkpoints on network volume. Check volume region availability and image SSH config before launch, not assumed here. |
| Vast | Per-second compute; host-priced continuously billed storage; both inbound/outbound bytes can cost extra; interruptible lower priority can pause. [Pricing docs](https://docs.vast.ai/guides/instances/pricing) | [INFERENCE] Filter reliable hosts and include model download/checkpoint transfer in total cost. Ordinary allocated storage surviving stop is not backup protection against host loss; require independent copy. |
| Lium | Docker containers with root SSH/GPU passthrough; per-second/no-minimum pricing; persistent volume management; reboot recreates container, losing everything outside volume. Template verification expects Debian/Ubuntu apt-based image. [Reference](https://lium.io/llms-full.txt), [pricing](https://lium.io/pricing) | [INFERENCE] Attractive exact-GPU pilot; put HF_HOME and outputs on explicit volume. Volume/egress tariff and host/preemption guarantees [UNVERIFIED]; do not call storage/egress free. |
| Lambda | Pay by minute, no egress fees; 1/2/4/8 GPU offerings, bundled local SSD in advertised node table. Persistent filesystem billed independently in hourly increments while it exists. [Instances](https://lambda.ai/instances), [billing](https://docs.lambda.ai/public-cloud/billing/) | [INFERENCE] Convenient node for full FT; verify specific SKU/count price, persistent filesystem region, SSH and Docker installation. $.20/GiB-month in billing docs is an example, not verified universal tariff. |
| TensorDock | Resource pricing separate: $.003/vCPU-hour, $.002/GB-RAM-hour, $.00005/GB-NVMe-hour, per-second. Advertised 1Gbps included; not a verified unlimited-egress promise. Balance zero auto-deletes servers. [2024 rate page](https://marketplace.tensordock.com/cloud-gpus.html) | [INFERENCE] Needs fresh dashboard quote. 8vCPU/32GB RAM/200GB disk adds $.098/h using old tariff. Treat server data as at risk, keep export. |
| Prime Intellect | Live availability CLI exposes GPU count/region/stock/prices; disks continuously billed until termination, whether attached or not, provider-dependent rate. [Availability](https://docs.primeintellect.ai/cli-reference/check-gpu-availability), [disk docs](https://docs.primeintellect.ai/api-reference/managing-disks) | [INFERENCE] Use as broker, not a single uniform tariff. Interconnect, SSH/image support, egress and billing granularity need offer-level confirmation [UNVERIFIED]. Prime CPU sandboxes are a different product than GPU Pods. |
| Hyperbolic | On-demand usage billing, VM/bare-metal, InfiniBand/Ethernet choice; no charges for failed instances; reserved capacity described as without preemption. [Marketplace](https://www.hyperbolic.ai/marketplace) | [INFERENCE] Credible full-node comparator; exact billing quantum, persistent disk/egress price, on-demand interruption policy and minimum GPU count [UNVERIFIED]. |
| Nebius | 1-GPU H100/H200/B200/PRO6000/L40S presets verified; shared filesystem $.08/GiB-month, object storage standard $.0147/GiB-month + $.015/GiB object egress. [Types](https://docs.nebius.com/compute/virtual-machines/types), [prices](https://nebius.com/prices) | [INFERENCE] Enterprise-ish 1-GPU fallback; do not apply object egress price automatically to all VM traffic. Separate boot disks/network items can add cost. |
| CoreWeave / Together | CoreWeave lists 8-GPU generic nodes vs explicitly single-GPU **inference** pricing. Together sells managed endpoints separately from clusters with Kubernetes/Slurm + SSH. [CW](https://www.coreweave.com/pricing), [Together](https://www.together.ai/pricing), [cluster workflow](https://docs.together.ai/docs/gpu-clusters-quickstart) | [INFERENCE] Exclude from cheap SSH single-GPU training shortlist unless sales/console verifies exact product. Together shared FS $.16/GiB-month. |

**Date-sensitive trap:** Nebius announces **October 1, 2026** changes: H100 $3.85→4.50; H200 $4.50→5.40; B200 $7.15→8.50, while PRO6000 remains $1.80. Its spot pricing is currently a *from* floor: H100/H200/PRO6000 $.79; B200 $.99. These are not guaranteed stable offers. [Prices](https://nebius.com/prices). Together H100 dedicated inference promo $3.99 expires Sept 30; regular display $5.49. [Together](https://www.together.ai/pricing).

## Experiment budgets and feasibility

### Explicit assumptions, not benchmark claims

All duration assumptions below are **[INFERENCE]**. No measured tokens/s exists for our final recipe. Costs = GPUs × actual occupied hours × rate + storage + transfer + failed/restarted work. Eval can be prompt-prefill dominated; training depends on sequence lengths, packing, checkpointing, optimizer, attention kernels, and MoE dispatch. Do not represent these as reproduction costs established by the leaderboard.

| Experiment | Proposed resource / duration assumption | Compute budget (derived from cited rates) |
|---|---|---:|
| Exact-GPU evaluation pilot | 1×PRO6000, 1h, Lium server $1.19–1.29 | $1.19–1.29 |
| Full evaluation allowance | 1×PRO6000, 6–12h, Lium server | $7.14–15.48 |
| Same evaluation, operational default | 1×PRO6000, 6–12h, Runpod Secure | $12.54–25.08 |
| 9–12B QLoRA short-context trial | 1×4090, 8–24h, Runpod Community $.34 | $2.72–8.16 |
| 9–12B BF16 LoRA trial | 1×L40S, 8–24h, Runpod Community $.79 | $6.32–18.96 |
| Memory-comfort LoRA | 1×A100-80 PCIe, 8–24h, Runpod Community $1.19 | $9.52–28.56 |
| Longer-context / larger batch LoRA | 1×PRO6000, 8–24h, Runpod Secure | $16.72–50.16 |
| Dense 27B full FT moderate pilot-to-run allocation | 8×H100 SXM, 12–48h, Lambda $3.99 | $383.04–1,532.16 |
| Same dense run, larger memory alternative | 4×H200, 12–48h, Hyperbolic $3.99 **if 4-GPU topology offered** | $191.52–766.08 |
| Same dense run, Blackwell alternative | 4×B200, 12–48h, Hyperbolic $5.99 **if 4-GPU topology offered** | $287.52–1,150.08 |
| 26B-A4B MoE full FT pilot-to-run allocation | 8×H100 SXM, 6–24h, Lambda $3.99 | $191.52–766.08 |

Rate citations: [Runpod](https://www.runpod.io/pricing), [Lium](https://lium.io/pricing), [Lambda](https://lambda.ai/instances), [Hyperbolic](https://www.hyperbolic.ai/marketplace). Different durations across dense/MoE are exploratory allowances, **not a guaranteed MoE speedup**. If the provider requires 8 H200/B200, double the corresponding 4-GPU numbers. If the job takes 100h, charge for 100h; these are not caps enforced by provider.

**[INFERENCE] Memory arithmetic:** HF's conventional mixed-precision Adam accounting is 18 bytes/parameter before activations/temp: 27B ≈486GB, 26B ≈468GB. Eight H100s provide 640GB aggregate; four H200s 564GB (tight at long context); four B200s about 720GB. ZeRO-3/FSDP sharding and a competent interconnect are prerequisites, not optional optimizations. BF16/no-master-weight or 8-bit optimizer implementations can reduce this materially, but that is a different validated recipe—not grounds to promise single-GPU full FT. MoE has 26B stored/trainable parameters even if about 4B are active per token. [HF memory anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy), [GPU capacity](https://docs.nebius.com/compute/virtual-machines/types).

**[INFERENCE] Storage allowance:** a 200GB Runpod network cache costs $14/month; 500GB $35/month; one week approximately $3.27/$8.17 using a 30-day month. Full Adam resumable checkpoints for 27B can approach hundreds of GB (weights+optimizer, implementation-dependent); keep 1–2 rolling resume checkpoints and a separate weights-only export, rather than filling the local 440GB with many optimizer snapshots. Source rates [Runpod](https://docs.runpod.io/pods/pricing); tensor accounting [HF](https://huggingface.co/docs/transformers/model_memory_anatomy).

### How to make estimates decision-grade before buying a long run

**[INFERENCE] Proposed trial protocol:** cache only required checkpoint revisions on a persistent mount (`HF_HOME` there), use pinned Docker/CUDA/PyTorch/kernel versions, then measure a representative 1,000–3,000 eval rows and 100–200 warmed training steps. Record actual token count, p95 context, microbatch, packing efficiency, peak VRAM, GPU utilization, total wall time including download/setup. Extrapolate `total_tokens / measured_tokens_per_second` plus checkpoint/eval time; add a 25–50% first-run reserve. Check actual GPU name/VRAM/power limit and `nvidia-smi topo -m` on multi-GPU nodes. Never compare a sharded 8-GPU run against eight independent cheap cards as though those were interchangeable.

## Contrarian recommendations

- **[INFERENCE] Optimize useful experiments per dollar, not cheapest advertised GPU-hour.** Repeated 30-minute environment/HF-download startup can erase a nominal $.20/h advantage. A shared persistent cache and one stable image may save more than switching providers.
- **[INFERENCE] Run the main modeling ablations on local hardware/smaller models and spend rental money on falsification.** A cheap 80/96GB rental removes quantization confounds for the decisive comparison; it should not be a reason to start with 27B full FT.
- **[INFERENCE] Refuse MoE “4B cost” arithmetic.** Active FLOPs do not imply 4B optimizer memory or frictionless expert-parallel training. Prove memory and dispatch throughput in a short run before selecting it as the cheap large-model path.
- **[INFERENCE] Spot is for restartable experiments, not default full-FT savings.** Define remote checkpoints and recovery first; Daytona's no-warning preemption makes this non-negotiable. [Daytona interruption contract](https://www.daytona.io/docs/sandboxes.md).
- **[INFERENCE] Alternative to owning a full-FT stack:** Together currently lists full supervised FT of Qwen3.6-27B at $1.16/M tokens (minimum $4): a 100M-token run would be ~$116 plus validation-token charges. This is a genuine cost comparator, **not an equivalent recipe** if custom typed-probability losses, candidate masking, calibration objectives, checkpoint exports, or required model are unsupported. The LoRA table separately includes Gemma4-26B-A4B at $1.05/M tokens; don't misread it as full-FT availability. [Together fine-tuning pricing](https://www.together.ai/pricing).

## Verification limits and URL discrepancies

The brief had no provider pricing URLs to invalidate. Our attempted `/pricing` URLs at TensorDock, Prime Intellect and Hyperbolic returned 404; canonical working pages are cited above. Daytona `/docs/en/gpu/` returned 404, while `/docs/sandboxes.md` explicitly verifies GPU support. Prime's search-indexed old H100-multinode page returned 404. Search summaries often disagreed with the direct live pages (notably Runpod Secure increases, Daytona GPU existence, Nebius future/current prices); tables above prioritize loaded pages over summaries. No authenticated console booking, spend, SSH session, bandwidth test, or provider reliability measurement was performed. Remaining offer-specific uncertainties are explicitly labeled rather than filled with historical/aggregator prices.
