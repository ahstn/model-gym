# RunPod budget for a 200,000-image training run

Planning estimate, checked 2026-11-17. This is not a measured training benchmark
or authorization to launch training. The completed labeled corpus currently has
20,000 images; acquiring and labeling another 180,000 is outside this estimate.

## Working assumption

Use the documented Qwen3.5-9B training control with LoRA, two epochs, one image
per training example, and a short joint technical/aesthetic target. Use the
accepted ensemble targets (37.5% Luna, 37.5% DeepSeek, 25% Liquid). Keep image
statistics out of model inputs. Memory fit and throughput still need a pilot
with the actual image processor, resolution/token cap, LoRA targets, precision,
batch size and checkpointing settings. Neither the old grading latencies nor
Q-ReAlign inference timings predict training throughput.

The conservative calculation treats all 200,000 images as training images:
400,000 image presentations. If 200,000 is the total corpus with an 80/10/10
family-disjoint split, 160,000 train images produce 320,000 presentations, reducing
the training portion below by 20%. Two separate task prompts per image double
presentations; a joint two-score target does not. Full-model fine-tuning, extra
crops, long reasoning targets and hyperparameter searches need separate budgets.

## Current rental rates and sensitivity

The authenticated RunPod GPU catalog returned these Secure Cloud, on-demand,
single-GPU prices on November 17. All three had LOW availability; deployment
availability and quoted prices must be checked again before launch.

| GPU | VRAM | USD/GPU-hour |
|---|---:|---:|
| A100 SXM | 80 GB | 2.19 |
| RTX PRO 6000 Blackwell Server | 96 GB | 2.69 |
| H100 SXM | 80 GB | 4.39 |

Source: RunPod `list_gpu_types`, `include=AVAILABILITY`, `product=POD`,
`cloud=SECURE`, `count=1`. RunPod's [pricing documentation](https://docs.runpod.io/pods/pricing)
explains billing and directs users to the deployment catalog for live GPU quotes.
No claim is made that these GPUs achieve equal throughput.

`training_hours = training_images × epochs × task_passes / (images_per_second × 3600)`

The following rates are **hypothetical sensitivities, not expected or measured
performance**. Effective images/second must include forward/backward passes,
optimizer steps and the training data loader. Additional evaluation, checkpoint
and startup time is accounted for separately.

| Effective images/sec | Hours for 400K presentations | A100 GPU cost | PRO 6000 GPU cost | H100 GPU cost |
|---:|---:|---:|---:|---:|
| 1 | 111.1 | $243 | $299 | $488 |
| 2 | 55.6 | $122 | $149 | $244 |
| 4 | 27.8 | $61 | $75 | $122 |
| 8 | 13.9 | $30 | $37 | $61 |

If a pilot achieves **2–4 images/sec**, allow **$200 on A100 or $350 on H100**
for one run: this includes a provisional 25% allowance for evaluation,
checkpoints and setup, plus roughly $20 for the timing pilot and short-lived
storage. Round upward instead of treating these allowances as measurements.
At only 1 image/sec, the same calculation approaches **$650 on H100**.
Performance below 1 image/sec, extra epochs, failures or repeated experiments
can exceed this. Prefer a small initial credit allocation for the pilot, then
fund the measured projection rather than depositing the worst-case estimate now.

H100 costs about 2.00 times as much per hour as A100 here; it must deliver over
2.00 times the effective throughput to reduce compute cost for the same workload.
PRO 6000 must exceed A100 throughput by about 23% to be cheaper. Compare the
actual training stack as well as memory fit before choosing a GPU.

## Storage and next measurement

The current 20K exact scored views occupy 35.22 GiB. Linear extrapolation gives
about 352 GiB (378 decimal GB) for 200K at the same size distribution. Allow
roughly 500 GB for views, weights, checkpoints and temporary files, then verify
against the actual corpus. A standard 500 GB network volume costs $35/month,
or about $8.17 for seven days using a 30-day month; capacity and rates vary by
storage product. Container disks add their own charges. RunPod charges no
transfer fee; other services' request/storage costs, labeling, tax and retained
storage beyond the run are excluded. See [RunPod storage pricing](https://docs.runpod.io/pods/pricing#storage-pricing).

Before the full run, time 500–2,000 representative images, including the largest
processed views. Record warmup separately, peak VRAM, effective images/sec,
training loss, checkpoint/save time and validation time. Confirm checkpoint
resume, then substitute the measured rate into the formula. Keep the existing
training masks disabled until labels and evaluation splits are approved.

Remaining account credit could not be verified in this session: the connector
exposes spending history, not live balance, and no local RunPod API credential or
accessible signed-in desktop was available. Spending history is not a balance.
The required top-up is `max(0, measured run budget + reserve - available credit)`;
confirm credit on the [RunPod Billing page](https://console.runpod.io/user/billing)
before funding. No training resources were started for this estimate.
