# Experiment plan for a Q-ReAlign re-train

This file gives a staged plan to re-train or improve Q-ReAlign and to compare it with other assessors. It lists the runs, controls, gates, compute assumptions, and the first competitor set. It is a plan, not a result: no run in this file has been done.

Last verified: 2026-09-27

## Working hypothesis

- The cheapest likely gain is better supervision and better visual input on the same 5-word scorer. A newer backbone is a separate question. Test the two apart.
- General VLM benchmarks do not show better quality, aesthetics, or video-quality scoring. Only task results count.
- A new architecture cannot load Q-ReAlign weights. Start from the new VLM, port the scorer, and re-train on the mix. Further tuning of the released Q-ReAlign checkpoint is a separate, cheaper control.
- Method evidence is in [methods.md](methods.md). Data evidence is in [data-mixture-evidence.md](data-mixture-evidence.md). Metrics, split rules, and the baseline code audit are in [evaluation/protocol.md](../evaluation/protocol.md).

## 1. Baseline recipe to hold fixed

| Item | Value | Source | Status |
|---|---|---|---|
| Task mix (README claim) | KonIQ + SPAQ + KADID + AGIQA-20K + AVA + LSVQ | [README](https://github.com/Q-Future/Q-ReAlign#results) | verified |
| Task mix (public config) | `mix: [koniq, spaq, kadid, ava, lsvq]` (no AGIQA-20K) | [onealign.yaml L67](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/configs/onealign.yaml#L67) | verified |
| Labels | 5 words (excellent, good, fair, poor, bad), hard bins; score = probability-weighted mean | [protocol audit, rows 1 and 6](../evaluation/protocol.md#8-q-realign-baseline-code-audit) | verified |
| Training | Full SFT, DeepSpeed ZeRO-2, lr 2e-5, 2 epochs, batch 4, grad accum 2, 2 GPUs, vision tower and projector trainable, cosine schedule, warmup 0.03 | [onealign.yaml L70-L78](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/configs/onealign.yaml#L70-L78), README L290 | verified |
| Video input | 8 uniform frames, long side 448, JPEG q90 cache | [protocol audit, row 2](../evaluation/protocol.md#8-q-realign-baseline-code-audit) | verified |
| Sizes | Mini 0.8B, Lite 4B, Pro 9B (Qwen3.5) | [unified models](../models/unified-and-backbones.md) | verified |

Two config items must change before any run we report. The example config selects checkpoints on test-named sets, and it caps eval at 200 items (audit rows 9 and 10). Use a validation split for selection and pass `--limit 0` for final eval.

## 2. Separate "backbone" from "recipe"

Run a 2 x 2 grid before anything else. Keep data, steps, visual-token budget, and eval fixed.

| | Recipe A: Q-ReAlign as released (hard 5-bin labels) | Recipe B: soft 5-level distribution + in-dataset fidelity loss (DeQA-style) |
|---|---|---|
| Qwen3.5-4B (same family as Lite) | R1: recipe control | R2: recipe-only change |
| Gemma 4 E4B (new architecture) | R3: backbone-only change | R4: both |

- R3 minus R1 = backbone effect. R2 minus R1 = recipe effect. R4 tells you if they add up.
- Why these two first: Qwen3.5-4B is the Lite base, so R1 should land near released Lite. Gemma 4 E4B is the only newer backbone with a direct quality-scoring result (BrightRate-LM HDR video; see [methods.md](methods.md#2-controlled-evidence)).
- Equal steps are not equal compute across VLMs. Log visual tokens per sample, examples seen, wall time, and GPU-hours for each run.

## 3. Staged runs

Do not run the full grid of every option. Screen on a fixed, representative subset. Keep the best settings. Spend repeat seeds only on the confirmation stage.

| Stage | Run | Controlled change | Question | Gate to go on (proposed) |
|---|---|---|---|---|
| 0 | Anchors | Evaluate released Q-ReAlign Mini, Lite, Pro, and the [competitor set](#6-first-competitor-set) with no weight change | Does our pipeline reproduce the expected order on locked manifests? | Manifests, processor settings, label-token check, score direction, and decode/cache path logged. Released Lite/Pro close to README values (after order conversion). |
| 1 | R1 recipe control | Qwen3.5-4B, released recipe, LoRA pilot budget | What does our stack reach at pilot cost? | Stable loss; no task loader failures; R1 not far below released Lite on validation. |
| 1 | R3 backbone screen | Gemma 4 E4B, same data, loss, and visual budget | Does the newer model help after task tuning? | Better validation correlation, or equal correlation at clearly lower cost. |
| 2 | R2 / R4 supervision | Hard bins vs soft distributions + fidelity loss | Does keeping rating spread help without longer output? | Gain on most tasks on locked validation; no task drops outside its bootstrap interval. |
| 2 | Task losses | Add within-dataset ranking loss for IAA; task-balanced batches (TATAR idea) | Does AVA stop pulling on authentic IQA? | LIVE Challenge and AVA both hold or improve (see [Q-Align Table 7 note](data-mixture-evidence.md#21-q-align-one-align-task-mixing)). |
| 3 | Spatial input | Lossless image path; global view + a few native-resolution crops, some clean | Does preprocessing remove low-level cues? | Gain holds when extra visual tokens are reported or matched. |
| 3 | Temporal input | 8 uniform frames vs short contiguous clips at the same token budget | Can the scorer catch flicker, stutter, and short defects? | Better LSVQ / cross-set VQA and temporal diagnostics; no aesthetics loss. |
| 4 | Compact student | MiniCPM-V-4.6 with the best recipe; 4x vs 16x visual compression | Can a 1.3B model keep fine-detail sensitivity? | Holds on blur, noise, compression, and temporal defects, not only semantic content. Compare with Q-ReAlign Mini. |
| 5 | Data expansion | Add one dataset family at a time (PIPAL, DQ-495K auxiliary, target video slices) | Which data fixes a measured failure? | No overlap with val/test; per-task gains outweigh regressions ([evidence](data-mixture-evidence.md#8-what-this-means-for-a-q-realign-re-train)). |
| 6 | Teacher distillation | Offline teacher (Q-ReAlign Pro or Qwen3.8-27B tuned) gives 5-level distributions; student mixes them with human MOS | Does a teacher distribution help the one-pass scorer (Z-Reward idea)? | Student gains on cross-sets; teacher labels audited on a human-labeled subset. |
| 7 | RL (optional) | GRPO-style fidelity or ranking reward on the best supervised model | Is RL worth its cost here? | Only if stages 2-6 leave a clear gap. Budget reference: Q-Ponder RL took 3 days on 8 A100 ([paper](https://arxiv.org/abs/2506.05384)). |
| 8 | Confirmation | Combine only useful parts; 3+ seeds; repeat content splits | Is the combined gain robust? | Paired bootstrap intervals exclude zero on the main tasks; held-out domains reported. |

## 4. Controls

- **Same data, same exposure.** Fixed manifests, same sampling weights, same number of examples seen. Report tokens seen as well as steps.
- **Same visual budget.** Cap max pixels and frames per sample. When a run adds crops or frames, also run a matched-token variant.
- **Same scorer.** Same 5 words, same prompt stem, same expectation formula. Check that each word is a single, distinct token at the answer position for every backbone ([protocol audit, row 7](../evaluation/protocol.md#8-q-realign-baseline-code-audit)).
- **Same precision.** Start in BF16. If QLoRA is needed for a large model, compare it to a BF16 reference on subtle-quality pairs and calibration.
- **Continue-tuning control.** Fine-tune released Q-ReAlign Lite with the new recipe. This separates "new recipe" from "re-train from a base VLM".
- **Random-subset control** for any data-selection method (same count as the selected set).
- **Dedupe first.** Hash and perceptual-hash all training media against all val and test media ([split rules](../evaluation/protocol.md#3-split-rules)).

## 5. Evaluation gates

- Use the locked suite in [benchmarks.md](../evaluation/benchmarks.md#7-proposed-locked-suite-for-this-project). Report IQA, IAA, and VQA in separate tables.
- Select checkpoints on validation data only. Never tune on cross-dataset tests ([model selection](../evaluation/protocol.md#4-model-selection-seeds-and-repeated-runs)).
- Report raw PLCC, SRCC, KRCC, and `n` per set. Name any logistic-mapped PLCC as such ([raw vs mapped](../evaluation/protocol.md#2-raw-plcc-vs-logistic-mapped-plcc)).
- A claimed gain needs a paired bootstrap interval over source units that excludes zero ([intervals](../evaluation/protocol.md#5-confidence-intervals)), at least 3 seeds, and a list of every regression.
- A better macro average alone is not a pass. A drop on any in-domain task outside its interval blocks promotion until explained.
- Also log latency, peak VRAM, and parse or scoring failures (the scorer drops failed items silently; audit row 8).

## 6. First competitor set

Run these next to frozen Q-ReAlign Mini, Lite, and Pro on the same locked manifests. Licenses, checkpoints, and published numbers are in the model docs. Do not copy paper numbers into our tables; re-score.

| Task | Model | Why | Caveat | Details |
|---|---|---|---|---|
| IQA | DeQA-Score-Mix3 | Closest control for soft distributions vs hard 5-word targets | Trained on KonIQ + SPAQ + KADID: those are in-domain for it | [models/iqa.md](../models/iqa.md#4-shortlist-for-the-first-comparison) |
| IQA | TOPIQ-NR | Cheap specialist; tests whether a big VLM is worth the cost | pyiqa code and weights are non-commercial | [models/iqa.md](../models/iqa.md#4-shortlist-for-the-first-comparison) |
| IQA | SigLIP2 + AGM | Compact, recent, MIT | Released runs cover CLIVE and KonIQ only | [models/iqa.md](../models/iqa.md#4-shortlist-for-the-first-comparison) |
| IAA | ArtiMuse-AVA | Recent MLLM aesthetics specialist | One checkpoint per dataset | [models/iaa.md](../models/iaa.md) |
| IAA | MUSIQ-AVA | Small aesthetics control | Checkpoint license not stated | [models/iaa.md](../models/iaa.md) |
| VQA | DOVER | Fast video specialist; technical vs aesthetic views | Default checkpoint is LSVQ-trained | [models/vqa.md](../models/vqa.md) |
| VQA | FineVQ-LSVQ | Fine-grained LMM video assessor | FineVD-split results do not apply to the LSVQ checkpoint | [models/vqa.md](../models/vqa.md) |
| Unified | Q-Align OneAlign | Original recipe on mPLUG-Owl2 | Code non-commercial; README baseline row differs from paper | [unified models](../models/unified-and-backbones.md) |

Second wave, if budget allows: Q-Insight and RALI (reasoning teacher and cheap student), VQ-Insight natural-video (temporal reasoning), and MR-IQA-2 (shares the Qwen3.5-4B base). See [models/iqa.md](../models/iqa.md#4-shortlist-for-the-first-comparison).

## 7. Compute assumptions

All GPU sizes below are **estimates for planning, not measurements**. Memory depends on crops, frames, sequence length, batch size, trainable vision blocks, optimizer, and kernels. Stored parameter counts are from the HF metadata in [unified models](../models/unified-and-backbones.md). BF16 weights = stored params x 2 bytes.

| Candidate | Stored params | BF16 weights | Role | Pilot budget (estimate) |
|---|---:|---:|---|---|
| Qwen3.5-4B | 4.66B | ~9.3 GB | Recipe control (Lite base) | 48 GB, LoRA |
| Gemma 4 E4B | 8.00B ("4.5B effective") | ~16.0 GB | New-architecture screen | 48 GB LoRA; 80-96 GB for many frames or native crops |
| MiniCPM-V-4.6 | 1.30B | ~2.6 GB | Compact student | 24 GB LoRA; 48 GB broader tuning |
| Q-ReAlign Pro (Qwen3.5-9B) | 9.41B | ~18.8 GB | Continue-tuning control; teacher | 48 GB LoRA; full tune needs sharding or several GPUs |
| Qwen3.8-27B | 27.78B | ~55.6 GB | Offline teacher or late challenger | 80-96 GB constrained LoRA or larger; not a first run |

Planning notes:

- Full-parameter Adam tuning stores weights, gradients, and two moments (often FP32 master weights too). That is several times the BF16 weight size before activations. The released recipe is a full tune on 2 GPUs (GPU type not stated).
- Start with BF16 LoRA, micro-batch 1-2 with accumulation, activation checkpointing, and a visual-token cap. Measure rank 16, 32, and 64; none is a proven best. BrightRate-LM used rank 16 ([repo](https://github.com/shreshthsaini/BrightRate-LM)).
- First tune language adapters and projector with a frozen vision tower. Then test partial vision tuning at a lower learning rate if low-level defects stay weak. Gemma 4 12B is encoder-free, so this plan does not apply to it.
- Before renting for days, profile 100-300 mixed image/video steps and the longest sample. Record peak allocated and reserved VRAM, examples per second, tokens, GPU use, data-loader time, and checkpoint size. Estimate the full run from measured throughput plus eval, save, and restart time.

Published compute for reference (author-reported; different models and data):

| Work | Compute | Source |
|---|---|---|
| Q-Align / OneAlign (mPLUG-Owl2) | 4x A100 80G; 2 epochs; batch 64 (128 with AVA) | [Q-Align section 4.1](https://arxiv.org/html/2312.17090) |
| VisualQuality-R1 (Qwen2.5-VL-7B GRPO) | 16x A100, about 5 h for 10 epochs on KADID | [paper section 4.1](https://arxiv.org/html/2505.14460) |
| Q-Ponder (7B, SFT + RL) | RL stage: 3 days on 8x A100 | [paper](https://arxiv.org/abs/2506.05384) |
| VQ-Insight (Qwen2.5-VL) | 8x A100 80G | [paper](https://arxiv.org/html/2506.18564) |
| TATAR (Qwen2.5-VL-7B) | 8x A800 | [paper](https://arxiv.org/html/2603.19779v1) |
| Q-SiT | ~720 A800 GPU-h for the 0.5B mix-ratio search; ~72 A800 GPU-h for the 7B model | [paper](https://arxiv.org/html/2503.09197v1) |
| RED-20k (7B IAA) | 572 H200 GPU-h for SFT + RL; 2,220 H200 GPU-h for data construction | [Table 1](https://arxiv.org/html/2606.05778v3) |

## 8. Records to keep

- Pinned revisions of code, model, processor, tokenizer, and datasets. Dataset terms and access dates ([IQA](../datasets/iqa.md), [IAA](../datasets/iaa.md), [VQA](../datasets/vqa.md) data docs).
- Source-grouped manifests with hashes, dedupe logs, and the rating transform per dataset (score direction, range, bins).
- Full training config, hardware, runtime versions, wall time, and GPU-hours.
- Raw per-item predictions and level probabilities for every eval set, so metrics can be recomputed.
- Per-task and cross-domain results, including regressions. An average gain alone does not show a better unified judge.
