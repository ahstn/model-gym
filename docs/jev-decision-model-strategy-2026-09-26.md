# Open Jev-style decision model: trials, experiments and hardware

Date: 2026-09-26

This document is the result of a multi-agent research run. Three model viewpoints did the work: a balanced default agent, GPT-6-Astra (ambitious, contrarian) and GPT-6-Sol (pragmatic, cost-first). Each wrote an independent plan. Then each model red-teamed the others, and separate agents checked the load-bearing facts against primary sources. All evidence is in [jev-decision-model-2026-09-26/](jev-decision-model-2026-09-26/):

- `brief.md`: the input brief.
- `research/`: 17 source notes (Jev, leaderboard, datasets, 5 competitors, papers, base models, hardware).
- `plans/`: the three independent plans.
- `critique/`: two cross-critiques, three fact checks and a completeness check.

No model was trained or evaluated for this document. All times and costs are estimates unless a line says "measured".

## 1. Decision summary

1. **The target moved.** The live board is Decision Index **v0.2.1**, not the v0.2 screenshot. Jev 57.89, Rune v3 57.44, AutoJev-27B 56.40, Decider-chat (stock Qwen3.6-27B, no training) 51.35, Winnow-12B 50.02, Lux 43.49, Decider 4B 40.70, Decider 2B 28.97. The public harness runs only edition 0.2. See [critique/verify-board.md](jev-decision-model-2026-09-26/critique/verify-board.md).
2. **No new architecture wins here.** The winners use a strong pretrained decoder, a one-pass option-code logit readout that covers all options (up to 255), clean data with checkable labels, and a held-out temperature. None of the DeepSeek, NSA or linear-attention ideas pays off for inputs under about 4K tokens.
3. **Data recipe is worth more than base size.** Winnow-12B and Jev-Omni use the same base and differ by about 9.5 points. Nimble 9B reached 39.57 with 4,926 rows. Xor lost more than 10 points because its 26-option cap left 12.09% of decisions unanswered.
4. **Measure the size frontier first, then choose the tier.** Trial 1 scores stock 2B, 4B, 12B and 27B models on our own dev panel. It runs on the 3090 at no cost. A matched 12B LoRA pilot costs about $5–10, so we run it early (trial 5). The data selects the ship tier. We do not assume it.
5. **The 3090 does trials 1–4 and 6–7.** Rent only for the 12B pilot, teacher labels at scale, and the board-parity suite run. The expected total GPU rental is **$30–120**. Set a cap of **$250**. A 27B full fine-tune is not authorized.
6. **Do not train on Jev outputs.** A secondary source says the TypeSafe API terms forbid distillation to train a competitor. We could not get the primary terms document. Do not use SargeDev's Jev-labelled streams until someone reads the actual terms.

## 2. What the research found

### 2.1 Jev (reference)

Source: [research/jev-reference.md](jev-decision-model-2026-09-26/research/jev-reference.md).

- **Interface:** `POST /v1/systemone` with `{state, model, questions}`. The model never sees question ids.
- **Question types:**
  - Noul: returns one probability.
  - Choice: up to 255 options.
  - Score: 2–10 ordered levels, returns an expected level.
- **Confidence is a fixed formula, not a learned output.** For Choice it is `(p_max − 1/K)/(1 − 1/K)`.
- **Limits and price:** 64K tokens per request, $0.042 per million input tokens, 70–500 ms latency.
- **Training claims:** "new architecture + parallel sampler", RLCD training on a pretrained model, all-synthetic data. The base model, readout and RLCD details are not disclosed.
- **Outside probes:** questions are isolated from each other, options are order-sensitive, and MMLU ECE is 0.031. TypeSafe itself documents weaknesses with negation, counting, dates, indirection and injection. These are places where we can beat Jev.

### 2.2 Leaderboard

Source: [research/leaderboard.md](jev-decision-model-2026-09-26/research/leaderboard.md).

- **Index formula:** chance-corrected skill per task, then weighted by area. Knowledge and language 0.258 each, retrieval 0.200, tools 0.183, arts 0.100. Gold tasks count ×1.2.
- **Calibration and latency do not count in the index.**
- **Open models lead on 29 of 43 displayed tasks.** Jev keeps GPQA, MMLU-Pro, BBH, HLE, API-Bank, ANLI and WinoGrande.
- **The 27B-vs-4B gap comes mostly from a few tasks:** Home appliances (structured exactness), RAGTruth, CRUXEval, HoVer, API-Bank and BBH. Target these task types in the training data.
- **Harness:** [apolinario/decision-index](https://github.com/apolinario/decision-index) at commit `19ad28e` supports editions 0.1 and 0.2 only. The Space publishes per-benchmark totals, not per-request rows. Building the suite needs about 7 GB of downloads and 17 GB of working space. It also needs acceptance of the HLE terms.
- **Treat the board as a final exam.** It is not a development set. Several entries have checkpoint-selection or training exposure. AutoJev selected checkpoints on WinoGrande, BBH and RAGTruth rows.

### 2.3 Competitors: what worked

| Entry (v0.2.1) | Recipe | Lesson for us |
|---|---|---|
| AutoJev-27B, 56.40, ECE 1.78 pts | Qwen3.8-27B full FT, LR 2e-6, about 51k rows seen (update 200 of 286), 8K context, 255-code readout initialized from LM-head rows, scalar T = 2.21 | A good recipe on modest private data. Full FT of a 27B needs about 450–490 GB of weight and optimizer state. Its $3.1k cost was agents and data, not GPU. [research](jev-decision-model-2026-09-26/research/competitor-autojev.md) |
| Rune v3 26B-A4B, 57.44, ECE 11.96 pts | Gemma 4 26B-A4B. v2 used rank-8 LoRA self-distillation on 241k rows. The v3 method is not disclosed | High index but poor calibration. Not a good teacher. [research](jev-decision-model-2026-09-26/research/competitor-rune.md) |
| Decider-chat, 51.35, ECE 2.09 pts | Stock Qwen3.6-27B, softmax over answer-letter logits, T = 1.943 | **Zero training beats every LoRA entry.** Every fine-tune must beat its own stock base. [research](jev-decision-model-2026-09-26/research/competitor-decider.md) |
| Decider 2B / 4B / 35B-A3B | Qwen3.5 Base, full FT, 1 epoch, LR 1e-5; per-type temperatures. RL raised browser success from 83% to 93% but lowered OpenJev by 0.8 | The Apache-2.0 harness, readout and about 95 dataset builders are reusable. Skip RL at first. |
| Winnow-12B, 50.02, ECE 16.79 pts | LoRA r32 on gemma-4-12B-it; contrastive fact-flip data; gold CE plus teacher soft labels (only when the teacher agrees with gold); no calibration | 12B LoRA can reach about 50. Adding a temperature would fix its ECE. [research](jev-decision-model-2026-09-26/research/competitor-winnow.md) |
| Xor 35B-A3B, 41.48, ECE 1.49 pts | Full FT, forward plus reversed option order, 26-option cap | Coverage matters: never cap options. The full FT added almost nothing over a stock 27B plus readout. [research](jev-decision-model-2026-09-26/research/competitor-xor.md) |
| Diffusion entries (diffgemma 26B) | One-step denoise readout | Fast, but ECE is 0.20–0.23. Do not use this readout. |

Cross-entry analysis: [research/top-vs-mid.md](jev-decision-model-2026-09-26/research/top-vs-mid.md).

### 2.4 Datasets

Source: [research/datasets.md](jev-decision-model-2026-09-26/research/datasets.md).

| Dataset | Size and content | Verdict |
|---|---|---|
| tasksource/tasksource-jev-typed-decisions | 2.5M rows, 667 human-labelled sources, per-row `license_use` | **Core source.** Use rows with `license_use == commercial` only. Drop every source that is also a suite benchmark: WinoGrande, HellaSwag, ANLI, BANKING77, CLINC, ARC, HoVer, CLadder, ContractNLI, NLI4CT, ESCI, CSQA, OpenBookQA and sarcasm sources. Cap each source. |
| ZefanCai/Open-Jev-v1.1 | 147k train rows, 56% WANLI, rule-generated structured states | **Use the non-WANLI part.** Cap WANLI at ≤5%, because WANLI is MNLI-seeded and ANLI is in the suite. Keep `group_id` splits. |
| SargeDev/jev-distill-corpus-v3 | 741k rows. 498k have Jev soft labels on short templated states with ≤16 options. 148k are labelled by an unnamed 32B model. 95k are a copy of Open-Jev v2 | **Exclude from the core recipe.** The Jev output rights are unresolved and the states have low diversity. The CC0 `openjev_v2` stream duplicates Open-Jev. |

Known contaminated corpora to avoid: pngwn typed-decisions (MMLU-Pro test), the JevK5 data, and the FrontiersMind kevv datasets (BANKING77 test).

### 2.5 Papers: verdicts

Sources: [papers-deepseek-nsa.md](jev-decision-model-2026-09-26/research/papers-deepseek-nsa.md), [papers-hybrids.md](jev-decision-model-2026-09-26/research/papers-hybrids.md), [papers-classification.md](jev-decision-model-2026-09-26/research/papers-classification.md) and [papers-sweep.md](jev-decision-model-2026-09-26/research/papers-sweep.md). All numbers in the brief were checked. The table lists the corrections.

| Paper | Verdict | Why / what we take |
|---|---|---|
| DeepSeek-V4.1-Flash (CED) | **Conditional, late** | It is a 20+20-layer causal encoder-decoder, not "shallow encoder, deep decoder". The only cheap transfer is layer truncation or an early readout at 0.75L with LoRA healing. Try it only if measured prefill time dominates latency. |
| DeepSeek-V4 (CSA/HCA) | Reject | It needs pretraining-scale work. It gives no gain under about 4K tokens. |
| FlashMemory-DeepSeek-V4 | Reject | The indexer is rank 2048, not small. MRCR falls from 76.0 to 48.0 on multi-evidence tasks. |
| Native Sparse Attention | Reject | The 9×/6× speedups are at 64K context. It needs pretraining. |
| LoLCATs | Reject | Short inputs give no gain. Keep one lesson: conversion on Alpaca data gave 0% passkey retrieval, and on passkey data gave 100%. Distil only on data in our decision format. |
| Implicit Hybrids | Reject as training work | Keep it only as an optional 2–3 h head-attribution diagnostic if layer truncation loses dispersed-evidence accuracy. |
| Sparse Frontier | **Adopt for evaluation** | Build a slice suite over length, evidence count, dispersion and aggregation. Our own axes add label count, negation and label order; these axes are not in the paper. Gate every trial on its worst slice. |
| On-Demand Attention | **Conditional (cascade)** | Its learned gate beats entropy (AUROC 0.643 vs 0.486). Use its target, "gold-NLL reduction from the big model", for a small→large router, but only if a large model wins on quality and fails on latency. |
| Limits of Speculation | Reject for decoding | We decode 0 tokens. Keep the idea of "gain per unit of added cost" for the router rule. |
| GLiClass | **One control arm** | Joint text-label encoder, all labels in one pass. The default output is a multi-label sigmoid, so Choice needs our own softmax. Quality drops with dense label sets. The encoder tier best on the board is only 14.32. |
| Calibration-Aware RL | Defer | The 92.79→93.70 accuracy gain is on an out-of-domain set. On Qwen3-4B, plain SFT has better ECE in 3 of 4 cells. SFT plus a temperature is enough at first. |
| DHRD (train-time reasoning) | Defer | The gain on Qwen3-4B is only +0.65%, from single runs with no calibration reported. |
| Sweep extras | **Adopt two** | (1) Soft-label KD over the full candidate set; never renormalize over a top-K subset. (2) Compare hard, soft and mixed supervision before buying more synthetic data. |

### 2.6 Base models

Sources: [research/base-models.md](jev-decision-model-2026-09-26/research/base-models.md) and [critique/verify-models-data.md](jev-decision-model-2026-09-26/critique/verify-models-data.md).

| Base | Facts (checked) | Role |
|---|---|---|
| **MiniCPM5-2B** | 2.517B parameters, `LlamaForCausalLM`, Apache-2.0, 131K context. Official TRL recipe: BF16 LoRA r16/α32, 2K sequences. Needs **transformers 5.6.2** (the repo pins 4.57.6) | Harness shakedown and fast tier. Lowest tooling risk. |
| **Qwen3.5-4B** (Base and instruct) | 4.66B parameters, hybrid: 24 Gated DeltaNet layers and 8 attention layers. Unsloth says **do not use QLoRA** on it. BF16 LoRA needs about 10 GB | Main local candidate. Question isolation needs a copy of the recurrent state or one row per question. A tree mask does not work. |
| **K2-Horizon-3.7B** | 5.06B stored parameters (3.78B core), Apache-2.0, needs `trust_remote_code` | Inference-only row in trial 1. Train it only if it wins by ≥2 points. |
| **gemma-4-12B-it** | 11.96B parameters, Apache-2.0 | 12B challenger. Q8 inference fits on the 3090. BF16 LoRA needs a rented GPU. |
| Qwen3.6-27B / Qwen3.8-27B | 27.78B parameters, Apache-2.0. Q4_K_M is 16.5 GB and fits on the 3090 | Stock reference and teacher. Not a training target. |
| ModernBERT-large / gliclass-instruct-large | About 0.4B parameters | One encoder control arm. |

## 3. How the three viewpoints differed, and the resolution

| Topic | Balanced (task) | Astra | Sol | Resolution |
|---|---|---|---|---|
| Primary tier | 4B primary, 2B distilled | 2B + 12B; later a 12B primary | 2B primary, 4B conditional | **Decide with data.** Stock frontier in trial 1 (local, $0), matched LoRA at 2B/4B, then an early 12B pilot (about $5–10). Promote the smallest model that passes the gates. Astra's point stands: gating 12B behind a 4B success is backwards when the pilot is this cheap. |
| Teacher | Mean of stock Qwen3.6-27B and AutoJev | Qwen3.8-27B in reasoning mode, plus a Gemma cross-check | Jev soft labels (if the terms allow) | **One stock teacher: Qwen3.6-27B with the Decider readout.** Audit AutoJev and Qwen3.8 reasoning mode on 1k adjudicated rows. Select by gold NLL. Use Gemma-12B as a second-family check on disagreements. Do not average two Qwen-family teachers by default, because their errors correlate. |
| Jev-distilled data | Internal ablation only | Optional 10%, after the terms are checked | 25% of the core mix | **Exclude it.** Internal use does not make the rights question go away. |
| Eval gate | Proxy regression against ≥10 board models | Pinned 0.2, sealed panels | Pinned 0.2, source-held-out panels | **No proxy-index gate.** Select on dev task-macro NLL, accuracy and safety cost. Run the pinned official 0.2 suite once per frozen release. Label any 0.2.1 reconstruction as a reconstruction. |
| Custom readout | Letter slot vs pointer head | Permutation-equivariant candidate-set head | Letter slot | **Letter slot by default.** Try Astra's equivariant head only as a 2B pilot, and only if shuffle augmentation still leaves ≥3 points of order sensitivity. |
| Full FT | 2B local, 4B on A100 | 27B full FT on 8×H100 if earned | None | **LoRA only at first.** A 2B full FT fits only with pure-BF16 weights and 8-bit Adam (about 17–19 GB). With FP32 master weights it needs 25–40 GB. A 27B full FT is out of scope. |
| Budget (GPU only) | $35 / $150–230 / $650–900 | $250 / $700 / $2,500 (all-in) | $0 / $100 / $250 | **Cap $250 GPU; expect $30–120.** Approve annotation and API spend separately. |

Two other views came out of the critique. Record both:

- **"Do not train."** A stock 27B with a readout and a temperature already scores 51.35. If the product can accept a 27B server, trial 1 answers the question. On a rented PRO 6000 that costs about $1.2/h to serve.
- **"The encoder is enough for command risk."** This stays as the fallback. The existing ModernBERT ONNX classifier is kept as a separate product.

## 4. Evaluation protocol (build before any training)

1. **Harness:** the separate uv project [`decision/`](../decision/README.md) holds the readout, evaluate and calibrate code. Keep our metrics code.
2. **Board runner:** apolinario/decision-index at commit `19ad28e`, edition 0.2, through a `/v1/systemone` HTTP engine. Check the suite hashes that the kit publishes.
3. **Splits (freeze once, split by source task and generator family, not by row):**
   - dev: all trial decisions
   - calibration A: fit temperatures
   - calibration B: choose the calibration map; it counts as dev after that
   - sealed transfer panel: opened once per release
   - sealed command-risk panel: ≥500 dangerous groups; with fewer, a zero-error result still has a 95% upper bound of about 1.5%
4. **Decontamination:**
   - Rebuild the suite rows.
   - Drop whole suite-source datasets.
   - Drop exact matches and 13-gram overlaps against the suite and the sealed panels.
   - Treat MinHash (5-word, Jaccard ≥ 0.8) matches as near-duplicates and review them.
   - Log every dropped row.
5. **Slice suite (template-built gold):**
   - input length 128 / 512 / 2K / 4K / 8K, with evidence at the start, middle or end
   - K = 2 / 8 / 32 / 128 / 255 options
   - evidence count 1 / 2 / 4 / 8, contiguous or dispersed
   - negation and exception minimal pairs
   - counting and aggregation
   - label order and description paraphrase
   - sibling-question isolation and injected text in the state
   - Choice vs equivalent Noul, where the logic proves they are equivalent
   - a 200-item non-English slice, for monitoring only (v1 is English-only)
6. **Metrics:**
   - task-macro accuracy with a group bootstrap CI
   - NLL (primary selection metric), multiclass Brier, ECE per type, AURC
   - ≥95%-confident-and-wrong share
   - unanswered %
   - permutation agreement (target ≥99%) and remapped total-variation distance (≤0.01)
   - worst-slice accuracy
   - command-risk unsafe-allow rate at a matched benign-intervention rate
   - latency p50/p95 including prefill at 512 / 2K / 8K tokens, with 1 / 8 / 32 questions per state

## 5. Experiment ladder

"Local" means the RTX 3090 with $0 rental. Rental prices are live quotes from 2026-09-26: see [critique/verify-hardware.md](jev-decision-model-2026-09-26/critique/verify-hardware.md). Run a 100-step warmed pilot before each full job, then re-cost the job from measured tokens/s.

| # | Question | Setup | Gate to promote | Hardware and cost |
|---|---|---|---|---|
| **T0** | Does the contract work? | Typed schema, 255-code readout check in context (MiniCPM5, Qwen3.5 and K2 all have ≥255 single-token code candidates; recheck after the chat prefix), isolated Score levels, sibling isolation, explicit `Unsupported` for inputs that are too long | 100% schema coverage up to K = 255. Sibling drift ≤1e-5 (FP32) or ≤1e-3 (BF16) | Local, about 1 day of engineering |
| **T1** | Where is the stock frontier? | Stock MiniCPM5-2B, Qwen3.5-4B Base and instruct, K2 (inference only), gemma-4-12B-it at Q8, Qwen3.6-27B at Q4 with the Decider readout. External controls: the released Decider 2B/4B. One temperature per type | Freeze the readout. The size gaps set the tier plan. Keep 12B in play if it beats 4B by ≥3 points of accuracy or ≥5% NLL | Local, about 25 GPU-h, $0 |
| **T2** | Does SFT on clean data help at 2B? | MiniCPM5-2B BF16 LoRA r16 (plus one r64 arm), candidate-slot CE only. 10k-row pilot, then 40k rows. Mix: 40% tasksource (clean), 25% Open-Jev non-WANLI, 30% executable-gold contrastive families, 5% command-risk as Score | ≥2 points accuracy or ≥5% NLL over the same stock base. No worst-slice loss >2 points | Local, 15–30 GPU-h, $0 |
| **T3** | Does 4B add capacity? | Qwen3.5-4B BF16 LoRA, same 40k rows (Base vs instruct only if their stock scores are within 2 points) | ≥2 points or ≥5% NLL over the T2 winner, at an acceptable measured latency | Local, 25–55 GPU-h, $0 |
| **T4** | Is one teacher worth it? | Audit Qwen3.6-27B (Q4 locally, or BF16 on a rented GPU) against AutoJev and Qwen3.8 reasoning mode on 1k adjudicated rows. Then run a matched comparison on the winning base: hard CE vs 0.5 CE + 0.5 KL over 5–10k rows | ≥3% relative gold-NLL gain, ≤0.5 points accuracy loss, no safety loss | Audit local. Full labels: rented H100 SXM, about 3.7 h, **$6–10** |
| **T5** | Does 12B beat 4B on the same data? | gemma-4-12B-it BF16 LoRA r32, the same 40k rows, stock BF16 and Q8 controls | ≥3 points or ≥5% NLL over the best 4B, and the Q8 local deployment keeps the gain | Rented A100-80 SXM (Vast, about $4) or PRO 6000 (Lium, about $8) per 40M tokens |
| **T6** | Does robustness need new structure? | Shuffle, description and minimal-pair augmentation vs equal-token replay. Try the equivariant candidate-set head (2B pilot) only if ≥3 points of order sensitivity remain | Permutation agreement ≥99%, TV ≤0.01, ≥3 points robust-dev gain, ≤1 point transfer loss, K32 p95 ≤1.5× baseline | Local, 10–30 GPU-h |
| **T7** | Scale the winner | Expand to 120–200k rows. Add a 2–8K long-context cohort and many-label cases. Target the gap tasks: structured exactness, hallucination, code effects, multi-hop and tool selection. Replay with KL to the parent | ≥2 points over T2/T3/T5 on dev. No Brier regression >0.01 on any type | 4B local (about 30–60 h), or 12B rented (about $15–40) |
| **T8** | Release check | Freeze weights, prompt, precision and temperature. Run the sealed panels once, then the full pinned 0.2 suite once. Add seeds for the finalist and one encoder control (GLiClass/ModernBERT) | All gates pass. Report official 0.2 and any 0.2.1 reconstruction separately. Publish failures too | Suite on Lium PRO 6000 Server: 4B about $3, 12B about $8, 27B about $19. Use RunPod Secure ($2.09/h) for the published run |

Deferred, and run only on a trigger:

- small→large router (On-Demand Attention target): only if a large model wins on quality and fails the latency target
- 0.75L layer truncation: only if measured prefill time dominates
- calibration-aware RL and DHRD: only after SFT stops improving
- 2B distillation from the winner

## 6. Hardware decision

### 6.1 The local RTX 3090

- **Fix first:** the loaded kernel module is 595.84 and the userspace driver is 595.91.07, after an apt upgrade today. `nvidia-smi` and anything that uses NVML fail, and vLLM probably fails too. PyTorch CUDA still works. Reboot, then run `apt-mark hold nvidia-driver-595-open libnvidia-compute-595` for the duration of the program. See [research/hw-3090.md](jev-decision-model-2026-09-26/research/hw-3090.md).
- **Environment:** add a separate `decision` extra or environment with transformers 5.x, peft, trl and vLLM. Do not change the ModernBERT and ONNX export environment.
- **It fits:**
  - 2B and 4B BF16 LoRA at 2K–4K tokens
  - stock inference up to 12B (Q8) and 27B (Q4)
  - encoders
  - calibration and routers
- **It does not fit:**
  - 12B BF16 LoRA (QLoRA fits, but is slow and does not match the served precision)
  - any 27B training
  - any MoE training
  - BF16 27B teacher labels at scale
- **Throughput (estimated):** 2B LoRA about 2.5k tok/s and 4B about 1.25k tok/s. So 40M tokens take about 4.5 h at 2B and about 9 h at 4B, before eval and overhead. Measure this in T2.

### 6.2 Rental triggers (rent only when one is true)

1. A job needs more than 24 GB after LoRA, gradient checkpointing and batch 1.
2. A local job is forecast at more than 48 h, and a rental gives ≥2× more completed experiments per day, including setup.
3. A board-parity suite run.

A 4090 rental does not solve an out-of-memory error, because it also has 24 GB.

### 6.3 What to rent and where

| Job | First choice | Fallback | Est. cost |
|---|---|---|---|
| 12B BF16 LoRA (40M tokens) | Vast A100-80 SXM (p10 $0.53/h, filter for reliability) | Lium PRO 6000 Server $1.19–1.29/h; RunPod A100 Community $1.19/h | $4–10 |
| 27B teacher labels (about 90M prefill tokens) | Vast or Lium H100 SXM ($1.69–1.73/h) | Lium PRO 6000 | $6–15 |
| Board-parity suite eval | Lium RTX PRO 6000 **Server Edition** (the board uses this card; Max-Q and Workstation cards are not latency-comparable) | Vast "RTX PRO 6000 S" ($1.20/h p10) | 4B about $3; 27B about $19 |
| Published final run | RunPod Secure PRO 6000 $2.09/h | Lium | $5–32 |

Providers not to use for this work:

- **Daytona:** it has GPUs, but the low prices are preemptible (PRO 6000 $1.74/h) and it can kill a sandbox without warning. On-demand is $3.03/h, about 2.5× Lium. The sandboxes are ephemeral.
- **Modal:** PRO 6000 is $3.03/h, plus CPU and RAM. Use it only for serverless burst inference.
- **Prime Intellect:** its front-page prices look like placeholders. Get a quote from its CLI if you need it.
- **Lambda:** $3.99/h per H100. Use it only for an 8×H100 full fine-tune, which is out of scope.
- **Buying hardware:** no evidence says we need it.

Workflow notes:

- `scripts/sync_to_server.sh` excludes `runs/` and `artifacts/`. Copy results back explicitly before you delete an instance.
- Put `HF_HOME` and checkpoints on a persistent volume. Vast bills storage while an instance is stopped. Lium loses data outside the volume on reboot.

### 6.4 Budget (GPU rental only)

| Scenario | Scope | Expected |
|---|---|---|
| Lean | T0–T4 and T6 local; one 4B suite run | about $5–15 |
| **Recommended** | Lean plus T4 teacher labels, T5 12B pilot, T7 at 12B, and the T8 suite runs | **about $30–120, cap $250** |
| Not authorized | 27B LoRA or full FT (8×H100, about $400–800 for the full FT alone) | Needs a separate decision after T5 and T7 show a 12B gain |

## 7. Kill criteria

- **T2 and T3 both fail to beat their stock bases** (by <2 points and <0.03 NLL): stop fine-tuning. Ship the stock model with the readout and temperature as a baseline. Audit the data before any more work.
- **T5 12B gains <3 points over 4B**, or Q8 loses the gain: stay at 4B. Do not look at 27B.
- **Teacher labels improve teacher agreement but not gold NLL:** drop the teacher stream and spend on verified contrastive data.
- **Contamination found:** remove the whole source, retrain the affected stage, and disclose it.
- **After T7, no win over Jev on any predeclared robustness macro and no ≥2× local latency gain:** stop calling this a Jev replacement. Publish the recipe as a narrower decision model.

## 8. First two weeks

New code lives in the separate uv project [`decision/`](../decision/README.md). Keep the existing five-level classifier, its CLI commands and its ONNX export unchanged.

- **Day 1:**
  - Reboot and hold the driver.
  - Add the `decision` extra.
  - Accept the HLE terms and start the suite rebuild (pinned `19ad28e`).
  - Read the TypeSafe and OpenRouter terms.
- **Days 2–3:**
  - `decision/schema.py`: typed Choice/Score/Noul, the confidence formulas, the 255/10-level limits.
  - `decision/readout.py`: per-tokenizer code validation in context, masked softmax / T.
  - `decision/engine.py`: one-pass scorer with isolated questions; recurrent-state copy for Qwen3.5.
- **Days 3–4:**
  - `decision/data.py`: three dataset adapters, license allowlist, suite blocklist, group splits, hash and MinHash dedup, frozen manifests.
  - `decision/probes.py`: executable-gold families and the slice suite.
- **Day 5:**
  - `decision/evaluate.py` and `decision/calibration.py`: the metrics in §4, per-type T.
  - `decision serve`: HTTP `/v1/systemone`.
  - Run T0.
- **Days 6–7:** run T1 (stock frontier) and profile latency and VRAM.
- **Days 8–10:** `decision/train.py` (candidate CE/KL, LoRA, token accounting). Run the T2 pilot, then 40k rows.
- **Days 11–12:** run T3. Do the T4 teacher audit locally.
- **Days 13–14:** first rental: T5 12B pilot on an A100 or PRO 6000. Decide the tier.

Realistic calendar: T0–T5 take about three weeks of mixed engineering and GPU time. The critiques showed that the three plans' two-week schedules underestimated the GPU hours.

## 9. Open items

- **TypeSafe terms for distillation:** only a secondary source says they forbid it. Read the primary terms before using any Jev-labelled rows.
- **vLLM scoring:** check that vLLM can return logits for 255 arbitrary codes at one position without generating. Also check the copy of the Qwen3.5 DeltaNet state. Start from Decider's `serve_vllm`.
- **Suite splits:** check which splits the suite uses for CLadder, ContractNLI, NLI4CT and VAST.
- **Exact overlap check:** run the item-level overlap check between tasksource and the suite rows on a local parquet copy (about 1.45 GB).
- **Lium storage and egress:** the prices are not published. Get a quote before the first rental.
