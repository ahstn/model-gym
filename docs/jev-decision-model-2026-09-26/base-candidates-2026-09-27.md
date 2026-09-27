# Base candidates, techniques and 12B fit (2026-09-27)

Context: [results for MiniCPM5-2B LoRA](results-minicpm5-2b-lora.md) (our 0.2 index 29.38, stock 18.91). Sources: HF API (`/api/models/<id>`), model cards, the live board data (`multimodalart-jev-decision-index.static.hf.space/data/index.json`, generated 2026-09-27 03:42 UTC, edition release-v2.1), and one measurement on our pod.

## 1. Does the RTX PRO 6000 (96 GB) work for 9–12B?

Yes, for bf16 LoRA. Measured on the pod with `google/gemma-4-12B-it` (11.96B), SDPA attention, 2,048-token rows, gradient checkpointing, AdamW, all 7 projections of the language model:

| Mode | Batch (rows × 2,048 tok) | Throughput | Peak memory |
|---|---:|---:|---:|
| Inference, last-position logits | 16 | 8,789 tok/s | 36.7 GiB |
| Inference, last-position logits | 32 | 8,785 tok/s | 51.1 GiB |
| LoRA r16 (66M trainable) | 8 / 16 / 32 | 1,503 / 1,485 / 1,505 tok/s | 38.8 / 54.6 / 86.1 GiB |
| LoRA r64 (262M trainable) | 8 / 16 / 32 | 1,488 / 1,468 / 1,494 tok/s | 41.1 / 56.9 / 88.4 GiB |

- Weights use 22.3 GiB. Throughput does not change with batch size, so the GPU is compute-bound at batch 8.
- R3 used 10.7M training tokens. At 1,500 tok/s a matched 12B run takes about 2 h of training, about $4 at $2.09/h. [INFERENCE] Dev evals add to this.
- Suite run: [INFERENCE] 5–10 h for a 12B LoRA (our 2B took 70 min stock and 2 h 10 min with the adapter). Merging the adapter before the suite run saves time.
- Full fine-tuning of 9–12B does not fit: bf16 weights, grads and fp32 Adam moments need 12–16 bytes per parameter, which is 108–192 GB before activations.
- MoE: `gemma-4-26B-A4B-it` has 51.6 GB of bf16 weights. Rune v3 serves it on one RTX PRO 6000. LoRA should fit with batch ≤8. [INFERENCE] We did not measure this. The experts are fused 3D tensors in transformers 5.x, so `target_modules` does not reach them; use PEFT `target_parameters` and check the trainable count.
- Disk: the 12B checkpoint uses 24 GB of the 60 GB container disk. `/root/suite-work` (18.7 GB) moved to `/workspace/backup/suite-work.tar` (verified with `tar --diff`). About 11 GB is free now.
- FlashAttention and FP8: not needed. SDPA works. Public evidence for FP8 training on sm_120 (Transformer Engine, torchao) is incomplete.

## 2. Is Qwen3.5 outdated?

Only above 27B. The HF API lists no Qwen3.6 or Qwen3.8 model below 27B. Qwen3.6 has only 27B and 35B-A3B; Qwen3.8 has only 27B, 180B (Flash-Next) and 2.4T. For 0.8–9B, Qwen3.5 is still the newest Qwen, and most small board entries use `Qwen3.5-*-Base`.

## 3. Board evidence by size (live board, balanced skill)

| Size class | Best entry | Index | Kind | Base |
|---|---|---:|---|---|
| ~2B | Decider 2B | 28.97 | full FT | Qwen3.5-2B-Base |
| ~2B (ours, 0.2 scorer) | R3 r64-40k | 29.38 | LoRA r64 | MiniCPM5-2B |
| 4–5B | Decider 4B | 40.70 | full FT | Qwen3.5-4B-Base |
| 4–5B | Hopper | 39.67 | LoRA | Qwen3.5-4B-Base |
| E4B (8B physical) | Winnow-E4B | 39.89 | LoRA | gemma-4-E4B |
| 9B | JPT-9B | 46.89 | LoRA r16 | Qwen3.5-9B-Base |
| 12B | **Winnow-12B** | **50.02** | LoRA r32 | gemma-4-12B-it |
| 12B | Jev-Omni | 40.53 | LoRA + head | gemma-4-12B |
| 26B-A4B MoE | **Rune v3** | **57.44** | full FT | gemma-4-26B-A4B-it |
| 27B | AutoJev-27B | 56.40 | full FT | Qwen3.8-27B |
| 31B, stock | Decider chat · Gemma-4-31B | 57.33 | prompt only | gemma-4-31B-it |
| 35B-A3B MoE | Decider 35B-A3B | 47.11 | full FT | Qwen3.5-35B-A3B-Base |
| reference | Jev 1.13 | 57.91 | closed | — |

- Base size explains most of the spread. A stock 31B with a good prompt (57.33) nearly matches Jev.
- At 12B, the same base gives 50.02 (Winnow) or 40.53 (Jev-Omni). The recipe still moves the score by about 10 points.
- No board entry uses MiniCPM5, K2-Horizon or Granite 4.2.
- Our 29.38 uses the 0.2 scorer. The board uses 0.2.1 (different area weights and rescored benchmarks), so the comparison is approximate.

## 4. Base candidates

All IDs verified with the HF API. Card scores are vendor numbers with different protocols (often thinking mode), so compare them only within one card.

| Candidate | Params | Architecture | Licence | Notes |
|---|---:|---|---|---|
| `google/gemma-4-12B-it` / `google/gemma-4-12B` | 11.96B | dense, sliding-window + global attention, 48 layers, vocab 262,144 | Apache-2.0 | Card: MMLU-Pro 77.2, GPQA-D 78.8, HLE 5.2. Winnow-12B 50.02. Attention-only KV, so our shared-prefix engine works. Fit measured above. Code readout needs `check-readout` (262k vocab). |
| `Qwen/Qwen3.5-9B-Base` | 9.65B | hybrid Gated DeltaNet + attention | Apache-2.0 | Card (thinking): MMLU-Pro 82.5, GPQA-D 81.7. JPT-9B 46.89. Recurrent layers: our prefix engine needs a state copy, or one row per question. |
| `google/gemma-4-E4B-it` | 8.0B (4.5B effective) | dense + per-layer embeddings | Apache-2.0 | Card: MMLU-Pro 69.4, GPQA-D 58.6. Winnow-E4B 39.89. |
| `Qwen/Qwen3.5-4B-Base` | 4.66B | hybrid DeltaNet | Apache-2.0 | Decider 4B 40.70 (full FT). Full FT fits on 96 GB [INFERENCE]. |
| `IFM/K2-Horizon-7B` | 9.00B | `K2HorizonForCausalLM`, `trust_remote_code` | Apache-2.0 | Released 2026-09-01. Card: HLE 18.6. No board entry. Also 0.9B, 3.7B (5.06B), 32B, MoVA-36B-A4B. |
| `ibm-granite/granite-4.2-8b` | 8.79B | dense `GraniteForCausalLM` | Apache-2.0 | Released 2026-08-07. No board entry. |
| `mistralai/Ministral-3-14B-Base-2512` | 13.95B | dense + vision encoder | Apache-2.0 | Card (Reasoning variant): GPQA-D 71.2. No board entry. |
| `google/gemma-4-26B-A4B-it` | 25.81B (3.8B active) | MoE, 128 experts, 8 active | Apache-2.0 | Card: MMLU-Pro 82.6, GPQA-D 82.3. Rune v3 57.44 (full FT). |
| `google/gemma-4-31B-it` | 31.27B | dense | Apache-2.0 | Stock 57.33 with a prompt. bf16 weights about 63 GB; LoRA fits at small batch [INFERENCE]. |
| `Qwen/Qwen3.6-35B-A3B` | 35.95B (3B active) | hybrid DeltaNet MoE, 256 experts | Apache-2.0 | Card: MMLU-Pro 85.2, GPQA 86.0. Xor 41.48, so full FT on it has not beaten Gemma MoE yet. 72 GB weights: tight. |

Rejected: `LiquidAI/LFM2.5-2.6B` (its card says "not recommended for knowledge-heavy tasks"; board 6.76), `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (Mamba, custom code, NVIDIA licence), `allenai/Olmo-3-7B-Think` (card GPQA 46.2), RWKV-7 (recurrent, 16K context), DiffusionGemma (no one-position logits).

## 5. Techniques

Ranked by evidence for this task:

| Technique | Evidence | Use for us |
|---|---|---|
| Two shuffled option orders per row | JPT-9B trains on two option-shuffled copies (46.89). PA-GRPO (ACL 2026) on Qwen3-8B: consistency +10.4, GPQA +9.6. | Add a two-view consistency term (symmetric KL after remapping to option identity). Our permutation agreement is .894. |
| Selective thinking | Rune v3 card: think ≤512 tokens only when top probability <0.7 (about 1 question in 10). 0.2 index 53.39 → 54.89; GSM8K +15.2, CRUXEval +11.4, BBH +8.0; ForecastBench −9.8. | Targets our GSM8K and knowledge losses. Needs a base that can reason; costs about 5 s per gated question. |
| Brier loss over options | JPT-9B uses multi-class Brier. No controlled comparison with soft CE. | Cheap ablation against our soft CE. |
| Teacher CE only when teacher agrees with gold | Winnow-12B recipe. | Applies if we add teacher distributions. |
| Post-hoc temperature | Rune: ECE 12.5% → 2.2% at T=2, same argmax. | We already fit per-kind temperatures. |
| LoRA on all layers, LR ≈10× full-FT LR, moderate batch | "LoRA Without Regret" (Thinking Machines, 2025): attention-only is worse; LR depends little on rank; LoRA loses more at large batch. | We already target all 7 projections. Keep 64 rows per step. r16 vs r64 at 12B is a cheap check. |
| DoRA, rsLoRA, PiSSA, LoRA+, LoRA-GA | Gains in their papers, none on this task. | Low priority. |
| PriDe, content-free calibration | Letter-prior correction; tested only with 2–5 options. | Diagnostic only. |
| Liger fused CE, FP8, FA3/FA4 | Our loss uses only the code rows of `lm_head`, so fused CE does not help. FP8 and FA on sm_120 are not proven for training. | Skip. |

## 6. Effect on the approved plan

- Stock frontier (step 1): add `google/gemma-4-12B-it`, `Qwen/Qwen3.5-9B` and `IFM/K2-Horizon-7B`. Keep the MoE check on `google/gemma-4-26B-A4B-it`. Drop `Qwen/Qwen3.6-35B-A3B`: it is larger, hybrid, and its board result is lower.
- Matched LoRA (step 3): the evidence favours `gemma-4-12B-it` over a 4B base. Board: 50.02 vs 40.70. It fits, it trains in about 2 h, and our prefix engine works with it.
- Recipe fixes (step 2) stay the same. Add two-view consistency as the option-order loss.
