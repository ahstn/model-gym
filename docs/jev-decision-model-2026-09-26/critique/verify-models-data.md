# Verify: models, data, licences, 3090 memory (verify-2)

Checked 2026-09-26 against HF API/raw files, vendor docs, and ToS pages. `[INFERENCE]` marks my own arithmetic or reasoning.

## 1. openbmb/MiniCPM5-2B
| Claim | Verdict | Evidence / corrected value |
|---|---|---|
| ~2.517B params | **CONFIRMED** | safetensors total = 2,516,756,480 ([API](https://huggingface.co/api/models/openbmb/MiniCPM5-2B)) |
| Llama-compatible | **CONFIRMED** | `LlamaForCausalLM`, `model_type: llama`, 42 layers, hidden 2048, 16 heads / 2 KV heads (GQA), head_dim 128, vocab 130,560, untied embeddings ([config](https://huggingface.co/openbmb/MiniCPM5-2B/raw/main/config.json)). The config was written with transformers 5.6.2, but it is a stock Llama class, so plain PEFT/TRL apply. Unsloth works in principle through its Llama path `[INFERENCE]`. |
| Apache-2.0 | **CONFIRMED** | API `license:apache-2.0` |
| Context 131,072 | **CONFIRMED** | `max_position_embeddings: 131072`, rope_theta 5e6 |
| OpenBMB TRL recipe (BF16 LoRA r16, grad-ckpt, 2K) | **CONFIRMED** | [trl.md](https://raw.githubusercontent.com/OpenBMB/MiniCPM/main/docs/finetune/trl.md): `LoraConfig(r=16, lora_alpha=32, dropout 0.05, q/k/v/o/gate/up/down)`, `bf16=True`, `max_length=2048`, gradient checkpointing (`use_reentrant=False`), sdpa, `assistant_only_loss=True`, lr 2e-4, grad-accum 4. The recipe covers LoRA only and states no VRAM figure. |

## 2. Qwen3.5-4B / 2B
| Claim | Verdict | Evidence |
|---|---|---|
| Hybrid Gated DeltaNet + attention | **CONFIRMED** | Card: "8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))", 32 layers ([README](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/README.md)). Config `layer_types` has 24 `linear_attention` and 8 `full_attention` layers, `full_attention_interval: 4` ([config](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json)). |
| Breaks tree-mask question isolation and shared-prefix KV forking | **CONFIRMED in substance** `[INFERENCE]` | A Gated DeltaNet layer carries a recurrent state rather than a per-token KV cache. An arbitrary 2-D tree/block attention mask therefore cannot isolate sibling questions inside one packed sequence in 24 of the 32 layers. Prefix forking is still possible, but only by snapshotting and copying the recurrent state (plus the conv state) at the fork point, not by sharing KV pages. Any plan that packs N questions after one state with a tree mask must use one sequence per question on Qwen3.5 or require state-copy support in the serving engine. Decider notes that the bases need `flash-linear-attention` (research/competitor-decider.md). |
| Required transformers version | **CONFIRMED: v5** | Unsloth: "Please use `transformers v5` for Qwen3.5. Older versions will not work." ([Unsloth Qwen3.5 FT](https://unsloth.ai/docs/models/qwen3.5/fine-tune.md)). The local transformers 4.57.6 in hw-3090.md is therefore insufficient. |
| Base variants exist | **CONFIRMED** | [Qwen3.5-4B-Base](https://huggingface.co/api/models/Qwen/Qwen3.5-4B-Base) has 4,659,865,088 params; [Qwen3.5-2B-Base](https://huggingface.co/api/models/Qwen/Qwen3.5-2B-Base) has 2,274,069,824. Both are Apache-2.0 and use `Qwen3_5ForConditionalGeneration`, so they are multimodal wrappers. |
| base-models.md: "Qwen3.5-4B, 4-bit QLoRA" | **REFUTED as a recommendation** | Unsloth: "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense." BF16 LoRA needs about **10 GB** for 4B and 5 GB for 2B (same page). Correction: use BF16 LoRA, which fits easily on 24 GB. |

## 3. IFM/K2-Horizon-3.7B
**CONFIRMED.** The API reports 5,058,255,360 total params, license Apache-2.0, architecture `K2HorizonForCausalLM`, which needs custom code / trust_remote_code ([API](https://huggingface.co/api/models/IFM/K2-Horizon-3.7B)). The [config](https://huggingface.co/IFM/K2-Horizon-3.7B/raw/main/config.json) shows a dense model (num_experts 0), 36 layers, hidden 2560, 32/8 heads, 524,288 context, bf16. The "3.7B" name refers to core parameters, not stored parameters.

## 4. google/gemma-4-12B-it
**CONFIRMED.** The model exists with 11,959,730,224 params (11.96B), Apache-2.0, `Gemma4UnifiedForConditionalGeneration`, revision sha `707f0a3b…`, the same revision Winnow pins ([API](https://huggingface.co/api/models/google/gemma-4-12B-it)).

## 5. Qwen3.6-27B / Qwen3.8-27B; 27B teacher on a 3090
- **Existence and licence: CONFIRMED.** Both models have 27,781,427,952 params, Apache-2.0, `Qwen3_5ForConditionalGeneration` (hybrid DeltaNet) ([3.6](https://huggingface.co/api/models/Qwen/Qwen3.6-27B), [3.8](https://huggingface.co/api/models/Qwen/Qwen3.8-27B); the 3.8 sha `1d4bf0f…` matches AutoJev's pin).
- **Fits on 24 GB at Q4/Q5: CONFIRMED.** File sizes from the [unsloth GGUF tree](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/tree/main), with KV/state room left on a 24 GB card:

  | Quant | File size | Headroom |
  |---|---|---|
  | UD-Q4_K_M | 16.46 GB | about 7 GB |
  | UD-Q4_K_XL | 17.56 GB | about 6 GB |
  | UD-Q5_K_M | 19.77 GB | about 4 GB (tight but workable at short context) |
  | Q6_K_L | 24.19 GB | ❌ does not fit |

  The text-only path does not need the mmproj file (0.93 GB). AWQ-INT4 repos exist, e.g. [cyankiwi/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4), and would load in vLLM at about 15–17 GB `[INFERENCE]`.
- **Throughput: PARTLY; figures are third-party.** Measured 3090 Q4_K_M decode is about 36–42 tok/s ([dev.to](https://dev.to/sysoft/doubling-qwen36-27b-on-one-rtx-3090-ollama-llamacpp-mtp-lever-by-lever-357-802-toks-4i8), [club-3090 #94](https://github.com/noonghunna/club-3090/issues/94)). Logit extraction is prefill-bound. Compute ceiling: 27.8B × 2 FLOP ≈ 56 GFLOP/token against about 71 dense FP16 TFLOPS gives at most ~1.3k tok/s. Realistic Q4 prefill is about **300–800 tok/s** `[INFERENCE, not measured]`. For 500-token items that is roughly 1–2 items/s, or **~50–150k items/day**. Measure with `llama-bench -p 512` before planning corpus sizes. Keep teacher logits only on candidate label tokens.

## 6. Jev / TypeSafe and OpenRouter terms: can SargeDev jev-distill-corpus-v3 go into released weights?
- **The corpus uses Jev labels: CONFIRMED.** The yuri_v3 stream has 498,010 rows labelled by "**Jev 1.13** (TypeSafe) via OpenRouter", but the card still declares the whole dataset Apache-2.0 ([card](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3/raw/main/README.md)).
- **TypeSafe forbids distillation: PARTLY; strong secondary evidence.** The public [typesafe.ai/terms](https://typesafe.ai/terms) (Site terms) contains no distillation clause. According to a legal review, the paid-API Master Customer Agreement §2.3 prohibits "distilling Output to train a competitor" ([Wunderlandmedia](https://wunderlandmedia.com/typesafe-ai-jev-terms-of-service-gdpr)). A second review says the paid API terms are not public ([Timewell](https://timewell.jp/en/columns/jev-typesafe-enterprise-terms-review)). I could not retrieve the MCA text itself.
- **OpenRouter terms: CONFIRMED.** [openrouter.ai/terms](https://openrouter.ai/terms) requires users to comply with each model's Model Terms "and determin[e] whether the applicable Model Terms allow you… to use the… Outputs as you intend". It also bans using the Service for "reselling API access… or otherwise developing a competing service". Output rights therefore pass through to TypeSafe's terms.
- **Verdict: do not put yuri_v3 (or yuri_v1, whose teacher is unnamed) into released weights.** The MCA binds SargeDev, not us directly. However, the Apache-2.0 label cannot grant rights SargeDev did not have, and a decision model that clones Jev is exactly a "competitor". Treat those streams as internal ablation only, or drop them. The `openjev_v2` stream (94,801 rows, CC0) is usable. **Impact:** datasets.md allocates 30% + 10% of the mixture to SargeDev yuri streams; that share must be replaced for any release.

## 7. tasksource-jev-typed-decisions
**CONFIRMED.**
- **`license_use` field:** a string with values `commercial`, `non-commercial`, or `unspecified`; `license` is `other` ([card](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/raw/main/README.md), [datasets-server info](https://datasets-server.huggingface.co/info?dataset=tasksource/tasksource-jev-typed-decisions)).
- **Splits:** 2.5M train / 15k validation / 15k test. Rows follow each source's own train/dev/test split, and the original split is kept in a `split` column.
- **Excluded benchmarks:** the card leaves out BIG-bench, MMLU, BLiMP and MATH test.
- **Suite datasets present:** [sources.yaml](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/raw/main/sources.yaml) lists facebook/anli, allenai/winogrande, Rowan/hellaswag, kiddothe2b/contract-nli and tasksource/contract-nli, allenai/ai2_arc, cosmos_qa, and medmcqa. Several of these are leaderboard suite tasks (ANLI, WinoGrande, HellaSwag, ContractNLI, ARC per research/leaderboard.md).

Filter by `split` plus the suite-overlap list, and dedup against the suite test items. Using train splits is legitimate, but it makes "zero-shot" claims on those tasks invalid.

## 8. 3090 memory feasibility; hw-3090 vs plan-astra
- **Qwen3.5-4B BF16 LoRA at 2K: CONFIRMED to fit.** Unsloth measures about 10 GB for 4B ([Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune.md)). Plain TRL/PEFT without Unsloth kernels will need more, since Unsloth claims "50% less VRAM than FA2": roughly 14–18 GB `[INFERENCE]`, still under 24 GB. The 248K vocabulary makes the logits large (2048 × 248k × 4 B ≈ 2 GB fp32), so use chunked cross-entropy or keep loss only on the answer slot. Unsloth also notes full fine-tuning takes about 4× the LoRA VRAM.
- **MiniCPM5-2B full FT, 2.517B params `[INFERENCE, byte arithmetic]`:**

  | Precision setup | Bytes/param | Weights + optimizer state | Total with activations (2K, ckpt) | Fits 24 GB? |
  |---|---|---|---|---|
  | (a) Standard AMP AdamW (fp32 master + fp32 m, v + grads) | 16 | ≈ 40 GB | — | ❌ |
  | (b) fp32 master + bf16 grads + 8-bit Adam | 4 + 2 + 2 + bf16 copy 2 ≈ 10 | ≈ 25 GB | — | ❌ |
  | (c) Pure bf16 weights + bf16 grads + 8-bit Adam (bnb `AdamW8bit`) | 2 + 2 + 2 = 6 | ≈ 15.1 GB | ≈ **17–19 GB** | ✅ |

  Activation terms for (c): checkpointed hidden states 42 × 2048 × 2048 × 2 B ≈ 0.35 GB; logits plus their gradient 130,560 × 2048 × (4 + 2) B ≈ 1.6 GB unless chunked; CUDA context and fragmentation ~1.5 GB.
- **Resolution: PARTLY; both notes are right under different assumptions.** hw-3090.md's "2B full FT fits with 8-bit Adam (~14–18 GB)" holds only for (c). plan-astra's warning holds for (a) and (b), the default HF Trainer `bf16=True` setup, which keeps fp32 master weights. Setup (c) also carries a real accuracy risk: with no fp32 master, bf16 weight updates at lr ~1e-5 are largely rounded away. It needs Kahan summation or stochastic rounding (e.g. optimi/torchao) to train properly `[INFERENCE]`.
- **Recommendation:** run BF16 LoRA first (the author recipe; Unsloth-class footprint ~5–8 GB). Try full FT on the 3090 only as setup (c) with a Kahan/stochastic-rounding optimizer and a measured peak-memory smoke step. Otherwise rent.

## Corrections to feed back
1. base-models.md Qwen3.5-4B "4-bit QLoRA" → **BF16 LoRA**; Unsloth says not to use QLoRA on Qwen3.5.
2. Qwen3.5 and Qwen3.6/3.8 need **transformers v5** and linear-attention kernels. Tree-mask packing and KV-page prefix sharing do not apply to them; this favours MiniCPM5-2B (pure Llama GQA) for the multi-question packing design.
3. SargeDev yuri streams: **exclude from released weights**. Only openjev_v2 is clean.
4. tasksource: filter suite-source train rows and state the overlap.
5. A 27B teacher runs locally at Q4_K_M (16.5 GB). Throughput is prefill-bound and still to be measured.
