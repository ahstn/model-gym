# Competitor deep-dive: juspay/xor + "inference technique" entries

Sources: model card https://huggingface.co/juspay/xor ; board data https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/index.json and .../data/methodology.json (live = v0.2.1, generated 2026-09-26T11:54Z). NOTE: live board scores differ from the brief's screenshot (brief: Xor 38.8, AutoJev 50.9). Live `scores` fields: Xor balanced_skill 41.48 / breadth_skill 39.78; AutoJev 56.40/54.93; Rune v3 57.44/56.47 (per research-astra-1, Jev 57.89). Relative ordering is what matters below.

## 1. Xor (juspay)
| Field | Value | Source |
|---|---|---|
| Base | Qwen/Qwen3.6-35B-A3B (MoE, ~3B active), BF16, fully merged | model card |
| Trained params (board) | 35.1B (full FT), served 35.95B | methodology.json |
| Recipe / data / compute | **Not disclosed.** Card has no training section; no code_url on board; repo holds only weights (~70 GB, 2 shards), chat template, `serving/` tarball | HF tree API |
| Prior name | Submitted to JEVBench as "JevOne" (`juspay/jev-one`, same base, same serving bundle layout) | https://github.com/fstandhartinger/jevbench/issues/20 |
| Readout | "deterministic single-token candidate readout, forward and reverse option-order evaluation, probability calibration, schema conversion" in the serving layer; one batched SGLang /generate with fwd+reversed orders per question | card; methodology latency note |
| Option cap | Server rejects >26 options (`422: 2..26 options`) → letter A–Z single-token readout [INFERENCE from cap] | methodology gaps |
| Self-run JEVBench public | easy 1.00, original 0.972, hard 0.775 (ECE 0.058), p50 77–136 ms on 2×RTX PRO 6000 TP2 | card |
| Board calibration | acc 0.709, ECE **0.0149** (best among peers listed), Brier 0.388 | index.json |
| Board latency | median 179.5 ms, p95 1009 ms (1 GPU, TP1; author validated TP2) | index.json |
| Juspay blog/GitHub | none found; juspay/decision-engine is an unrelated Rust payment router | web search |

### Why Xor < AutoJev-27B and Rune 26B-A4B
1. **12.09% of decisions unanswered = scored wrong.** Unsupported reasons: >26 options on CLINC150 (151), BANKING77 (77), POP909 chord (129), API-Bank tool (53). Board rule "Unanswered = wrong"; benchmarks score × answered share. Xor gets 0 on API-Bank, BANKING77, CLINC150, POP909 where AutoJev gets 0.84/0.79/0.88/0.37 skill; ChessBench coverage 0.32. Retrieval area coverage 0.64, tools 0.78. This alone costs roughly 10+ index points.
2. **On covered benchmarks it is still ~4.8 skill points below AutoJev** (mean over 35 fully-covered benchmarks, computed from index.json). Big losses: CRUXEval 0.37 vs 0.60, CLadder 0.27 vs 0.49, NLI4CT 0.47 vs 0.70, RAGTruth 0.38 vs 0.59, BBH 0.58 vs 0.68, ESCI 0.31 vs 0.44, WinoGrande 0.62 vs 0.70. Wins: GSM8K 0.66 vs 0.61, BPoMP 0.93 vs 0.88, Habermas.
3. **Xor ≈ stock Qwen3.8-27B + readout (reflex-27b).** Same 12.09% gap profile (same 26-letter cap), balanced_skill 41.48 vs reflex 41.87, near-identical per-benchmark rows. Full FT of a 3B-active MoE bought ~nothing over a zero-training dense 27B readout. [INFERENCE] 3B active compute limits reasoning-heavy tasks (CRUXEval/CLadder/BBH) vs 27B dense (AutoJev) or 4B-active Rune; and without disclosed data we can't tell whether the FT data was narrow.
4. No run-to-run repeat on file; run at TP1 not author's TP2 (methodology "other").

Lesson: support arbitrary option counts (multi-token/trie or per-option scoring, not A–Z letters) — it's worth more than the fine-tune.

## 2. Inference-technique entries (no own weights; stock checkpoint + custom readout)
| Entry | Base | Technique | balanced_skill (live) | ECE | median ms | Unanswered | Code |
|---|---|---|---|---|---|---|---|
| Decider chat | Qwen3.6-27B stock | Chat-layout prompt, options lettered, softmax of option-letter logits at answer slot / T=1.943 (fitted temp) | 51.35 | 0.021 | 917 | 0 | https://github.com/Mapika/decider (decider.serve_vllm, 1.5.0) |
| Jevfire | Qwen3.8-27B-FP8 | vLLM; each field batched sharing prefix; LM head scores verified single-token labels; JSON assembled in code; context lifted 16K→262K | 49.37 | 0.052 | 78 | 0 | https://github.com/kikoncuo/jevfire |
| JoshuaSP open-jev | diffusiongemma-26B-A4B-it | Diffusion LM: constrained JSON canvas, 1 denoising step, pick highest-logit allowed token; trie for multi-token options | 49.47 | 0.216 | 266 | 0 | https://github.com/JoshuaSP/open-jev |
| reflex 27B | Qwen3.8-27B-FP8 | Single forward pass, lettered answer, Evidence/Criterion framing, average 2 option orders, calibration; ≤26 options | 41.87 | 0.021 | 88 | 12.09% | https://github.com/kshetrajna12/reflex |
| djev | diffusiongemma-26B-A4B | One denoising read over compact seeded answer canvas, exact label-ID probs (large choice sets ok) | 40.28 | 0.212 | 84 | 0.97% | https://github.com/Davipar/djev-dev |
| razorback16 | diffusiongemma NVFP4 | "one read" via vLLM PR 57250 | 37.25 | 0.232 | 77 | 0 | vLLM PR 57250 |
| mmastrac | diffusiongemma | vLLM PR 57250 (closed): fixed canvas positions, logprobs → entropy; resample if low confidence | 32.24 | 0.202 | 125 | 12.07% | https://github.com/vllm-project/vllm/pull/57250 |

Key facts:
- reflex README: "What is proven not to help. Fine-tuning, in every form we tried" (4 LoRA mixes + 27B distillation lost generality); what helped: lettered answers, Evidence/Criterion framing, two option orders.
- Decider README: on an earlier Decision Index edition, the two entries above decider-35b-a3b were zero-training wrappers (jevfire 55.7, joshua-diffusion 55.6).
- Diffusion readouts are fast but badly calibrated (ECE ~0.20–0.23 vs ~0.02 for AR + temperature scaling).

### How far zero-training gets
Best zero-training: Decider chat 51.35 vs Jev ~57.9, AutoJev 56.4 → ~89% of Jev, **~5 pts below the best full FT**, and **above Xor, Decider-35B FT (47.11), all LoRA ≤12B**. Cost: 27B dense at inference (Decider chat 917 ms median; Jevfire 78 ms with prefix batching FP8). The fine-tune gain that remains (AutoJev over Decider chat) concentrates in knowledge/tools/language areas (0.41 vs 0.37, 0.79 vs 0.71, 0.63 vs 0.57).

## Recommendations
1. Build a zero-training baseline first (stock Qwen3.6/3.8-27B or smaller, letter/label-token logit readout, 2 option orders, per-type temperature fit on held-out) — this is the bar any FT must beat; Xor shows FT can fail to beat it.
2. Readout must handle >26 options (trie/first-differing-token or per-option likelihood) — 12% of board decisions need it.
3. Calibrate with held-out temperature per answer type; avoid diffusion readouts unless calibrated separately.
