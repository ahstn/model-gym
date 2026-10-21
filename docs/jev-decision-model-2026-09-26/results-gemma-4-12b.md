# Results: stock frontier, clean recipe and gemma-4-12B-it LoRA

Date: 2026-09-27. Code: [`decision/`](../../decision/README.md). Earlier runs: [MiniCPM5-2B LoRA results](results-minicpm5-2b-lora.md). Base choice: [base candidates and 12B fit](base-candidates-2026-09-27.md).

All dev numbers are on the same dev panel (5,500 rows) with one temperature per kind fit on the calib panel (3,000 rows). "No-SargeDev" is the Open-Jev + tasksource part of dev (3,500 rows). The suite and the board have no SargeDev rows, so this is the fair slice for runs that do not train on SargeDev.

## Summary

- **Stock frontier:** `google/gemma-4-12B-it` is the best stock base on dev (0.722 / 0.779 calibrated agreement / NLL). `gemma-4-26B-A4B-it` is level (0.717 / 0.797), `Qwen3.5-9B` is lower (0.671 / 0.832) and `K2-Horizon-7B` is far lower (0.498 / 0.989).
- **Clean recipe:** the decontaminated pool without SargeDev beats R3 on the no-SargeDev dev slice (0.790 / 0.511 vs 0.774 / 0.547). The two-view option-order consistency loss adds more (0.805 / 0.486) and lifts permutation agreement from 0.868 to 0.910.
- **gemma-4-12B-it LoRA** (r32, lr 5e-5, clean pool, consistency loss): dev 0.782 / **0.575** vs R3 0.780 / 0.637 (NLL −9.7%), no-SargeDev **0.851 / 0.380** vs 0.774 / 0.547. It passes the promotion gate. The first try at lr 1e-4 collapsed at step ~210.
- **Decision Index 0.2.1 (latest kit, full suite, exact rescore): 50.44**, rank 10 of 70 on the live board and the best entry of 12.5B or fewer parameters (Winnow-12B 50.02, JPT-9B 46.89). R3 is 31.49 (rank 34), stock MiniCPM5-2B 19.32. Under 0.2 the same run scored 45.70.
- **Baseline: stock gemma-4-12B-it ≈49.2** (estimate from a paired 15,000-request sample). Our LoRA adds only about +1.3 points over its base on the suite: it gains on language (+4.7) and loses on retrieval (−3.0). Almost all of the gain over R3 comes from the base.

## 1. Stock frontier on dev

No adapter. The same readout (`Answer:` + one-token option codes) and calib-fit temperatures for every model.

| Model | Params | Dev agreement / NLL | No-SargeDev | Tasksource | Perm. agreement / TV | Temperatures (choice, score, noul) |
|---|---:|---|---|---|---|---|
| `openbmb/MiniCPM5-2B` (R0) | 2.5B | 0.447 / 1.051 | 0.466 / 0.990 | 0.489 / 1.029 | 0.668 / 0.286 | — |
| `google/gemma-4-12B-it` | 12.0B | **0.722 / 0.779** | **0.765 / 0.655** | **0.705 / 0.775** | **0.900** / 0.101 | 4.95, 8.16, 5.66 |
| `google/gemma-4-26B-A4B-it` | 25.8B (4B active) | 0.717 / 0.797 | 0.760 / 0.676 | 0.702 / 0.783 | 0.892 / 0.106 | 6.48, 11.31, 8.90 |
| `Qwen/Qwen3.5-9B` | 9.7B | 0.671 / 0.832 | 0.709 / 0.713 | 0.667 / 0.798 | 0.824 / 0.129 | 0.97, 1.74, 0.97 |
| `IFM/K2-Horizon-7B` | ≈9.0B stored (18.0 GB bf16) | 0.498 / 0.989 | 0.497 / 0.925 | 0.538 / 0.970 | 0.716 / 0.231 | 2.07, 3.01, 15.75 |
| R3 LoRA (MiniCPM5-2B, for reference) | 2.5B | 0.780 / 0.637 | 0.774 / 0.547 | 0.674 / 0.776 | 0.894 / 0.082 | — |

- Stock gemma-4-12B-it is the best stock model. On tasksource it is already better than every 2B LoRA (0.705 vs 0.674–0.683).
- The 26B-A4B MoE is not better than the dense 12B on this readout, and it needs 52 GB of weights.
- The gemma models are very over-confident before calibration (temperatures 5–11; raw NLL 2.14 and 2.92). After calibration their ECE is 0.034 and 0.041. Qwen3.5-9B is almost calibrated as is (T ≈ 1).
- K2-Horizon-7B is weak with this zero-shot readout. It needs `--trust-remote-code` and has no pad token.
- Board entries on these bases (live 0.2.1): gemma-4-12B-it: Winnow-12B 50.02 (LoRA), Jev-Omni 40.53 (LoRA + head). Qwen3.5-9B: JPT-9B 46.89, Decision 1.0 Lux 43.49, Bespoke Nimble 9B v2 39.57, Kev 9B 38.48. gemma-4-26B-A4B: Surogate Rune v3 57.44 (full fine-tune).

## 2. Recipe fixes

1. **Decontaminated pool.** `decision filter-pool` removes every pool row that shares a 13-gram with a suite row (`selected-rows` + `added-rows`, 10.8M 13-gram hashes) and drops SargeDev. The row order is kept, so the dev and calib panels stay disjoint from the pool. Result (`results/pool-clean.manifest.json`): 127,909 of 200,000 rows kept. Dropped: 71,529 SargeDev, 536 tasksource and 26 Open-Jev rows for suite 13-grams. The runs use the first 40k rows (39,781 after the 2,048-token cap).
2. **SargeDev excluded** from training (`exclude_datasets: [sargedev]`). Reasons: unresolved output rights (Jev labels), and R4 showed that it helps only on SargeDev itself.
3. **Option-order consistency loss** (`consistency_weight`). Each choice row gets a second, different option order in the same micro-batch. Loss per row = mean soft CE of both views + `consistency_weight` × symmetric KL between the two distributions, compared per original option (`p_a[inv_a]` vs `p_b[inv_b]`). Score and noul rows keep one view (their order has a meaning). The micro-batch row and token budgets are halved so that a pair fits the old budget.

## 3. MiniCPM5-2B runs on the clean pool

Same recipe as R3 (r64, alpha 128, lr 1e-4, 64 rows per step, 1 epoch of 40k rows).

| Run | Dev agreement / NLL | SargeDev | Open-Jev | Tasksource | No-SargeDev | Perm. agreement / TV | Train tokens | Wall |
|---|---|---|---|---|---|---|---:|---:|
| R3 `r64-40k` (old pool, with SargeDev) | **0.780 / 0.637** | **0.798 / 0.874** | 0.833 / 0.411 | 0.674 / 0.776 | 0.774 / 0.547 | 0.894 / 0.082 | 10.7M | 27 min |
| `r64-40k-clean` | 0.716 / 0.685 | 0.519 / 1.146 | 0.862 / 0.361 | 0.670 / 0.763 | 0.790 / 0.511 | 0.868 / 0.080 | 13.6M | 47 min* |
| `r64-40k-clean-cons` (weight 1.0) | 0.729 / 0.667 | 0.525 / 1.148 | **0.879 / 0.331** | **0.683 / 0.746** | **0.805 / 0.486** | **0.910 / 0.060** | 22.1M | 60 min* |

\* Shared the GPU with the stock evals.

- On the no-SargeDev slice the clean run beats R3 (+1.6 points, NLL −6.6%). The loss on full dev is only the SargeDev slice.
- The consistency loss helps on every slice: no-SargeDev +1.5 points and NLL −4.9% over `r64-40k-clean`. Permutation agreement rises from 0.868 to 0.910 and TV falls from 0.080 to 0.060. It costs 1.6× the training tokens.

## 4. gemma-4-12B-it LoRA

Recipe: [`configs/g12-r32-40k-clean-lr5e5.yaml`](../../decision/configs/g12-r32-40k-clean-lr5e5.yaml). Same data as `r64-40k-clean-cons` (clean pool, consistency weight 1.0). Winnow-12B adapter shape: r32, alpha 64, dropout 0, all 7 projections of the text decoder only (a PEFT regex on `get_decoder()` skips the vision tower). 64 rows per step, micro-batches of at most 16 rows or 32k tokens.

Code changes for Gemma 4: the backbone is `get_decoder()` (not `.model`); the code logits apply the final logit soft-cap (`tanh(x/30)·30`); the shared-prefix engine copies the prefix cache layer by layer, so the sliding-window layers (window 1,024) keep their state. Checks: new unit tests on a tiny Gemma 3 model (sliding window 16 < prefix 100, soft-cap 3.0); on 150 real suite requests the shared-prefix engine and independent rows agree on 99.3% of argmaxes (mean TV 0.004; MiniCPM: 99.7%, TV 0.009).

### First attempt: lr 1e-4 collapsed

`g12-r32-40k-clean` used the 2B learning rate (1e-4). Grad norms spiked (84 at step 70, 360 at step 210), and at step ~210 the loss jumped from about 0.5 to 1.2–1.8 and stayed there (grad norm then ≈0.3: a collapsed, near-uniform output). The dev NLL on the 3,000-row selection subset went 0.632 (step 125) → 1.218 (step 250). We stopped it at step 440 and kept the step-125 adapter (`best/`).

Even that step-125 adapter (8k rows seen) is better than R3 on the fair slice:

| Run | Dev agreement / NLL | SargeDev | Open-Jev | Tasksource | No-SargeDev | Perm. agreement / TV |
|---|---|---|---|---|---|---|
| R3 `r64-40k` | 0.780 / 0.637 | 0.798 / 0.874 | 0.833 / 0.411 | 0.674 / 0.776 | 0.774 / 0.547 | 0.894 / 0.082 |
| stock gemma-4-12B-it | 0.722 / 0.779 | 0.606 / 1.111 | 0.801 / 0.583 | 0.705 / 0.775 | 0.765 / 0.655 | 0.900 / 0.101 |
| 12B lr 1e-4, step 125 | 0.774 / 0.621 | 0.656 / 1.068 | 0.875 / 0.322 | 0.723 / 0.671 | 0.818 / 0.453 | 0.910 / 0.070 |

### Main run: lr 5e-5

Only the learning rate changed (5e-5). No collapse: the loss stayed at about 0.5–0.7. Grad norms (before the clip to 1.0, logged every 10 steps) had median 10.7 and p90 40.7, with single spikes up to 398 that the model recovered from. Selection-subset NLL by step: 0.625 (125), 0.646 (250), 0.599 (375), 0.587 (500), **0.583 (623, best)**.

| Item | Value |
|---|---|
| Rows | 39,818 first views + 28,970 second views (choice rows) |
| Tokens | 22.8M, 2,033 tok/s, 3 h 7 min, peak 48.7 GiB |
| Trainable params | 131M (r32) |
| Temperatures (choice, score, noul) | 0.99, 1.34, 1.27 (stock: 4.95, 8.16, 5.66) |

| Run | Dev agreement / NLL | SargeDev | Open-Jev | Tasksource | No-SargeDev | Perm. agreement / TV | ECE |
|---|---|---|---|---|---|---|---:|
| R3 `r64-40k` (2B) | 0.780 / 0.637 | **0.798 / 0.874** | 0.833 / 0.411 | 0.674 / 0.776 | 0.774 / 0.547 | 0.894 / 0.082 | 0.050 |
| `r64-40k-clean-cons` (2B) | 0.729 / 0.667 | 0.525 / 1.148 | 0.879 / 0.331 | 0.683 / 0.746 | 0.805 / 0.486 | **0.910** / 0.060 | 0.027 |
| stock gemma-4-12B-it | 0.722 / 0.779 | 0.606 / 1.111 | 0.801 / 0.583 | 0.705 / 0.775 | 0.765 / 0.655 | 0.900 / 0.101 | 0.034 |
| **`g12-r32-40k-clean-lr5e5`** | **0.782 / 0.575** | 0.597 / 1.092 | **0.907 / 0.243** | **0.756 / 0.611** | **0.851 / 0.380** | 0.906 / **0.053** | **0.025** |

By kind (calibrated agreement / NLL), 12B vs R3: choice 0.770 / 0.610 vs 0.720 / 0.758; noul 0.910 / 0.253 vs 0.891 / 0.311; score 0.511 / 1.228 vs 0.703 / 1.031. The score loss is SargeDev: 504 of the 842 dev score rows are SargeDev (agreement 0.325 vs 0.715). On Open-Jev and tasksource score rows the 12B is better (0.833 vs 0.742 and 0.663 vs 0.551).

**Promotion gate** (beat R3 by ≥2 points or ≥5% NLL on dev): passed. On full dev, NLL is −9.7% (0.575 vs 0.637) although the run saw no SargeDev rows; agreement is level (+0.2 points). On the no-SargeDev slice: +7.7 points and NLL −30.5%.

**Merged adapter.** For the suite, the adapter is merged into the bf16 weights (`--merge`) and the temperatures are fit again on the merged model (`eval-merged`). Merged vs unmerged on dev: the same metrics (0.782 / 0.575 both), argmax agreement 98.5%, mean TV 0.011, temperatures 1.03 / 1.37 / 1.28. The dev eval runs 1.56× faster merged (9,299 vs 5,967 tok/s).

## 5. Decision Index (full suite)

One run of the full, hash-verified suite with HLE (the same files as R3: rows sha256 `b2b56d6f…`). Merged adapter, merged-fit temperatures, 2 shards, `--batch-tokens 32768`, max 8,192 tokens per question. 151,020 of 151,034 requests answered; 14 unsupported (too long). Wall time 5 h 11 min (R3: 2 h 10 min with 6 shards). Median latency 102 ms, p95 676 ms per request (the board admits models up to a 1,000 ms median).

### 5.1 Decision Index 0.2.1 (current board edition)

The kit's latest version is 0.2.1 ([`apolinario/decision-index@87d4650`](https://github.com/apolinario/decision-index/commit/87d4650b42b377c0291a89c1f1a879f9b31082bf), 2026-09-27). It uses the same suite files as 0.2, so a complete 0.2 run is also a complete 0.2.1 run. What changes is the scoring:

- Area weights by the square root of the benchmark count (knowledge .258, language .258, retrieval .200, tools .183, arts fixed at .100). Gold benchmarks weigh 1.2 inside their area.
- SGD and RouterBench leave the index (38 benchmarks). ACOS is scored with F1 per review. RAGTruth chance is 0.518 (always "hallucinated").
- ToolRet and BRIGHT keep only queries with a relevant candidate (685 of 1,000 and 220 of 550). Home appliances drops duplicate and dev-identical rows (88 of 160 kept).

We pinned the kit to 0.2.1 (`suite score` now defaults to 0.2.1; `--edition 0.2` gives the old index) and rescored the existing `results.jsonl` of every full run with the kit's own scorer. These are exact 0.2.1 numbers, not estimates. Score files: `decision/results/<run>/suite-0.2/score-0.2.1/`.

| Run | Index 0.2.1 | Raw | Knowledge | Language | Retrieval | Tools | Arts | Board rank (of 70) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stock MiniCPM5-2B (R0) | 19.32 | 39.04 | 12.3 | 12.7 | 28.0 | 32.6 | 12.8 | 45 |
| R3 MiniCPM5-2B LoRA | 31.49 | 48.64 | 17.6 | 30.0 | 36.7 | 55.6 | 16.8 | 34 |
| Stock gemma-4-12B-it (estimate, see below) | ≈49.2 | — | 32.0 | 51.5 | 58.8 | 68.3 | 33.2 | 13 |
| **gemma-4-12B-it LoRA (ours)** | **50.44** | **61.97** | **33.4** | **56.2** | **55.8** | **69.3** | **34.4** | **10** |

- Our 12B LoRA is +18.95 over R3. It is better on 36 of the 38 index benchmarks, level on HLE (0 for both) and worse only on GPQA Diamond (−1.4). R3 would rank 34th, just below Jobe (32.35).
- The rescore lifts the 12B LoRA from 45.70 to 50.44. Most of the gain is from ToolRet (39.6 → 60.6), BRIGHT (15.0 → 40.4) and ACOS (2.5 → 18.1), and from SGD (skill 0) leaving the index. RAGTruth drops (58.9 → 49.9) with the new chance level.
- Our earlier 0.2.1 estimates (≈48.2 for the 12B LoRA, ≈29.6 for R3) were 2.3 and 1.9 points too low. They reused the 0.2 skills of the five rescored benchmarks, and the answerable-query cut raises ToolRet and BRIGHT a lot.

**Stock gemma-4-12B-it baseline (estimate).** The stock model has no full-suite run. We ran it on a 15,000-request stratified sample (`decision suite sample --n 15000`: whole case groups, about 295 requests per benchmark, 0.2.1 rows only), with its dev-fit temperatures (4.95 / 8.16 / 5.66), the same readout and 2 shards (1 h 18 min). Then we scored all runs on the sample rows only, with the 0.2.1 scorer:

| Run | Index on the sample | Index on the full suite | Sample error |
|---|---:|---:|---:|
| Stock MiniCPM5-2B (R0) | 19.72 | 19.32 | +0.40 |
| R3 MiniCPM5-2B LoRA | 31.56 | 31.49 | +0.07 |
| gemma-4-12B-it LoRA (ours) | 49.51 | 50.44 | −0.93 |
| Stock gemma-4-12B-it | 48.23 | — | — |

The estimate in the main table is paired: for each benchmark, stock sample skill + (LoRA full skill − LoRA sample skill), then the 0.2.1 weights. That gives **≈49.16**. [INFERENCE] From the sample errors above, it is good to about ±1 point.

**What the LoRA adds over its base.** On the same 15,000 requests the LoRA is only +1.3 points over the stock 12B (49.51 vs 48.23). It is better on 20 benchmarks and worse on 17:

- Better: SATA-Bench +21.3, ANLI +18.6, VAST +16.2, When2Call +12.2, WinoGrande +10.2, CRUXEval +8.7, CLadder +8.2, New Yorker +8.1, BPoMP +7.8.
- Worse: PhishNChips −15.0, API-Bank −12.8, GSM8K −9.2, ContractNLI −7.6, GPQA Diamond −6.1 (196 requests), Habermas −5.4, ChessBench −5.1, MMLU-Pro −3.8.
- By area: language +4.8, knowledge +1.3, arts +1.2, tools +1.0, retrieval −3.0.

So almost all of the gain over R3 comes from the stronger base, not from our training. Several large "gains over R3" in §5.2 (PhishNChips, GSM8K) are base effects, and the LoRA loses part of them. This matches the dev picture: stock 12B dev NLL 0.779 → LoRA 0.575 is a large gain on the training-like panel, but the suite has many tasks that the pool does not cover.

### Comparison with the live board (0.2.1)

Board data generated 2026-09-27 16:59 UTC (68 entries + Jev). Our rows are exact 0.2.1 scores of our own runs of the public kit (the stock 12B row is the sample estimate). They are not board submissions.

| Entry | Base | Kind | Params | Index | Knowl. | Lang. | Retr. | Tools | Arts |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Jev 1.13 | closed | — | — | 57.91 | 51.4 | 62.0 | 55.4 | 75.1 | 37.7 |
| Surogate Rune 26B-A4B v3 | gemma-4-26B-A4B-it | full FT | 25.8B | 57.44 | 43.4 | 63.1 | 63.5 | 71.2 | 41.9 |
| AutoJev-27B | Qwen3.8-27B | full FT | 27.8B | 56.40 | 40.9 | 63.5 | 54.9 | 79.3 | 39.4 |
| Eikos-27B | Qwen3.8-27B | LoRA | 27.8B | 53.13 | 39.9 | 54.3 | 55.9 | 74.4 | 39.8 |
| Decider chat · Qwen3.6-27B | Qwen3.6-27B | prompt only | 27.8B | 51.35 | 37.0 | 57.1 | 52.2 | 71.4 | 35.1 |
| **Ours: gemma-4-12B-it LoRA** | gemma-4-12B-it | LoRA r32 | 12.0B | **50.44** | 33.4 | 56.2 | 55.8 | 69.3 | 34.4 |
| Winnow-12B | gemma-4-12B-it | LoRA r32 | 12.0B | 50.02 | 33.8 | 56.0 | 54.0 | 71.0 | 30.0 |
| JoshuaSP diffusiongemma | diffusiongemma-26B-A4B-it | prompt only | 25.8B | 49.47 | 32.7 | 53.5 | 58.4 | 70.2 | 26.7 |
| Jevfire | Qwen3.8-27B | prompt only | 27.8B | 49.37 | 30.5 | 53.3 | 56.2 | 72.3 | 32.2 |
| Ours: stock gemma-4-12B-it (estimate) | gemma-4-12B-it | none | 12.0B | ≈49.2 | 32.0 | 51.5 | 58.8 | 68.3 | 33.2 |
| Decider 35B-A3B | Qwen3.5-35B-A3B-Base | full FT | 36.0B | 47.11 | 31.8 | 55.5 | 54.7 | 56.5 | 32.6 |
| JPT-9B | Qwen3.5-9B-Base | LoRA r16 | 9.7B | 46.89 | 31.7 | 56.7 | 44.6 | 67.0 | 28.6 |
| Decision 1.0 Lux | Qwen3.5-9B-Base | head / adapter | 9.7B | 43.49 | 30.9 | 48.0 | 50.0 | 57.2 | 26.4 |
| Decider 4B | Qwen3.5-4B-Base | full FT | 4.7B | 40.70 | 25.7 | 46.0 | 44.7 | 58.6 | 25.0 |
| Jev-Omni | gemma-4-12B-it | LoRA + head | 12.0B | 40.53 | 25.5 | 50.5 | 35.8 | 60.8 | 26.1 |
| Ours: R3 MiniCPM5-2B LoRA | MiniCPM5-2B | LoRA r64 | 2.5B | 31.49 | 17.6 | 30.0 | 36.7 | 55.6 | 16.8 |
| Decider 2B | Qwen3.5-2B-Base | full FT | 2.3B | 28.97 | 14.9 | 32.6 | 37.3 | 42.4 | 14.6 |
| Ours: stock MiniCPM5-2B (R0) | MiniCPM5-2B | none | 2.5B | 19.32 | 12.3 | 12.7 | 28.0 | 32.6 | 12.8 |

- The 12B LoRA would rank **10th of 70** (Jev and eight entries above it; the table leaves out four more 27B entries between 52 and 56). It is the highest entry with 12.5B or fewer parameters, +0.42 over Winnow-12B. The kit's tie margin is 0.25, so this is not a tie, but it is within the noise of one training run.
- Against Winnow-12B per benchmark: better on 18, worse on 19, level on HLE (both 0). Ahead on arts (+4.4) and retrieval (+1.8), level on knowledge and language, behind on tools (−1.7). Worst: Home appliance simulator −11.4, iSarcasmEval −9.4, When2Call −9.3, ContractNLI −4.9, CRUXEval −4.7. Best: BPoMP +17.7, PhishNChips +13.9, ANLI +9.5, API-Bank +7.6, CLadder +7.2, ACOS +5.3.
- [INFERENCE] The stock gemma-4-12B-it with our calibrated readout (≈49.2) is already close to Winnow-12B. Winnow runs on llama.cpp with its own prompt and no fitted temperature, so its own stock baseline may be different.

### 5.2 Decision Index 0.2 (first scoring)

The same run scored with the 0.2 rules (equal area weights, 40 benchmarks). Score files: [`decision/results/g12-r32-40k-clean-lr5e5/suite-0.2/`](../../decision/results/g12-r32-40k-clean-lr5e5/suite-0.2/).

| Run | Index 0.2 | Raw index | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stock MiniCPM5-2B (R0) | 18.91 | 39.40 | 11.9 | 14.2 | 25.5 | 29.7 | 13.2 |
| R3 MiniCPM5-2B LoRA | 29.38 | 47.47 | 17.4 | 30.9 | 32.4 | 49.4 | 16.9 |
| **gemma-4-12B-it LoRA** | **45.70** | **57.88** | **33.6** | **55.1** | **43.2** | **62.0** | **34.6** |

- +16.3 points over R3. It is better than R3 on 37 of the 40 index benchmarks and worse on 2.
- Largest gains over R3: BPoMP +46.1, PhishNChips +41.6, Home appliance simulator +39.4, WinoGrande +38.4, CRUXEval +37.9, CLadder +33.7, GSM8K +33.3, RAGTruth +30.7.
- Losses: GPQA Diamond −1.4 (noise level, 196 requests) and **SGD/SGD-X −38.1 (skill 0.0, macro-F1 0.023)**. The model answers `NONE` on 99% of the SGD intent questions (R3: 5%). A 200-request check shows that the stock gemma-4-12B-it does the same (99.5% `NONE`; R3 0%), so this comes from the base, not from our training. [INFERENCE] The base reads "Choose NONE when no service intent is active" as the default. SGD is not in the 0.2.1 index.
- No overlap discount is needed: the training pool had the suite 13-gram filter.

## 6. Cost and time

| Phase | Wall time |
|---|---|
| Stock frontier (4 models, dev + calib) | under 2 h, next to the 2B training |
| 2B clean and clean-cons training + evals | 47 + 60 min (shared GPU) |
| 12B lr 1e-4 (stopped at step 440) | about 2 h |
| 12B lr 5e-5 training + dev evals (unmerged and merged) | 3 h 7 min + 12 min |
| Full 0.2 suite, 12B merged, 2 shards | 5 h 11 min |
| 0.2.1 rescore of three runs (CPU) | 3 min |
| Stock gemma-4-12B-it, 15,000-request sample, 2 shards | 1 h 18 min |

The pod (RTX PRO 6000 96 GB, $2.09/h) ran from 2026-09-26 18:41 UTC. The 12B session (from 08:08 UTC) used about 13.5 h, about $28; the 0.2.1 re-evaluation added about 1.5 h, about $3.

## 7. Next trials

1. Make the LoRA add more than +1.3 over its base. Find why it loses on PhishNChips, API-Bank, GSM8K, ContractNLI and the retrieval area: check which pool sources teach the wrong prior (for example a skew to "safe" answers), and try a lower LoRA strength or fewer steps with the suite sample as a second gate next to dev.
2. Use the 15,000-request sample (1 h 18 min) as the promotion gate for suite-level changes. Dev alone did not predict the small suite gain.
3. Tools gap to Winnow-12B: Home appliance simulator and When2Call. Check the option counts and prompt lengths of these requests against our 2,048-token training cap.
4. Look at the SGD `NONE` default of the gemma base. Check whether a few SGD-like training rows (active intent with a `NONE` option) fix it without loss elsewhere.
5. Try the consistency weight and the Brier loss as ablations on 12B, and a second seed to measure the noise of one run (our lead over Winnow is 0.42).
6. Selective thinking (Rune v3) for low-confidence decisions.
