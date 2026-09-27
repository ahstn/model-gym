# Results: stock frontier, clean recipe and gemma-4-12B-it LoRA

Date: 2026-09-27. Code: [`decision/`](../../decision/README.md). Earlier runs: [MiniCPM5-2B LoRA results](results-minicpm5-2b-lora.md). Base choice: [base candidates and 12B fit](base-candidates-2026-09-27.md).

All dev numbers are on the same dev panel (5,500 rows) with one temperature per kind fit on the calib panel (3,000 rows). "No-SargeDev" is the Open-Jev + tasksource part of dev (3,500 rows). The suite and the board have no SargeDev rows, so this is the fair slice for runs that do not train on SargeDev.

## Summary

- **Stock frontier:** `google/gemma-4-12B-it` is the best stock base on dev (0.722 / 0.779 calibrated agreement / NLL). `gemma-4-26B-A4B-it` is level (0.717 / 0.797), `Qwen3.5-9B` is lower (0.671 / 0.832) and `K2-Horizon-7B` is far lower (0.498 / 0.989).
- **Clean recipe:** the decontaminated pool without SargeDev beats R3 on the no-SargeDev dev slice (0.790 / 0.511 vs 0.774 / 0.547). The two-view option-order consistency loss adds more (0.805 / 0.486) and lifts permutation agreement from 0.868 to 0.910.
- **gemma-4-12B-it LoRA** (r32, lr 5e-5, clean pool, consistency loss): dev 0.782 / **0.575** vs R3 0.780 / 0.637 (NLL −9.7%), no-SargeDev **0.851 / 0.380** vs 0.774 / 0.547. It passes the promotion gate. The first try at lr 1e-4 collapsed at step ~210.

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

## 5. Decision Index 0.2 (full suite)

Status: running on the pod since 2026-09-27 15:59 UTC (151,476 rows, 2 shards, `--batch-tokens 32768`, merged adapter).
