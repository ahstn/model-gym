# Results: MiniCPM5-2B LoRA decision model (first runs)

Date: 2026-09-26/27. Code: [`decision/`](../../decision/README.md). Strategy: [strategy doc](../jev-decision-model-strategy-2026-09-26.md).

## Summary

- A LoRA on `openbmb/MiniCPM5-2B` with 40k mixed rows raises dev agreement from 0.447 (stock) to 0.780. It lowers calibrated soft NLL from 1.051 to 0.637. This is on the selection set; the untouched part of dev gives the same result (see [Dev holdout](#dev-holdout)).
- Rank 64 is better than rank 16, but only by a small amount: NLL −0.022 (95% CI −0.033 to −0.010).
- Without SargeDev rows (R4), the model is as good as R3 on the other two datasets, but it loses a lot on SargeDev (NLL +0.27). SargeDev-style data does not help the other datasets.
- Decision Index 0.2 on the full, hash-verified suite with HLE: stock 18.91, R3 LoRA **29.38**. The LoRA gains on 33 of the 40 index benchmarks and loses on 6. If the benchmarks with text overlap get no credit for their gain, R3 is at 27.41. This is our own run of the public kit, not a board submission.
- Estimated on the live 0.2.1 board: R3 ≈29.6, about rank 33 of 68. That is level with Decider 2B (28.97), the best entry of 2.5B or less. The 4–12B entries score 40–50 (see [board comparison](#comparison-with-the-live-board-021)).
- Permutation robustness is far from the strategy target. Agreement is 0.894 (target ≥0.99), and TV is 0.082 (target ≤0.01).

## Setup

| Item | Value |
|---|---|
| Base model | `openbmb/MiniCPM5-2B`, bf16, sdpa |
| Readout | Chat template, empty thinking, assistant turn starts with `Answer:`. Softmax over 255 one-token option codes at the answer position. Backbone only; code rows of `lm_head`. |
| Loss | Soft cross-entropy over the K valid options. Option order is shuffled every epoch (`shuffle_options: true`). |
| Calibration | One temperature per kind (`choice`, `score`, `noul`), fit on the calib panel. |
| LoRA | All attention and MLP projections, dropout 0.05, alpha = 2r, lr 1e-4, cosine to 10%, warmup 3%, 1 epoch, 64 rows per step, max 2,048 tokens. |
| Hardware | RunPod `scrawny_amber_whippet`: RTX PRO 6000 Blackwell Server (96 GB), torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.0. |

## Data

Sources and revisions:

| Dataset | Revision |
|---|---|
| `SargeDev/jev-distill-corpus-v3` | `fc99c6357a9f89f7512c4a987314352addead049` |
| `ZefanCai/Open-Jev-v1.1` (config `community-hard-mix-v2-redistributable`) | `10ad6888333fa97f8c948192797bad3de3040802` |
| `tasksource/tasksource-jev-typed-decisions` | `85afbc161c36db908a6edd8adaa95f20afb4e2db` |

Pool (200,000 rows; the runs use the first 40k):

| Stream | Target | Realized | Rows |
|---|---:|---:|---:|
| tasksource | 0.37 | 0.414 | 82,706 |
| SargeDev `yuri_v3` | 0.32 | 0.358 | 71,529 |
| Open-Jev, not WANLI | 0.16 | 0.179 | 35,765 |
| Open-Jev WANLI (capped) | 0.05 | 0.050 | 10,000 |
| SargeDev `yuri_v1` | 0.10 | 0 | 0 |

- All 137,203 `yuri_v1` rows have P(true) in [0.45, 0.55]. The entropy filter (>0.97 bits) drops all of them, and their share goes to the other streams.
- The tasksource suite-source blocklist dropped 131,718 train rows from 41 sources. `openjev_v2` is dropped because it duplicates Open-Jev.
- Kinds: choice 117,051, noul 46,048, score 36,901. tasksource `license_use` in the pool: commercial 44,610, non-commercial 5,711, unspecified 32,385. All are accepted for research use.
- Dev panel (5,500 rows): SargeDev `test_set_30k` 1,500, Open-Jev test 1,500, Open-Jev OOD 1,000, tasksource test 1,500. Calib panel: 1,000 from each dataset. Dev and calib share no group.

## Training runs

| Run | Config | Steps | Rows | Tokens | Wall | tok/s | Peak GiB | Trainable | Best step (dev-subset NLL) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| R1 | `pilot-r16-10k` | 152 | 9,959 | 2.68M | 417 s | 6,419 | 28.3 | 25.1M | last |
| R2 | `r16-40k` | 608 | 39,851 | 10.73M | 1,620 s | 6,622 | 28.4 | 25.1M | 600 (0.655) |
| R3 | `r64-40k` | 608 | 39,851 | 10.73M | 1,603 s | 6,691 | 29.3 | 100.5M | 608 (0.630) |
| R4 | `r16-40k-nosarge` | 601 | 39,781 | 13.62M | 2,143 s | 6,357 | 28.4 | 25.1M | 600 (0.726) |

- No run needed an OOM retry. Peak memory is about 29 of 96 GB. [INFERENCE] A larger `batch_tokens` would train faster.
- R4 has more tokens because tasksource and Open-Jev rows are longer than SargeDev rows.
- The R4 best NLL is on its own dev subset, which includes SargeDev rows. Do not compare it with R2/R3.

## Dev results

Calibrated. Cells are agreement / soft NLL. "Agreement" is the argmax match on rows with one top target.

| Run | All | SargeDev | Open-Jev | tasksource | choice | score | noul |
|---|---|---|---|---|---|---|---|
| R0 stock | .447 / 1.051 | .398 / 1.214 | .452 / .967 | .489 / 1.029 | .511 / 1.150 | .275 / 1.586 | .435 / .696 |
| R1 r16 10k | .722 / .742 | .716 / .960 | .781 / .545 | .630 / .854 | .670 / .872 | .593 / 1.173 | .845 / .390 |
| R2 r16 40k | .772 / .658 | .786 / .889 | .831 / .438 | .660 / .795 | .711 / .768 | .691 / 1.078 | .886 / .337 |
| R3 r64 40k | **.780 / .637** | .798 / .874 | .833 / .411 | .674 / .776 | .720 / .758 | .703 / 1.031 | .891 / .311 |
| R4 r16 40k, no SargeDev | .705 / .718 | .504 / 1.159 | .843 / .425 | .677 / .767 | .687 / .800 | .401 / 1.385 | .857 / .330 |

| Run | ECE cal / raw | Brier | Permutation TV / agreement | T (choice, score, noul) | Eval rows/s |
|---|---|---:|---|---|---:|
| R0 | .067 / .276 | .467 | .286 / .668 | 3.01, 5.20, 20.0 (clamped) | 120.7 |
| R1 | .047 / .058 | .282 | .113 / .838 | 1.04, 0.95, 0.77 | — |
| R2 | .056 / .059 | .234 | .085 / .906 | 0.98, 1.00, 0.95 | — |
| R3 | .050 / .060 | .222 | .082 / .894 | 0.93, 0.98, 0.94 | 60.4 |
| R4 | .038 / .027 | .258 | .087 / .862 | 1.07, 1.76, 1.19 | 61.9 |

- Stock needs large temperatures, and its noul temperature hits the clamp at 20. After LoRA training, all temperatures are near 1.
- R0 left 0.15% of dev rows unanswered. The LoRA runs left none.
- The `score` kind occurs only in the dev panel. The kit has only `choice` and `noul` questions, so `score` never occurs in the suite.

### Dev holdout

Each run selects its best checkpoint on a 3,000-row dev subset. The other 2,496 decisive dev rows were never used for selection:

| Run | Holdout NLL | Holdout agreement |
|---|---:|---:|
| R0 | 1.049 | .452 |
| R1 | .752 | .712 |
| R2 | .662 | .777 |
| R3 | .646 | .779 |

Paired bootstrap (2,000 resamples, per-row NLL difference, 95% CI):

| Change | All dev (5,492) | Holdout (2,496) |
|---|---|---|
| R0 → R1 | −0.309 (−0.324, −0.293) | −0.297 (−0.320, −0.275) |
| R1 → R2 | −0.084 (−0.096, −0.072) | −0.090 (−0.109, −0.071) |
| R2 → R3 | −0.022 (−0.033, −0.010) | −0.016 (−0.033, −0.001) |

### SargeDev ablation (R4)

| Rows | R2 → R4 NLL change | R3 → R4 NLL change |
|---|---|---|
| SargeDev dev (1,500) | +0.269 (+0.250, +0.289) | — |
| Open-Jev dev (2,500) | −0.013 (−0.032, +0.006) | — |
| tasksource dev (1,492) | −0.028 (−0.045, −0.011) | — |
| Not SargeDev (3,992) | −0.018 (−0.032, −0.005) | +0.006 (−0.009, +0.020) |

On the not-SargeDev rows: R3 NLL 0.547, agreement .774; R4 NLL 0.553, agreement .781. R4 uses the same 40k-row budget, all on the other two datasets, and it matches R3 there.

## Decision Index 0.2 suite

- Suite: the official 0.2 edition, rebuilt from pinned sources with kit `apolinario/decision-index@19ad28e` after HLE access was granted. The rebuild reports `byte_identical: true` for the v1 rows, the v2 rows and the added rows. The import check against the edition (expected gz `25aac5e8…`) reports `match: true` and `uncompressed_match: true` (uncompressed sha256 `b2b56d6f…`). The added rows (`7429f3c9…`) and the exclusions also match. The runs are strict-verified: `complete: true` and `frozen_corpus_sha256` is set.
- History: we first ran both models on the suite without HLE. When HLE access came, we rebuilt the full suite and resumed into the same run folders. The kit resumes by `run_id`, so only the 513 HLE requests were new. The final scores are the kit's own scoring of the full suite.
- HLE: stock raw .136, R3 raw .124. Both are below chance (.164), so HLE skill is 0 for both. The index is the same as it was without HLE.
- Engine: one shared prompt-prefix KV cache for each request, then one short pass for each question. On a 500-request sample, it matched the per-question engine on 99.25% of argmax answers (mean TV 0.006, max 0.039). Stock and LoRA use the same engine.
- R3 runs as the unmerged adapter with its dev temperatures, which is the same model as in the dev tables. A bf16 merge was about 22% faster, but it changed 1.5% of the argmax answers on the sample, so we did not use it.
- Both runs ran 151,476 requests. The kit scores 151,034 of them: ok 151,027, unsupported 7 (4 ToolRet, 1 BRIGHT, 1 POP909-CL, 1 HLE), errors 0. Every unsupported request has a question prompt over the 8,192-token limit. Outside the scored subsets, 305 more ToolRet requests and 5 more BRIGHT requests also go over the limit.

| | Stock | R3 r64 40k | R3, overlap benchmarks at stock skill |
|---|---:|---:|---:|
| Decision Index 0.2 (balanced_skill) | 18.91 | **29.38** | 27.41 |
| balanced_raw | 39.40 | 47.47 | — |
| breadth_skill | 18.08 | 27.70 | — |
| Area skill: knowledge / language / retrieval / tools / arts | 11.9 / 14.2 / 25.5 / 29.7 / 13.2 | 17.4 / 30.9 / 32.4 / 49.4 / 16.9 | — |
| Request latency median / p95 (6 shards, one GPU) | 71 / 357 ms | 125 / 623 ms | — |

Per-benchmark skill (index benchmarks; ¹ = shares 13-grams with R3 train *state* text, see [Contamination](#contamination-audit)):

| Benchmark | Area | Stock | R3 | Change |
|---|---|---:|---:|---:|
| FinEntity | language | .175 | .702 | +.527 |
| BFCL | tools | .402 | .851 | +.449 |
| When2Call MCQ | tools | .347 | .633 | +.286 |
| VAST | language | .066 | .340 | +.274 |
| SATA-Bench¹ | knowledge | .000 | .259 | +.259 |
| ANLI | language | .079 | .324 | +.245 |
| CLINC150+OOS | retrieval | .470 | .710 | +.240 |
| API-Bank | tools | .376 | .615 | +.239 |
| ContractNLI¹ | language | .044 | .282 | +.238 |
| BANKING77 | retrieval | .442 | .620 | +.178 |
| Amazon ESCI | retrieval | .042 | .179 | +.137 |
| HellaSwag | language | .451 | .581 | +.130 |
| ForecastBench | arts | .000 | .119 | +.119 |
| BBH fixed-option¹ | knowledge | .277 | .389 | +.112 |
| RouterBench¹ | tools | .346 | .457 | +.111 |
| RAGTruth¹ | language | .181 | .282 | +.102 |
| Humicroedit | arts | .035 | .132 | +.097 |
| CRUXEval | knowledge | .000 | .090 | +.090 |
| MuSR | knowledge | .256 | .342 | +.087 |
| WinoGrande | language | .048 | .133 | +.085 |
| CLadder | knowledge | .047 | .106 | +.059 |
| HoVer¹ | retrieval | .161 | .214 | +.053 |
| NLI4CT | language | .329 | .381 | +.052 |
| ToolRet | tools | .312 | .362 | +.050 |
| cfcolor | arts | .017 | .061 | +.045 |
| Home appliance simulator | tools | .000 | .044 | +.044 |
| GPQA Diamond | knowledge | .150 | .170 | +.020 |
| iSarcasmEval | language | .043 | .062 | +.019 |
| SGD/SGD-X | retrieval | .369 | .381 | +.012 |
| BRIGHT¹ | retrieval | .111 | .121 | +.010 |
| Habermas Machine | arts | .132 | .141 | +.009 |
| New Yorker caption matching | arts | .340 | .344 | +.005 |
| ChessBench | knowledge | .025 | .026 | +.002 |
| HLE (below chance) | knowledge | 0 | 0 | 0 |
| ACOS | language | .003 | .000 | −.003 |
| POP909-CL | arts | .003 | .000 | −.003 |
| MMLU-Pro¹ | knowledge | .268 | .263 | −.005 |
| BPoMP | arts | .398 | .384 | −.014 |
| GSM8K | knowledge | .168 | .092 | −.075 |
| PhishNChips | retrieval | .193 | .041 | −.152 |

Not in the index (shown only): MMLU .460 → .485, ARC-Easy .859 → .879, ARC-Challenge .672 → .733, SimpleBench 0 → 0.

- The largest gains are on tool selection, intent classification, NLI and stance. The training mix has these task shapes.
- The two real losses are PhishNChips (accuracy .597 → .521, chance .5) and GSM8K (accuracy .312 → .249, chance .25). [INFERENCE] The LoRA loses answer-checking skill on math, and on phishing it moves toward the base rate. We did not investigate.
- ACOS (5,479 requests, 58 questions each, exact-case scoring) stays at 0. It is the largest compute cost in the suite.

### Comparison with the live board (0.2.1)

Source: board data `multimodalart-jev-decision-index.static.hf.space/data/index.json`, generated 2026-09-27 03:42 UTC, edition release-v2.1, 67 entries. The 0.2.1 scorer is not public, so we estimate our 0.2.1 index:

- Take our per-benchmark 0.2 skill, average it over the board's 0.2.1 panel for each area (knowledge 10, language 10, retrieval 6, tools 5, arts 7 benchmarks), then weight the areas with weights fitted on all 67 board rows (least squares: .259 / .259 / .200 / .183 / .100).
- Check: the same method on the board's own per-benchmark numbers gives the board index with a mean error of −0.41 points (sd 0.33, max 1.10).
- Not covered: 0.2.1 rescores five benchmarks. We cannot apply that change, so the estimate is approximate (about ±1–2 points).
- These are our own runs, not board submissions. Our copies of the score files are in `decision/results/`.

| Entry | Base | Kind | Params | Index | Knowl. | Lang. | Retr. | Tools | Arts |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Jev 1.13 | closed | — | — | 57.91 | 51.4 | 62.0 | 55.4 | 75.1 | 37.7 |
| Surogate Rune 26B-A4B v3 | gemma-4-26B-A4B-it | full FT | 25.8B | 57.44 | 43.4 | 63.1 | 63.5 | 71.2 | 41.9 |
| Decider chat · Gemma-4-31B | gemma-4-31B-it | prompt only | 32.7B | 57.33 | 44.3 | 60.4 | 63.1 | 75.6 | 38.3 |
| AutoJev-27B | Qwen3.8-27B | full FT | 27.8B | 56.40 | 40.9 | 63.5 | 54.9 | 79.3 | 39.4 |
| Winnow-12B | gemma-4-12B-it | LoRA r32 | 12.0B | 50.02 | 33.8 | 56.0 | 54.0 | 71.0 | 30.0 |
| JPT-9B | Qwen3.5-9B-Base | LoRA r16 | 9.7B | 46.89 | 31.7 | 56.7 | 44.6 | 67.0 | 28.6 |
| Decider 4B | Qwen3.5-4B-Base | full FT | 4.7B | 40.70 | 25.7 | 46.0 | 44.7 | 58.6 | 25.0 |
| Winnow-E4B | gemma-4-E4B | LoRA | 8.0B | 39.89 | 22.3 | 45.1 | 43.8 | 62.5 | 22.8 |
| Hopper | Qwen3.5-4B-Base | LoRA | 4.7B | 39.67 | 23.3 | 44.4 | 44.5 | 59.1 | 24.8 |
| **Ours: R3 r64-40k (estimate)** | MiniCPM5-2B | LoRA r64 | 2.5B | **≈29.6** | 17.4 | 30.9 | 31.4 | 50.1 | 16.9 |
| Decider 2B | Qwen3.5-2B-Base | full FT | 2.3B | 28.97 | 14.9 | 32.6 | 37.3 | 42.4 | 14.6 |
| this-that 1.2 | decider-2b | full FT | 1.9B | 28.14 | 15.1 | 31.9 | 32.8 | 45.0 | 11.8 |
| Decision 1.0 Sol | Qwen3.5-2B-Base | head / adapter | 2.3B | 25.32 | 11.5 | 25.4 | 32.7 | 46.1 | 8.2 |
| Bosun v3.1 1.7B | Qwen3-1.7B-Base | LoRA + head | 1.7B | 20.10 | 4.7 | 12.4 | 42.8 | 34.8 | 7.5 |
| Intern-Decision-2B | Qwen3.5-2B-Base | full FT | 2.3B | 19.38 | 13.8 | 20.6 | 15.7 | 33.3 | 12.7 |
| **Ours: R0 stock (estimate)** | MiniCPM5-2B | none | 2.5B | **≈18.1** | 11.9 | 14.2 | 23.7 | 28.7 | 13.2 |

- R3 would rank about 33rd of 68. It is level with the best entry of 2.5B or less (Decider 2B, 28.97), within the error of the estimate. It is ahead on knowledge and tools, and behind on retrieval.
- With the conservative 0.2 score (overlap benchmarks at stock skill, 27.41), R3 is about 1.5 points below Decider 2B.
- The gap to the 4–12B entries is 10–20 points, and it is largest on knowledge and language. The base size explains most of it. See [base candidates](base-candidates-2026-09-27.md).

### Contamination audit

The pool was built before the suite existed, so the 13-gram suite filter did not run (`suite_rows: null` in the manifest). Only the tasksource source blocklist ran. We checked the overlap after training (`runs/audit-ngram.log`):

- Train side: 73 of the 40k R2/R3 rows share a word 13-gram with suite state text, and 57 share one with suite question text. SargeDev has 0 hits. R4 has 110 and 82.
- Suite side: rows per benchmark with a 13-gram in R3 train text. State hits: ContractNLI 19 of 123 (15.4%), RouterBench 50 of 10,000, SATA-Bench 46 of 1,650, HoVer 15 of 4,000, RAGTruth 12 of 2,700, MMLU-Pro 6, BBH 3, BRIGHT 1. BRIGHT (52), ToolRet (53), WinoGrande (32) and ANLI (9) have hits only in question text. [INFERENCE] Those are shared instruction phrases, not leaked items.
- `read_suite_texts` did not read the suite `questions` field. This is fixed (`SUITE_TEXT_KEYS`), so a future `decision data --suite-rows` build filters question text too.

## Throughput and cost

| Phase | Wall time |
|---|---|
| Pod setup (first torch cu130 download about 20 min; one corrupt model copy from `/workspace`) | about 1.5 h |
| Data build (200k pool and panels) | about 2 min |
| Training R1 / R2 / R3 / R4 | 7 / 27 / 27 / 36 min. Dev and calib eval (8,500 rows) runs at 120.7 rows/s stock and about 60 rows/s LoRA. |
| Suite source download and v1 rebuild without HLE | 24 min (CPU, next to training). A second rebuild from the cache (`--skip-hle` check) took 20 min. |
| Suite, stock, 6 shards | 70 min (about 36 requests/s) |
| Suite, R3 LoRA, 6 shards | 2 h 10 min (about 19 requests/s) |
| Full rebuild with HLE, then resume of both runs (HLE rows only) | 22 min rebuild, about 7 min resume |

Compute time was 18:41 to 02:14 UTC plus 06:48 to 07:17 UTC. The pod also stayed on, idle, from 02:14 to 06:48. From 18:41 to 07:20 UTC is 12.7 h at $2.09/h, which is about $26. The strategy expected $30–120 and set a cap of $250.

## Caveats

- Dev results come from the selection set. The holdout table shows the same order and effect sizes, but it is small.
- The index is our own run of the public 0.2 kit on the verified suite, not a board submission. The live board is v0.2.1, so board numbers are not directly comparable.
- Training rows share some 13-grams with suite rows (see the audit). The conservative index gives the overlap benchmarks no credit.
- The pod ran the train code from before the review fixes for R2–R4. The only visible effect is the logged lr value: it is one step late. There were no OOM retries, and a pool rebuild with the fixed code gave the same sha256.
- Permutation robustness is poor. Agreement is .894 and TV is .082; the targets are ≥.99 and ≤.01. Option shuffling in training was on.
- Each configuration has one seed.
- The strategy doc says to exclude Jev-distilled data. SargeDev `jev-distill-corpus-v3` is Jev-distilled. We used it here because the user asked for it. R4 shows that the other datasets do not need it.

## Next trials

1. Rebuild the pool with `--suite-rows` (13-gram filter on state, question and options), then retrain R3. This removes the overlap caveat.
2. Permutation robustness: add a consistency loss over two option orders, or average the logits over K cyclic shifts at inference. Gate on agreement ≥.99.
3. Scale: all 200k rows, r64, and a larger `batch_tokens` (only 29 of 96 GB is in use).
4. Look at the PhishNChips and GSM8K losses. Check the base rate and answer distribution per option before changing the data.
5. Run the same recipe on a 4B or 12B base (strategy trial 5) to measure the size frontier. See [base candidates and 12B fit](base-candidates-2026-09-27.md).

Follow-up (done): [stock frontier, clean recipe and gemma-4-12B-it LoRA](results-gemma-4-12b.md).
