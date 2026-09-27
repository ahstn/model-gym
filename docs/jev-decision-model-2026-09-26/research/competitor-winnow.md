# Competitor deep-dive: Winnow-12B / Winnow-E4B and the 3B–12B LoRA/head tier

Sources fetched 2026-09-26. Leaderboard numbers are taken from the Space's `data/*.json`: https://huggingface.co/spaces/multimodalart/jev-decision-index (files `data/index.json` = panel `decision-index-0.2.1`, generated 2026-09-26T11:54Z; `data/index-v2.json` = panel `decision-index-0.2`, 09:43Z).

## 0. Brief correction: which score is it?
The brief's numbers (Winnow-12B 45.0, Winnow-E4B 36.0, Hopper 37.0, Lux 39.0, …) are **`balanced_skill` on the older panel 0.2** (`index-v2.json`). The live board is now **panel 0.2.1**, where every entry moved and the order shifted. Formulas are in `suite.formulas`: balanced = 100·Σ 0.2·category; "skill" normalisation; breadth = a geometric mean over categories. `frozen_scores` is a separate, much lower number (the board note says rows an entrant trained on are counted as wrong in calibration).

| Entry | kind (board) | base (board) | 0.2 skill | **0.2.1 skill** | 0.2.1 raw | frozen skill | cal acc | ECE | Brier | p50 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev (ref) | – | – | – | 57.89 | 68.08 | – | .739 | .074 | .356 | 253 |
| AutoJev-27B | full FT | Qwen3.8-27B | – | 56.40 | 66.89 | 38.56 | .730 | .018 | .361 | 104 |
| **Winnow-12B (Q8_0)** | LoRA | gemma-4-12B | 45.05 | **50.02** | 61.91 | 32.90 | .670 | **.168** | .486 | 48 |
| Decision 1.0 Lux | head/adapter | Qwen3.5-9B-Base | 38.98 | 43.49 | 57.43 | 27.79 | .664 | .076 | .440 | 47 |
| Decider 4B | full FT | Qwen3.5-4B-Base | – | 40.70 | 55.45 | – | .649 | .084 | .454 | 23 |
| Jev-Omni | LoRA+head | gemma-4-12B | 37.11 | 40.53 | 53.50 | 28.69 | .614 | .161 | .534 | 169 |
| **Winnow-E4B (Q8_0)** | LoRA | gemma-4-E4B | 35.98 | **39.89** | 54.84 | 26.64 | .636 | .058 | .461 | 169 |
| Hopper | LoRA | Qwen3.5-4B(-Base) | 37.00 | 39.67 | 54.31 | 25.13 | .643 | .083 | .465 | 46 |
| Bespoke Nimble 9B v2 | LoRA | Qwen3.5-9B | 36.68 | 39.57 | 53.61 | 27.11 | .643 | .024 | .467 | 76 |
| JevK5 | LoRA (merged) | Qwen3.5-4B | 36.44 | 38.81 | 53.68 | 25.98 | .648 | .027 | .459 | 22 |
| Kev 9B | LoRA+head | Qwen3.5-9B-Base | 35.41 | 38.48 | 54.04 | 24.78 | .652 | .138 | .487 | 52 |
| Kev 4B | LoRA+head | Qwen3.5-4B-Base | 31.31 | 34.64 | 50.99 | 21.74 | .616 | .176 | .545 | 53 |
| Decision 1.0 Nox | head/adapter | Qwen3.5-4B-Base | – | 34.36 | 50.99 | 19.40 | .614 | .140 | .530 | 52 |

Category raw % at 0.2.1 (know/lang/retr/tools/arts): Winnow-12B 47.6/69.6/66.6/73.6/48.2; Jev 61.5/73.3/68.2/77.6/53.7; Lux 45.6/64.4/64.5/62.2/47.0; Hopper 40.0/62.2/59.6/63.3/43.9; Nimble 44.3/63.0/**49.7**/62.8/44.4; Kev9B 42.2/60.5/60.9/59.2/44.7. The gap between Winnow-12B and the 9B tier is mostly in **tools (+11)** and retrieval.

## 1. Winnow-12B
Card: https://huggingface.co/EldanRing/Winnow-12B · benchmark report: https://huggingface.co/EldanRing/Winnow-12B/raw/main/docs/BENCHMARKS.md · runtime: https://github.com/EldanRing/winnow-inference

- **Base:** `google/gemma-4-12B-it` @ `707f0a3b…` (instruct model, not the base). Apache-2.0. Multimodal; the vision projector ships separately.
- **LoRA:** r=32, α=64, dropout 0, targets q/k/v/o/gate/up/down. Merged into BF16, then exported as GGUF BF16 and Q8_0. The board runs **Q8_0**, and the card reports Q8 ≈ BF16 (+0.43 pp JevBench-public).
- **Data (private):** synthetic scenarios, teacher-supervised examples, and labelled semantic tasks covering routing, policy/rule application, evidence selection, workflow, ordinal, entailment, paraphrase and answerability. **Contrastive examples vary facts that should flip the answer.** Two stages: an initial decision FT, then a 26k-example refinement mixture with replay. The release was selected after **19,200** refinement presentations, with effective batch 8 and an 8,192-token cap.
- **Objective:** gold-label CE **plus teacher-distribution CE only when the teacher agrees with gold**. This is gated soft-label distillation. The teacher is not named [UNVERIFIED].
- **Readout:** prefill the state once, fork a KV branch per question, read the answer-token logits of the verified candidate tokens only (no generation), then softmax over the supplied options. Runs in a llama.cpp server with `/v1/systemone`.
- **Calibration:** none. Default T=1.0 with no fitted map (card). This is why the board shows ECE **.168** and Brier .486, the worst calibration in the top tier; its rank is driven by accuracy alone.
- **Compute:** not disclosed [UNVERIFIED]. Evaluated on an RTX PRO 5000 Blackwell and deployed on a 16 GB RTX 5070 Ti (Q8, 64K ctx, 15 GiB peak). Board p50 is 48 ms on an RTX PRO 6000.
- **Exposure caveat (author's own):** "Earlier Kev-v4 measurements informed Winnow's later data refinement". The refinement mixture was scanned against the 231 JevBench and 1,264 Kev-v9 records. For E4B, "public evaluation suites were known before the targeted continuation was designed and informed task selection". Its frozen skill (32.9) still ranks it right below AutoJev (38.6), so the lead holds on that view too.
- Card-reported: JevBench public subset 85.71% for Q8 vs Jev 85.71%; Kev-v9 clean 81.55% vs Jev 87.00%.

## 2. Winnow-E4B
Card: https://huggingface.co/EldanRing/Winnow-E4B. Base `google/gemma-4-E4B-it` (the board counts it as 8.0B served params). Same LoRA r32/α64, language tensors only, vision and audio frozen. A selected decision adapter was **continued on targeted hard cases**, and step 438 was exported. The adapter was merged in FP32, then exported to Q8_0/BF16. It **does ship a temperature (1.2574)** in the example request, and its ECE is .058, much better than 12B. Same private data family. Board: 39.89 skill, #3 among sub-12B LoRA/head entries, but p50 is 169 ms (the llama.cpp/E4B path is slow compared with the ~22 ms Qwen3.5-4B runtimes).

## 3. Why does a 12B LoRA beat several 27B+ entries? [INFERENCE, grounded in the rows above]
1. **The "27B" entries it beats are mostly inference-only or weak FTs.** Jevfire, reflex, diffusiongemma and Solomon use no decision training, or a head on a frozen model. The only trained 27B+ entries above it are AutoJev (full FT) and Rune (full FT). Decider-35B-A3B has 3B active params, and Xor is a 35B-A3B MoE whose retrieval score is 35.6.
2. **Strong instruct base.** Gemma-4-12B-it already carries instruction-following and tool/format priors. Winnow is LoRA on the **-it** model, while most Qwen entries start from **-Base**.
3. **Data design over size.** Winnow uses contrastive fact-flip pairs, a broad task taxonomy that matches the board's categories (routing, evidence selection and workflow map onto tools and retrieval), two-stage refinement with replay, and gated teacher-soft-label distillation. It is strongest exactly where the 9B entries are weakest (tools 73.6 vs ~62).
4. **Higher LoRA capacity:** r32 on all linear layers, versus r16 for Hopper, JevK5, Kev and Nimble.
5. **Benchmark-informed refinement (author-admitted).** This probably adds some points. frozen_scores shrinks the gap but does not remove it (32.9 vs Lux 27.8).
6. Accuracy is not calibration. Winnow is uncalibrated (ECE .168). A single fitted T would likely cut Brier with no accuracy change, though that does not move `balanced_skill`.

## 4. Other small entries: what distinguishes the better ones

| Entry | Base | Adaptation | Data | Readout | Calibration | Notes |
|---|---|---|---|---|---|---|
| **Decision 1.0 Lux 9B** ([card](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B), [METHODS](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B/raw/main/METHODS.md)) | Qwen3.5-9B text backbone | **Full-parameter** adaptation (the board labels it "head/adapter"; trained_params 7.94B) plus a new 4.2M-param **candidate head**: a bilinear/MLP readout over contextual candidate-endpoint vectors × final query vector (256-d) | 24k-example mix of decision tasks and human-annotated NL judgments. Checkpoint selected on 3,419 separate selection examples | All candidates scored in one pass per question, up to 255 choices; LM head removed | One T fitted on 1,814 independent calibration examples (ECE .076) | Validated on AMD gfx942. Best sub-12B entry |
| **Hopper** ([card](https://huggingface.co/HopitAI/hopper)) | Qwen3.5-4B | LoRA r16 | LLM-generated synthetic families with **code-computed labels**; JevBench-style LLM items kept only when independent solvers agreed, with 8-gram dedup against JevBench; train splits of ARC, CSQA, MMLU-aux (RACE, non-commercial), SNLI/MNLI, VitaminC, BoolQ, SQuAD2, CLINC, DBpedia, HelpSteer2 | Softmax over option-letter logits, thinking off, one forward pass | **Per-type temperatures** (choice .790 / noul .753 / score .900). An ablation showed the per-type T beat a fancier linear map on held-out sources (+4.5 vs +0.1 cal) | Non-commercial weights |
| **Bespoke Nimble 9B v2** ([card](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2), [training_summary.json](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2/raw/main/training_summary.json)) | Qwen3.5-9B | LoRA r16/α32, dropout .05 | **Only 4,926 rows**: 2,826 original, 1,000 contrastive, 100 long-form, 1,000 decision skills. **1 epoch, 616 steps, 22.5 min, 24.5 GiB peak** | A–Z letter-token logits, up to 255 choices via extra single-token codes | One T=2.179 (ECE .024) | Fixed-epoch run with no eval-based selection. Weak on retrieval (49.7) |
| **JevK5 v0.3** ([card](https://huggingface.co/alibiserikbay/JevK5)) | Qwen3.5-4B | LoRA r16, attention-only, merged | 17,408 teacher questions (Qwen3.6-27B + "GPT-6 Luna") + 30,052 human-labelled train-split items from 26 datasets. 1 epoch over 47,460 rows | SemIf letter-softmax. >16 options handled by multi-pass knockout with a second T | T=1.22 plus knockout T=0.93 (ECE .027) | ~13 ms/decision on H100 with CUDA graphs. 22 ms p50 on the board |
| **Kev 4B / 9B** ([4B](https://huggingface.co/jaredpalmer/kev-4b), [9B](https://huggingface.co/jaredpalmer/kev-9b), [code](https://github.com/jaredpalmer/kev)) | Qwen3.5-4B/9B-**Base** | LoRA r16/α32 on attention, MLP and **DeltaNet projections** + a **pointer head** trained from scratch | decision-v7: 10k public records (10 sources), 896 policy minimal pairs, 1,680 generated rule-structure records. 2 epochs, lr 5e-5, effective batch 8. Delta FTs on hard cases (9B: 15 min on 1 H100, 39.5 GB peak). "No Jev outputs" | Pointer head over options. Questions run as separate causal rows from a shared state (DeltaNet can't block-mask) | 9B served T=2.30. Board ECE .138/.176 (poor) | Classic classification-dataset mix, with no teacher distillation |
| **Decision 1.0 Nox 4B** ([card](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox-4B)) | Qwen3.5-4B | Same family recipe as Lux (candidate head) [INFERENCE: METHODS not re-read for Nox] | – | – | – | Board 34.36, well below Hopper/JevK5 at the same size |
| **Jev-Omni** ([card](https://huggingface.co/akhilaaa3/Jev-Omni)) | Gemma-4-12B-it | Merged FT + classifier head | 30k questions | Head, ≤20 options reliably (accepts 256) | Board ECE .161 | Same base as Winnow, ~9.5 skill points lower. **The base alone does not explain Winnow** |

### Distilled patterns
- **Base matters, but recipe matters more.** Winnow-12B (50.0) and Jev-Omni (40.5) share a base yet sit ~9.5 points apart. Nimble uses only 4.9k rows and ~22 min on a single GPU, yet ties Hopper and JevK5 at 4B, which trained on 47k rows.
- **Letter-token logit readout ≈ trained heads** at this tier. Hopper, JevK5 and Nimble (letter logits) match or beat Kev (pointer head). Lux's bilinear candidate head plus full FT is the best head result, but it is confounded by full-param training.
- **The best data ingredients recur:** contrastive/minimal-pair fact flips (Winnow, Nimble, Kev), code-computed synthetic labels (Hopper, Kev), teacher agreement filtering or gating (Winnow, Hopper), and broad train-split replay (JevK5, Hopper).
- **Calibration is cheap:** one global T (JevK5 .027, Nimble .024) or per-type T (Hopper). Pointer-head Kev and uncalibrated Winnow are worst. All the small entries fit T on held-out non-benchmark data.
- **Compute is small.** Every LoRA entry here is trainable on one ≤80 GB GPU. Nimble's run peaked at 24.5 GiB for 9B at 8k seq (micro-batch 2, grad ckpt [INFERENCE]), which is borderline for our 3090 24 GB. 4B r16 fits comfortably [INFERENCE]. Gemma-4-12B LoRA at 8k would need QLoRA or a 48 GB+ card on our side [INFERENCE].
