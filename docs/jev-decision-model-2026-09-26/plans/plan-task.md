# Experiment program: an open Jev-style decision model (balanced senior-MLE view)

Date 2026-09-26. The live board is **Decision Index v0.2.1**: Jev 57.89, Rune v3 57.44, AutoJev-27B 56.40, Decider-chat (zero training) 51.35, Winnow-12B 50.02. The v0.2 figures in the brief (51.7 and so on) must not be compared with v0.2.1 figures ([leaderboard.md], [top-vs-mid.md]).

---

## 1. Thesis

### What wins
The evidence points to four things.

1. **Base capability sets the ceiling. The readout and the data decide how close you get to it.**
   - A stock Qwen3.6-27B with only a letter-slot readout and one temperature scores **51.35**. That is above every trained entry at or below 35B except AutoJev and Rune ([competitor-xor.md], [competitor-decider.md]).
   - Winnow-12B (50.02) and Jev-Omni (40.53) share a base but are ~9.5 points apart ([competitor-winnow.md]).
   - Xor (full-FT 35B-A3B, 41.48) roughly ties reflex, a stock 27B plus readout (41.87) ([competitor-xor.md]).
2. **The readout must be a one-pass, constrained candidate readout that covers up to 255 options.**
   - AutoJev, Decider, Winnow, JevK5 and Hopper all read option-code logits at one answer slot ([competitor-autojev.md], [competitor-winnow.md]).
   - Xor's 26-option cap left **12.09 %** of decisions unanswered, which cost it more than 10 index points ([competitor-xor.md]).
   - Letter logits are as good as trained pointer heads at 4–9B: Kev (pointer head) is worse than Hopper and JevK5 ([competitor-winnow.md]).
3. **Data design matters more than volume.** The ingredients that recur in the top entries are:
   - contrastive fact-flip pairs
   - code-computed labels
   - teacher-agreement gating
   - replay
   - Evidence: Nimble reached 39.57 with **4,926 rows**; Decider 4B reached 40.70 with 742M tokens ([competitor-winnow.md], [competitor-decider.md]).
4. **Calibration comes cheaply from a proper-scoring SFT loss plus held-out temperature.** RL is not needed for it.
   - AutoJev ECE .018; JevK5 .027 (one T); Hopper per-type T.
   - CalRL loses to SFT on ECE in 3 of 4 Qwen3-4B cells ([papers-classification.md]).
   - Decider's RL helped browser tasks but cost 0.8 pt on OpenJev ([competitor-decider.md]).

### What we will not do (and why)
- **No new attention architecture.** That rules out NSA, the YOCO/CED retrofit, the FlashMemory indexer and LoLCATs. All of them need pretraining-scale changes, and their speedups disappear below 4K tokens, while our prompts are mostly under 2K ([papers-deepseek-nsa.md], [papers-hybrids.md]).
- **No MoE training and no 27B full FT in the core program.** Two reasons:
  - MoE is not a reliable win: Decider-35B-A3B scores 47.11 and Xor 41.48.
  - Full FT of a 27B needs roughly 470–490 GB of optimizer state across 8×H100 ([hw-rental-astra.md], [competitor-autojev.md]).
- **No diffusion readouts.** Their ECE is .20–.23 ([competitor-xor.md]). SetScore also shows joint-canvas collapse ([papers-sweep.md]).
- **No RL before SFT plateaus**, and even then only on 2B (E16).
- **No selection on the public suite.** AutoJev's WinoGrande/BBH/RAGTruth checkpoint-selection exposure is flagged on the board ([competitor-autojev.md]).
- **No Jev-distilled data in released weights.** SargeDev yuri_v3 is labeled by Jev through OpenRouter, and the terms of service are unverified ([datasets.md]). It may be used only as an internal ablation arm (E5c), never in shipped checkpoints.
- **No known-contaminated datasets:**
  - pngwn typed-decisions (built from MMLU-Pro test)
  - JevK5 data
  - FrontiersMind kevv (contains BANKING77 test)

  ([datasets.md])

---

## 2. Target

| Tier | Model | Index target (v0.2.1-equiv, our run) | Calibration | Latency | Why |
|---|---|---|---|---|---|
| **Primary: 3–10B** | Qwen3.5-4B (Base vs instruct, decided in E7), LoRA-r64 and/or full FT | **≥ 42.0**, beating Decider 4B 40.70, Winnow-E4B 39.89 and Hopper 39.67 | ECE ≤ 0.030, ≥95%-and-wrong ≤ 1.0 %, 0 % unanswered | p50 ≤ 30 ms on PRO 6000; ≤ 80 ms at 1K tokens on 3090 | Best score per 3090-hour. All ≤4B LoRA work fits locally ([hw-3090.md]). The tier has real, beatable anchors, and Decider 4B has a 22.5 ms board median. |
| **Secondary: 0.7–3B** | Qwen3.5-2B-Base or MiniCPM5-2B, distilled from the 4B | **≥ 31.0**, beating Decider 2B 28.97 | ECE ≤ 0.04 | p50 ≤ 20 ms on 3090 | Latency and edge tier. A full FT fits on the 3090 with 8-bit/bf16 Adam. |
| Stretch (gated, E15) | gemma-4-12B-it LoRA-r32 | ≥ 50 **and** ECE ≤ 0.03 | — | p50 ≤ 60 ms PRO 6000 | Winnow already reaches 50.02 uncalibrated (ECE .168), so a calibrated equal is a real contribution. |
| Product check | same 4B | ≥ 47.3 % exact-level on our command-risk set (Jev's number, docs/jev-laya-comparison) with no weight updates for command-risk beyond its ≤2 % mix share | — | — | This is the use case we actually care about. |

- **Deploy target:**
  - `/v1/systemone`-compatible wire format with the Jev confidence formulas ([jev-reference.md] §1).
  - Question isolation; question ids never reach the model.
  - HF/vLLM serving with shared-state prefix caching.
  - GGUF/Q8 for 16–24 GB cards.
- We do **not** chase Jev's 64K context. The training cap is 8,192 tokens, and anything past the model's reach returns an explicit `Unsupported`.

---

## 3. Eval harness first (week 1, before any training)

### 3.1 Stack
1. **Dev harness (every run):** port the Apache-2.0 `decider/prompt.py`, `model.py`, `evaluate.py` and `calibrate.py` into `src/model_gym/decision/`. This gives us:
   - letter-slot readout with 255 codes
   - isolated Score levels
   - acc / NLL / Brier / ECE / AURC per task

   The metrics module stays ours (extend `metrics.py`). This harness scores the proxy panel and the slice suite.
2. **Suite harness (rare):** `apolinario/decision-index`, pinned to a commit SHA, `edition=0.2`, run through a custom `Engine(state, questions)`.
   - v0.2.1 weights and exclusions come from our own `index021.py`, which re-weights the 0.2 per-benchmark results using `methodology.json` (area weights 0.258494 / 0.258494 / 0.200229 / 0.182783 / 0.1; gold ×1.2; the 5 excluded tasks; ToolRet 685 / BRIGHT 220 answerable; Home-appliances 88; ACOS per-review F1).
   - **Parity proof:** recompute Rune 57.44, AutoJev 56.40 and Decider-chat 51.35 from the published per-benchmark skills to ±0.01 before trusting the port.
   - Report both the 0.2 and the 0.2.1 numbers.
   - The ~17 GB suite build and gated HLE terms are prerequisites.

### 3.2 Held-out and decontamination protocol
- **Three-way split by source task** (not by row), done once and frozen as a SHA-pinned manifest:
  - *train-tasks*
  - *select-tasks*: ~40 sources, ~8K rows; used for checkpoint selection and early stop
  - *calib-tasks*: ~20 sources, ~5K rows; used only to fit temperature
  - *test-tasks*: ~40 sources plus 3 whole held-out families (e.g. legal, logs, tables), ~10K rows; opened only at ladder milestones E4, E8, E10 and E12
- Grouping keys: `group_id`, dataset alias, and 8-gram MinHash clusters (Jaccard ≥ 0.8). This keeps paraphrases and variants on one side.
- **Suite decontamination:**
  - Rebuild the suite rows.
  - Drop any training row with a normalized-exact or **13-gram** overlap with a suite state, question or option.
  - Drop entire sources that are known suite datasets from tasksource/Decider: winogrande, hellaswag, anli, banking77, clinc, esci, hover, cladder, contract-nli, nli4ct, humicroedit, openbookqa, commonsense_qa, ARC, sarcasm, and the list in [datasets.md] §3.
  - Decontaminate the teacher prompt pool the same way.
  - Log the dropped counts per source.
- **Option-shuffle audit** (the board's test) before any submission: acc must drop < 2 pt under shuffle on 5 knowledge sets.

### 3.3 Slice suite (Sparse-Frontier style, [papers-hybrids.md] §3)
- ~200 items per cell, ~8K items in total, all with template-constructed gold (no model labels).
- Axes:
  - length: 64 / 256 / 1K / 4K tokens × evidence depth 0 / 50 / 100 %
  - label count K: 2 / 4 / 8 / 16 / 32 / 64 / 255
  - evidence count 1 / 2 / 4 / 8
  - dispersion: contiguous / spread / spread + near-miss
  - negation minimal pairs, applied to the state, to label descriptions, and as double negation
  - aggregation over N = 5–50
  - label order: original / reversed / 3 permutations
  - description form: name / short / long / paraphrase
  - type: Choice / Score / Noul
- **Jev semantics probes** ([jev-reference.md] §1):
  - isolation (adding sibling questions changes nothing; target |Δp| < 1e-3)
  - irrelevant-option addition
  - p(q) + p(¬q) coherence
  - swapped true/false criteria
  - injected content in the state

### 3.4 Metrics (every eval report)
- **Quality:**
  - v0.2 and v0.2.1 index plus per-area skill (suite runs only)
  - proxy-predicted index
  - macro accuracy over test-tasks with a 95 % bootstrap CI over **tasks**
  - Score: Spearman and MAE
  - unanswered %
- **Probability:**
  - NLL (primary selection metric after T)
  - Brier (sum over classes, row mean)
  - ECE-15 per type
  - ≥95 %-and-wrong share
  - AURC and selective accuracy at 80 % coverage
  - all reported raw and temperature-scaled
- **Robustness:**
  - permutation consistency (target ≥ 0.95)
  - std of the gold probability across permutations
  - negation pair-consistency
  - worst-cell accuracy
- **Latency:**
  - CUDA-synchronized p50/p95 **including prefill** at 512 / 2K / 8K input tokens, batch 1 and batch 16
  - both 3090 and PRO 6000 (PRO 6000 for suite runs only)
  - tokens/s
  - peak VRAM

### 3.5 Checkpoint-selection leakage controls
1. Selection and early stop use **select-tasks NLL after T-fit on calib-tasks**. Tie-breaks: accuracy, then the earlier step.
2. The suite is a one-shot final exam. **At most 3 suite runs over the whole program** (E11 plus two more), each on a checkpoint frozen beforehand, with its hash committed in `runs/registry.jsonl`. No hyperparameter is chosen from a suite result. A suite result can only stop or continue a *tier*.
3. **Proxy-to-index regression without suite exposure.** In E1, run the proxy panel on ≥10 public board models whose v0.2.1 scores are known: Decider 2B / 4B, Hopper, JevK5, Winnow-E4B Q8, Nimble 9B v2, Kev 4B / 9B, Lux, and stock Qwen bases. Fit `index ≈ a + b·proxy_area_skills`. We then track this prediction without ever touching the suite.
4. Test-tasks are opened at the 4 milestones only, and each opening is logged.

---

## 4. Data

### 4.1 Mixture v1 (for the 4B main run; ~300K rows, ~120M tokens, 1 epoch)
| Share | Rows | Source | Label type | Filter |
|---|---:|---|---|---|
| 33 % | 100K | Decider public mixture builders (Apache, ~95 datasets, `decider/data/`) | gold hard; soft where the source has ≥5 annotators | Suite source blocklist; ≤3K rows per source; options subsampled to ≤10, with a 10 % stratum kept at 11–255 options |
| 25 % | 75K | tasksource-jev-typed-decisions, `license_use=="commercial"` | gold (≈10 % graded soft) | Blocklist; ≤3K per source; `packed_derived` ≤ 5 %; keep `criteria_permutation` and `instruction_paraphrase`; states ≤ 8K tokens |
| 20 % | 60K | **Our programmatic families**, all with code-computed gold. There are 12: dates, counting / aggregation, arithmetic, JSON path lookup, table filters, policy / rule application with exceptions, negation minimal pairs, tool selection from specs, schedule conflicts, log triage, own-generated code-output prediction (not CRUXEval), and ordinal thresholds (Score) | exact hard | ≥1 contrastive fact-flip twin per item (Winnow/Nimble pattern) |
| 15 % | 45K | **Teacher-labeled realistic states.** Synthetic business, tool, RAG and agent documents are generated by Qwen3.6-27B (thinking on). Questions are answered by 2 teachers (§4.3) | soft (teacher mean) | Kept only if both teachers agree on the argmax and a fresh reasoning-mode answer matches it (Decider found 89–91 % kept) |
| 5 % | 15K | Open-Jev-v1.1 non-WANLI sources, including `abstain` rows | mostly one-hot | Keep `group_id`; drop the SargeDev openjev_v2 duplicates |
| 2 % | 6K | Our command-risk corpus recast as Score (5 levels, low→high) and Noul ("should block?") | gold | Train split only |

- **Augmentations** (all from Decider):
  - 10 % of ≥3-option questions get an abstain option; in ¼ of those, the other labels are unrelated, so abstain is correct
  - state-first vs schema-first layout, 50/50
  - Score rows are also emitted as isolated per-level Nouls
  - candidate order shuffled every step, with soft targets remapped
- **Kind balance target:** Choice 55 / Noul 25 / Score 20.

### 4.2 Soft vs hard labels
- **Default loss:**
  - `CE(gold)` + **gated** `λ·KL(teacher‖model)` only when the teacher's argmax equals gold (Winnow), with λ = 0.5
  - on teacher-only rows, KL to the teacher mean at T = 1
  - on replay rows in refinement stages, `KL(parent‖model)` (Decider v2.1 found hard replay labels inflated T to 1.94)
- Alternatives are tested in E5.
- **Never** treat teacher-verbalized JSON probabilities as distributions. Store only candidate-slot logprobs over the complete finite candidate set, tagged with teacher and readout version ([papers-sweep.md] §12).

### 4.3 Teachers
- **T1: Qwen3.6-27B stock + Decider-chat readout, T = 1.943.** Board 51.35, ECE .021, Apache, and never selected on the suite.
- **T2: AutoJev-27B.** Board 56.40, ECE .018, Apache. It has suite-selection exposure, but that is harmless here: it only labels non-suite prompts that we have decontaminated.
- **Teacher distribution = mean of T1 and T2.** This mirrors TypeSafe averaging two frontier models ([jev-reference.md] §3).
- Both run on a rented PRO 6000 in BF16. Measured audit: in E0, 300 human-checked rows per family. A teacher is dropped from a family where its agreement with gold is < 80 %.
- Rune v3 is rejected as a teacher: ECE .120 at T = 1, and its v3 training is undisclosed.

---

## 5. Experiment ladder

Assumptions:
- 3090 effective throughput: ~2.5K tok/s for 2B LoRA, ~1.25K tok/s for 4B LoRA, ~1.9K tok/s for 2B full FT ([hw-3090.md]).
- Pilot = 60K rows ≈ 24M tokens ≈ 2.7 h on 2B LoRA, 5.3 h on 4B LoRA.
- Rental rates: Lium PRO 6000 $1.29/h (RunPod Secure fallback $2.09); RunPod A100-80 $1.19–1.59/h.
- 3090 hours cost $0.
- "Proxy" = the proxy-predicted v0.2.1 index.

| # | Hypothesis | Base | Method | Controls / ablations | Data | Success gate | HW | GPU-h | $ |
|---|---|---|---|---|---|---|---|---|---|
| **E0** | The harness reproduces published numbers | Decider-2B, Decider-4B (released) | Ported readout, our metrics | Compare with Decider's own regression / OpenJev numbers | Proxy + slices | Within ±1.0 pt of Decider 4B's own OpenJev 66.0 and regression 0.784; 0.2.1 port reproduces 3 board indices ±0.01 | 3090 | 6 | 0 |
| **E1** | The proxy tracks the board; we learn the zero-training floors | Qwen3.5-2B-Base, MiniCPM5-2B, Qwen3.5-4B-Base, Qwen3.5-4B, gemma-4-E4B-it, K2-Horizon (inference only), + ≥10 released board models | Zero-shot letter slot, 1 scalar T | 2 option orders vs 1 | Proxy + slices | **Spearman ≥ 0.80** between proxy and board over ≥10 models; else rebuild the proxy before any training | 3090 (Q8 for 9–12B) | 25 | 0 |
| **E2** | A joint-label encoder is enough for the <700M tier (GLiClass / Ettin) | gliclass-instruct-large, ModernBERT-large, Ettin-400M | Single-label softmax over label-marker tokens, full FT | vs E4 2B decoder; K-scaling 2 → 64 | 60K pilot | Keep the encoder tier **only if** within 5 proxy pts of the 2B decoder at ≥5× throughput (ONNX INT8); otherwise drop it for good | 3090 | 12 | 0 |
| **E3** | The readout choice matters less than coverage | Qwen3.5-2B-Base | LoRA r64 on the 60K pilot × readouts: (a) letter slot 255 codes, (b) pointer head over option-marker states, (c) isolated per-option Noul | Order averaging on/off; schema-first vs state-first | 60K | Pick (a) unless another beats it by ≥1.0 proxy pt **and** ≤ equal ECE; 0 % unanswered at K = 255 | 3090 | 12 | 0 |
| **E4** | SFT adds ≥ +10 at 2B (Decider saw +19) | Qwen3.5-2B-Base vs MiniCPM5-2B | LoRA r64 α128 lr 1e-4 vs full FT bf16-param AdamW lr 1e-5 (Decider found no-FP32-master better) | Same data and steps; 3 seeds on the winner | 60K | ≥ +10 proxy over E1 zero-shot for the same base; full FT vs LoRA decided by ≥ 1 pt | 3090 | 20 | 0 |
| **E5** | Soft labels improve NLL at equal accuracy | best E4 2B | (a) hard CE, (b) gated soft (λ = 0.5), (c) full KD α = 0 / 0.5, (c′) *internal only*: + 30K SargeDev Jev-soft rows | 3 seeds each; per-type T after | 60K + teacher rows | Winner: ≥ 3 % rel. NLL reduction on test-tasks with acc ≥ −0.3 pt; (c′) is only used to quantify the value of "Jev agreement" | 3090 + teacher rent (shared with E-T) | 30 | — |
| **E-T** | Teacher labels are worth their cost | T1, T2 | Label 150K prompts × 2 teachers, avg ~600 tok; also generate 45K synthetic states | Human audit of 300 rows/family | 150K | Agreement filter keeps ≥ 80 %; teacher-vs-gold ≥ 80 % per family | **Rent PRO 6000** | 22 | 28–46 |
| **E6** | Every data group earns its place | best E5 2B | Leave-one-group-out × 5: programmatic, contrastive twins, abstain aug, tasksource, teacher rows | vs full mixture | 60K each | Drop a group if removing it costs < 0.3 proxy pt **and** no slice cell drops ≥ 3 pt | 3090 | 15 | 0 |
| **E7** | Which 4B base | Qwen3.5-4B-Base vs Qwen3.5-4B (instruct) vs K2-Horizon (5.06B) | LoRA r64, 100K rows | Same seed and data | 100K | Take the best by proxy; K2 has to win by ≥ 1.5 pt to justify remote-code risk | 3090 | 30 | 0 |
| **E8** | Main 4B run reaches the tier target | E7 winner | (a) LoRA r64 1 epoch on the full mixture, **on the 3090** (~27 h); (b) full FT bf16 AdamW lr 1e-5 on **A100-80** | a vs b | 300K / 120M tok | Proxy ≥ 42, macro-acc CI excludes E7's, ECE ≤ 0.03 after T, perm-consistency ≥ 0.95 | 3090 + A100-80 | 27 + 10 | 12–16 |
| **E9** | Calibration map choice | E8 best | Scalar T vs per-type T vs per-type × K-bucket (≤4, 5–16, >16) | Fit on calib-tasks only | 5K | Pick the simplest map within 0.002 NLL; ECE ≤ 0.03, ≥95 %-wrong ≤ 1 % | 3090 | 1 | 0 |
| **E10** | A hard-case refinement stage adds points | E8 best | LoRA r64 stage: 8K new programmatic hard rows + teacher-agreed hard rows + 10K replay with KL(parent) | vs no stage; vs hard-label replay | ~30K | ≥ +1.5 proxy; no area regresses > 1 pt; T stays within 0.9–1.4 | 3090 | 8 | 0 |
| **E11** | Suite confirmation (one-shot #1) | E10 frozen | apolinario kit 0.2 + our 0.2.1 port | Board-style latency | Full suite 120K | **≥ 42.0 v0.2.1**; unanswered 0 %; board-style ECE ≤ 0.03; shuffle audit clean | **Rent PRO 6000** | 5 | 7–11 |
| **E12** | Distil to the 2B tier | Qwen3.5-2B-Base (or E4 winner) | Full FT, KD from E10 4B + gold on the full mixture | vs E6 2B trained directly | 300K | Proxy ≥ 31; p50 ≤ 20 ms on 3090 | 3090 | 45 | 0 |
| **E13** | Learned escalation beats confidence thresholds (ODA + Speculation λ-rule) | 2B → 4B cascade | ≤30M MLP on frozen 2B features regressing Δ gold-NLL; escalate iff pred/Δcost > λ | vs max-prob / margin / entropy; oracle upper bound | 60K logged pairs | At 20 % escalation, recover ≥ 60 % of the 2B → 4B NLL gap; beat the entropy baseline AUROC by ≥ 0.05 | 3090 | 6 | 0 |
| **E14** | Early readout cuts prefill cheaply (CED-inspired) | E10 4B | Readout at 0.75L and 0.5L + LoRA heal + KD from full depth | Full depth | 60K | Keep if ≤ 0.5 proxy loss and ≥ 1.25× p50 speedup at 2K | 3090 | 12 | 0 |
| **E15** | 12B calibrated reaches ~Winnow (**gated**) | gemma-4-12B-it | LoRA r32 BF16, the E10 recipe | vs E10 4B | 300K | **Trigger:** the E4 → E8 2B → 4B slope implies ≥ +5 at 12B **and** E11 passed. **Pass:** proxy ≥ 50, ECE ≤ 0.03 → suite one-shot #2 | Rent PRO 6000 | 30 + 8 eval | 50–75 |
| **E16** | CalRL / proper-scoring GRPO helps after the SFT plateau (**deferred**) | 2B | R2 (λ = 1e-3, uniform on wrong) vs R4 (log-score reward) | vs SFT + T | CSQA-like + programmatic | ≥ +1 acc and ECE no worse **on test-tasks**; else close the line | 3090 | 40 | 0 |
| **E17** | Train-time rationales (DHRD) help (**conditional**) | 2B | Aux LM loss α ∈ {0, 0.5} on T1 rationales + shuffled-rationale control | 3 seeds | 60K + rationale gen | D1 > D0 by > 2σ seed noise; otherwise reject (paper: 4B gain +0.65 %) | 3090 + ~6 h rent | 20 | 8–13 |

**Core path:** E0 → E1 → E3/E4 → E-T → E5 → E6 → E7 → E8 → E9 → E10 → E11 → E12. E2, E13 and E14 run in parallel slack. E15, E16 and E17 are gated.

### Paper ideas: verdicts
- **Earned a slot:**
  - Sparse Frontier: the slice design and worst-slice gates (§3.3)
  - GLiClass / Ettin encoders (E2)
  - On-Demand Attention and Limits-of-Speculation λ-rule (E13)
  - DeepSeek V4.1 CED → layer truncation only (E14)
  - CalRL (E16, deferred)
  - DHRD (E17, conditional)
  - LFM2: KD over the full candidate set, never renormalized top-K (§4.2)
  - Implicit-Hybrids: its head-attribution diagnostic becomes an optional 3 h analysis inside E14 (which heads to keep when truncating)
- **Rejected:**
  - NSA, V4 CSA/HCA and the YOCO retrofit: pretraining-scale work, and no gain below 4K tokens
  - FlashMemory indexer: only pays at ≥100K context, and it cuts aggregation (MRCR 76 → 48)
  - LoLCATs: short inputs; a 64-token window would break label↔evidence matching
  - speculative decoding: we decode 0 tokens
  - diffusion readouts: ECE ~0.2
  - SiDyP: premature until E5 shows label noise is the bottleneck

**Total GPU-hours (core + parallel):** ~290 h on the 3090, ~37 h rented. With gated arms: ~390 h on the 3090 and ~85 h rented.

---

## 6. Hardware decision

- **Day 0:** reboot to clear the NVML 595.84/595.91 mismatch. Then `apt-mark hold nvidia-driver-595-open libnvidia-compute-595` for the duration of the program ([hw-3090.md]).
- **The 3090 runs:**
  - all ≤4B LoRA work
  - 2B full FT
  - all dev-harness and proxy evals for models up to 9B (Q8 for 9–12B)
  - encoders, routers, and calibration
  - This is the whole core ladder except E-T, E8b and E11.
- **Rent triggers** (any one is enough):
  1. A job needs > 24 GB **after** grad-checkpointing, a valid LoRA/QLoRA setup, and batch 1. Examples: 27B teachers in BF16, 4B full FT, 12B BF16 LoRA.
  2. A suite run that needs board-comparable latency or throughput. This means a PRO 6000, never a Max-Q.
  3. A single 3090 job forecast at > 36 h, when a rental finishes it in < 8 h for < $25 **and** it blocks the critical path.
- **Which GPU and provider:**
  - **Teachers, suite and 12B:** 1× RTX PRO 6000 96 GB server edition. Lium at $1.19–1.29/h, falling back to RunPod Secure at $2.09/h ([hw-rental-astra.md]).
  - **4B full FT:** 1× A100-80 PCIe on RunPod ($1.19 Community / $1.59 Secure). Switch to H100 only if a measured speedup is > 1.67× ([hw-rental-sol.md]).
  - **Not used:** Daytona or other spot capacity for runs > 2 h (no-warning preemption); Modal (premium on long jobs).
  - Keep HF_HOME and checkpoints on a network volume. **Rsync `runs/` back explicitly**, because `sync_to_server.sh` excludes it.
- **Budget scenarios** (rental only; +20 % reserve included):

| Scenario | Contents | Rented GPU-h | $ |
|---|---|---:|---:|
| **Lean** | E-T (1 teacher only, T1), E11 once | ~18 | **≈ $35** |
| **Medium (recommended)** | E-T with 2 teachers + synthetic generation, E8b full FT, E11, E15 12B + suite #2, E17 rationales | ~85 | **≈ $150–230** |
| **Ambitious** | Medium + 27B LoRA-BF16 of Qwen3.6-27B on PRO 6000 (~60 h) + suite #3 + 8×H100 short 27B full-FT pilot (12 h) | ~230 + 96 node-GPU-h | **≈ $650–900** |

The ambitious tier requires E15 to reach ≥ 50 first.

---

## 7. Risks and kill criteria

| Risk | Detection | Kill / pivot |
|---|---|---|
| The proxy doesn't track the board | E1 Spearman < 0.80 | Stop training; rebuild the proxy with per-area weighting; allow one extra suite run on stock bases only |
| SFT doesn't beat the zero-training floor | E4 < +10 proxy at 2B | Audit data, readout and prompt; if E5 and E6 still show < +6, drop our data pipeline and adopt Decider's public mixture wholesale |
| 4B plateaus below target | E8 proxy < 39.7 (Hopper) | Kill the 4B ship target; ship 2B/4B as a command-risk product only; do not escalate to 12B |
| Calibration regresses on hard items | ECE on hard slice > 0.10 after T | Add a per-K-bucket T; if still failing, add a gated soft-label weight on hard rows; ship with a documented limit rather than delaying |
| Contamination found after the fact | 13-gram audit hit or shuffle drop ≥ 5 pt | Retract the suite number, remove the source, retrain the affected stage, disclose |
| Teacher bias (Decider saw 72 % self-agreement on generic options) | Human audit < 80 % | Drop the teacher per family; fall back to gold plus programmatic data |
| Qwen3.5 hybrid DeltaNet blocks tree-mask isolation | Isolation probe |Δp| > 1e-3 | Use KV-cache forking per question (Kev approach) instead of tree masks, or switch to MiniCPM5 (Llama) |
| Tooling breaks (transformers 4.57.6 predates 2026 archs) | E0 load failure | Pin transformers ≥ the version each card requires in a separate `llm` extra |
| Moving board (0.2.1 → 0.3) | Edition change | Keep reporting pinned editions; never re-select on the new edition |
| Budget overrun | Rented spend > 1.5× the scenario | Stop renting; finish on the 3090 |

**Program kill:** by the end of week 6, if the best 4B is < 38 proxy **and** the command-risk exact-level < 45 %, stop the Jev-style effort and return to the encoder product.

---

## 8. First two weeks

### Week 1: harness, data plumbing, zero-training floors
- **Day 1**
  - Reboot and hold the driver.
  - Add a `llm` extra to `pyproject.toml`: peft, bitsandbytes, flash-linear-attention, `transformers>=<Qwen3.5 min>`, `decision-index @ git+…@<sha>`.
  - Create `src/model_gym/decision/`:
    - `schema.py`: Question / Answer dataclasses mirroring `/v1/systemone`; validates the 255-option and 10-level limits; `confidence_choice` = `(p_max−1/K)/(1−1/K)`; `confidence_score` = MAD formula; tests that check the doc examples (0.81 / 0.92).
- **Day 2**
  - `render.py`: state-first and schema-first layouts, letter codes, JSON state serialization, structured instructions.
  - `readout.py`: builds up to 255 single-token codes per tokenizer and validates them in context; slices LM-head rows; masked softmax / T. Tests: masking, remapping under shuffle, K = 255.
- **Day 3**
  - `engine.py`: HF causal-LM batch scorer, shared-state prefix caching via KV fork, isolated Score levels.
  - `metrics.py` extension: Brier, NLL, AURC, ≥95 %-wrong, permutation consistency, bootstrap-over-tasks CI (reuse the existing ECE).
  - `calibration.py`: per-type T (reuse the validation-only fitting pattern).
- **Day 4**
  - `data/decisions/`: unified JSONL (`state, questions{type, instructions, criteria}, target{dist|index}, source, group_id, license, label_kind`).
  - Loaders for the Decider builders, tasksource (commercial filter), Open-Jev (non-WANLI) and command-risk.
  - `decontam.py`: 13-gram and normalized exact matching plus the MinHash grouper; blocklist file `configs/decision/suite_blocklist.yaml`.
- **Day 5**
  - Build the suite rows with the apolinario kit (17 GB).
  - Run the decontamination pass and write the frozen split manifest `data/decision/splits-v1.json` with its SHA.
  - `index021.py`, plus a test that recomputes 57.44 / 56.40 / 51.35 from the published skills.
- **Days 6–7:** E0, then E1.
  - `model-gym decide-eval --model … --panel proxy,slices`.
  - Fit the proxy→index regression. Gate: Spearman ≥ 0.80.

### Week 2: slices, first training, teachers
- **Day 8:** `slices.py` template generator (all §3.3 axes, ~8K items) plus the Jev semantics probes. Run E1 models on the slices.
- **Day 9**
  - `train_decision.py`: LoRA/full FT, readout-slot CE, gated KL, per-step option shuffle, token-budget microbatches, checkpoint eval on select-tasks with T-fit on calib-tasks.
  - CLI: `decide-train`, `decide-calibrate`.
  - Smoke config `configs/decision/smoke.yaml` (Qwen3.5-0.8B, 200 rows).
- **Days 10–11:** E3 readout ablation and E4 (2B LoRA vs full) on the 60K pilot, overnight runs. Start the E2 encoder run in the gaps.
- **Day 12**
  - Rent a Lium PRO 6000 for E-T.
  - `teacher.py` (T1/T2 candidate-logprob labeling with provenance tags; agreement filter), plus the 300-row human-audit sheet.
  - Rsync results back.
- **Day 13:** Programmatic-family generators (first 6 of 12: dates, counting, JSON path, rules-with-exceptions, negation pairs, tool selection) with fact-flip twins.
- **Day 14**
  - Launch E5 (3 arms × 3 seeds, overnight chain).
  - Write `docs/decision-program-status.md` covering the E0–E4 numbers and the go/no-go on E2's encoder tier.
  - Queue E6.

**Exit criteria for week 2:**
- harness parity (E0) passed
- proxy validated (E1)
- a 2B SFT that is ≥ +10 over its zero-shot floor
- teacher labels on disk with audit results
- ≤ $50 rented

---

**Research notes cited in this plan:**
- ../research/jev-reference.md
- ../research/leaderboard.md
- ../research/datasets.md
- ../research/competitor-autojev.md
- ../research/competitor-rune.md
- ../research/competitor-decider.md
- ../research/competitor-winnow.md
- ../research/competitor-xor.md
- ../research/top-vs-mid.md
- ../research/papers-deepseek-nsa.md
- ../research/papers-hybrids.md
- ../research/papers-classification.md
- ../research/papers-sweep.md
- ../research/base-models.md
- ../research/hw-3090.md
- ../research/hw-rental-astra.md
- ../research/hw-rental-sol.md
