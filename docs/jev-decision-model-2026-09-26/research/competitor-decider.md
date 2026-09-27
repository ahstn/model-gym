# Competitor deep-dive: Mapika/decider (read 2026-09-26)

Sources: repo README https://github.com/Mapika/decider · docs/RL.md https://raw.githubusercontent.com/Mapika/decider/main/docs/RL.md · scripts/train.sh https://raw.githubusercontent.com/Mapika/decider/main/scripts/train.sh · docs/SERVING.md §8 https://raw.githubusercontent.com/Mapika/decider/main/docs/SERVING.md · cards: https://huggingface.co/Mapika/decider-2b, https://raw.githubusercontent.com/Mapika/decider/main/MODEL_CARD_4B.md, https://raw.githubusercontent.com/Mapika/decider/main/MODEL_CARD_35B.md · leaderboard data https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/main/data (index-v2.json, index.json).

License: **Apache-2.0** for code and all weights (repo LICENSE + HF card tags). PyPI `decider-ai`. Author states nothing distilled from Jev; teacher = local Qwen3.5-27B (custom questions) and Qwen3.6-27B (v2.x document questions).

## 1. Family / base models
| model | base | params | method | weights |
|---|---|---|---|---|
| decider-0.8b | Qwen3.5-0.8B-Base | 0.8B | full SFT | 1.4 GB |
| decider-2b v11 (v10, v8 tags) | Qwen3.5-2B-Base | 1.9B | full SFT (v1–v8) → RL (v10) → LoRA r64 merged (v11) | 3.8 GB |
| decider-4b v2.1 (v2, v1 tags) | Qwen3.5-4B-Base (32 layers, 8 full-attn + 24 gated delta-net) | 4.2B | full SFT (v1) → LoRA r64 merged (v2/v2.1); **no RL** | 8.4 GB |
| decider-35b-a3b v1 (+NVFP4 19.6 GB) | Qwen3.5-35B-A3B-Base | 34.7B / 3B active | SFT, routed experts frozen, 2.45B trainable, Muon; **no RL** | 65 GB |
| "Decider chat · Qwen3.6-27B" | stock Qwen/Qwen3.6-27B instruct | 27.8B | **zero training**: chat template, thinking off, same letter-slot readout, T=1.943 | — |
| decider-2b-vision | Qwen3.5-2B VL, v5 text weights | 1.9B | transplant | 4.1 GB |
All bases are hybrid linear-attention (needs `flash-linear-attention`). Context 32k.

## 2. Data
- Public mixture: ~95 public decision datasets (intent, routing, topic, sentiment, NLI, moderation, fact verification, relevance, RC, MCQA, ordinal ratings, pairwise preference, tool selection) + AgentGym next-action, Mind2Web element choice, game states, teacher-written situations/custom questions, Jev input shapes (described options, up to 255 options, JSON states w/ path refs, long inputs). 1.47M examples / 455M tokens (README) — 35B card says 1,543,567 items / 463M tokens. Builders in `decider/data/` (`core.py`, `mixture.py`, `teacher_*`), teacher data committed in `teacher_data/`.
- Augmentations: 10% of ≥3-option questions get an abstain option; in ¼ of those the options are swapped for unrelated labels so abstain is correct. Label sets sub-sampled to ≤10 options per example (gold kept, shuffled). Two layouts 50/50 (state-first / schema-first). Isolated Score levels (each level judged as its own yes/no row).
- Mixture v2 (4B): public mixture (60% tok) + 26 extra public datasets (8%; code, logs HDFS/BGL, legal LEDGAR/CaseHOLD/…, tables TabFact/WikiTQ/TAT-QA, temporal MC-TACO/TimeQA, ProofWriter, XNLI…) + ten programmatic families with verifiable gold (32%; code, dates, logs, long docs, plans, policies, probability, schedules, tables, tools). 1.89M items / 742M tokens. **v2 builders not yet published** — only the public 60% reproducible.
- Hard-decision stage (4B v2.x, 2B v11): 8,000 generated-family rows (code-computed answers, families from JevBench's *published family names*), 11,356 business-document questions by Qwen3.6-27B (thinking on) kept only when 2 fresh independent answers agree (89–91% kept), 3,293 human-labelled training halves (MMLU, ARC, CSQA, BoolQ, MNLI, SNLI, Banking77, RACE, OBQA, LogiQA2, MedQA, Winogrande), + replay (6,676 for 4B; 20,100 for 2B v11). Decontamination vs every eval file stated; no JevBench/Decision-Index items used.

## 3. SFT recipes
| | 2B (train.sh full) | 4B v1 | 35B v1 | LoRA stage (4B v2.1 / 2B v11) |
|---|---|---|---|---|
| trainable | all | all 4.2B | 2.45B (attn, delta-net, shared experts, routers, norms, emb, head); 256 routed experts frozen | LoRA r64 α128 on attn+MLP, merged |
| optimizer | AdamW (bf16 params, no master copy for 2B/4B) | AdamW on bf16 params, no FP32 master, β 0.9/0.95, no WD | Muon (block matrices, NS 5 steps, momentum .95) + AdamW rest, FP32 master | — |
| LR / sched | 1e-5 (delta: 8e-6), 150 warmup, cosine | 1e-5, 150 warmup, cosine→0, 26,729 steps × 32,768 tok, clip 1.0 | 1e-5, 150 warmup, cosine, 16,287 steps × 32,768 tok | 1e-4, 5% warmup, cosine, 2 epochs, 65,536 tok/step |
| seq len | `--max_tokens 16384 --accum 2 --max_ctx 16384 --max_options 255` | 16,384-tok micro-batches | 8,192-tok micro-batches | — |
| epochs | 1 | 1 | 1 | 2 |
| hardware/time | 1 GH200, 5.3 h + 45 min eval | 2× B300, 577 min, 39.6 GB/GPU peak, ~25k tok/s | 4× B300, 394 min, 100 GB/GPU peak | 1 B300, 94 min (4B) / 150 min shared (2B) |
| loss | CE on slot readout (proper scoring rule) | same | same | CE on labelled rows; **KL(p_parent‖p_model) on replay rows** |
Other flags: `--none_prob 0.1 --schema_first_prob 0.5`. Key finding: AdamW-with-FP32-master moved weights further from base: on a quarter-data 4B ablation, bf16-param AdamW beat it by +3.3 held-out pts (0.791 vs 0.758) and +7 on knowledge tasks (4B card). 35B used master weights; author flags it as a known weakness, retrain planned. v2 (hard labels on replay) sharpened logits → fitted T rose to 1.935 → hurt sampled play; v2.1 fixed with self-distillation KL on replay (T 1.099).
Muon vs AdamW on 35B: lower CE in 75/94 windows at 11% of epoch (no full AdamW run).

## 4. Readout
Prompt: `Context: <state>\n\nQuestion: …\nOptions:\n(A) …\n(B) …\nAnswer: (` — hidden state at each answer slot projected with the LM-head rows for option-letter tokens (A–Z, then two-letter tokens up to 255), softmax over valid letters / T. No generation; multiple questions can share one pass (packed) but default is **one row per question** (independence) and one row per Score level ("isolated levels", normalised; `level_fit`, `fit_mass` reported). Schema-first layout enables prefix caching (1.2–2.4× faster/request, up to 19×/batch) at ~1.5 pts accuracy cost on fixed labels, ~5 on per-example options. `confidence` follows TypeSafe's formula `(n·p_max−1)/(n−1)` since 1.3.0. Plain-transformers reproduction snippet on the 2B card.

## 5. Calibration
Post-hoc temperature scaling by NLL on in-task half of the public regression set (61 tasks, 102,804 rows, excluding Banking77, CLINC-OOS, MMLU, ARC, Winogrande, HellaSwag). Since decider-ai 1.4.0: **per-type temperatures** (`python -m decider.calibrate`, `fit_by_type`): 2B v11 {choice 1.164, noul 1.624, score 1.124}; 4B v2.1 {1.110, 1.560, 1.287}; 35B single T 1.08; chat-27B T 1.943 [how 1.943 was fitted not found — UNVERIFIED]. Choice pool is 94% everyday rows so Choice T barely moves; hard-item overconfidence persists (JevBench hard-tier ECE: 2B v11 0.175, 4B v2.1 0.184, 35B 0.15, 4B v2 0.104). Their own release limit (ECE ≤0.08 on held-out generated families) was **failed** by v11 (0.156) and v2.1 (0.147); released anyway, failures listed.

## 6. RL stage (v8 → v10, 2B only) — docs/RL.md
- Environments: 16 live MiniWoB++ click tasks (6 held-out), exact 4×4 minesweeper, 5×5 slippery grid, bag-draws. No gold labels.
- Loss = 0.1·PPO clipped (clip 0.2, terminal ±1 reward, leave-one-out baseline over 4 repeats) + **0.2·belief log-score** of the model's one-pass prediction of the next outcome vs exact law (proper scoring rule, two option orders averaged) + 1.0·rendering-consistency KL (reversed order, schema-first, both) + 0.7·retention KL(v8‖student) on 8 replay rows/step with hard gate (mean >0.01 or any row >0.05 nats → drop reward terms, zero Adam m; 94/576 steps).
- AdamW FP32 master, LR 1e-6 peak (2e-6 drifted), 16 warmup, cosine over 576 steps, selected step 384; 4 arms on 4 GPUs; 6 pre-registered gates vs v8 (incl. general acc ≥ −0.002, NLL ≤ +0.005).
- So "calibration-aware" = proper-scoring belief term on outcome prediction, not an ECE reward. RL training code **not released** (separate research repo).

### Claim check: "calibration-aware RL improves browser-task success while broad OpenJev accuracy decreases slightly"
**Verified** (docs/RL.md table, v8→v10): live browser sampled 83.0%→93.2% (+10.2, CI +5.1…+15.9); never-rewarded 6 tasks 72.9→91.7; greedy only +0.6; OpenJev 5,252 rows 64.1→63.3% (−0.8, CI −1.3…−0.3, significant). Also: belief excess 0.473→0.219 nats; Mind2Web +1.5; TypeSafe rows +2.0 (CI incl. 0); 847 in-task −0.4 (n.s.); Bespoke 0.706→0.704. Nuance: the browser gain is mostly **sampled-play sharpening** (greedy flat), i.e. the RL makes the served distribution peakier on in-domain tasks; the 35B without RL has better greedy browser (97.2%) but worse sampled (86.4%). The v11 LoRA later recovered OpenJev (+1.4 vs v10) while losing 2.8 pts sampled browser (n.s.).

## 7. Results
### Decision Index (Space data). NOTE brief numbers = `index-v2.json` (edition 0.2, balanced_skill). The live `index.json` was re-generated 2026-09-26T11:54 as **edition 0.2.1** with different numbers.
| entry | kind | v0.2 balanced_skill | v0.2.1 balanced_skill | v0.2.1 raw | ECE (0.2.1 calib) | acc | median/p95 latency ms |
|---|---|---|---|---|---|---|---|
| Jev 1.13.0 | ref | 51.67 | 57.89 | 68.08 | 0.074 | 0.739 | — |
| Decider chat · Qwen3.6-27B | inference | 46.08 | 51.35 | 63.06 | 0.021 | 0.697 | 917 / 6813 |
| Decider 35B-A3B NVFP4 | full FT | 43.50 | 47.11 | 59.72 | 0.023 | 0.696 | 99 / 209 |
| Decider 4B | full FT | 36.58 | 40.70 | 55.45 | 0.084 | 0.649 | 22.5 / 171 |
| Decider 2B FP8 | full FT | 26.11 | 28.97 | 45.85 | 0.077 | 0.586 | 40.5 / 995 |
v0.1 (2026-09-22): Jev 46.26, 35B 39.37, 2B 24.8 skill (README quotes raw 59.5/54.3/44.0). Category (0.2.1 raw): 35B knowledge 0.46, language 0.70, retrieval 0.69, tools 0.61; chat-27B knowledge 0.51, tools 0.74. MiniWoB++ is **dropped** from the index panel, so the RL gain doesn't score there. 4B entry's `served_checkpoint` null — which revision was run [UNVERIFIED]. Note the zero-training 27B wrapper beats all trained deciders and has the best calibration (ECE 0.021 < Jev 0.074).
### Own evals (selected)
| | 2B v10 | 2B v11 | 4B v1 | 4B v2.1 | 35B v1 |
|---|---|---|---|---|---|
| regression in-task / held-out acc | .805/.755 | .802/.752 | .834/.788 | .831/.784 | .855/.810 |
| OpenJev 5,252 | 63.3 | 64.6 | 63.9 | 66.0 | 68.3 |
| JevBench hard (Jev .730) | .459 | .577 | .550 | .649 | .676 |
| Bespoke macro (Jev .760) | .704 | .706 | .757 | .756 | .774 |
| browser sampled | 93.2 | 90.3 | 90.9 | 93.2 | 86.4 |
Base zero-shot with same readout: Qwen3.5-2B-Base .620/.642; 35B-Base .732/.749 → SFT adds ~+19 pts at 2B, ~+12/+6 at 35B.

## 8. Failure modes (author-stated)
No multi-step arithmetic (one pass); hard-item overconfidence; knowledge MCQ gains small at 2B; rules written in the question not followed at small size (form probe 0.67 short vs 0.24 with paragraph of rules); positional lookup in long JSON arrays (0.51 @64 records); generic options must look like buckets; schema-first cache costs accuracy; teacher bias (27B teacher agreed with itself 72% on generic-option labels); English only; regressions per stage (TREC-fine, Freeway, BabyAI, bag-draw); FP32-master drift on 35B; releases failing own pre-registered gates.

## 9. Reusable for us (Apache-2.0)
- `decider/prompt.py`, `model.py`, `systemone.py`: letter-slot readout, isolated Score levels, independence rows, TypeSafe `/v1/systemone` wire format — directly usable as our eval/serving harness (also lets us evaluate any stock HF causal LM zero-shot via `DECIDER_LAYOUT=chat`).
- `decider/data/` (~95 dataset converters + mixture + committed teacher_data) → drop-in SFT corpus; `decider/evaluate.py` (acc/NLL/Brier/ECE/AURC per task); `decider/calibrate.py` per-type T fit; `decider/bench/public_suite.py` (Bespoke suite); `serve_vllm.py`.
- Not reusable: RL training loop (unreleased), mixture-v2 builders (unreleased).
- Hardware reality [INFERENCE]: 2B full SFT = 5.3 GH200-h at 16k tokens/micro-batch; on our 3090 24 GB a 2B full FT with bf16 AdamW (≈2B×(2+4)B ≈12 GB states+grads) plus grad-ckpt at shorter micro-batches is feasible but ~3–5× slower (≈1 day/epoch); 4B full FT needs ≥48 GB → rent; LoRA-r64 stages fit a 3090.
