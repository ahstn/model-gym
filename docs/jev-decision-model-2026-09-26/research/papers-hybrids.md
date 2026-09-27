# Papers: hybrids, linearization, sparse eval, on-demand recall, speculation limits → Jev-style one-pass decision classifier

Sources: full arXiv HTML fetched 2026-09-26 (curl + tag strip; the reader tool truncated at 50 KB, so numbers come from the full HTML).
- Hybrids: https://arxiv.org/html/2609.02986 (v2, 24 Sep 2026)
- LoLCATs: https://arxiv.org/html/2410.10254
- Sparse Frontier: https://arxiv.org/html/2504.17768v3
- ODA: https://arxiv.org/html/2609.20734 (v1)
- Limits of Speculation: https://arxiv.org/html/2609.22156

Our regime: short inputs (usually <2K tokens), 2–~50 runtime candidate labels with descriptions, one forward pass, calibrated typed probabilities. Long-context efficiency is mostly **not** our bottleneck. So what transfers is (a) diagnostics and eval design, and (b) "decide whether to spend more compute" routing. Architecture changes transfer much less.

Note: the leaderboard is now suite v0.2.1 (per peer research-astra-1). That doesn't change anything below.

## Brief verification

| Brief claim | Verdict | Evidence |
|---|---|---|
| Hybrids: 1.4B/100B tok, retrieval 44.64 vs 40.17 | **Correct.** Table 2 real-world retrieval average (FDA, SWDE, SQuAD, NQ, TriviaQA, DROP): HwH-std 44.64 vs Transformer 40.17. Missing context: Inter (1:3 layer hybrid) scores 43.08 and GDN 29.33. Commonsense average is roughly a tie (HwH 52.70, Transformer 52.61, Inter 53.05). At 4K, the Transformer scores 0 on RULER NIAH (no length extrapolation) while HwH holds 99.6–100 on S-1/S-2. | 2609.02986 Table 2, App. D |
| Hybrids: "global retrieval heads vs local heads" | Correct but simplified. The paper uses intervention metrics RFIS/RPD on Qwen3 and Llama3.1 to get a two-type taxonomy: retrieval heads (low frequency, position-invariant) and positional heads (mid/high RoPE band, the "GPBand"). HwH = NoPE full attention for retrieval heads plus linear attention for positional heads, FA:LA ≤ 1:3, allocated per head and per layer. | abstract, §3 |
| LoLCATs: retrieval-focused conversion data needed | **Correct, and stronger than stated.** Llama 3 8B linearized on packed Alpaca gets **0** passkey samples right. Linearizing on 10K synthetic ~8K-token passkey samples recovers **100%**. Default recipe: sliding window 64 + Hedgehog linear attention, attention-transfer MSE, then LoRA, 40M tokens, a single 40 GB GPU for 8B. Llama 3 8B zero-shot LM-Eval 73.1 vs 74.2 for the Transformer. Up to 42.4 MMLU points lost for low-rank linearizing without the paper's fixes. | 2410.10254 §3, App. B.5.2, Table 21 |
| Sparse Frontier: eval across "length, label count, evidence count/dispersion, negation, aggregation" | **Partly wrong.** The paper's axes are **dispersion** (how hard the evidence is to find), **scope** (how much information is needed), **naturalness** (synthetic vs natural), and **sequence length** (16K–128K). Its 9 tasks: RULER NIAH, VT (multi-hop), CWE (aggregation), SQuAD, QuALITY, TOEFL, plus new Story Retrieval / Multi-hop / Filtering. "Negation" and "label count" appear nowhere (0 matches); they are our extensions. Models: Qwen 2.5, Llama 3.1, Gemma 3, 4B–72B, sparsity up to 0.95. | 2504.17768v3 §3.3, Limitations |
| ODA: learn when global recall helps | Correct. A recall head (28.3M params for Qwen3-1.7B, 196,608 training examples, 1,024 updates) predicts the gain g_t = NLL_local − NLL_full from the local hidden state, previous state, and token embedding. Base weights stay frozen. On RULER16K: Local 19.23 → ODA 81.17 using Full on 41.6% of steps, vs Full 81.94. On Qwen3-8B: Full 92.59, Local 25.43, ODA 91.07 at 47.32% Full. | 2609.20734 §1, App. |
| Speculation: MoE bound ~2.34x | Correct but narrow. 2.34x is an **oracle** plateau for Qwen3-Coder-30B-A3B + EAGLE-3 on the Math Reasoning subset (80 questions) on an A100 80GB. Accepted length saturates at ≈2.1 and the oracle caps drafts at ≈2.8. Verification time is linear in unique experts loaded (α≈0.26 ms/expert). The brief's "cascade escalation trained on error-reduction per unit cost" is our analogy, not the paper's content. The transferable part is the Δcost/Δexpected-progress stopping rule, which the paper finds forms a linear boundary. | 2609.22156 §5, App. A.6 |

## Per-paper analysis

### 1. Modern Transformers Are Implicit Hybrids (2609.02986)
- **Mechanism:** RoPE heads split into retrieval heads (low-frequency, position-invariant matching) and positional heads (local). The design rule is: global access through position-free retrieval heads, local modeling through cheap linear attention, mixed at head granularity.
- **Transfer:** For short-input classification we don't need linear attention or extrapolation. The useful part is the **diagnostic**. Retrieval heads are the heads that match a candidate-label description to evidence in the text. [INFERENCE] Pruning or LoRA-freezing them should hurt label-conditioned accuracy most on high-dispersion and multi-evidence slices, and should hurt label-order robustness.
- **Experiment (E-H1):** On the candidate base (MiniCPM5-2B / Qwen3-class):
  1. Use the cheap behavioral proxy: attention mass from label-description tokens to gold evidence spans on the slice set below.
  2. Rank the heads by that mass.
  3. Ablate the top-k heads (mean-ablation) and measure per-slice accuracy, NLL, and ECE.
  4. Separately, try LoRA only on the top-k retrieval-head q/k/o projections vs uniform LoRA.
  
  RFIS/RPD proper need a frequency-intervention implementation. Skip them unless the proxy is noisy.
- **Cost:** 2B model inference plus hooks, about 1–3 h on the 3090. No rental.
- **Priority:** Medium-low. It's an interpretability and targeted-LoRA hint, not a leaderboard lever.

### 2. LoLCATs (2410.10254)
- **Mechanism:** Replace softmax attention with sliding-window (w=64) + learned linear attention. Train the feature maps with output MSE against the frozen softmax layer, then recover quality with LoRA. 40M tokens total.
- **Transfer:** Weak. Our inputs are short, so quadratic attention costs little, and a 64-token window plus a linear state would sit exactly on the label↔evidence matching we depend on. The data lesson does transfer: conversion or distillation data must exercise the target skill. Generic Alpaca data destroyed retrieval completely.
  - Applied to us: any cheap-student distillation, shallow-encoder variant (the DeepSeek-V4.1 idea), or quantization-aware step must use decision-format data with many labels and dispersed evidence, not generic chat.
- **Experiment (E-L1, optional):** Only as a latency probe for a >8K-context product mode. Linearize a 2B model with LoLCATs using (a) Alpaca and (b) our decision corpus, then compare on the slice set at 4K/8K inputs. Expect (a) to collapse on high-dispersion slices.
- **Cost:** A 2B model fits a 3090 (the paper used a single 40 GB GPU for 8B). A few GPU-hours. [INFERENCE]
- **Priority:** Low. Skip unless long inputs become a requirement. Keep the data lesson.

### 3. The Sparse Frontier (2504.17768 v3)
- **Findings:**
  - Single-QA tasks tolerate 0.95 sparsity.
  - Multi-QA tasks degrade at 0.8–0.9.
  - **High-scope or high-dispersion tasks degrade even at 0.5–0.67.** Averaging across task types hides this.
  - Longer sequences tolerate more sparsity: relative error at a 1/20 budget is ≈0.33 at 16K, 0.26 at 32K, 0.20 at 64K.
  - Larger sparse models beat smaller dense models at equal cost (isoCost).
- **Transfer:** High for **evaluation methodology**. Leaderboard averages (43 benchmarks, 121K decisions) will hide failures in the same way. We need stratified slices.
- **Priority:** **High.** It's cheap and it applies to every trial.

#### Slice set design (cheap to build)
Build by synthetic templating plus re-labeling rows we already have (tasksource/Open-Jev style decisions and our command-risk set). Target about 200 items per cell and ~6–10K items total. Generation: a local 2B–8B model for fillers plus deterministic templates. Gold labels come from construction, not a model.

| Axis | Levels | Construction | Metric |
|---|---|---|---|
| Input length | 64 / 256 / 1K / 4K tokens | Pad a fixed decision item with neutral distractor text, holding evidence fixed; evidence position at 0/50/100% depth | acc, NLL, ECE vs length and depth |
| Label count | 2 / 4 / 8 / 16 / 32 / 64 candidates | Same item; add distractor labels (random, then hard: semantically near) | acc, top-1 prob of gold, ECE; slope vs log(K) (GLiClass-style degradation) |
| Evidence count (scope) | 1 / 2 / 4 / 8 facts needed | "Choose the label satisfied by all conditions" templates; Score items computed from k facts | acc vs k |
| Evidence dispersion | contiguous / spread / spread + near-miss distractors | Place the k facts in adjacent sentences vs across the input, and optionally add near-duplicate distractors (Sparse Frontier's "difficult to locate") | acc gap between contiguous and spread |
| Negation | none / negated evidence / negated label description / double negation | Minimal pairs: "X is **not** approved" flips gold; label "does NOT involve deletion" | minimal-pair flip consistency (both halves right) |
| Aggregation | count / majority / threshold / min-max over N items (CWE analog) | Lists of N=5–50 items with attributes; label = "more than 3 failed" etc. | acc vs N |
| Label order | original / reversed / 3 random permutations; gold position first/middle/last | Permute candidate order on the same item | permutation consistency, std of gold prob, position bias |
| Label-description form | name only / short desc / long desc / paraphrased desc | Reuse items | acc delta (runtime-description sensitivity) |
| Type | Choice / Score / Noul(abstain-like) | Balanced across all axes | per-type ECE |

- **Reporting:** Report per-cell results, plus the worst-cell score and a macro average. Never report only the mean.
- **Primary gates** for any trial:
  - Worst-slice accuracy.
  - Permutation consistency ≥ 0.95.
  - Negation minimal-pair consistency.
  - ECE per type after temperature scaling on a disjoint calibration split.
- **Cost:** Generation is about 1–2 h on the 3090 for fillers. Evaluating a 2B model on ~10K items with ≤32 labels is well under 1 h. No rental needed.

### 4. On-Demand Attention (2609.20734)
- **Mechanism:** Run a cheap Local branch (4 sinks + 2,048 window) first. A small trained head predicts whether Full attention would reduce NLL for this step (g_t), and only then recomputes with Full.
  - Key result: this beats confidence heuristics. On the 675 Local-incorrect positions, head AUROC is 0.643 vs 0.486 for Local entropy (95% CI of the difference: [0.069, 0.230]). Max-prob, margin, and log-ratio all score 0.486–0.491.
  - Caveat from the paper: a recall rate ρ doesn't imply a 1/ρ speedup, because fixed costs dominate.
- **Transfer:** High as a **cascade router**, not as an attention mechanism.
  - Cheap pass: small model (2B) or a truncated input.
  - Expensive pass: 9–27B model, full input, or a reasoning mode.
  - Router target: the **decrease in candidate-set log loss** g = NLL_cheap(gold) − NLL_expensive(gold), predicted from the cheap model's pooled hidden state plus label features.
  - [INFERENCE] The ODA evidence that entropy/margin are near chance on hard cases suggests a learned router will beat thresholding the cheap model's max-prob.
- **Experiment (E-O1):**
  1. Run cheap and expensive models on the training corpus plus the slice set and log per-item gold NLL.
  2. Train an MLP (≤30M params) on frozen cheap-model features to regress g.
  3. Compare against max-prob / margin / entropy thresholds on the curve of accuracy (and log loss) vs % escalated.
  4. Report the result per slice. Escalation should concentrate in high-dispersion, aggregation, and negation slices.
  5. Recalibrate the final mixture, since routing shifts the probability distribution.
- **Cost:** The cheap model and router run on the 3090. Labeling the expensive side needs one pass of a 27B model over ~50–100K items: about a day on the 3090 with a 4-bit quantized model, or a few hours on a rented RTX PRO 6000 / H100 at ~$1–3/h [UNVERIFIED pricing, see the hardware report].
- **Priority:** **High.** It directly trades Jev-like latency (p50 311 ms) against accuracy.

### 5. The Limits of Speculation (2609.22156)
- **Mechanism:** An offline SSP oracle over the speculation budget K for MoE targets. Verification cost is linear in unique experts. The oracle speedup plateaus at ≈2.34x beyond L_max=6. The optimal stopping rule is linear in (Δcost, Δexpected progress).
- **Transfer:**
  - To one-pass classification: nearly none. We decode ≤ a few tokens (constrained readout), so speculative decoding buys little.
  - Caution for MoE bases (Rune 26B-A4B, 35B-A3B): batched multi-label verification still loads input-dependent experts, so latency is less predictable than the active-parameter count implies [INFERENCE].
  - For the cascade (E-O1): use the paper's rule, escalate iff predicted Δ(expected log-loss reduction) / Δcost exceeds λ. Fit λ on a validation set, and use an offline oracle (known g for every item) as an upper bound for the router.
- **Experiment:** Fold into E-O1. Compute an oracle-routing curve as an upper bound, and report the router's efficiency against the oracle.
- **Cost:** Negligible beyond E-O1.
- **Priority:** Low as its own item; medium as a method inside E-O1.

## Priority summary

| Rank | Item | Hardware | Est. cost |
|---|---|---|---|
| 1 | Sparse-Frontier-style slice set + worst-slice/permutation/negation gates | 3090 | ~3 h |
| 2 | E-O1 learned escalation router (ODA target + speculation-style λ rule + oracle bound) | 3090 + optional rented 96 GB GPU for 27B labeling | ~$10–30 rented [INFERENCE] |
| 3 | E-H1 retrieval-head ablation / targeted LoRA | 3090 | ~3 h |
| 4 | LoLCATs linearization | 3090 | skip unless long inputs matter; enforce the "task-matched conversion data" rule in all distillation |
