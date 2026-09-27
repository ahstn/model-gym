# Papers: classification-head / calibration / reasoning-distillation — verified notes + experiment designs

Sources read: arXiv 2508.07662 PDF (https://arxiv.org/pdf/2508.07662), GLiClass GitHub README (https://github.com/Knowledgator/GLiClass), HF API for `knowledgator/*gliclass*`, ACL Findings 2026.610 PDF (https://aclanthology.org/2026.findings-acl.610.pdf), arXiv 2509.21487 PDF (https://arxiv.org/pdf/2509.21487). All URLs loaded OK.

## 1. Verification of brief claims

| Brief claim | Verdict | Evidence |
|---|---|---|
| GLiClass: joint text+label encoding, all labels in one pass | **Correct** | Uni-encoder; each label prefixed `«LABEL»` and concatenated with text; one encoder pass (PDF §2.1) |
| GLiClass degrades with large dense label sets | **Correct, nuanced** | Throughput only −7…−20% from 1→128 labels (Table 6, A6000); *quality* degrades: attention between label tokens diminishes as labels grow, text reps degrade at extreme label:text ratios (§2.4); banking77 declines; ~1024-token context limit for 1000+ labels; "cross-encoders handle dense information better" (§4) |
| Cal-aware RL: Qwen3-4B OpenBookQA acc 92.79→93.70, ECE 6.74→1.91 | **Numbers correct, context missing** | Table 4: OBQA GRPO 92.79/6.74 → Ours 93.70/1.91. **OBQA is out-of-domain** (trained on CSQA). SFT on same row: 89.80/4.76. Title is actually "Balancing Classification and Calibration Performance in Decision-Making LLMs via Calibration Aware Reinforcement Learning" (Yaldiz et al., Amazon authors list) |
| DHRD = train-time-only reasoning | **Correct** | Pooled last-token classification head + train-only LM/reasoning head over input+teacher rationale; reasoning head disabled at test (abstract, §2) |

### GLiClass key numbers (paper Table 2/3/6, A6000)
| Model | Params | Avg zero-shot F1 (14 ds) | ex/s |
|---|---|---|---|
| gliclass-edge-v3.0 | 32.7M | 0.4900 | 97.29 |
| gliclass-modern-base-v3.0 | 151M | 0.5577 | 54.46 |
| gliclass-modern-large-v3.0 | 399M | 0.6197 | 43.80 |
| gliclass-base-v3.0 (DeBERTa-v3) | 187M | 0.6764 | 51.61 |
| gliclass-large-v3.0 (DeBERTa-v3) | 439M | 0.7193 | 25.22 |
| deberta-v3-large-zeroshot-v2.0 (cross-enc) | — | 0.6821 | 6.03 (0.25 at 128 labels) |

- DeBERTa backbones "consistently outperform" ModernBERT ones (§2.1) — directly relevant to our repo's ModernBERT vs DeBERTa choice.
- 8-shot adds +0.106 (large) … +0.209 (modern-base) F1 (Table 5).
- Training: pre-train on GPT-4o–annotated FineFineWeb (50 true + 50 false labels/text), mid-training with a PPO variant, LoRA post-training on logic/NLI; *huge* LoRA ranks (edge r=1536, large r=384) (Table 1).
- Paper's own limitations: "residual calibration differences across datasets", sensitivity at extreme label-text lengths (§4/§5). No calibration metric reported.
- Default pipeline is `classification_type='multi-label'` with sigmoid threshold 0.5 (README) — i.e. scores are not a normalized distribution over Choice candidates; we would need a softmax/single-label head for Jev Choice.
- HF: all Apache-2.0. Newer checkpoints not in paper: `gliclass-instruct-{edge,base,large}-v1.0` (2026-02; large = 438.7M params, supports label descriptions, task prompts, few-shot examples, hierarchical labels — close to Jev's Choice-with-descriptions interface), `gliclass-multilang-{mini,edge,ultra}` (2026-04), and decoder-backbone variants `gliclass-qwen-0.5B/1.5B-v1.0`, `gliclass-llama-1.3B-v1.0` (2024) (https://huggingface.co/api/models?author=knowledgator&search=gliclass; https://huggingface.co/knowledgator/gliclass-instruct-large-v1.0).

### Calibration-aware RL key facts (Yaldiz et al., 2026)
- Setup: Qwen3-1.7B/4B/8B; train on CSQA (4-way) and OpenAI Moderation (500-ex train, binary); OOD eval OBQA and XSTest. Reward 1 iff decision token = gold. GRPO.
- Finding: RLVR → accuracy↑, extreme overconfidence; SFT (CE on decision token) → best calibration, smaller acc gain, holds under shift.
- Diagnosis: decision token is an *extraction* of the reasoning: swapping reasoning to the opposite label flips the prediction 92–100% of the time (Table 3).
- Method: L = L_GRPO + λ·CE(decision token; q), q = one-hot if rollout correct, **uniform over C if incorrect**; λ=0.001 (λ sweep 0.0005–0.005 is a real accuracy/calibration trade-off, Table 8).
- Full Qwen3-4B Table 4 (Acc/ECE):

| Split | Base | SFT | GRPO | CalRL |
|---|---|---|---|---|
| CSQA (ID) | 76.97/20.96 | 79.12/**8.81** | 81.84/16.98 | 79.16/12.91 |
| OBQA (OOD) | 88.40/10.67 | 89.80/4.76 | 92.79/6.74 | **93.70/1.91** |
| OpenAI mod (ID) | 74.07/24.29 | 87.54/**5.26** | 88.98/11.00 | 89.56/7.11 |
| XSTest (OOD) | 80.89/18.18 | 86.22/8.26 | 87.78/12.21 | 87.56/8.89 |

- Takeaway the brief omits: **SFT beats CalRL on ECE in 3 of 4 Qwen3-4B cells**; CalRL on CSQA-ID loses the GRPO accuracy gain (79.16 vs 81.84). The paper itself says it "does not always reach the calibration level of SFT". Post-hoc isotonic/Platt on top helps further (App. B). Only ECE (plus SCE/MCE, AUROC in appendix) — no Brier/NLL.
- Consistent with the brief's note that Decider's calibration-aware RL helped browser tasks but slightly hurt broad OpenJev accuracy [UNVERIFIED here; other agent's scope].

### DHRD key facts (Xu et al., NeurIPS 2025 workshop-style, 2509.21487)
- Backbones Llama-3.1-8B, Llama-3.2-3B, Qwen3-8B, Qwen3-4B; LoRA r=16/α=32 on all proj; teacher Gemini 2.5 Flash rationales given gold label (label words forbidden in rationale).
- Loss = β·CE(label via last-input-token pooled head) + α·LM loss over [input, <REASON>, rationale, <ANS>, label].
- Macro SuperGLUE gains: Llama-3B +5.47% rel (77.34→81.57), Llama-8B +1.43%, Qwen3-8B +1.32%, **Qwen3-4B only +0.65% and −1.05% at α=1** (Table 3). Single runs, no CIs (App. A) → sub-1-point gains are within noise [INFERENCE].
- Ablation: label-only LM loss −1.75%; shuffled rationale −5.7%; so benefit comes from aligned rationales.
- QPS pooled vs CoT: 96–142× (Table 5).
- Important caveat for us: **task-specific supervised** (train on each SuperGLUE task), not zero-shot/task-disjoint. No calibration reported. Pooled head has a fixed K classes → incompatible with runtime-defined labels unless we replace it with a per-candidate scoring head (see exp c).

## 2. Experiment designs

Common protocol (all experiments):
- **Task-disjoint split**: split source *tasks/benchmarks* (not rows) of the training corpus (e.g. tasksource-jev-typed-decisions / Open-Jev) into train-tasks / calib-tasks (for temperature) / test-tasks; test tasks never seen in training or checkpoint selection (avoids the AutoJev selection-exposure issue noted in the brief). Also report our command-risk set as an external OOD probe.
- Metrics: accuracy (Choice), Spearman/MAE (Score), NLL, Brier (multiclass), ECE-15 (report but don't select on), risk–coverage AURC, and selective accuracy at 80% coverage. Bootstrap 95% CI over *tasks*, ≥3 seeds for anything with Δ<1 pt.
- Readout for LLM variants: constrained next-token distribution over candidate letter/ID tokens (AutoJev-style), renormalized over valid candidates.

### (a) GLiClass-style joint label encoding vs LLM answer readout
| Arm | Model | Training |
|---|---|---|
| A0 | `gliclass-instruct-large-v1.0` zero-shot (439M) | none — baseline |
| A1 | A0 + fine-tune on our train-tasks, **single-label softmax over label tokens** (replace sigmoid multi-label) + label descriptions as label text | full FT bf16, 1–2 epochs |
| A2 | A1 on `gliclass-base-v3.0` (187M) | same — latency tier |
| A3 | Our existing DeBERTa/ModernBERT cross-encoder (per-label pairs) | same data |
| B1 | Small decoder (Qwen3-0.6B-class or MiniCPM5-2B) with constrained answer readout, LoRA r16 SFT | same data |
| B2 | Same decoder, GLiClass head (`gliclass-qwen-*` style: label-token pooling + scorer) | same data |

Ablate label count K∈{2,4,8,16,32,64} and label-description length on held-out tasks to reproduce/measure the documented dense-label degradation; measure p50 latency on 3090 + ONNX INT8 (reuse repo export path). Decision rule: pick encoder route for the <700M tier if A1 ≥ B1 −2 pt acc at ≥5× throughput.
Cost (3090) [INFERENCE]: A1/A2 ~2–4 GPU-h each at ~200K examples, 512 tok; B1 (0.6B LoRA) ~4–6 h; B2 ~6 h. Total ≈ 20 GPU-h. **Priority: P1.**

### (b) SFT+TS vs GRPO vs calibration-aware RL (vs soft-label KD)
Base: 1.7B–4B decoder (fits 3090 with LoRA; Qwen3-4B GRPO with 8 rollouts needs vLLM colocate — tight on 24 GB) [INFERENCE].
| Arm | Recipe |
|---|---|
| S0 | Hard-label SFT (CE on decision token), no TS |
| S1 | S0 + single temperature fit on calib-tasks (AutoJev-style) |
| S2 | S0 + per-type temperature (Choice/Score/Noul) |
| K1 | **Soft-label KD**: KL(teacher‖student) over candidate distribution, teacher = strongest available open model (e.g. AutoJev-27B/Decider-27B probabilities) or Jev API probs where licensable; T∈{1,2}; mix α·CE(gold)+(1−α)·KL, α∈{0,0.5} |
| K2 | K1 + TS |
| R1 | S0 → GRPO (reward 1/0 on decision), reasoning on |
| R2 | S0 → GRPO + λ·CE(decision; one-hot if correct else uniform), λ∈{5e-4,1e-3,5e-3} (paper recipe) |
| R3 | R2 + TS |
| R4 [our variant] | S0 → GRPO with Brier/log-score reward on the decision-token distribution (proper scoring rule instead of 0/1) |

Hypotheses from paper evidence: S1 ≈ best NLL/ECE on task-disjoint; R1 best raw accuracy but worst ECE; R2 closes most of ECE gap; after TS the gap between S1 and R3 is mainly accuracy. K1 should beat S0 on NLL/Brier on held-out tasks at equal accuracy (teacher dark knowledge over candidates) [INFERENCE]. Primary selection metric: **NLL on test-tasks after TS**, secondary AURC.
Cost [INFERENCE]: S0/K1 1.7B LoRA ~4–6 GPU-h each; teacher-prob generation for KD: 27B teacher won't fit bf16 on 3090 → use existing corpus teacher probs if present (check SargeDev/jev-distill-corpus-v3 for probability fields — other agent) or rent 1×H100/RTX PRO 6000 for ~10–20 h; GRPO arms 1.7B: ~15–25 GPU-h each on 3090, 4B: rent (48–96 GB). Total ≈ 80–120 GPU-h. **Priority: S0/S1/K1 = P0 (cheap, most likely win); R1/R2 = P2 (only after SFT plateau); R4 = P3.**

### (c) DHRD train-time reasoning head
Adapt to runtime labels: instead of fixed-K pooled head, score each candidate with a per-candidate head (pool at candidate-ID token or constrained readout), keep the auxiliary LM loss on [prompt, <REASON>, rationale, <ANS>, label].
| Arm | α (LM weight) | rationale source |
|---|---|---|
| D0 | 0 | — (= S0) |
| D1 | 0.5 | teacher rationales (label-conditioned, label words banned, as in paper) |
| D2 | 1.0 | same |
| D3 | 0.5 | **shuffled** rationales (control; paper shows −5.7%) |
| D4 | 0.5 | LM loss on input only (control for generic regularization) |
Backbone: 1.7B–4B (paper: 4B gains tiny/negative at α=1; 3B Llama largest gain) and optionally 8B QLoRA. Evaluate accuracy + NLL/Brier after TS on test-tasks; watch entailment/rule-following task subsets (paper's biggest gains on CB/RTE/COPA).
Rationale generation: ~100–200K rationales × ~100 tok via an open 27–32B model on rented GPU or cheap API [INFERENCE]; training ~1.5–2× S0 cost due to longer sequences. ≈ 15–25 GPU-h on 3090 + rationale gen. **Priority: P2** (evidence is single-run, task-specific, small gains on 4B; worth it only if D1 > D0 by > seed noise on 3 seeds).

## 3. Priority summary
1. P0: S0 vs S1 vs K1(+TS) on 1.7B–2B decoder, task-disjoint NLL/Brier/AURC — 3090, ~15 GPU-h.
2. P1: GLiClass-instruct fine-tune (softmax single-label) vs small-decoder readout for <700M tier incl. K-scaling and INT8 latency — 3090, ~20 GPU-h.
3. P2: DHRD α∈{0,0.5,1}+controls; GRPO vs CalRL (λ=1e-3) — 3090 for ≤1.7B, rent 48–96 GB GPU for 4B GRPO.
