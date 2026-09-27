# Astra experiment program: beat Jev where its interface promises more than its behavior delivers

## 1. Thesis

Build an open, runtime-rubric decision model, not a five-label classifier with a Jev-shaped HTTP wrapper. The first credible win is **better instruction robustness, calibrated selective decisions, and local latency**, not a claim that 2B parameters will beat Jev on GPQA. The ambitious path is a 12B model plus a structurally permutation-equivariant readout; 27B is an earned escalation, not the starting point.

Evidence fixes the priorities:

- The live **Decision Index v0.2.1** is Jev 57.89, Rune v3 57.44, AutoJev 56.40, zero-training Decider-chat 51.35, Winnow-12B 50.02. The older screenshot is not the current objective. Open models already win 29 of 43 displayed task metrics; Jev retains large knowledge/reasoning advantages. Calibration does not enter the headline index. [Leaderboard audit](../research/leaderboard.md), [live source](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/index.json).
- Jev officially acknowledges context rot, negation, counting, dates, indirection and injection weaknesses. Third-party probes find candidate-order effects and incoherence across equivalent primitives. These are testable openings, not proof that we already beat it. We can make sibling-question isolation and candidate permutation equivariance structural, then determine whether that costs useful listwise reasoning. [Reference research](../research/jev-reference.md), [official weaknesses](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md), [third-party probes](https://archerhume.com/posts/jevs-architecture-unmasked/).
- AutoJev's effective recipe is modest-size curated data, decision-only CE, vocabulary-initialized readout and temperature scaling. Its 73k training set is private and checkpoint selection touched public benchmarks. Rune v2 disclosed rank-8 LoRA self-distillation, whereas v3's training is not sufficiently disclosed to reproduce. Thus neither leaderboard label proves full FT is necessary. [AutoJev](../research/competitor-autojev.md), [Rune](../research/competitor-rune.md).
- Winnow and Jev-Omni share a 12B base but differ by approximately 9.5 index points. Xor loses about 12% request coverage to its 26-option cap. Data and complete typed readout deserve money before custom attention. [Winnow](../research/competitor-winnow.md), [Xor](../research/competitor-xor.md), [top versus middle](../research/top-vs-mid.md).

**We deliberately will not:** pretrain a backbone; copy Jev's undisclosed RLCD; start with RL, diffusion, MoE full FT or sparse-attention replacement; treat soft teacher probabilities as objective truth; claim all-task superiority from a cherry-picked robustness panel; truncate benchmark inputs or remove candidates; train on known public evaluation sources in our clean-transfer track; let the public index choose checkpoints.

All proposed gains, durations, costs and gates below are **[INFERENCE: experimental design or planning estimates]**, not observed results. No training, inference or tests were run for this plan.

## 2. Target and product contract

### Compete in two tiers, keep a third conditional

1. **700M–3B: MiniCPM5-2B, actually 2.517B stored.** Milestone: exceed Decider 2B's 28.97 by reaching **32.0 v0.2.1** after parity is established. More importantly, deliver >=5 percentage points above Jev on a sealed, balanced robustness panel, with no increase in unsafe-allow rate. This is a target, not a forecast.
2. **10B+: Gemma-4-12B-it, about 11.95B stored.** Practical ambitious tier: **>=52.0 v0.2.1**, beyond Winnow-12B, with Brier and AURC superior to our stock 12B baseline and ECE <=0.04 on source-held-out calibration evaluation. Stretch: **>=58.5** and a positive paired confidence interval against a same-edition Jev run. Meeting 52 does not justify saying we beat Jev generally.
3. **Conditional 27B dense challenger:** Qwen3.8-27B, 27.78B stored, only after the 12B recipe beats stock by >=3 points on our independent development macro score and a same-data scaling pilot demonstrates another >=3-point gain. No under-10B claim for Rune's 25.81B or Gemma E4B's 8B.

A **350M–700M encoder** is a cheap falsification/control and potentially a cascade first stage, not our assumed general decision champion. Board best in this tier is only 14.32; encoder supervised results do not establish unseen rubric following. Qwen3.5-4B, actually 4.66B, is the bridge experiment; K2-Horizon is a 5.06B custom-code inference control, not a mandatory training line. [Base inventory](../research/base-models.md), [tier results](../research/leaderboard.md).

### Deployment target

- One local RTX 3090; model fits with >=2GiB operating headroom. MiniCPM BF16 LoRA first, Qwen4B QLoRA if needed. 12B deployment may use validated 4-bit weights; do not assume training-quantized and deployed distributions match.
- For **512-token total prompts, one question, eight candidates, batch 1**: target local p50 <=100ms and p95 <=200ms including tokenization, prefill, readout and serialization; at 2K total tokens target p95 <=500ms. These are acceptance aspirations to measure, not extrapolations of 96GB-board latency. Report failures, rather than shrinking the benchmark input.
- Text-only Choice 1–255 candidates, Score 2–10 ordered levels, Noul Bernoulli; structured state and instructions; question IDs excluded from model input. Probabilities retained in FP32, score is expected zero-based level. Derive Jev-compatible confidence from probabilities, but never use that heuristic as the correctness probability for calibration.
- Question isolation is mandatory. Start with separate rows; optimize shared state prefill only after equality proof. Do not apply a tree attention mask to Qwen DeltaNet and assume recurrent-state isolation. MiniCPM's ordinary Llama attention is the safer experimental architecture.
- Advertise initially validated context limits, not the backbone's nominal 128K/256K. An unsupported request is explicit. Full-index coverage is independently reported and can prevent a release.
- Preserve the existing command-risk classifier/export as a distinct supported product. An arbitrary-label decision model does not fit the existing fixed-five-class Rust artifact contract.

## 3. Evaluation harness before training

### Exact stack and edition handling

Use **apolinario/decision-index** as the authoritative public benchmark runner, not Decider's private regression suite. Pin commit **`19ad28ec9485493cc4f7fc07d91c178f948e6434`**, observed through [GitHub's commit API](https://api.github.com/repos/apolinario/decision-index/commits/19ad28ec9485493cc4f7fc07d91c178f948e6434). At this commit the public package supports **0.2**, not live **0.2.1**. Run its `pipeline --edition 0.2 --engine http` against our `/v1/systemone` implementation. Preserve untouched result rows and environment metadata. Use a custom Engine only for in-process synchronized latency; it must call the same probability path.

Rebuild the suite locally; verify uncompressed base-row hash `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5`, added-row hash `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89`, and exclusions hash `331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81`. Respect HLE access and source redistribution restrictions; a public result bundle is not permission to publish suite text.

Parallel reporting track: obtain the maintainer's v0.2.1 builder/scorer, or implement the published changes in an explicitly named **local 0.2.1 reconstruction**, pinned by commit and manifest. Required differences: 38-task panel; remove RouterBench/SGD; 1.2 gold-task weight; area weights 0.258494/0.258494/0.200229/0.182783/0.1; answerable ToolRet/BRIGHT filtering; Home-appliance duplicate/dev exclusions; ACOS per-review F1; revised baselines and linked-case handling. Matching the displayed aggregate arithmetic is necessary but not sufficient for row/scorer parity. Require official row hashes or raw-result rescoring parity before calling our number official 0.2.1. Until then report official 0.2 and reconstruction separately; **never compare 0.2 output to Jev 57.89**. [Harness limitation evidence](../research/leaderboard.md).

Reuse Decider's Apache-licensed prompt/readout and metric ideas, especially NLL/Brier/AURC and type calibration, but do not substitute its regression accuracy for the Decision Index. Pin its code revision before copying code; preserve attribution. Stock Decider-chat is a model/readout baseline, not the selected evaluation authority. [Decider research](../research/competitor-decider.md).

### Locked splits and leakage protocol

Create immutable manifests before E1. Distinguish:

- Train: up to 200k rows, source/task/family tagged.
- Development: 6k rows from separate source groups/templates, used for checkpoint and trial selection.
- Calibration A/B: 3k+3k from additional source tasks; A fits temperature, B chooses global versus per-type calibration and cascade thresholds. Calibration B becomes development after that decision, never test.
- Sealed transfer test: 8k rows from >=20 entirely unseen source tasks and >=8 held-out generator/rule families.
- Sealed robustness test: 4k independent base scenarios plus predefined paired perturbations. Separate 4k development scenarios expose the same phenomena but not the same rules, entities, documents or generators.
- Command-risk test: existing frozen test plus >=1k independently reviewed new command groups; no reuse of the already analyzed Jev comparison as pristine model-selection evidence. Keep old comparison as historical regression only.
- Public index: one final run per frozen release candidate; weights, prompt, quantization, temperature and policy are committed before opening its results. No subsequent model selection from that release's public score.

Group splits by original document, semantic scenario, translation family and generator lineage, not rendered row. Exclude **whole Decision Index source datasets and aliases** from clean-transfer training, then exact normalized text/candidate-map hashing and 13-word shared-span screening against every public suite and private holdout input. Audit near matches with 5-word MinHash/Jaccard >=0.8; quarantine suspected semantic duplicates for review. Do not blindly reject generic boilerplate matches, but record adjudications. Group-derived examples and option permutations stay together. No metadata/latents/correct-option index enters prompts. Licenses and benchmark exposure travel with rows. Base-model pretraining contamination remains unknowable and must be disclosed.

Use a fixed 1-epoch schedule with checkpoints at 25/50/75/100%; choose on development **task-macro NLL**, subject to <=0.5-point macro-accuracy loss and command-risk constraints. Fit calibration only after choosing weights. Compare trial finalists with 3 seeds; paired bootstrap over source tasks and linked scenario groups, 10k resamples. Public-test results cannot reopen selection. A later release needs a newly sealed test panel. Do not select on the SargeDev `test_set_30k` merely because a research note suggested it; a test used to select is development.

### Metrics and robustness slices

Report official index, per-area and native task metrics, coverage and unsupported reasons. Also report task-macro and pooled **NLL**, **multiclass Brier = mean sum_k(p_k-y_k)^2**, binary Brier separately, 15 equal-width-bin ECE plus reliability counts, classwise ECE, **AURC** from risk ordered by raw max probability, and risk at 50/80/95% coverage. Score adds MAE and ranked probability score; soft-label agreement metrics are separate from correctness against gold. ECE is diagnostic, not the training reward or sole selection metric. Define binary-vs-multiclass Brier conventions explicitly; do not claim parity with the board's partly undocumented calibration aggregation.

For command risk retain runtime-policy critical misses, unsafe allows, approval bypass, allow precision/coverage, and benign interventions. Thresholds are cross-fitted on development/calibration groups. Primary product win: **>=25% relative reduction in unsafe-allow rate at matched benign-intervention rate**; require >=500 dangerous cases and report binomial intervals. Zero observed failures is not proof of zero risk.

Construct a balanced covering design, not an impossible full factorial, with >=200 independent scenarios per named primary slice:

1. Negation, double negation, exceptions, quantifiers, conflicting instruction/criteria.
2. Quoted command versus executed command; permissions, shell expansion, environment-dependent consequences.
3. Dates, hex/decimal, counts, thresholds and multi-hop JSON references with executable gold, never execution of untrusted shell commands.
4. Fixed evidence plus neutral distractors at 256/1K/4K/8K tokens; evidence at start/middle/end.
5. Scope 1/2/4/8 facts; contiguous versus dispersed evidence; aggregation versus single-needle lookup.
6. Candidate count 2/8/32/128/255, near-miss labels, long descriptions, paraphrases, opaque label IDs.
7. Original/reversed/three seeded random candidate permutations; map distributions back to semantic IDs before comparing.
8. Choice versus equivalent Noul; Noul negation complement; Score level-description changes. Only enforce equivalence when the logical construction proves it, not for arbitrary natural-language complements.
9. Prompt injection embedded as state data; unrelated sibling-question addition; structured state-key order.
10. No valid candidate, explicit abstain, overlapping labels and ambiguity; Spanish/Chinese instruction/text mismatch as secondary stress tests, without claiming multilingual product support.

Perturbation gates: top-choice permutation agreement >=99%; mean total-variation distance after remapping <=0.01; minimal-pair both-correct accuracy >=85% on verified development tasks; neutral-distractor accuracy drop <=3 points from 256 to 4K; sibling isolation max probability drift <=1e-5 in FP32 reference or <=1e-3 in validated reduced precision. Do not enforce irrelevant-option independence universally: genuine relative choices can change when options change. For duplicate labels evaluate combined semantic probability, not arbitrary identity wins.

Benchmark latency at 128/512/2K/8K tokens; K=2/8/32/128/255; questions=1/8/32; batch/concurrency=1 and 8. Record p50/p95/p99, prefill GPU time, tokenizer/CPU time, queueing, total synchronized wall time, tokens/s, decisions/s, cold start, peak VRAM and operating power limit. Separate local loopback from Jev WAN measurements. Run the same prompts on the same GPU when claiming speed ratios. [Evaluation ideas](../research/papers-hybrids.md), [classification metrics](../research/papers-classification.md).

## 4. Data program

### Initial 80k-row mixture; expand to 200k only after gates

| Share | Source | Purpose and restrictions |
|---|---|---|
| 35% | `tasksource/tasksource-jev-typed-decisions` | Commercial-permitted, source-license verified rows only; whole public-eval-source exclusion; cap any source at 2%; retain human vote distributions where available. |
| 20% | `ZefanCai/Open-Jev-v1.1` | Non-WANLI structured controls preferentially; WANLI <=5% of total mixture and only after license/provenance approval; group_id and latent rule-family separation. |
| 30% | Newly generated, executable-gold typed decisions | Dates, numerical representations, rule exceptions, scopes, policy hierarchies, document references, shell semantics in a simulator; half minimal-pair scenarios, all variants grouped. |
| 10% | New semantic judgments with teacher distributions | Independent evidence/rubric/ambiguity cases, not benchmark rewrites; balance difficulty rather than retain only confident teacher agreements. |
| 5% | Existing command-risk training groups plus newly reviewed training scenarios | Convert current L1-dangerous to L5-safe taxonomy into explicitly described score levels; do not accidentally reverse safety ordering. |

Target kind ratio Choice/Noul/Score **50/30/20**; 15% of Choice rows K>=32, 3% K>=128; 15% of rows 2–8K tokens reserved for rented/long-context phase. Start local training at <=2K without truncation: shorter cohort is an explicitly different training cohort. Token budget, not row count alone, is the compute control. Reserve 20% general-task replay in any targeted continuation.

**SargeDev is an optional ablation, not the core:** replace 10% of the initial mixture with yuri_v3 Jev soft labels only after TypeSafe/OpenRouter distillation terms are verified. Exclude openjev_v2 duplicate stream and unnamed-teacher yuri_v1 from the clean recipe. The dataset has 740,957 rows but short templated states and <=16 candidates; a million-row imitation run would copy blind spots. Its Apache metadata does not independently settle upstream output rights. Do not discard near-uniform labels indiscriminately: legitimate ambiguity is useful calibration supervision. [Dataset audit](../research/datasets.md).

Teacher first choice: **stock Qwen3.8-27B** in bounded reasoning mode for semantic labels/rationales, versus AutoJev's one-pass candidate distribution as a cheaper control. Evaluate 1k independently adjudicated development items before selecting the teacher. Use a second-family Gemma-4-12B-it check for 20% random rows and all high-risk disagreements. Select by gold NLL/Brier and high-risk error, not agreement with Jev. Tag probabilities as token-derived, verbalized, human-vote, or sampled-frequency; they are not interchangeable. Gold deterministic labels override a conflicting teacher. Retain difficult ambiguous cases with reviewed soft targets; quarantine unresolved semantic labels rather than confidently relabel them.

Base training objective: CE on hard gold; KL(teacher||student), equivalent to soft CE up to a constant, on teacher distributions; BCE for absolute Noul; categorical CE for Score plus a separately ablated 0.1 cumulative-distribution penalty. On rows with both gold and teacher compare hard-only, soft-only and **0.5 CE + 0.5 KL**. Mask padding/candidate slots only, never score all vocabulary positions unnecessarily. Add pair consistency only for transformations with proven unchanged or remapped semantics. No inverse-class cost weighting in the base probability objective: apply safety costs in the decision policy, otherwise probabilities no longer estimate the observed class distribution.

## 5. Experiment ladder

All costs below are incremental GPU compute estimates, excluding generation/API, storage and human review. A local GPU-hour has **$0 rental**, with explicit planning electricity **$0.10/h** (assumed 0.4kW system draw at $0.25/kWh). Rates: Lium PRO6000 Server **$1.29/h**; Runpod Secure PRO6000 fallback **$2.09/h**; Runpod Community A100-80 **$1.19/h**, subject to quote; Lambda 8xH100 **$3.99/GPU-h**. Ranges include listed controls but not arbitrary sweeps. Measure 100 warmed training steps and a 1k-request pilot before authorizing each full job; recost using actual token throughput plus 30% overhead.

### E0. Establish the ceiling and failure map, no training

- Hypothesis: readout/coverage/calibration fixes are competitive with casual fine-tuning; Jev fails systematically on independently generated perturbations.
- Bases: existing ModernBERT risk checkpoint, MiniCPM5-2B, Qwen3.5-4B, K2-Horizon-3.7B (5.06B); remotely AutoJev-27B, stock Qwen3.8-27B, and stock Gemma4-12B. Jev-1.13.0 API is a reference, not a training target.
- Method: candidate-token head initialized from validated single-token vocabulary codes, thinking off; verify all 255 codes in the actual answer prefix. No adapter/full FT. Controls: raw/global/per-type T, one/two option orders, original shipped competitor adapter versus our shared adapter on 500 rows. K2 gets inference only.
- Data: 6k dev + 6k calibration + 4k development robustness scenarios, plus 1k throughput profiling; no public-index checkpoint selection.
- Gate: 100% supported-schema correctness; 255-option path; model-selection opportunity >=3-point error gap on at least two primary robustness slices with >=200 groups each. Teacher must beat small baseline by >=5 macro-accuracy points or >=0.05 NLL to justify distillation. Drop K2 unless it wins >=2 accuracy points at <=1.5x Qwen4B latency.
- Hardware/cost: **3090 8–16h, $0 rent/$0.8–1.6 electricity; PRO6000 6–12h, $7.74–15.48**. API reference allowance $20, separately accounted. Stock reasoning labels are E4 cost, not silently included here.

### E1. Conventional decision SFT, the control we must beat

- Hypothesis: 20–80k clean, described-label decisions plus a proper scoring rule produce most affordable gain.
- Base: MiniCPM5-2B, BF16 LoRA r16/alpha32 on attention and MLP; LR 5e-5, effective 32 examples, 5% warmup, one epoch, 2K context, gradient checkpointing. Train only candidate-slot readout loss; not prompt-token SFT. Calibrate after freezing checkpoint.
- Controls: frozen-backbone trained head; no-training E0; identical-data hard CE versus available soft-target CE. Use 20k nested pilot then 80k (approximately 40M tokens at measured mean 500).
- Gate: >=3-point source-task-macro accuracy gain and >=5% lower NLL than E0; no primary slice >2 points worse; unsafe-allow upper confidence bound not worse at matched intervention.
- Hardware/cost: **3090 18–32h, $0 rent/$1.8–3.2 electricity** across pilot and two main arms. Do not try full 2.517B Adam FT on 24GB merely because a rough 2B table said it fits.

### E2. Architecture falsification: instruction-conditioned encoder

- Hypothesis: a native discriminative model is sufficient for operational decisions even if it cannot win the broad knowledge index.
- Bases: `answerdotai/ModernBERT-large` (~396M) and `knowledgator/gliclass-instruct-large-v1.0` (~439M). Full FT with joint text+instruction+candidate-marker readout and masked candidate softmax; Noul BCE, Score bins. GLiClass default sigmoid is not a Choice distribution.
- Controls: ModernBERT independent per-candidate cross-encoder versus joint markers; zero-training GLiClass; same 80k E1 data and no task-specific prompts. Compare K-scaling separately. Keep 512-token GLiClass/truncation limits explicit; no hidden pruning to claim 255-choice coverage.
- Gate: within 2 points of E1 task-macro accuracy, <=1% relative NLL degradation, >=3x measured throughput at K=8, and no >5-point many-label regression. Otherwise retain only as risk/cascade baseline, not the general model.
- Hardware/cost: **3090 10–18h, $0 rent/$1–1.8 electricity**, approximately 40M tokens/arm, two primary trained arms and one limited cross-encoder control.
- Earned literature slot: GLiClass, Ettin's controlled architecture lesson, BTZSC reranker/cross-encoder counterargument. No extra Ettin/mmBERT/EuroBERT training sweep yet: their marginal value is smaller than controlling current architecture and data. [Classification](../research/papers-classification.md), [sweep](../research/papers-sweep.md).

### E3. Our contrarian readout: equivariant candidate set, isolated questions

- Hypothesis: remove meaningless candidate-order information without losing useful interactions among alternatives.
- Base: E1 MiniCPM5 weights. Architecture: encode state+question as common causal prefix; encode each candidate description as a separate branch with identical branch position coordinates and no candidate-to-candidate attention; pool candidate-end states; apply **two 256-d set-attention blocks with no candidate positional encoding**, scoring each candidate with shared parameters. This is permutation-equivariant by construction, still listwise through set attention, and does not assume irrelevant-option independence. Start with repeated-prefix candidate rows; only then share cached prefix. Keep instructions in the prefix, runtime label IDs in response mapping, not learned class IDs. Noul uses an explicit true/false pair; Score order is supplied as semantic level information, not discarded.
- Method: train head plus r16 LoRA, CE/KL, optional 0.1 consistency loss between random orderings for E1 control. No full FT.
- Controls: E1 sequential candidate-token readout; independent candidate scoring without set blocks; shuffled-order augmentation alone. Same 80k rows, hard negatives and <=2K total serialized budget; record expanded candidate-branch tokens, because independent branches can cost more.
- Gate: permutation TV <=0.005 and agreement >=99.5%; >=3-point improvement on development robustness macro with <=1-point transfer-accuracy loss; p95 <=1.5x E1 at K=32 and memory <=21.5GiB. If it only buys a formal invariant but harms useful decisions, stop. Sibling isolation must pass by construction and measurement.
- Hardware/cost: **3090 24–48h, $0 rent/$2.4–4.8 electricity**, three 20k pilots followed by one 80k winner. This is our own design, not an architecture claimed by a cited paper. Lux/GLiClass motivate candidate-conditioned scoring; Jev order probes motivate the invariant. [Reference](../research/jev-reference.md), [Winnow/head comparisons](../research/competitor-winnow.md).

### E4. Distillation that can beat its teacher's blind spots

- Hypothesis: verified counterfactual data plus teacher uncertainty beats wholesale Jev imitation; aligned training-only reasoning may improve hard decisions without runtime CoT.
- Base: best MiniCPM E1/E3 path, with untouched E1 control. 80k identical rows; 20k hard rows receive full teacher candidate probabilities; 10k receive <=128-token verified rationales, only if E0 teacher gate passed.
- Method: r16 LoRA, gold-only versus soft-only versus 0.5 gold/0.5 KL on the same 20k overlap. Winning arm continued with train-only auxiliary rationale LM loss weight 0.25; decision head reads only the pre-rationale position. No test-time generated rationale. Controls: no rationale, shuffled rationale, equal-token generic text loss on a 5k pilot. An optional 8k SargeDev substitution is reported separately, terms permitting.
- Gate: soft mixture reduces held-out NLL >=5% and Brier >=3%, with accuracy loss <=0.5 points; rationale arm adds >=1.5 accuracy points on hard task-family macro, repeated on 3 seeds, without >2-point context-rot regression. Reject rationale generation if cheap controls explain the gain.
- Hardware/cost: **3090 24–44h, $0 rent/$2.4–4.4 electricity; PRO6000 teacher 12–24h, $15.48–30.96**. Cap external semantic-generation API spend at $100; cannot call this guaranteed sufficient data until a 100-example token-cost pilot establishes it.
- Earned slots: DHRD small controlled arm; LFM2's complete finite-distribution lesson; SiDyP's noisy-label audit, not full simplex diffusion. Reject GRPO/CalRL here: SFT beats CalRL ECE in three of four cited Qwen4B cells, and Decider RL traded broad accuracy for sampled browser success. [Classification](../research/papers-classification.md), [Decider](../research/competitor-decider.md), [sweep](../research/papers-sweep.md).

### E5. Scale to 4B and 12B, same data before more data

- Hypothesis: the quality ceiling is pretrained capability, not a missing exotic optimizer.
- Bases: Qwen3.5-4B instruct (4.66B stored) and Gemma4-12B-it. Use conventional E1 token readout on both first; transfer E3 only after a positive small-model result. Qwen: r32 QLoRA, attention/MLP/DeltaNet projections only after verified module coverage. Gemma: r32 BF16 LoRA all text linear layers; use 80/96GB rented memory. LR 5e-5, same schedule and targets as winning E4.
- Controls: stock corresponding base, same 80k-row recipe, identical 2K cohort; then a single 200k-row expansion including 2–8K examples for winning 12B. Compare BF16 versus deployment 4-bit inference and recalibrate independently. Do not give larger model more data until same-data scaling is measured.
- Gate: Qwen adds >=2 points over MiniCPM development macro at <=1.7x latency; Gemma adds >=4 over Qwen or >=0.05 NLL reduction at acceptable deploy cost. Expanded mixture improves >=2 points without >0.01 absolute Brier regression on any primary type. Final aspirational index milestones: Qwen >=43.5, Gemma >=52.0 v0.2.1 only after genuine parity and final freeze.
- Hardware/cost: **3090 16–28h, $0 rent/$1.6–2.8 electricity; A100-80 24–48h, $28.56–57.12**, or PRO6000 equivalent at measured throughput. Approximately 40M tokens for controlled runs; expanded 200k corpus budget 160M tokens, one epoch.

### E6. Spend extra compute only where it buys lower loss

- Hypothesis: a learned gain predictor beats max-probability escalation and provides a product-level Jev win without pretending the small model is universally better.
- Base: frozen best MiniCPM/Qwen as cheap path, frozen 12B or stock 27B as expensive path. Train a <=1M-parameter MLP on cheap hidden state, token count, K and margin to predict **NLLcheap(gold)-NLLexpensive(gold)**. Data: 30k independent training decisions with both outputs; 6k router development/calibration distinct from base model test.
- Controls: random escalation, entropy, margin, max-p, oracle loss-gain router; escalation budgets 10/20/40%. Select gain-per-added-millisecond threshold on development, calibrate each routed branch and evaluate final mixture on untouched calibration B.
- Gate: at <=20% escalation, recover >=70% of the cheap-to-expensive NLL gap and reduce AURC >=15% relative to cheap alone; average latency <=1.5x cheap and p95 within declared product SLA. Require >=10% NLL improvement over entropy router at matched escalation. Include model loading or dual-GPU residency in timing; no free hot 27B cache assumption on a 3090.
- Hardware/cost: **3090 4–8h, $0 rent/$0.4–0.8 electricity; PRO6000 labeling 4–8h, $5.16–10.32**. Deployment may be 3090 plus remote paid large-model service; report this as a cascade, not a single local 2B model or single-tier board entry.
- Earned slots: ODA's gain target; speculation paper only supplies the cost-benefit analogy, not a claimed speculative-decode speedup. [Hybrid papers](../research/papers-hybrids.md).

### E7. Cheap efficiency research after accuracy stabilizes

- Hypothesis: 75%-depth MiniCPM with decision-matched distillation recovers nearly all quality while reducing prefill.
- Base: winning MiniCPM; compare full 42 layers versus 32 and 21 layers; r16 LoRA healing, 40k decision rows, KL to full-depth probabilities plus gold CE, same K/dispersion/aggregation mix.
- Controls: truncation without healing, full-depth equal-token continuation. Gate: >=25% lower measured 2K prefill latency, <=1-point task-macro accuracy loss, <=2% NLL increase, no primary worst-slice loss >2 points. Retain full-depth model if the kernel-level gain is too small.
- Hardware/cost: **3090 12–24h, $0 rent/$1.2–2.4 electricity**.
- Optional long-context selector is not automatically authorized: only if >8K prompts are >=10% traffic and >40% measured latency. Then a separate 20k-example 200M scorer trial, 3090 6–12h ($0 rent), must show >=30% latency gain with <=1-point aggregation accuracy loss; its evaluation must never alter official no-pruning benchmark inputs. Label importance by actual loss impact, not attention weight alone.
- Earned idea: DeepSeek-V4.1 early readout analogy; Sparse Frontier stress axes. Reject YOCO/CSA/NSA retrofit, FlashMemory architectural replication and LoLCATs linearization: evidence is long-context/pretraining scale, and FlashMemory MRCR drops 76 to 48. Retrieval-head diagnostics get at most a 2h local investigation if E7 specifically loses dispersed-evidence accuracy, not another whole training line. [DeepSeek/NSA](../research/papers-deepseek-nsa.md), [hybrids](../research/papers-hybrids.md).

### E8. Ambitious 27B challenger, only after evidence

- Hypothesis: a clean, robust decision recipe at stronger dense scale can clear Jev rather than just specialize below it.
- Base: Qwen3.8-27B (not AutoJev's already benchmark-selected checkpoint). First r32 BF16 LoRA on PRO6000, 80k same-data cohort, compare frozen stock and E5 Gemma. Then, only if LoRA is within 2 points of the sealed-development target and training loss suggests capacity limitation, compare full FT on the same 200k/160M-token cohort.
- Controls: same data/readout/schedule LoRA versus full FT; two seeds for LoRA, one full pilot followed by replication only if >=1.5-point gain on independent dev. Full FT candidate-only CE/KL, LR 2e-6, 5% warmup, conventional FSDP/ZeRO3 Adam state sharding; no single-H200 memory fiction.
- Gate: >=3-point development gain over 12B, <=0.04 ECE by type, >=5-point robustness gain over Jev on the sealed paired panel, then one-shot **>=58.5 v0.2.1** with uncertainty. Broad superiority claim additionally requires same-protocol Jev results and positive grouped-bootstrap difference; a scalar alone is not enough.
- Hardware/cost: **PRO6000 32–64h, $41.28–82.56** for LoRA and small inference pilots; conditional **8xH100 node 12–24 wall-h = 96–192 GPU-h, $383.04–766.08** at Lambda advertised rate. Validate topology and actual node quote; CPU RAM and checkpoint volume >=1TB for full-optimizer snapshots. No assumption that MoE active size makes its full training cheaper to implement.

### E9. Frozen release falsification, not another optimization trial

- Base/method: selected trained model and fixed quantized deployment artifact; no weight updates, no calibration refits on tests. Run sealed transfer/robustness/risk panels plus full public suite, raw and calibrated model, with baseline outputs fixed in advance.
- Data: 8k transfer, 4k robustness base scenarios plus variants, risk test and 120,340 live-edition requests if v0.2.1 parity is available; otherwise use the pinned 0.2 counts and label them correctly.
- Gate: all claimed tier/product gates and <=0.1% non-capacity schema/runtime failures; all unsupported capacity cases remain scored wrong. Deployment quantization <=0.5-point accuracy loss and <=0.01 absolute Brier increase; permutation/isolation gates still pass. Release negative results if targets fail; do not tune on failures and call the rerun held out.
- Hardware/cost: **3090 6–12h, $0 rent/$0.6–1.2 electricity; PRO6000 24–72h, $30.96–92.88** for one final candidate, baseline/precision comparisons and broad-suite overhead. The broad Rune path alone has a 46.3h serial-equivalent published mean; do not budget every model at six hours. Jev reference calls capped separately at $30, measured from actual usage.

## 6. Hardware decision and budgets

The local note observes **torch CUDA working despite broken NVML**, caused by loaded 595.84 kernel versus 595.91 userspace after upgrade. Do not rerun to dispute that finding. Schedule a reboot and verify recovery before a long job; avoid risky module unloading on a workstation in use. Current lock has Transformers 4.57.6 and lacks PEFT/TRL/bitsandbytes/vLLM; introduce an isolated decision-training extra/environment with model-supported versions. Do not accidentally break current ModernBERT export to pursue a new model. [3090 audit](../research/hw-3090.md).

Run CPU manifests, dedup, scoring and calibration locally; encoder full FT, MiniCPM LoRA, Qwen4B QLoRA, readout research and router fitting on the 3090. 60GB RAM and 440GB free disk do not support casual 27B full-optimizer offload/checkpoint sprawl.

**Exact rental trigger:** E0 gets one bounded 6–12h large-model reference rental because falsifying the small-model/teacher premise requires a BF16 large comparator. Further rental occurs only after (a) E1's >=3-point/5%-NLL gate passes, and (b) a named E4/E5 job has measured >21.5GiB peak or >48h local forecast, or requires the reference precision unavailable locally. For calendar acceleration, demand >=2x useful completed experiments/day after setup. No 4090 rental to solve a 24GB OOM. E8 has its additional model-quality gates above.

Provider choice: **Lium RTX PRO6000 Server 96GB at <=$1.35/h quoted GPU cost** for reference/final evaluation and memory-comfort LoRA, persistent mounted HF cache/checkpoints; confirm storage/egress before purchase. Fallback **Runpod Secure $2.09/h**, not the $1.69 Community tariff mislabeled secure. Cheap **Runpod Community A100-80 $1.19/h** for restartable 12B LoRA if its complete quote is favorable; fallback PRO6000. If privacy/reliability rules require Secure, use its $1.59 A100 rate and update the budget. Vast is a conditional price comparator with reliable-host filters, not a promise of the minimum advertised price. Daytona is real GPU compute but the verified prices are preemptible and CPU/RAM may be additive; skip for first long runs. Modal serverless adds CPU/RAM and orchestration costs; not needed for continuous training. [Astra hardware audit](../research/hw-rental-astra.md), [second opinion](../research/hw-rental-sol.md).

Budgets are **cash caps, not commitments to run every arm**. They exclude labor and tax, include explicit data/annotation allowances, and treat owned hardware capital cost as sunk:

| Scenario | Authorized scope | Local GPU-h | Rented allocation and compute | Other + reserve | Total planning cap |
|---|---|---:|---|---|---:|
| Lean | E0, E1, one E2 arm, three 20k E3 pilots then one winner, limited E4 soft-target arm, E9 single small release | 90 | 36 PRO6000-h = $46.44 | $9 electricity + $100 data/annotation + $15 storage + $79.56 contingency | **$250** |
| Medium | E0–E7 with gates, finalist seeds, E9; no 27B full FT | 210 | 110 PRO6000-h = $141.90; 48 A100-h = $57.12 | $21 electricity + $250 data/annotation + $40 storage/transfer + $189.98 contingency | **$700** |
| Ambitious | Medium plus E8 LoRA/full-FT comparison, independent release confirmation | 260 | 200 PRO6000-h = $258; 64 A100-h = $76.16; 192 H100 GPU-h = $766.08 | $26 electricity + $750 data/annotation + $150 checkpoint storage/transfer + $473.76 contingency | **$2,500** |

These allocations may cover the lower/middle trial-duration ranges, not every maximum and every rejected branch. If measured throughput exhausts a scenario, stop at its predeclared gate; never silently skip controls or tests and call the full program complete. Replacing all medium PRO6000 hours with Runpod Secure adds $88, funded from contingency. Full E8 training memory uses total stored parameters: conventional Adam can approach ~500GB before activation/temporary buffers; eight H100s are an actual sharded node, not eight independent cheap marketplace GPUs. Budget checkpoint/download/setup time and retain independently copied resume checkpoints before releasing the machine.

## 7. Risks and kill criteria

1. **No architecture signal:** E3 fails to beat order-augmented E1 after 3 pilots, or exceeds 1.5x p95 at K=32. Stop custom readout work, ship the conventional model. Structural elegance is not a consumer benefit on its own.
2. **Teacher imitation ceiling:** E4 lowers agreement loss but not gold NLL/Brier, or harms dangerous minority cases. Drop the teacher stream; spend on verified counterfactuals. Jev-output rights unresolved means SargeDev distillation remains excluded, not hand-waved.
3. **Narrow transfer:** >5-point gap between unseen-task and unseen-family performance, or broad performance drops >2 points after targeted continuation. Increase replay only in a new preregistered trial; do not declare a general model from domain wins.
4. **Calibration failure:** global ECE improves while worst-type Brier rises >0.01 or high-confidence errors rise. Reject the calibration map. A large temperature that makes everything uncertain is not sufficient if AURC/operational utility does not improve.
5. **Safety uncertainty:** fewer than 500 dangerous held-out cases or confidence bounds cannot establish matched-intervention improvement. Claim only provisional accuracy gains, never safer deployment. Human rubric disagreement remains visible.
6. **Scale does not pay:** 12B adds <4 dev points over 4B or cannot meet validated deployment latency/memory. Stop at small model/cascade; E8 is not authorized. K2 custom-code adaptation gets no open-ended engineering allocation.
7. **Index mismatch:** v0.2.1 artifact/scorer parity unavailable. Release reproducible 0.2 and private-panel findings; leave live ranking claim unmade. This is a comparison blocker, not an excuse to relabel scores.
8. **Benchmark leakage:** any exact held-out duplicate in training blocks release pending removal and complete retraining of affected model. Source overlap that is merely pretraining-unknown is disclosed, not falsely claimed solved.
9. **Framework/runtime fragility:** model loader or expert/DeltaNet kernels cannot reproduce raw logits between training and serving within tolerance. Resolve on a 100-row pilot or abandon that base; do not train first and hope export works.
10. **Ambition test:** after medium program, if we cannot outperform Jev by >=5 points on any predeclared robustness macro and cannot offer >=2x local product latency improvement with acceptable accuracy, stop calling this a Jev replacement. Publish the open recipe and position it as a narrower command-risk decision model.

## 8. First two weeks in this repository

Grounding: current `modeling.py` uses `AutoModelForSequenceClassification`, fixed `NUM_LABELS`, and command-only tokenization with truncation; `data/build.py` has group-aware train/validation/test manifests; README describes validation-only temperature scaling, cross-fit policy selection and fixed-five-class ONNX schema v2. Reuse those disciplines, not their task-specific shape. Files inspected: `README.md:239-329`, `src/model_gym/modeling.py:1-130`, `src/model_gym/data/build.py:1-110`; `find` also located `cli.py`, `train.py`, `evaluate.py`, `metrics.py`, `policy.py`, `data/schema.py` and `export.py`. Code below is proposed, not already present.

**Day 1:** Recover driver via scheduled reboot; pin separate decision dependencies and Docker image, tokenizer/model revisions, HF dataset SHAs and harness commit. Write `src/model_gym/decisions/schema.py`: immutable typed question/candidate/target/provenance records, explicit candidate mapping, validation, Score/Noul conversions. Extend `cli.py` with a cohesive `decision` command group, not a second unrelated executable. Existing risk commands remain unchanged.

**Day 2:** Write `decisions/data.py`: three dataset adapters, license allowlist, duplicate grouping, whole-source blocklist and immutable manifests. Adapt command-risk records only at this boundary. Generate tiny probability/gold examples by hand and verify sum/type/order semantics. Write `decisions/probes.py` for executable-gold rule/date/count/negation and shell-simulator scenarios; no actual execution of hostile shell text. Separate dev and sealed generator families now.

**Day 3:** Write `decisions/evaluate.py` with candidate-aware NLL/Brier/ECE/AURC, group bootstrap and all coverage denominators. Reuse stable numeric temperature fitting from existing calibration where compatible, but put variable-K masking in `decisions/calibration.py`. Write `decisions/index_engine.py` and the System One HTTP adapter. Build/import pinned official 0.2 suite and store it privately; scope v0.2.1 parity evidence independently of training. Test real probability boundary behavior, not wiring assertions.

**Day 4:** Implement conventional `decisions/modeling.py` readout, 255-code tokenizer validation, no-generation inference, explicit capacity errors and immutable model/tokenizer hashes. Run actual API smoke scenarios with 2/32/255 candidates, 1/8 sibling questions and Score/Noul. Write `decisions/profile.py` for synchronized end-to-end timings. Profile MiniCPM/Qwen/K2 on development cases; no public benchmark tuning.

**Day 5:** E0 bounded PRO6000 reference job; compare stock 12B/27B and AutoJev on identical dev records. Query Jev only on predeclared reference cases. Record native adapter differences rather than repairing them with label-aware prompt tricks. Teacher calibration audit on 1k reviewed cases. Decide whether E1, teacher and E3 gates are plausible.

**Days 6–7:** Add `decisions/train.py` for candidate CE/KL, dynamic K batching, LoRA-only updates, example/token accounting, grouped checkpoint metrics and portable manifests. Run E1 20k then 80k; E2 one encoder arm while CPU data auditing proceeds. Keep tiny contract tests for label remapping, candidate masks, isolated questions, score expectation, split leakage and proper losses. Actual forward/backward and serving smoke exercises are required; passing unit tests alone is not proof.

**Day 8:** Select E1 checkpoint using frozen development rule, then fit global/per-type T on calibration A/B. Freeze and retain raw/calibrated outputs. Review worst-slice failures without reading sealed tests. Implement E3 as a separate model class within `decisions/modeling.py` initially; split candidate-set model into its own file only if it becomes an independent substantial implementation. Do not modify Qwen recurrence to imitate a tree mask.

**Days 9–10:** E3 three 20k pilots; early stop losers numerically. E4 gold/soft small matched experiment and rationale control only if teacher advantage is real. Use shared-prefix caching only after candidate-row reference logits agree. Extend invariance tests and report expanded token cost so shared encoding does not conceal candidate scaling.

**Day 11:** E5 Qwen4B pilot locally; launch 12B 80k run only after rental gate. Prepare persistent cache/checkpoint volume and independent export before renting. Reuse SSH/bootstrap scripts after examining their exclusions: existing sync does not carry `runs/`/`artifacts`, so explicitly transfer outputs and hashes before termination.

**Day 12:** Finish two additional finalist seeds or, if throughput makes that impossible, report no stability claim and continue the predeclared work rather than pretending the two-week calendar is proof. Build 30k paired-loss data for E6 if medium path is funded; no public-test outputs become router training data.

**Day 13:** Freeze release candidates, prompt, tokenizer, precision, calibration and policy. Quantized local inference parity and real HTTP load smoke. A decision model gets a new explicit deployment artifact contract; leave existing `export.py`/Rust five-level schema untouched. Encoder ONNX proof is optional only if selected for deployment, and must preserve candidate-aware behavior instead of exporting a fake fixed head.

**Day 14:** One-shot sealed transfer/robustness/risk evaluation and start E9 final public run on exact-GPU rental; broad suite may extend past day 14 at measured throughput. Publish findings with clear edition, uncertainty, exposure, licenses, memory and latency. Update README with actual new commands and model contract, record performed experiments and limitations in a release model card. If a gate fails, publish that result; do not reopen the sealed panel to rescue a release claim.

### Final decision rule

The medium program is the recommendation. It risks hundreds, not thousands, of dollars to establish whether **robust runtime semantics plus a competent pretrained backbone** is the missing ingredient. The ambitious node rental is earned only by a clean 12B scaling result. Our strongest plausible win is a model that follows changed rubrics, handles all 255 candidates, knows when to defer, and can be audited and run locally. Beating Jev's knowledge score is a separate, substantially harder claim.
