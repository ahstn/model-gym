# Other 2025–2026 papers and reports: adversarial sweep for an open decision model

## Scope and evidence discipline

Primary-source literature research, not benchmark replication. No repo edits, model weights, or datasets downloaded; no runtime/model claims independently measured. All proposed experiments and extrapolations below are **[INFERENCE]**. Reported numbers are authors' results, not our results. Sources were loaded directly unless explicitly labeled otherwise. This is 15 numbered research items; closely related older foundations and companion papers are identified separately rather than represented as new discoveries.

**Bottom line [INFERENCE]:** challenge the assumption that the next trial should be another 4–27B generative SFT model. Run a genuinely task-held-out comparison of a compact instruction-conditioned encoder, a relevance reranker, and a decoder with identical decision targets. Test output semantics and calibration before architecture cleverness. The strongest counterargument to encoder enthusiasm is that strong supervised encoder classification is not evidence of following arbitrary runtime rubrics; the strongest counterargument to decoder enthusiasm is that next-token generation is not necessarily the most efficient way to score labels.

**Brief consistency:** this sweep does not independently verify leaderboard scores. The brief's shorthand “diffgemma 26B” has a real Google counterpart, but the official report specifies **25.2B total / 3.8B active**, not a dense 26B; its generation-speed claims are not classification latency claims ([report](https://arxiv.org/abs/2608.00146), [model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)). No loaded brief URL returned 404 in this sweep. Related-paper discovery had noisy search summaries: mmBERT's direct abstract says **over 1,800 languages**, not the narrower initial-stage mix; LFM2's direct record is **November 2025**, not a 2026-only release ([mmBERT](https://arxiv.org/abs/2509.06888), [LFM2](https://arxiv.org/abs/2511.23404)).

## 1. Ettin / Seq vs Seq — July 2025, JHU-CLSP

**Evidence.** Paired encoder/decoder models, 17M–1B parameters, same data and recipe, up to 2T tokens; authors report a 400M encoder beating a 1B decoder on MNLI and poor efficiency of converting between objectives through continued training. They release data/order and 200+ checkpoints ([paper](https://arxiv.org/abs/2507.11412), [repository](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder)).

**Why relevant [INFERENCE].** Most comparisons of our ModernBERT with Jev confound architecture, pretraining, instruction data, and output heads. This is unusually good evidence for controlling those confounds.

**Concrete experiment [INFERENCE].** Train Ettin encoder-400M and its matched decoder on identical instruction+document+candidate descriptions. Compare (a) per-candidate scalar cross-encoder scores, (b) joint candidate-marker readout, and (c) decoder answer-token scores. Include existing ModernBERT under the same training set. Hold out complete task families, not just rows; measure NLL, Brier, macro-F1, risk/coverage, batch-1 latency and throughput.

**Limit.** MNLI/classification evidence is not proof of arbitrary-label zero-shot instruction following; the paper establishes architecture advantages on its evaluation, not on Jev typed Score/Noul contracts ([paper](https://arxiv.org/abs/2507.11412)). **Contrarian [INFERENCE]:** do not spend our budget “encoderizing” a decoder before testing native encoders.

## 2. mmBERT — September 2025, JHU researchers

**Evidence.** Encoder trained on 3T multilingual tokens across >1,800 languages; inverse mask-ratio and temperature-sampling schedules, adding >1,700 low-resource languages during decay. Abstract reports strong classification/retrieval and comparisons with o3/Gemini 2.5 Pro on low-resource classification ([paper](https://arxiv.org/abs/2509.06888)).

**Why relevant [INFERENCE].** Tests whether a small discriminative model can generalize label semantics beyond English without paying decoder-scale compute.

**Concrete experiment [INFERENCE].** Use the same text-label scoring head as item 1; train instructions in a few languages, then evaluate unseen languages, English labels paired with non-English text, non-English labels paired with English text, and code mixed with prose. Compare translated versus original rubrics to locate tokenizer/semantics failures.

**Limit.** The headline comparison must not be retold as “mmBERT generally equals frontier models”; it is bounded to reported classification conditions ([paper](https://arxiv.org/abs/2509.06888)). **[INFERENCE]** A multilingual model may lose English command-risk efficiency at equal size, so it is a multilingual stress-control, not an automatic replacement.

## 3. EuroBERT — March 2025, European academic/open-lab collaboration

**Evidence.** Multilingual encoders for European and widely spoken global languages, code and mathematics evaluations, native 8,192-token context, released training framework and intermediate checkpoints ([paper](https://arxiv.org/abs/2503.05500)).

**Why relevant [INFERENCE].** A second encoder family checks whether an observed gain comes from bidirectionality versus a particular English pretraining recipe; code/math exposure is potentially relevant to shell-command decisions.

**Concrete experiment [INFERENCE].** Match parameter/latency budgets to Ettin/ModernBERT; build long command-history tasks with decisive evidence at start/middle/end, conflicting permissions, quoted malicious commands versus executable commands, and English/multilingual policy instructions.

**Limit.** Native context capacity does not demonstrate reliable instruction-conditioned aggregation across that context ([paper](https://arxiv.org/abs/2503.05500)). Keep total tokenized length and truncation policy identical where feasible.

## 4. BTZSC — March 2026, zero-shot classification benchmark

**Evidence.** Compares 38 checkpoints from four families on 22 datasets. Authors report Qwen3-Reranker-8B macro-F1 0.72, instruction LLMs up to 0.67, strong GTE embedding accuracy/latency, and NLI cross-encoder plateau with scaling ([paper](https://arxiv.org/abs/2603.11991)).

**Why relevant [INFERENCE].** The most important missing baseline may be a reranker, not a newer chat model. Labels can be treated as candidate relevance descriptions.

**Concrete experiment [INFERENCE].** Score `(instruction + document, candidate description)` using a small Qwen reranker, a compact embedding model, Tasksource NLI, and decoder constrained readout. Add hard label pairs distinguished only by negation, quantifiers, policy exceptions, or severity. Audit prompt reuse and task overlap before claiming zero-shot. Run both per-label independent and mutually exclusive normalization.

**Limit.** These are sentiment/topic/intent/emotion results, not calibrated typed probabilities, ordinal scores, or null detection ([paper](https://arxiv.org/abs/2603.11991)). **[INFERENCE]** Relevance can prefer a topically matching but logically incorrect label; this is exactly why command-risk exceptions matter.

**NLI foundation/provenance note.** Tasksource is older than this sweep's date window; its model card explicitly includes 600+ tasks and many common benchmarks, including Banking77, GoEmotions, AG News, MMLU, and OpenBookQA, in training metadata. A row-held-out benchmark on those sources is not an unseen-task test ([card](https://huggingface.co/sileod/deberta-v3-base-tasksource-nli), [project](https://github.com/sileod/tasksource)). **[INFERENCE]** Recast tasks as instruction-conditioned entailment but split by source task before recasting and deduplicate across aliases.

## 5. Qwen3 Technical Report — May 2025, Qwen

**Evidence.** Dense and MoE open-weight models from 0.6B to 235B; unified thinking/non-thinking modes, adaptive thinking budget, smaller models learning from flagship knowledge, Apache-2.0 release ([report](https://arxiv.org/abs/2505.09388)).

**Why relevant [INFERENCE].** Same model supports a clean test of whether thinking actually helps decisions rather than changing model families.

**Concrete experiment [INFERENCE].** On Qwen3-1.7B/4B, compare no-think answer-token readout against 64/256-token reasoning budgets and 2/4-sample voting at equal wall-clock budgets. Train a gate on held-out **change in decision log loss**, not merely entropy, for escalation. Preserve final-answer token probabilities after reasoning; compare them against pre-reasoning probabilities.

**Limit.** The report's math/code/agent strengths are not calibration guarantees ([report](https://arxiv.org/abs/2505.09388)). **[INFERENCE]** Thinking may make confident wrong answers more stable; majority-vote agreement is not a probability target without validation.

## 6. Gemma 3 Technical Report — March 2025, Google DeepMind

**Evidence.** 1B–27B family, distillation training, increased local/global attention ratio to reduce KV cache, substantial instruction/multilingual improvements ([report](https://arxiv.org/abs/2503.19786)).

**Why relevant [INFERENCE].** Distilled pretrained competence may matter more than output-head novelty. A compact Gemma offers an independent teacher/student family rather than only Qwen-on-Qwen distillation.

**Concrete experiment [INFERENCE].** Compare Gemma-3-4B-IT with Qwen3-4B using identical candidate-ID verbalizers, then jointly train final-token scalar heads with the same decision data. Distill identical teacher decision distributions into both; include conflicting policies and dispersed evidence to stress local/global attention.

**Limit.** Reduced KV memory is not evidence of lower end-to-end classification latency; abstract family-level context statements should not be applied blindly to every size ([report](https://arxiv.org/abs/2503.19786)). **[INFERENCE]** Do not introduce a whole new sparse-attention architecture before trying an already distilled base.

## 7. DiffusionGemma Technical Report — July/August 2026, Google DeepMind

**Evidence.** Fine-tunes Gemma 4 MoE (25.2B total/3.8B active); 256-token parallel canvases; two-stage bidirectional denoising SFT then RL/sampler distillation; reported roughly 20 tokens/forward and 1,500 output tokens/s on H100. Official card describes autoregressive prompt encoder with cached context and bidirectional canvas decoder ([report](https://arxiv.org/abs/2608.00146), [card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)).

**Why relevant [INFERENCE].** Gives a route to parallel label-slot scoring with shared prompt processing, rather than autoregressively generating probabilities.

**Concrete experiment [INFERENCE].** Compare one masked categorical answer slot, multiple yes/no label slots in one canvas, and independent label queries with shared prompt cache if supported. Measure label-order permutations, duplicate labels, candidate count 2/5/20/100, and denoising steps 1/2/4/8. Contrast logits before/after refinement; add matched autoregressive Gemma 4 as a control if accessible.

**Limit.** High long-output tokens/s does not imply a fast single-label decision. Weight memory depends on total rather than active parameters; 25.2B×2 bytes is about 50.4GB before runtime state **[INFERENCE arithmetic]**, so not a BF16 24GB-local candidate. Card reports capability losses versus parent, e.g. MMLU-Pro 77.6 vs 82.6 and MRCR 32.0 vs 44.1 ([card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)).

## 8. Discrete Diffusion Language Models Are Training-Free Multi-Label Classifiers — 2026, Pawan Kumar / IIIT Hyderabad

**Evidence.** dLLM-SetScore uses one yes/no masked position per document-label pair, LLaDA-8B/Dream-7B, and **200 labeled validation examples for prompt/temperature/threshold selection**. Finds all-masked multi-slot first-position collapse (99.4% positive GoEmotions; 100% Reuters); independent per-label readout avoids ordering asymmetry. Joint Set Refinement was harmful in tested configurations ([full paper](https://arxiv.org/html/2608.14649v1)).

**Why relevant [INFERENCE].** The useful diffusion insight is not “generate all decisions at once”; naive simultaneous slots can fail catastrophically. This directly tests a likely leaderboard reproduction trick.

**Concrete experiment [INFERENCE].** Reproduce only the readout ablation: same prompt/evidence, independent label slots versus joint canvas, randomized label order, calibrated versus uncalibrated scores. Use a disjoint calibration source, and separately report target-task validation tuning rather than calling it pure zero-shot.

**Limit.** It is training-free backbone use, not label-free validation. Independent label querying adds compute with label count; results do not automatically transfer to DiffusionGemma's encoder/canvas architecture. The paper's explicit negative refinement result is a warning against assuming extra inference steps help ([paper](https://arxiv.org/html/2608.14649v1)).

## 9. SmolLM3 engineering report — July 2025, Hugging Face

**Evidence.** Fully open 3B model, 11T-token-scale pretraining, dual think/no-think, six named languages, long-context extension; open recipe includes synthetic Qwen3-32B reasoning data, SFT, APO preference optimization, and merging. Report documents reasoning training degrading long-context performance and a 0.9/0.1 merge recovering RULER performance ([report](https://huggingface.co/blog/smollm3)).

**Why relevant [INFERENCE].** A reproducible small decoder and concrete evidence that adding reasoning can damage a capability our decisions may need.

**Concrete experiment [INFERENCE].** Include 3B SmolLM3 in decoder baseline, with no-think final-token classification. Compare direct decision-only SFT against identical examples augmented with train-time rationales. Retain long-context evidence tests across checkpoints; if merging is attempted, select on held-out NLL and evidence retrieval, not chat scores.

**Limit.** Open recipes do not imply small-scale reproducing of pretraining is practical; this is a fine-tuning base/control, not a suggestion to repeat its full training ([report](https://huggingface.co/blog/smollm3)). **[INFERENCE]** Model merging can change calibration even when accuracy recovers.

## 10. Olmo 3 — December 2025, AI2

**Evidence.** 7B/32B fully open models; release includes model lifecycle, checkpoints, training data points and dependencies; targets long-context reasoning, instruction following, tools, coding, knowledge ([report](https://arxiv.org/abs/2512.13961)).

**Why relevant [INFERENCE].** The value is experimental traceability and contamination auditing, not necessarily best deployable model per GPU dollar.

**Concrete experiment [INFERENCE].** Use 7B stage checkpoints as a teacher-control: base versus instruct versus available reasoning stages, matched decision prompts. Determine whether decision calibration is gained/lost in post-training. Select a source-held-out subset whose training provenance can be audited; compare teacher distributions against opaque API teachers on the same human-reviewed examples.

**Limit.** Fully open does not mean contamination-free; general report strengths do not establish superior decision quality ([report](https://arxiv.org/abs/2512.13961)). **[INFERENCE]** Do not prioritize 32B local training just to satisfy an openness preference; use open provenance to explain errors.

## 11. Nemotron Nano 2 — August 2025, NVIDIA

**Evidence.** Hybrid Mamba-Transformer; 12B base pretrained on 20T tokens, compressed/distilled into 9B. Authors target 128K BF16 inference on 22GiB A10G and report up to 6× inference throughput versus comparable models in **8K-input/16K-output reasoning**, with checkpoints and much training data released ([report](https://arxiv.org/abs/2508.14444)).

**Why relevant [INFERENCE].** Strong test of the “hybrids are faster” hypothesis and training-data source for reasoning distillation.

**Concrete experiment [INFERENCE].** Profile 9B no-think/short-answer decision inference at batch 1 and batched, contexts 512/2K/8K. Compare prefill time separately from answer production to Qwen3-8B; measure long-range negation/exception recall before choosing a hybrid teacher or student.

**Limit.** The 6× claim is explicitly a long-output reasoning workload, not one-token classification; A10G capacity claim is inference, not full training ([report](https://arxiv.org/abs/2508.14444)). **[INFERENCE]** Renting for this family is premature without a measured prefill advantage.

## 12. LFM2 — November 2025, Liquid AI

**Evidence.** Hardware-aware design chooses short gated convolutions plus a minority of GQA blocks; dense 350M/700M/1.2B/2.6B and 8.3B-total/1.5B-active MoE, 32K context. Report describes tempered decoupled Top-K knowledge distillation addressing support mismatch, and CPU prefill/decode gains up to 2×. Hardware search says extra SSM/linear-attention machinery did not improve its on-device Pareto frontier ([abstract](https://arxiv.org/abs/2511.23404), [architecture/report](https://arxiv.org/html/2511.23404)).

**Why relevant [INFERENCE].** Most contrarian small-model candidate: cheap local operations plus occasional attention may be sufficient; “sophisticated state space” is not automatically better. Its truncated-distribution distillation issue is directly relevant to API logprobs.

**Concrete experiment [INFERENCE].** Trial LFM2-1.2B or 2.6B as decision-only student. For API teachers, prefer the complete finite **candidate distribution** over vocabulary Top-K logits; compare hard label, candidate soft KL, and mixed gold/soft loss. If only Top-K vocabulary logits are available, preserve an explicit omitted-mass bucket rather than pretending renormalized top-K is the full teacher distribution. Track CPU latency separately from GPU latency.

**Limit.** Liquid's CPU/mobile wins do not prove RTX-3090 wins; short-convolution architecture may need backend-specific kernels. Transferring its vocabulary KD objective to candidate distributions is a proposal, not a reproduced result ([report](https://arxiv.org/html/2511.23404)). **[INFERENCE]** Teacher-written JSON probabilities, sampled-vote frequencies, and API token probabilities are different supervision objects and must be tagged separately.

## 13. Granite 4.0 release/technical documentation — 2025–2026, IBM

**Evidence.** Official repository releases Apache-2.0 dense, dense-hybrid, and MoE-hybrid base/instruct variants, discusses SFT/RL/merging, coding/FIM, tools and structured JSON. Loaded README still has `extended evaluation ... [here](URL)` and a commented future technical-report sentence; therefore **a standalone formal Granite 4.0 paper was not verified** ([official repo](https://github.com/ibm-granite/granite-4.0-language-models), [official announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)). This item is an official technical release report, not an invented arXiv citation.

**Why relevant [INFERENCE].** Dense versus hybrid members provide a practical enterprise/coding-oriented contrast without switching providers.

**Concrete experiment [INFERENCE].** Compare `granite-4.0-micro` with `granite-4.0-h-micro` under identical decision data and no generated JSON: scalar or candidate-token readout. Include tool-availability/abstention, shell permissions, and natural-language policy violations. Match memory and runtime settings; report prefill and p95 end-to-end latency.

**Limit.** Structured JSON support does not establish calibrated probabilities, and the visible evaluation/report placeholder prevents treating the README as a complete formal empirical paper ([repo](https://github.com/ibm-granite/granite-4.0-language-models)). **[INFERENCE]** “Enterprise ready” is not evidence for command-risk robustness.

## 14. SiDyP: Calibrating Pre-trained Language Classifiers on LLM-generated Noisy Labels — May 2025

**Evidence.** Retrieves candidate true labels using embedding-neighborhood distributions and iteratively refines via simplex diffusion; reports average classifier performance increases of 7.21% / 7.30% on zero-/few-shot LLM-labeled datasets ([paper](https://arxiv.org/abs/2505.19675)).

**Why relevant [INFERENCE].** Distilling a teacher can faithfully learn teacher mistakes. Quality control of labels may dominate another round of architecture changes.

**Concrete experiment [INFERENCE].** Before full SiDyP, compare gold-only, hard-API-label, soft-API-label, and gold+soft mixtures. Audit disagreement strata with human labels; test neighborhood-label disagreement as an annotation prioritizer, not an automatic truth replacement. If these simple arms fail specifically on instance-dependent noise, add the paper's iterative refinement arm.

**Limit.** The abstract's “calibrating” terminology and reported classifier-performance improvement are not evidence of a particular ECE/NLL reduction; do not reinterpret its percentages as calibration-point gains ([paper](https://arxiv.org/abs/2505.19675)). **[INFERENCE]** Nearest neighbors can propagate correlated mistakes and erase legitimately rare high-risk cases.

## 15. Probabilistic Uncertain Reward Model (PURM) — March 2025; calibration companion: Rewarding Doubt

**Evidence.** PURM generalizes Bradley–Terry scalar reward modeling into reward distributions, uses distribution overlap to estimate uncertainty, and reports improved robustness under inconsistent/OOD preference data and longer effective RLHF optimization ([PURM](https://arxiv.org/abs/2503.22480)). Companion **Rewarding Doubt** trains expressed answer-confidence using a logarithmic scoring-rule RL reward and reports improved confidence calibration on unseen tasks ([paper](https://arxiv.org/abs/2503.02623)). PURM's abstract links released code/data via an anonymous repository; code availability was not independently cloned/validated.

**Why relevant [INFERENCE].** A reward-style head is a natural score readout, but raw scores, pairwise preference probabilities, correctness probabilities, and epistemic uncertainty must not be conflated. Rewarding Doubt provides a proper-scoring motivation but does not make RL mandatory.

**Concrete experiment [INFERENCE].** Attach a scalar candidate head and train (a) listwise categorical CE/KL, (b) pairwise BT, (c) pointwise candidate correctness BCE on the same data. For ordinal Score use explicit ordered-bin probabilities; for Noul retain an explicit null candidate with examples of incomplete candidate sets. Compare a distributional reward head only if disagreement/OOD evidence justifies it. Fit temperature on unseen calibration tasks; compare to direct NLL/Brier optimization before expensive RL.

**Limit.** Preference reward ranking is not mutually exclusive classification and does not identify a universal absolute score scale; distributional uncertainty is not automatically calibrated candidate correctness **[INFERENCE]**. Rewarding Doubt evaluates expressed confidence for factual answers, not Jev's complete typed candidate distribution ([PURM](https://arxiv.org/abs/2503.22480), [Rewarding Doubt](https://arxiv.org/abs/2503.02623)).

## Foundation cross-check: ModernBERT

ModernBERT's preprint is December 2024, but its publication is ACL 2025; it therefore belongs as the incumbent foundation, not as an undiscovered 2026 model. The paper reports 2T-token training and native 8,192-token context, classification and retrieval including code ([ACL 2025 paper](https://aclanthology.org/2025.acl-long.127/), [preprint](https://arxiv.org/abs/2412.13663)). **[INFERENCE]** Keep it in the same instruction-conditioned trial: its existing fixed five-class head is not a fair proxy for a runtime-label model.

## Three trials to run first [INFERENCE]

1. **Readout/architecture bakeoff, not leaderboard chasing.** ModernBERT + Ettin-400M instruction-conditioned candidate head vs a small Qwen reranker vs Qwen3-1.7B/4B non-thinking readout. Same source-held-out training/calibration/test sets, same descriptions, hard negation and exception tests, candidate permutations, duplicate-label and unknown-label cases. Record quality, NLL, Brier, risk/coverage, batch-1 p50/p95 and throughput. This directly tests the disagreement between Ettin's supervised encoder strength and BTZSC's zero-shot reranker strength ([Ettin](https://arxiv.org/abs/2507.11412), [BTZSC](https://arxiv.org/abs/2603.11991)).
2. **Distillation target experiment before more data.** Fixed examples/budget; hard teacher answers vs full candidate soft targets vs mixed gold+soft. Track teacher source, verbalized versus token-derived probabilities, omitted mass, prompt permutation, and provenance. Human-review disagreement and rare high-risk strata. Use disjoint calibration tasks; fit temperature after training rather than treating teacher confidence as ground truth ([LFM2](https://arxiv.org/abs/2511.23404), [SiDyP](https://arxiv.org/abs/2505.19675)).
3. **Buy test-time compute only where it improves log loss.** Compare no-think, bounded reasoning, and diffusion one-mask readout on hard examples; include random-label-order checks before any joint canvas trick. Train escalation from measured error reduction per millisecond. Rent a large-memory GPU for the diffusion experiment only after smaller local candidates establish a quality/latency floor; do not extrapolate H100 generation throughput to 3090 decisions ([Qwen3](https://arxiv.org/abs/2505.09388), [DiffusionGemma](https://arxiv.org/abs/2608.00146), [SetScore](https://arxiv.org/html/2608.14649v1)).

## Non-negotiable evaluation distinctions [INFERENCE]

- Separate **unseen labels on seen source tasks**, **unseen source tasks**, and **unseen task families**. Disclose target-task calibration distinctly from zero-label deployment.
- Keep Choice categorical normalization, independent multilabel decisions, ordinal Score, and null/abstention semantics separate. A shared backbone need not imply one identical loss.
- Score label-description changes, instruction paraphrases, label count and order, unsupported candidate sets, negation, rare high-severity mistakes, and distributed evidence—not merely average accuracy.
- Calibrate each inference regime separately: no-think and think, quantized and unquantized, joint and independent label scoring can change score distributions.
- Do not call reward confidence epistemic uncertainty without OOD validation. Do not call output-format compliance probability calibration.
- No hardware rental requirement is established by this literature sweep; the first compact-model experiments should be profiled on the available machine once its reported driver issue is resolved. The larger diffusion model is a conditional evaluation rental, not the first training commitment.
