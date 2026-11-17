# RL and preference optimization for the two-score Qwen scorer

Evidence checked on 2026-11-17 using Exa discovery/full-text retrieval and GitHub file/tree inspection. This is a research recommendation, not an instruction to launch training or change the accepted targets.

## Recommendation

Start with supervised adaptation to the existing ensemble labels, and compare a numeric/distributional scoring objective plus supervised ranking before spending on RL. Preserve independent technical and aesthetic outcomes. LoRA is a parameter-update strategy; SFT, DPO and GRPO are objectives and can each use LoRA. There is no requirement to progress through full fine-tuning before RL.

Our current labels make SFT immediately well-defined. A reward based on closeness to those same labels is mechanically checkable, but its perceptual correctness is not objectively verifiable. It can reinforce the ensemble's known softness/darkness bias as efficiently as SFT. RL does not introduce new human judgment. Better agreement with teacher labels is distillation success, not proof of better image quality assessment.

RL is a credible later experiment, not a dismissible fad: multiple task-specific papers report gains. However, their gains depend on human MOS, degradation supervision, generated rationales, ranking protocols, resolution handling and substantial rollout compute. None establishes that GRPO is the most cost-effective first treatment for our synthetic two-score corpus or Qwen3.5-9B.

## Options and what additional data they require

| Method | Potential benefit | Cost and limitations here | Recommendation |
|---|---|---|---|
| Supervised numeric/ordinal/distributional scoring | Direct, dense learning signal for the labels we already have; fast deterministic output | Learns teacher bias; loss/output representation and calibration matter | First baseline; preferably compare short JSON SFT against a score-aware objective |
| Supervised pairwise ranking auxiliary loss | Trains correct ordering without requiring generative sampling; naturally uses parent/distortion pairs and reviewed preference comparisons | Ranking alone does not preserve absolute 1–5 calibration; many tiny teacher-score differences are not reliable preferences | First extension, combined with absolute score loss and explicit ties |
| DPO | Offline preferred/rejected completion pairs; no online rollout loop or separately learned reward model required | Our rows are image+score, not chosen/rejected responses to the same image. Constructing pseudo-pairs from the same teacher adds no independent truth. Optimizes response likelihood ratios, not calibrated numeric error | Later, for reviewed correctness/grounding preferences or response behavior, if needed |
| IPO/robust DPO | Alternatives addressing some DPO overfitting or preference-noise assumptions | Do not eliminate systematic teacher errors; noise parameters need validation | Sensible controls if preference optimization is tested, not reasons to delay SFT |
| ORPO | Combines chosen-response SFT with an odds-ratio penalty; no reference model | Still requires reliable chosen/rejected responses; foundational evidence is general language alignment | Lower-priority preference baseline; no demonstrated special advantage for our IQA task |
| PPO/RLHF | Optimizes a scalar reward and can use human preference reward models | Typical language-model setup adds a critic/value estimate, reward model or rule, sampling, KL tuning; expensive and easy to exploit proxy rewards | Not first choice for a static two-number scorer |
| GRPO/RLVR | Group samples give relative advantages without a separate value critic; task-specific IQA evidence exists | Several completions per image, potentially long reasoning, reference-policy costs, unstable groups and noisy score rewards; not inherently cheap | Later capped experiment after SFT and data-quality baselines |
| Tool-using/agentic RL | Could learn where to inspect high-resolution details or request a crop | Adds tool/trajectory data, deployment complexity and latency; crop-selection shortcuts possible | Only if ordinary full-view/detail-view models demonstrably miss defects |

DPO distinction: preferring image A to image B is a ranking-supervision example. Standard DPO instead prefers completion A to completion B for the same input/prompt. We can turn a two-image comparison into a language task, but that is an explicit architecture/data choice, not directly applying DPO to two existing score rows.

## Research evidence and release checks

### Q-Insight (2025)

[Paper](https://arxiv.org/html/2503.22679). Uses Qwen2.5-VL-7B-Instruct with joint score regression and degradation classification/intensity rewards under GRPO. Reports stronger cross-dataset results, but slightly weaker in-domain KonIQ results than DeQA-Score. Training specification is eight samples per input, ten epochs, batch 128, about one day on 16 A100s. This supports testing degradation-aware auxiliary supervision; it does not isolate RL as the sole reason for improvement or establish a cheap one-GPU reproduction. Its MOS and distortion labels are richer than our current ensemble-score rows.

### VisualQuality-R1 (2025)

[Paper](https://arxiv.org/html/2505.14460), [official code](https://github.com/TianheWu/VisualQuality-R1), inspected master commit `a16dcdb64eea03a27afdb411d43cc1c9d723a1b2`. Uses six sampled scores per image, Thurstone comparative probabilities and continuous fidelity rewards, enabling multi-dataset ranking without directly aligning incompatible MOS scales. Paper reports 16 A100s, approximately five hours, ten epochs. This is strong motivation for a ranking objective, but ordinary supervised ranking should be tested before attributing benefits to policy optimization. Repository supports a non-thinking inference prompt; do not assume long reasoning is necessary at deployment. The README's inference example substitutes `random.randint(1,5)` when parsing fails: this must never be copied into our scorer, where missing/invalid results remain missing.

### TATAR (2026-03-20 preprint)

[Paper](https://arxiv.org/html/2603.19779v1), [official repository](https://github.com/yinwen2019/TATAR). Particularly relevant because it unifies IQA and aesthetics while using task-conditioned reasoning and different reward shapes. It reports SFT followed by GRPO, Gaussian score shaping for IQA and ranking for aesthetics. Uses Qwen2.5-VL-7B, eight generations, eight A800s, one SFT epoch followed by two–three RL epochs. Evidence supports keeping task losses and task behavior distinct; it does not prove our deployment needs lengthy aesthetic reasoning. Actual code exists for SFT and GRPO, but README still marks quick-start instructions/checkpoints as coming soon and licensing as unreleased. The launcher contains private absolute paths and eight-GPU assumptions. Treat as a research reference requiring adaptation and license clarification, not a ready package.

### Q-Hawkeye (2026)

[Paper](https://arxiv.org/html/2601.22920v2). Adds uncertainty-weighted GRPO updates and an implicit perception loss on original/degraded pairs. It argues that unstable rollouts can amplify noisy gradients and that textual reasoning alone does not guarantee perception. Relevant to our known soft-image errors and controlled distortion families. Its claimed official link, `https://github.com/AMAPML/Q-Hawkeye`, returned 404 through GitHub in this session; runnable code could not be verified. Downweighting difficult samples can also suppress precisely the rare cases we care about, so uncertainty weighting needs slice-level controls.

### PreResQ-R1 (EMNLP Findings, October 2026)

[Proceedings](https://aclanthology.org/2026.findings-emnlp.379/), [latest paper](https://arxiv.org/html/2511.05393), [official repository](https://github.com/DanceSkyCode/General-Visual-Quality-RL). Latest paper separates intra-image response stability from inter-image ranking/score preferences. It reports mean SRCC 0.811 and PLCC 0.795 across ten IQA benchmarks after KADID training; these are paper-specific protocol results, not comparable to our rubric MAE. It fine-tunes Qwen2.5-VL-7B without freezing layers, reports eight A800s and about twenty hours for IQA. The code/checkpoint links exist, but release fidelity needs work: tree `d4290479d90ae4f94cfef9c02f0e465da34a7efe` stores `src/grpo_preres_IQA_r1.py`, while the supplied launcher changes into absent `src/open-r1-multimodal` and calls a different nested entrypoint. README says MIT; actual LICENSE is Apache-2.0. Validate code against the selected paper revision before reproduction.

### Of Chains and Gains (2026-10-23)

[Full paper](https://arxiv.org/html/2610.33203). A highly relevant recent counterweight to claims that RL makes scoring better because explanations become more plausible. On Qwen2.5-VL 7B/32B, it finds genuine single-draw scoring gains, but gains beyond averaging concentrate in-distribution. Better scores do not consistently depend more on the explicit chain; larger models behave differently. Its controlled experiments show that score rewards can learn MOS-correlated text cues unrelated to image quality. This supports matched sampling/latency comparisons, image/text counterfactual tests and metadata-free inputs. It is a recent preprint, not a final consensus. No author-owned code link was verified in the retrieved page/searches.

### VGIQA (2026-10-27)

[Full paper](https://arxiv.org/html/2610.37186). Combines three-epoch SFT cold start and ten-epoch GRPO with local boxes, degradation tasks and global scores. VGSR-2K uses reference/generated-image pairs plus human voting/correction. Local rewards use matching, DIoU and F1, with task-isolated routing. Local grounding improves, but global score table 4 gives average PLCC/SRCC **0.791/0.762 versus Q-Insight 0.803/0.780**. Authors attribute this to stricter local inspection; for our goal it is a real tradeoff, not automatic success. This directly illustrates why improving targeted defects must be checked for global regression. Our scalar labels lack the spatial supervision needed. No runnable official repository was verified.

## Spinning Up: valuable foundations, unsuitable project runtime

Inspected the user's [repository](https://github.com/openai/spinningup/tree/master), [README](https://github.com/openai/spinningup/blob/master/readme.md), [setup.py](https://github.com/openai/spinningup/blob/master/setup.py), and [PPO explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html). It is explicitly educational and in maintenance mode. Dependencies include torch 1.3.1, TensorFlow below 2 and Gym approximately 0.15.3. Use it to understand policies, rewards, advantages, clipping and controlled evaluation. It does not provide a modern Qwen multimodal LoRA/GRPO training stack. PPO clipping constrains update incentives; it is not a guarantee against reward exploitation or instability.

For a maintained implementation route, official [TRL DPO documentation](https://huggingface.co/docs/trl/dpo_trainer) explicitly supports VLM image columns, adapters and IPO/robust losses; [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) provides current group-policy training. Pin and smoke-test the exact Qwen checkpoint/processor and library versions. Framework support for some Qwen VLMs does not by itself verify compatibility with our chosen newer checkpoint.

Foundational sources: [DPO](https://arxiv.org/html/2305.18290v3), [IPO](https://arxiv.org/abs/2310.12036), [ORPO](https://arxiv.org/abs/2403.07691). Their general alignment results do not establish wins on calibrated two-outcome photo scoring.

## How to test without circular evidence or overfitting

1. Use a family-disjoint frozen evaluation set that did not select prompts/ensemble weights. Keep human-reviewed judgment distinct from assistant references and from ensemble targets. Report teacher agreement and human agreement separately.
2. Retain the accepted ensemble weights, photography eligibility and null outcomes. Do not manufacture labels for incomplete rows. Keep technical and aesthetic losses/evaluation separate; do not collapse them into one weighted reward without explicit analysis.
3. Start with equal-image/equal-resolution/equal-token-budget comparisons: unchanged base; SFT-LoRA; score-aware supervised alternative; supervised-ranking auxiliary. Run learning curves rather than committing immediately to 200K.
4. For pairwise supervision, review high-margin pairs, represent ties/uncertainty and keep near-duplicate families in one split. Teacher disagreement is a review signal, not calibrated human uncertainty. Controlled corruption only gives a known ordering after checking that the delivered view actually became worse; blur may be intentional, denoising can remove noise, and crops can change composition.
5. If DPO is warranted, collect preferred/rejected **answers to the same image**, judged for score accuracy, scope and grounding. Match style and length to prevent preference for verbosity. Retain an absolute numeric calibration objective/evaluation, because likelihood preferences alone do not preserve 1–5 scale meaning.
6. If GRPO is warranted, initialize from the winning SFT checkpoint. Start with short rollouts and a small reviewed subset. Compare against additional SFT using the same GPU-hour budget, not merely the same epoch count. Track invalid completions, rollout variance/zero-advantage groups, token counts, KL and real end-to-end cost. Evaluate greedy one-pass output and matched-budget averaging separately.
7. Test counterfactuals: remove filenames/source IDs/stats/teacher hints from training input; swap irrelevant captions; perturb reasoning or compare short/long answer modes; inspect repeated-score stability. Audit darkness with intentional low-key controls, blur with intentional depth-of-field controls, and grain with artistic/film examples.
8. Promotion should require no material regression in either score, eligibility false-floor rates or targeted slices, using family-level bootstrap intervals. Select a practical minimum improvement before testing (for example 0.05 MAE on the 1–5 scale is a project choice, not a literature guarantee). Aesthetic gains cannot compensate for technical degradation without explicitly changing requirements.

The $50 should establish reliable data splits, train/evaluate the simplest viable adaptation and measure throughput. Do not spend it reproducing the multi-GPU RL papers. A tiny LoRA-GRPO feasibility test could be much cheaper than their full runs, but is justified only after a strong supervised baseline and a trustworthy reward exist. New capability such as localized evidence/crop selection would be a stronger reason for RL than merely fitting existing two scalar labels.
