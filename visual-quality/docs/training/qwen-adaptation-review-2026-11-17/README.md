# Choosing how to train our two-score Qwen assessor

Research cutoff: **17 November 2026**. Research used Exa search/full-text retrieval and GitHub source inspection across four workstreams: quality/aesthetics methods, RL/preferences, Qwen adaptation/tooling, and data/evaluation. Primary papers and executable configurations take precedence over vendor performance claims. Recent preprints are identified as such; published results have not been reproduced on our corpus. This is a comprehensive decision review, not a claim to enumerate every paper or an empirical comparison of trained models.

**Recommendation: start with supervised BF16 LoRA on Qwen3.5-9B, measure a cheap frozen-feature baseline, and test one score-aware or visual-detail improvement.** Full fine-tuning remains a legitimate challenger. Reserve RL for a demonstrated need and independently trustworthy rewards. Use the first $50 to establish a working, evaluated training pipeline and a measured full-run estimate.

No GPU was started, no training masks were enabled, and no labels or ensemble weights were changed by this research. The $50 balance is user-reported. This review extends the earlier cost estimate and qualifies its hardware assumptions; it does not authorize an unbounded experiment.

Evidence appendices: [quality-scoring methods](quality-scoring-evidence.md), [RL and preferences](rl-and-preferences.md), [Qwen implementation checks](implementation-evidence.md).

## What our data supports

The canonical `labeling-20000-2026-11-16/exports/labelled-images.jsonl` was inspected locally. It contains:

| Property | Actual state |
|---|---:|
| Images / source families | 20,000 / 18,500 |
| Complete ensemble score pairs | 19,997 |
| Assistant reference images | 1,500 |
| EmbeddingGemma 2 content groups | 64 |
| Ensemble photography-scope disagreements | 262 |
| Records approved for training | 0 |

Sources are 15,000 Megalith images, 3,000 VizWiz images, 500 PlotQA/CLEVR controls and 1,500 generated distortion variants. These are not 20,000 independent subjects, and there is not yet a completed 200K labeled corpus. The 200K calculation below is a future capacity estimate.

Every current record has `split="train"`. That is acquisition metadata, **not a safe training partition**. Reference membership lives in `benchmark_partition`: 100 development, 300 representative, 100 challenge, 500 expansion examples and 500 expansion confirmation. Create a new versioned training manifest, preserving the frozen exports, that excludes benchmark families from optimization. Review near-duplicate candidates before assigning groups. Keep already inspected/used examples diagnostic; call the final 500 confirmation images untouched only if their exposure history supports that claim. New human-reviewed test images would provide stronger confirmation.

The learning targets remain **0.375 Luna + 0.375 DeepSeek + 0.25 Liquid**, separately for technical quality and aesthetics, using native precision. Jev remains diagnostic. No new Q-ReAlign inference is proposed. Missing outcomes remain missing; a partial valid outcome may be used only with an explicit per-task training mask after approval. Existing all-false masks stay unchanged until review is complete. The first-1,000 Git allowlist remains unchanged.

These are synthetic teacher targets, not human mean opinion scores. A model can imitate them well while retaining the teachers' bias on soft, grainy or dark images. Training on each teacher separately does not automatically create new evidence: under weighted squared loss, summing losses to the three labels is equivalent to loss against their weighted mean plus a term independent of the student's prediction. Their disagreement is useful for review, but not a measured distribution of human preferences.

## Two separate choices: objective and parameters

**Objective:** what should the model learn? Examples are numeric regression, ordinal score prediction, next-token supervised fine-tuning (SFT), pairwise ranking, preference optimization and reinforcement learning.

**Parameter policy:** what can change? Examples are new prediction heads, LoRA adapters, selected original layers, or all weights. SFT and RL can both use LoRA or full weights. Full fine-tuning is not a mandatory stage before RL, and LoRA is already a form of fine-tuning.

| Approach | Advantages for this task | Disadvantages / risks | Decision |
|---|---|---|---|
| Frozen features + two small scoring heads | Cheap training, cached features, continuous deterministic outputs; tests whether a large language component adds value | Frozen semantic features can miss blur/noise; feature extraction and deployment still cost compute | Include as the low-cost control |
| BF16 LoRA | Small trainable state/checkpoints; easy versioning and rollback; enough flexibility for a focused scoring task | Low rank or wrong modules can underfit; still incurs backbone and image-activation costs; can still overfit | First Qwen adaptation |
| QLoRA | Keeps the frozen base in low-bit weights and trains adapters, reducing memory | Quantization can shift subtle score distinctions; dequantization can reduce speed; does not remove activation memory | Memory fallback, not default on 96GB |
| LoRA plus projector / selected vision adaptation | Can improve quality-specific visual features while limiting changes | More gradients/memory; layer choice and learning rate need testing | Strong next option when detail failures persist |
| Full supervised fine-tuning | Maximum flexibility to change representations; may beat adapters on larger domain shifts | Larger state/checkpoints; more exposure to noisy-label fitting and forgetting; implementation-specific memory | Matched challenger after a LoRA baseline, not automatically the final answer |
| DoRA / rsLoRA / initialization variants | Can improve adapter capacity or optimization | More choices and sometimes extra cost; general benchmarks do not prove wins here | One controlled follow-up only if needed |
| Supervised pairwise ranking | Learns ordering directly without rollout generation | Cannot establish absolute 1–5 calibration; tiny teacher differences are unreliable; pairs are correlated | Add to absolute scoring only on trustworthy pairs |
| DPO / IPO / ORPO | Learn preferred response behavior from offline comparisons; simpler than online RL | Need suitable chosen/rejected answers; preference likelihood is not numeric calibration | Later, if we collect response-quality preferences |
| PPO / GRPO | Can optimize non-differentiable rewards and potentially learn inspection policies | Repeated sampling, reward design, instability and proxy exploitation; expensive relative to two-number supervision | Defer beyond the first $50 |
| Crop/zoom agent training | May learn where to inspect local defects | Requires trajectories or localized supervision; higher latency and more failure modes | Only after static global+detail input is insufficient |

LoRA is not guaranteed equivalent to full tuning. Controlled work on code/math finds differences in learning and forgetting, but those results do not establish the winner for photo scoring. Conversely, DeQA's strong LoRA results also train visual components, so they cannot prove that language-only adapters suffice. [LoRA comparison](https://arxiv.org/html/2405.09673), [DeQA training script](https://github.com/zhiyuanyou/DeQA-Score/blob/main/scripts/train_lora.sh).

## The objective I would test

**First: a short-output SFT control.** Use the fixed photography rubric, actual image, and compact two-score answer. Disable long reasoning for this control. Mask input/padding/image tokens out of the answer loss. Keep any eligibility target independently reviewed. This is the simplest way to establish that our exact checkpoint, data loader, prompt and serving contract work together.

**Then: a score-aware supervised objective.** Compare two bounded regression heads with Huber loss, or two five-level ordinal distributions with continuous expected scores. Retain separate technical and aesthetic losses; normalize each over its available examples. Neither a single blended score nor a many-token essay is necessary to specialize the model.

A five-level distribution still supports a continuous result: `score = sum(k * p[k])`, for `k=1..5`. With only a scalar target, use a declared mean-preserving encoding, for example `3.7 -> 0.3 at level 3 + 0.7 at level 4`. This is **an encoding choice**, not an observed human rating distribution. Do not invent Gaussian human uncertainty from three teacher scores. Compare this against direct regression; DeQA's published human-distribution recipe is not an exact match to our data. [Q-Align](https://proceedings.mlr.press/v235/wu24ah.html), [DeQA-Score](https://arxiv.org/html/2501.11561).

Only add an auxiliary ranking loss if it improves validation beyond absolute-score learning. Start with reviewed, clearly separated same-domain pairs and verified parent/degradation pairs. Include ties or abstention rather than deciding that a difference of 0.01 establishes preference. Do not create every possible pair and count those as independent data. Aesthetic ordering is not guaranteed by technical degradation; intentional grain and blur can change mood positively. Plain supervised ranking is a cheaper control before RL-to-rank.

Output JSON can be enforced by a deterministic serializer around score heads, or constrained decoding for the generative control. Keep both results in [1,5] and display at most two decimal places. Formatting precision is not measurement accuracy. Head-only outputs need an explicitly versioned student contract: do not fabricate prose evidence to fill the richer teacher schema. Failed inference remains missing, never a random or zero score.

Treat **photo eligibility as a separate supervised task** with reviewed targets. A flat poster should receive the accepted scope floor, while photographed packaging with physical context remains eligible. Do not convert the 262 disagreements into a new majority-vote rule or silently overwrite ensemble targets. Any reviewed correction belongs in a versioned training-label layer linked to the original observations. Report raw score errors and eligibility errors separately so easy floor cases cannot hide poor photo discrimination.

## What recent task-specific research changes

The detailed matrix covers 16 quality/aesthetic methods; the RL appendix also examines newer policy-learning papers and their releases. The most decision-relevant evidence is:

| Work | Relevant result or idea | What it does not establish |
|---|---|---|
| Q-Align / Q-ReAlign | Ordinal supervised scoring; Q-ReAlign provides a closely related Qwen3.5 implementation | Their pretrained benchmark accuracy is not our photo-scope/rubric accuracy; released defaults need split and evaluation fixes |
| DeQA-Score, CVPR 2025 | Soft score targets and within-dataset ranking improve calibration/generalization in its experiments | We do not have its human score distributions; LoRA recipe is not language-only |
| ArtiMuse, CVPR 2026 | Aesthetics needs its own supervised treatment; visual training with language adapters is viable | Specialized checkpoint results do not establish one universal technical+aesthetic model |
| ReLIQS, 2026 | Native-detail information and multiscale inputs address information lost through resizing | Default released code does not deliver the advertised sparse top-k path; crops also affect cost and composition |
| LEAF, January 2026 preprint | Distill teacher perception then calibrate on a small human-rated subset | Teacher alignment alone is not human alignment; code promised on publication, not verified as runnable |
| TATAR, March 2026 preprint | Task-conditioned SFT+GRPO separates IQA and aesthetic behavior | Its richer supervision and multi-GPU recipe do not show RL is the cheapest first step |
| PortraitAes, October 2026 preprint | Intent-conditioned SFT, calibration and ranking; useful dark/blur counterexamples | Its 11K task records come from 1.6K portraits, not 11K independent images or broad human MOS |
| MetaRefine, October 2026 preprint | Small residual heads use frozen visual features and image geometry | It does not validate feeding sharpness/noise statistics to our model |
| Of Chains and Gains, October 23 2026 preprint | RL gains do not consistently imply stronger causal use of reasoning; irrelevant text can exploit score rewards | It does not show RL never helps; gains depend on distribution and sampling budget |
| VGIQA, October 27 2026 preprint | Localized supervision improves grounding | Its global scoring average is below Q-Insight in the reported comparison; local gains need regression checks |

Sources: [Q-ReAlign](https://github.com/Q-Future/Q-ReAlign), [ArtiMuse](https://github.com/thunderbolt215/ArtiMuse), [ReLIQS](https://github.com/hakan-emre-gedik/ReLIQS), [LEAF](https://arxiv.org/html/2601.20689), [TATAR](https://arxiv.org/html/2603.19779v1), [PortraitAes](https://arxiv.org/abs/2610.05010), [MetaRefine](https://arxiv.org/abs/2610.40487), [Of Chains and Gains](https://arxiv.org/html/2610.33203), [VGIQA](https://arxiv.org/html/2610.37186).

Two additional data findings matter. MOSAIQ uses human perceptual anchors to align dataset scales; simply stretching each dataset to 1–5 is not equivalent. SynDR-IQA supports studying diversity and redundant distortion families, but its similarity-based pseudo-label transfer is not something we should adopt: semantic similarity does not establish equal quality. Continue EmbeddingGemma 2 for organization, family review and stratification. [MOSAIQ](https://arxiv.org/abs/2609.20247), [SynDR-IQA](https://arxiv.org/html/2601.00225v1), [official implementation](https://github.com/Li-aobo/SynDR-IQA).

## Why RL should wait, and where it could earn a place

The existing labels directly define a supervised target. A reward such as `-abs(predicted_score - ensemble_score)` is computable, but its visual correctness is not independently verifiable. RL using that reward can reproduce the same teacher bias. SFT has this limitation too; RL does not supply missing human information.

There is credible positive evidence: Q-Insight combines score rewards with degradation classification/intensity, and VisualQuality-R1 learns comparative quality. Their reported training budgets are substantial: roughly 16 A100s for a day and 16 A100s for five hours, respectively. These figures describe the authors' workloads, not minimum costs or a forecast for our LoRA pilot. Smaller RL experiments are possible; they still need a baseline worth improving. [Q-Insight](https://arxiv.org/html/2503.22679), [VisualQuality-R1](https://arxiv.org/html/2505.14460).

**DPO is not the same as ranking two images.** Standard DPO prefers one completion over another for the same input. Our image-score rows do not contain those response preferences. Converting them to pseudo-preferences may be an experiment, but adds no independent truth. Reviewed preferences about score correctness, photography scope or grounded explanations would justify DPO/IPO. Keep absolute-score calibration alongside preference evaluation. [DPO](https://arxiv.org/html/2305.18290v3), [TRL VLM DPO support](https://huggingface.co/docs/trl/dpo_trainer).

A useful later GRPO trial would begin from the winning SFT model, use reviewed short-response rewards, and compete against **more supervised training for the same GPU-hours**. Record generated tokens, rollout count, KL, invalid outputs and zero-advantage groups. If we eventually need learned crop selection, a reward based on independently annotated localization and final scores would test a genuinely different capability. Long generated reasoning should justify its latency through controlled gains.

The supplied [OpenAI Spinning Up repository](https://github.com/openai/spinningup/tree/master) is useful for understanding policies, rewards, advantages and PPO clipping. Its README describes an educational maintenance-mode project; the inspected dependency file uses torch 1.3.1, TensorFlow <2 and Gym ~0.15.3. It is not our modern multimodal training runtime. Use current, pinned Transformers/PEFT/TRL or a framework with verified Qwen3.5 vision support. The appendix records code-release problems in several research repositories rather than treating every paper as ready to run.

## Qwen and Blackwell implementation choice

Start with the **post-trained `Qwen/Qwen3.5-9B` vision-language checkpoint**, not automatically the separate literal `-Base` checkpoint. This recommendation aims to retain existing instruction/rubric behavior with limited task labels. It is not a measured comparison; a 4B checkpoint is a later efficiency challenger. Use the multimodal conditional-generation class and processor, not the similarly named text-only causal-LM class. [Official model card](https://huggingface.co/Qwen/Qwen3.5-9B), [Base card](https://huggingface.co/Qwen/Qwen3.5-9B-Base), [architecture documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5).

My first implementation choice is **Transformers + PEFT, using TRL for the short-output SFT control and a small explicit trainer for custom score heads/losses**. Axolotl is a useful configuration-managed alternative with inspected 9B vision LoRA/full-tuning examples. LLaMA-Factory and ms-swift also have explicit Qwen3.5 support; these are alternatives, not packages to stack together. Pin the exact versions only after a forward/backward/save/resume/serve test. Source support is not proof that our checkpoint, kernels and images work together.

Use one **full RTX PRO 6000 Blackwell Server Edition, 96GB**, not a 24/48GB MIG partition. BF16 is the initial weight/compute choice; newer FP4 hardware does not make our BF16 training automatically receive its headline low-bit speedup. Start with LoRA rank 16 or 32, explicit adapter targets and a frozen vision tower; these are initial hypotheses, not a validated optimum. Qwen3.5 has hybrid GatedDeltaNet/full-attention layers, so a copied `q_proj/v_proj` recipe can miss much of the intended adaptation. Log all trainable modules and counts. [PEFT LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora), [Axolotl Qwen3.5 guidance](https://docs.axolotl.ai/docs/models/qwen3.5.html).

**Full fine-tuning may fit this GPU.** Conventional Adam state for the 9B language weights alone can take roughly 108–144 decimal GB before vision/activations, depending on state precision and master weights. However, Axolotl documents a 9B vision full-tuning recipe around **61 GiB**, using an 8-bit optimizer, microbatch 1 and activation checkpointing. That is a framework report for a particular workload, not our measured requirement. It means full tuning should be judged experimentally, not ruled out by simplistic memory arithmetic. Do not copy its learning rate or zero-sized validation split as our defaults. [Documented recipe](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/qwen3.5/9b-fft-vision.yaml).

QLoRA quantizes frozen base weights; an 8-bit optimizer compresses optimizer state. Those are different choices. Unsloth's current guide cautions against 4-bit Qwen3.5 adaptation because of quantization differences, reinforcing BF16-first on 96GB, but it is a vendor caution rather than proof of universal inferiority. [Guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune).

## Evaluation that can distinguish improvement from imitation

Keep a teacher-agreement report and an independent visual-correctness report. Assistant references must remain labeled as assistant judgments. Add human-reviewed development and test subsets, including random representatives as well as disagreements; selecting only difficult images would distort population estimates. Never calibrate on the test set. A monotonic calibration map can fix scale bias, but cannot fix incorrect ordering or recover unseen detail.

Use family-disjoint splits, keeping original/distorted/duplicate relatives together. Fit any new selection/calibration transform only on training or development data. Existing full-corpus content clusters may support declared transductive organization; they are not trained scoring features, and should not be presented as evidence of strictly inductive generalization. Build retrieval indices from training data only. [Why grouping matters](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0269715).

For **each** score, report raw MAE, signed bias, SRCC, raw PLCC, large-error rate, coverage and family-bootstrap intervals. Also report eligible-photo-only metrics, scope false floors and missed exclusions. Publish per-source, content-group and low/mid/high-score slices, with sample sizes. Predeclare practical regression tolerances before comparing models; a better average cannot silently trade worse technical quality for better aesthetics. A proposed 0.05 MAE improvement threshold is a project decision, not a literature guarantee. Use multiple seeds for final confirmation; a $50 screen is exploratory.

Keep these diagnostic pairs/slices explicit:

- Global blur versus intentional background blur, soft focus and smooth surfaces.
- Crushed shadows versus successful low-key/night photography.
- Grain/noise with lost detail versus intentional texture and film grain.
- Flat reproductions/interfaces versus photographed artwork, labels and packaging with physical context.
- Local defects and occlusions versus globally attractive composition.
- Portraits, landscape, objects, indoor scenes and VizWiz difficult captures; include higher-quality cases so penalties cannot simply lower every score.

For controlled corruptions, verify the delivered variant is actually worse before scoring monotonic violations. Count severe-degradation rank inversions as a local diagnostic; do not claim it is numerically the same as PACE's published holistic-override metric. PACE motivates checking whether attractive content overrides a damaged target dimension, but its agentic evaluation protocol is distinct. [PACE](https://arxiv.org/abs/2609.22942v1).

Prevent shortcuts: inputs contain the rubric and pixels, not teacher scores, filenames, source IDs, benchmark labels or image-stats. Preserve `sharpness_peak`, entropy, focus area, clipping and colourfulness as audit metadata only. No default blur, JPEG, brightness or random-crop augmentation with unchanged quality labels. A detail crop must accompany the global view, not inherit the image's aesthetic score as if it were the whole composition. Compare input variants at both matched token budgets and actual cost. Verify model export/quantization parity on the same benchmark.

## How to use the first $50

At the last checked Secure Cloud quote of **$2.69/GPU-hour**, $50 buys 18.59 hours before storage and other account charges. Budget **about 15 hours ($40.35)** and retain roughly $9.65 for storage, recovery and reserve. Prepare data, environments and scripts locally before renting. Use a small corpus subset rather than transferring a future 200K corpus for a pilot. Stop owned compute promptly and retain checkpoints outside the pod before cleanup. This is a proposed plan, not work already executed.

| Stage | Maximum GPU hours | Compute allowance | Required output |
|---|---:|---:|---|
| Load/backward/resume and throughput pilot | 2 | $5.38 | Correct masks and gradients, peak VRAM, image tokens, images/sec, restored-checkpoint parity |
| Frozen-feature control | 1 | $2.69 | Cheap two-score baseline, including feature extraction cost |
| BF16 LoRA supervised control | 5 | $13.45 | Fixed-subset training and validation learning curve |
| One selected ablation | 4 | $10.76 | Score-aware target **or** detail/vision adaptation **or** matched full-tuning challenger, chosen from observed failure |
| Evaluation, export and recovery allowance | 3 | $8.07 | Slice metrics, final checkpoint and serving parity |
| **Compute total** | **15** | **$40.35** | |

These are hard planning ceilings, not a promise each stage finishes within them. Initially profile 500–2,000 representative images including the largest processed views; select the training subset size only after timing. If setup fails or throughput is poor, stop and fix it rather than burning the remaining balance. Spend less on lower-priority ablations if the baseline needs more time. Do not attempt every adapter rank, backbone, objective and RL variant within $50.

There are two independent reasons to proceed: the score quality passes the frozen evaluation, and throughput makes the larger run affordable. If LoRA underfits despite useful learning curves, test broader adapters, selective visual training or full tuning. If it learns the teacher well but fails human-reviewed cases, improve supervision/input evidence rather than merely increasing parameter count or RL steps.

## Full-run estimate and required top-up

For a conservative 200,000 **training** images, two epochs, one joint two-score example per image:

`GPU hours = 400,000 / (effective training images per second × 3,600)`

| Measured images/sec | Training hours | Compute at $2.69/h | Planning total: compute ×1.25 + $20 |
|---:|---:|---:|---:|
| 1 | 111.1 | $298.89 | $393.61 |
| 2 | 55.6 | $149.44 | $206.81 |
| 4 | 27.8 | $74.72 | $113.40 |
| 8 | 13.9 | $37.36 | $66.70 |

These are sensitivity calculations, **not measured performance**. The 25% and $20 are provisional allowances for evaluation, saving, setup and short-term storage/contingency. They exclude the exploratory pilot already spent, acquiring/labeling another 180K images, extended retained storage, taxes and multiple full experiments. Two separate task passes double training presentations; two outputs in one example do not. For 200K total with 160K training after holdouts, multiply the training compute portion by 0.8. RL needs a new estimate including rollouts; full tuning needs its own measured throughput.

After the pilot, `top_up = max(0, measured full-run budget - remaining available credit)`. If about $5 remains and throughput is 2–4 images/sec, the illustrative addition is **about $110–$205**. Do not subtract the original $50 again after spending it on exploration. Reserve $250 only if the measured rate supports that range; a 1 image/sec result instead implies roughly $395 before subtracting remaining credit.

## Decision and open questions

The evidence supports **supervised task specialization first, using BF16 LoRA**, with score representation, preserved image detail and reviewed labels as the first improvement candidates. Full tuning is feasible enough to remain a serious control; RL is justified by a capability or evaluation gain, not by being the newest stage in an alignment pipeline.

Before the full run, we still need measured training/serving throughput, processor/token limits, deployment model-size requirements, a reviewed training manifest and an independent evaluation anchor. No paper, vendor VRAM table or teacher-agreement score resolves those questions for us. The 20K dataset is sufficient to begin that investigation once its training subset is approved; 200K should follow learning curves rather than be assumed necessary in advance.
