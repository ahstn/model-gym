# Training methods for quality and aesthetics assessors

This file lists training-method papers for IQA, IAA, and VQA assessors. For each method it gives the idea, what to borrow for a Q-ReAlign re-train, the controlled evidence, the limits, and code and weight licenses. Data-addition evidence is in [data-mixture-evidence.md](data-mixture-evidence.md). The run order is in [experiment-plan.md](experiment-plan.md).

Last verified: 2026-09-27

## How to read this file

- All pairs are `PLCC / SRCC`. We converted sources that use `SRCC / PLCC` (Q-Align, TATAR, MDS-VQA, Q-Probe, VQA², LMM-VQA).
- "Controlled" means the paper changes one thing and keeps the model, data, and split fixed. Numbers from different rows use different protocols. Do not rank them against each other.
- All numbers are author-reported. We did not run any of them.
- `Status: verified` = we read the table in the primary source today. `from dossier` = taken from the 26 September 2026 dossier and not re-checked.
- Model details for the assessors (weights, benchmark tables) are in [models/iqa.md](../models/iqa.md), [models/iaa.md](../models/iaa.md), [models/vqa.md](../models/vqa.md), and [models/unified-and-backbones.md](../models/unified-and-backbones.md).

## 1. Method summary

Priority: `1` = test in the first re-train cycle; `2` = test after the first cycle; `3` = later or reference only.

| # | Method | Year / venue | Core idea | What to borrow for Q-ReAlign | Priority | Code | Weights | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | Q-Align / OneAlign ([paper](https://arxiv.org/abs/2312.17090), [repo](https://github.com/Q-Future/Q-Align)) | ICML 2024 | Train on 5 rating words, not numbers. Score = probability-weighted mean of the words. One model for IQA + IAA + VQA. | This is the baseline recipe. Keep the level-word interface as the control. | baseline | S-Lab License 1.0 (non-commercial) | [one-align](https://huggingface.co/q-future/one-align): MIT | verified |
| 2 | DeQA-Score ([paper](https://arxiv.org/abs/2501.11561), [repo](https://github.com/zhiyuanyou/DeQA-Score)) | CVPR 2025 | Soft label = the discretized Gaussian MOS distribution over the 5 levels. Add a Thurstone fidelity (ranking) loss inside each dataset. | Replace hard 5-bin targets with soft targets. Add in-dataset pair loss so datasets with different MOS scales can co-train. | 1 | MIT | [Mix3](https://huggingface.co/zhiyuanyou/DeQA-Score-Mix3), [LoRA-Mix3](https://huggingface.co/zhiyuanyou/DeQA-Score-LoRA-Mix3): MIT | verified |
| 3 | Compare2Score ([paper](https://arxiv.org/abs/2405.19298), [repo](https://github.com/Q-Future/Compare2Score)) | NeurIPS 2024 | Train the LMM to compare image pairs. At test time, compare with anchor images and convert to a score. | Pairwise training signal is scale-free. Use as an idea for cross-dataset pairs; not for one-pass scoring. | 3 | MIT | [Compare2Score](https://huggingface.co/q-future/Compare2Score): MIT | verified |
| 4 | Z-Reward (GDSO teacher + RISD student) ([paper v3](https://arxiv.org/abs/2606.09076), [project](https://srameo.github.io/projects/z-reward/)) | arXiv 2026 | A Qwen3.5-27B teacher reasons and outputs a score distribution. A Qwen3.5-9B student learns that distribution and answers in 1 token. | Distill a teacher's 5-level distribution into the one-token scorer. Mix with human MOS loss. | 2 | Not stated (no code link found) | Not released | verified |
| 5 | ReLIQS ([paper v1](https://arxiv.org/abs/2608.01730), [CVPR page](https://openaccess.thecvf.com/content/CVPR2026/html/Gedik_Learning_Where_to_Look_and_How_to_Judge_Resolution-agnostic_Image_CVPR_2026_paper.html)) | CVPR 2026 | CLIP patches at 224, 512, and native resolution. A small saliency net picks patches. Per-dataset rank + PLCC losses. | Add native-resolution crops next to the global view. Keep the global view. | 1 | Not stated (no repo found) | Not released | verified |
| 6 | Q-Probe ([paper v2](https://arxiv.org/abs/2601.15356v2)) | arXiv 2026 (v5 exists) | Agent zooms into local crops. Crops keep clean context so the model does not learn "crop = damage". | If you add crops, include clean crops and context. | 2 | Not stated (no repo found) | Not released | verified (v2 only) |
| 7 | HFVQA ([paper](https://arxiv.org/abs/2609.16946)) | WACV 2027 (accepted) | Fixed-size spatio-temporal patches at several scales incl. native, dense frames, learned patch selection. | Keep dense contiguous frames; drop regions, not frames. | 3 | Not stated | Not stated | verified (abstract only) |
| 8 | VQA² ([paper](https://arxiv.org/abs/2411.03795), [repo](https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment)) | ACM MM 2025 | Add a motion (SlowFast) branch; 3-stage training: distortion pre-training, then scoring and QA. | Test a motion branch or clip input only if frames miss temporal defects. | 2 | Apache-2.0 | from dossier: released | verified |
| 9 | LMM-VQA ([paper](https://arxiv.org/abs/2408.14008), [repo](https://github.com/Sueqk/LMM-VQA)) | arXiv 2024 | Spatial tokens from key frames plus temporal tokens from a 3D encoder. | Evidence that temporal tokens help LSVQ scoring. | 3 | MIT (repo contents not audited) | Not stated | verified |
| 10 | Q-Insight ([paper](https://arxiv.org/abs/2503.22679), [repo](https://github.com/bytedance/Q-Insight)) | NeurIPS 2025 | GRPO on Qwen2.5-VL-7B. Joint score reward and distortion type/level reward. | Add distortion type and severity as an auxiliary task (supervised first). | 2 | Apache-2.0 | [ByteDance/Q-Insight](https://huggingface.co/ByteDance/Q-Insight): Apache-2.0 | verified |
| 11 | Refine-IQA ([paper v2](https://arxiv.org/abs/2508.03763), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39387/43348), [trainer repo](https://github.com/jzhws/VisualQualityAssessment-RL-Trainer)) | AAAI 2026 | Stage 1 RL on local distortion type, severity, and boxes. Stage 2 RL on scores with a reward on the thinking. | A short local-perception warm-up before score training. | 2 | Not stated (no license file) | Not stated | verified |
| 12 | SHDIQA ([paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Distilling_Spatially-Heterogeneous_Distortion_Perception_for_Blind_Image_Quality_Assessment_CVPR_2025_paper.pdf)) | CVPR 2025 | Synthesize different distortions per block. Distill local distortion features by contrastive and affinity losses. | Idea for vision-tower pre-training on local, mixed distortions. | 3 | Not stated | Not stated | verified |
| 13 | VITAL ([paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Jia_VITAL_Vision-Encoder-centered_Pre-training_for_LMMs_in_Visual_Quality_Assessment_CVPR_2026_paper.pdf), [repo](https://github.com/jzhws/VITAL-Series)) | CVPR 2026 | Pre-train the vision encoder on 4.5M machine-annotated quality pairs. New decoders need a warm-up of under 1/1000 of that data. | Idea: a quality-tuned vision tower that can move to a new decoder. | 3 | Not stated (no license file) | Not stated | verified (abstract) |
| 14 | TATAR ([paper](https://arxiv.org/abs/2603.19779), [repo](https://github.com/yinwen2019/TATAR)) | arXiv 2026 | One backbone. Short reasoning + Gaussian score reward for IQA. Long reasoning + ranking reward for IAA. | Task-specific loss weights: distribution/MOS loss for IQA, in-dataset ranking loss for IAA. | 1 (loss idea) | README: "License: To be released" | Not released | verified |
| 15 | Q-SiT ([paper](https://arxiv.org/abs/2503.09197), [repo](https://github.com/zzc-1998/Q-SiT)) | arXiv 2025 | Joint scoring + interpreting. Search the data-mix ratio on a 0.5B model, then reuse it on 7B. | Tune task/dataset ratios on the small model (MiniCPM-V-4.6 or Q-ReAlign Mini) first. | 2 | Not stated (no license file) | [q-sit](https://huggingface.co/zhangzicheng/q-sit), [q-sit-mini](https://huggingface.co/zhangzicheng/q-sit-mini): MIT | verified |
| 16 | VisualQuality-R1 ([paper](https://arxiv.org/abs/2505.14460), [repo](https://github.com/TianheWu/VisualQuality-R1)) | NeurIPS 2025 | GRPO with a continuous Thurstone fidelity reward over image pairs in a batch (rank, not raw MOS). | Try the same pairwise fidelity as a supervised loss first. | 3 | Apache-2.0 | [VisualQuality-R1-7B](https://huggingface.co/TianheWu/VisualQuality-R1-7B): MIT | verified |
| 17 | Q-Ponder ([paper](https://arxiv.org/abs/2506.05384), [repo](https://github.com/vivocameraresearch/Q-Ponder)) | arXiv 2025 | Distill long reasoning from Qwen2.5-VL-72B, then GRPO for score + reasoning. | Cost reference for an RL stage. Teacher rationales as optional auxiliary data. | 3 | LICENSE.txt points to Apache-2.0; no code released | Not released ("coming soon") | verified |
| 18 | VQ-Insight ([paper](https://arxiv.org/abs/2506.18564), [repo](https://github.com/xuanyuzhang21/VQ-Insight)) | AAAI 2026 | Image-quality warm-up, then video GRPO with a temporal modeling reward (shuffled vs ordered frames). | IQA warm-up before video; a temporal-order auxiliary signal. | 2 | Not stated (no license file) | [vqinsight-naturalvideo](https://huggingface.co/ByteDance/Q-Insight/tree/main/vqinsight-naturalvideo): Apache-2.0 (repo card) | verified |
| 19 | Aes-R1 ([paper](https://arxiv.org/abs/2509.21871), [repo](https://github.com/ssssmark/AesR1)) | ICLR 2026 | 1 epoch of reasoning SFT (AesCoT), then RL with absolute-error + ranking rewards (RAPO). | Short cold start only. Combine absolute and rank terms for aesthetics. | 3 | Apache-2.0 | Not stated | verified |
| 20 | EvoQuality ([paper](https://arxiv.org/abs/2509.25787), [repo](https://github.com/bytedance/EvoQuality)) | ICLR 2026 | No human labels. Pairwise majority voting makes pseudo-ranks; GRPO with a fidelity reward; repeat. | Pseudo-pairs on unlabeled training-only images, kept next to a human-labeled anchor set. | 3 | Apache-2.0 | from dossier: Apache-2.0 declared | verified |
| 21 | RALI / RACT ([paper](https://arxiv.org/abs/2510.11369), [repo](https://github.com/xuanyuzhang21/RALI)) | ICLR 2026 | Align a CLIP image encoder to Q-Insight reasoning text by contrastive loss. Score without the LLM. RACT: cross-dataset SFT on reasoning text. | A cheap student and a control for "is the LLM needed?". | 3 | Apache-2.0 | [RALI](https://huggingface.co/ByteDance/Q-Insight/tree/main/RALI): Apache-2.0 (repo card) | verified |
| 22 | BrightRate-LM ([repo](https://github.com/shreshthsaini/BrightRate-LM)) | submitted 2026 | Rank-16 LoRA on a VLM, 5 quality-word expectation score, HDR multi-exposure frames. | LoRA recipe precedent. Only direct Gemma 4 quality-scoring evidence. | 1 (recipe) | MIT | HF adapters (see repo); license per card | verified |

Watch list (not ranked; no controlled number checked today):

| Method | Year / venue | Why watch | Status |
|---|---|---|---|
| [UniPercept](https://arxiv.org/abs/2512.21675) | ICML 2026 | Domain-adaptive pre-training + task-aligned GRPO over IAA, IQA, and structure/texture. See [unified models](../models/unified-and-backbones.md). | from search summary |
| [LLaVA-Assessor](https://arxiv.org/abs/2609.26205) ([repo](https://github.com/jzhws/LLaVA-Assessor)) | arXiv 2026 | Image + video foundation assessor. "Prompt disentanglement" to reduce multi-task objective confusion. The abs page says submitted 11 Aug 2026, which does not match the 2609 ID. | verified (abstract only) |
| [Q-Adapt](https://arxiv.org/abs/2504.01655) ([repo](https://github.com/yeppp27/Q-Adapt)) | arXiv 2025 | Progressive tuning to avoid conflict between explanation and attribute tasks. | from search summary |
| [PreResQ-R1](https://arxiv.org/abs/2511.05393) | arXiv 2025 | Separates response stability from score and rank rewards. | from dossier |
| [Probabilistic Prompt Adaptation](https://openaccess.thecvf.com/content/CVPR2026/papers/Hara_Probabilistic_Prompt_Adaptation_for_Unified_Image_Aesthetics_and_Quality_Assessment_CVPR_2026_paper.pdf) | CVPR 2026 | User-chosen aesthetic criteria in one IQA/IAA model. | from dossier |
| [Temporal Gains, Spatial Costs](https://arxiv.org/abs/2603.17541) | arXiv 2026 | General VLMs: video SFT helps video but often gives small gains or losses on image benchmarks. Risk for a unified model. | verified (abstract only) |

## 2. Controlled evidence

Each row is a within-paper comparison. "Base" and "Change" use the same model, data, and split unless the row says otherwise.

| Method | Comparison (source table) | Base | Change | Protocol |
|---|---|---|---|---|
| Q-Align | Level words vs numeric score text; KonIQ-trained, KADID cross-set ([paper Table 11](https://arxiv.org/html/2312.17090)) | 0.524 | 0.679 | mPLUG-Owl2; metric is the mean of SRCC and PLCC; trained on KonIQ only. |
| DeQA-Score | One-hot vs soft vs soft + fidelity; PIPAL (held-out domain) ([Table 6](https://arxiv.org/html/2501.11561#S5.T6)) | 0.409 / 0.420 (one-hot) | 0.472 / 0.472 (soft); 0.495 / 0.496 (soft + fidelity) | mPLUG-Owl2; trained on KonIQ + SPAQ + KADID; same data and steps. |
| DeQA-Score | Same ablation on KonIQ test ([Table 6](https://arxiv.org/html/2501.11561#S5.T6)) | 0.945 / 0.938 | 0.954 / 0.943; 0.957 / 0.944 | As above. |
| DeQA-Score | Level-word order: common vs random; LIVE-Wild ([Table 7](https://arxiv.org/html/2501.11561#S5.T7)) | 0.900 / 0.887 | 0.863 / 0.851 (random order) | As above. Shows the word choice matters. |
| Z-Reward | 9B student: SFT vs RISD distillation; internal T2I test ([v3 Table 2](https://arxiv.org/html/2606.09076)) | 0.5296 / 0.4942; HPA 0.7459 | 0.7391 / 0.6882; HPA 0.8864 (teacher 27B: 0.7620 / 0.7132; HPA 0.8956) | Qwen3.5-27B teacher, Qwen3.5-9B student; authors' internal annotated set; RISD outputs 1 token vs about 750 for on-policy distillation. |
| ReLIQS | Scales: 224 vs 224+512 vs +native; UHD test ([v1 Table 5](https://arxiv.org/html/2608.01730v1)) | 0.686 / 0.680 (224) | 0.756 / 0.750 (224+512); 0.837 / 0.865 (+native) | CLIP ViT-B/16; trained on UHD + KonIQ + SPAQ + KADID. Saliency weighting vs averaging adds only 0.833 / 0.861 to 0.837 / 0.865. |
| ReLIQS | Same training data; UHD test ([v1 Table 3](https://arxiv.org/html/2608.01730v1)) | Q-Align 0.627 / 0.683; DeQA 0.654 / 0.701 | ReLIQS 0.837 / 0.865 | Different model families; shows the resize-to-448 LMMs lose on 4K images. |
| Q-Probe | Crop policy; Vista-Bench ([v2 Table 4](https://arxiv.org/html/2601.15356v2)) | 0.580 / 0.510 (damaged crops only) | 0.776 / 0.728 (all damaged + some clean crops) | Authors' local-artifact benchmark. Later versions report other numbers; cite v2. |
| VQA² | With vs without motion extractor; KoNViD-1k ([Table 7](https://arxiv.org/html/2411.03795v4)) | 0.865 / 0.873 | 0.884 / 0.894 | 7B VQA² UGC scorer; trained on its instruction data; KoNViD cross-set. |
| LMM-VQA | Add temporal tokens; LSVQ test ([Table VII](https://arxiv.org/html/2408.14008v1)) | 0.887 / 0.855 | 0.891 / 0.889 | Llama-2 + CLIP ViT-L/14; trained on LSVQ train. Swapping to Llama-3 then gives 0.914 / 0.913. |
| Q-Insight | Score-only vs score + distortion task; 7-set mean ([Table 3](https://arxiv.org/html/2503.22679v2)) | 0.765 / 0.739 | 0.806 / 0.783 | Qwen2.5-VL-7B GRPO; scores from KonIQ; 7K DQ-495K images for the distortion task. AGIQA falls slightly (0.816 / 0.766 to 0.811 / 0.764). |
| Refine-IQA | Without vs with Stage-1 perception RL; KADID ([v2 Table 5](https://arxiv.org/html/2508.03763v2)) | 0.671 | 0.709 | Metric is the mean of SRCC and PLCC; no-think inference; KonIQ 0.916 to 0.931. |
| SHDIQA | vs LoDa; FLIVE (LIVEFB) ([Table 1](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Distilling_Spatially-Heterogeneous_Distortion_Perception_for_Blind_Image_Quality_Assessment_CVPR_2025_paper.pdf)) | LoDa 0.679 / 0.578 | SHDIQA 0.735 / 0.639 | Dataset-specific training; different models (24.2M vs 118M); not an ablation. |
| TATAR | SFT + RL: rank-only reward vs task-conditioned reward; 4-set IQA mean ([Table 2](https://arxiv.org/html/2603.19779v1)) | 0.756 / 0.754 | 0.765 / 0.767 | Trained on 6,189 IQA + 7,726 IAA samples (KonIQ, ArtiMuse-10K); 3 IQA sets are cross-set. |
| TATAR | Same, 4-set IAA mean ([Table 2](https://arxiv.org/html/2603.19779v1)) | 0.516 / 0.503 | 0.525 / 0.510 | As above; 3 IAA sets are cross-set. |
| VisualQuality-R1 | Binary vs continuous fidelity reward; 8-set mean ([Table 4](https://arxiv.org/html/2505.14460)) | 0.809 / 0.772 | 0.814 / 0.777 | Qwen2.5-VL-7B; trained on KADID only; 16 A100, about 5 h. |
| VisualQuality-R1 | Rollouts K=4 vs K=6; 8-set mean PLCC ([Table 3](https://arxiv.org/html/2505.14460)) | 0.811 (K=4) | 0.814 (K=6) | PLCC only reported. |
| Q-Ponder | No-CoT SFT vs long-CoT SFT vs long-CoT + RL; 6-set mean ([Table 3](https://arxiv.org/html/2506.05384)) | 0.775 / 0.754 | 0.834 / 0.800 (SFT); 0.845 / 0.821 (+RL) | Qwen2.5-VL-7B; same 7K KonIQ subset; RL took 3 days on 8 A100. |
| VQ-Insight | Full vs no temporal reward vs no image warm-up; LGVQ multi-dimension ([Table 4](https://arxiv.org/html/2506.18564)) | 0.787 / 0.761 (no temporal reward); 0.716 / 0.690 (no warm-up) | 0.869 / 0.853 | AI-generated video (LGVQ); Qwen2.5-VL; 8 A100 80G. |
| Aes-R1 | RAPO after 0, 1, 2 SFT epochs; 5-set IAA mean ([Table 3](https://arxiv.org/html/2509.21871)) | 0.6297 / 0.6102 (0 epochs) | 0.6337 / 0.6186 (1); 0.6027 / 0.5903 (2) | Qwen2.5-VL-7B; 15K combined data. The 10-epoch + RAPO row repeats the 2-epoch no-RL row, so we do not use it. |
| EvoQuality | Zero-shot base vs self-evolved (round 2); weighted mean of 8 sets ([Table 2](https://arxiv.org/html/2509.25787)) | 0.615 / 0.570 | 0.770 / 0.726 | Qwen2.5-VL-7B; unlabeled KonIQ images + synthetic distortions; no MOS used. |
| RALI | Without vs with contrastive alignment; 7-set mean ([Table 4](https://arxiv.org/html/2510.11369v2)) | 0.748 / 0.727 | 0.798 / 0.779 | CLIP-based; KonIQ-trained. Full RALI vs Q-Insight: 0.798 / 0.779 vs 0.806 / 0.783 at 3.4% of the inference time. |
| BrightRate-LM | Gemma 4 E4B vs Gemma 4 12B, same recipe ([E4B metrics](https://github.com/shreshthsaini/BrightRate-LM/blob/main/results/metrics/sft-gemma4e4b-v0-split0.metrics.json), [12B metrics](https://github.com/shreshthsaini/BrightRate-LM/blob/main/results/metrics/sft-gemma4-v0-split0.metrics.json)) | 12B: 0.782 / 0.776 | E4B: 0.880 / 0.862 | HDR video (BrightVQ), split 0 only, n = 420; 8 tone-mapped frames at 448; LoRA SFT. The 12B model is encoder-free. |

## 3. Limits and what not to over-read

| Method | Main limit |
|---|---|
| Q-Align | Hard one-hot targets lose rater spread. Code license is non-commercial. |
| DeQA-Score | Tested on mPLUG-Owl2 and IQA only. If a dataset has no rating variance, the soft target is a modeled guess, not human disagreement. |
| Compare2Score | Inference needs anchor comparisons, so it is slower than one-token scoring. Checkpoint data mix must be pinned. |
| Z-Reward | Text-to-image preference on internal data. Not IQA/IAA/VQA MOS. No code. The teacher is expensive. |
| ReLIQS | Specialist CLIP model, not a VLM. Shows that resolution matters; does not show the gain transfers to Qwen. |
| Q-Probe | Agentic inference is costly. Numbers change across arXiv versions. |
| HFVQA | Only the abstract was read. No numbers checked. |
| VQA², LMM-VQA | Full-system gains. VQA² instruction data overlaps LIVE-VQC and LSVQ-1080p (from dossier); re-check overlap before any reuse. |
| Q-Insight | Gain mixes a new auxiliary task and RL. Supervised-only effect is not shown. |
| Refine-IQA | Metric is a mean of two correlations. Stage-1 data (Refine-Perception-20K) release not verified. |
| SHDIQA | Small specialist model; not a VLM fine-tune result. |
| VITAL | Only the abstract was read. |
| TATAR | No video. Code, weights, and license not released. Gains are small (about 0.01). |
| Q-SiT | Scores are a mean of SRCC and PLCC. Ratio search cost about 720 A800 GPU-hours on a 0.5B model. |
| VisualQuality-R1 | Reward gains are small (0.005 SRCC). Sampled output spread is decoding noise, not rater spread. |
| Q-Ponder | No code or weights. RL stage is 3 days on 8 A100. |
| VQ-Insight | Evidence is for its GRPO system. A cheaper supervised version is our hypothesis. |
| Aes-R1 | One table row looks duplicated. Reasoning output adds latency. |
| EvoQuality | Consensus is not truth. Can amplify model bias. Needs a human-labeled anchor set. |
| RALI | IQA only. Not shown for IAA or video. |
| BrightRate-LM | One task, one split, submitted study. Not a ONE-ALIGN result. |

## 4. What to borrow first

1. **Soft targets + in-dataset ranking loss (DeQA-Score).** Keep the 5 words and the expectation scorer. This changes only the target and loss. It is the cheapest method with a clean ablation.
2. **Task-specific losses (TATAR idea).** Distribution + MOS loss for IQA and VQA. Add a within-dataset ranking loss for IAA. Use task-balanced batches.
3. **Visual input fixes (ReLIQS, Q-Probe, VQA², LMM-VQA).** Global view plus a few native-resolution crops, some of them clean. For video, compare 8 uniform frames with short contiguous clips at the same token budget.
4. **Auxiliary distortion supervision (Q-Insight, Refine-IQA).** Supervised, low weight, then restore the score interface.
5. **Teacher distillation (Z-Reward).** Only after steps 1–4. Distill the 5-level distribution, not generated text.
6. **RL (VisualQuality-R1, Q-Ponder, VQ-Insight, Aes-R1).** Last. Expect small gains per unit of compute. Use RL for a teacher, not for the deployed one-pass scorer.

## 5. Porting notes for new backbones

- The five level words must each map to one unique token after the prompt stem, with the same leading-space rule. The Q-ReAlign card says it matches `" excellent"` etc. with a leading space ([card](https://huggingface.co/q-future/Q-ReAlign-Pro-9B)). BrightRate-LM used title-case words with no space for Gemma 4 (`level_token_strategy: title_no_space` in its metrics file). Check each tokenizer; do not assume.
- DeQA Table 7 shows word order and word choice change results. Keep the same 5 words across backbones unless the ablation is about the words.
- Full reproduction details for the Q-ReAlign scorer and cache are in the [baseline code audit](../evaluation/protocol.md#8-q-realign-baseline-code-audit).
