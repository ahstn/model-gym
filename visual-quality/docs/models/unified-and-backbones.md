# Unified quality assessors and VLM backbones

This file lists large multimodal model (LMM) assessors that cover two or more of IQA, IAA, and VQA in one model. It also lists vision-language backbones that we can fine-tune with the Q-ReAlign recipe.

Last verified: 2026-09-27

Related files: single-task models are in [iqa.md](iqa.md), [iaa.md](iaa.md), and [vqa.md](vqa.md). Metric rules and the Q-ReAlign code audit are in [../evaluation/protocol.md](../evaluation/protocol.md). Training recipes are in [../training/methods.md](../training/methods.md). The staged plan is in [../training/experiment-plan.md](../training/experiment-plan.md).

All correlation pairs are `PLCC / SRCC`. All numbers are author-reported. We did not run any model.

## Part 1: Unified LMM assessors

### 1.1 Overview

| Model | Year / venue | Tasks | Base model | Code | Weights | Commercial use? | Paper | Release | Status |
|---|---|---|---|---|---|---|---|---|---|
| Q-Align / OneAlign | 2024 / ICML 2024 | IQA + IAA + VQA | mPLUG-Owl2 (LLaMA-family LLM) | S-Lab License 1.0 (non-commercial) | MIT (HF card) | Unclear: code is NC; base LLM terms not checked | [arXiv 2312.17090](https://arxiv.org/abs/2312.17090) | [GitHub](https://github.com/Q-Future/Q-Align), [q-future/one-align](https://huggingface.co/q-future/one-align) | verified |
| Q-ReAlign Mini / Lite / Pro | 2026 / no paper (HF, June 2026) | IQA + IAA + VQA | Qwen3.5 0.8B / 4B / 9B | MIT declared in `pyproject.toml`; no LICENSE file in repo root | Apache-2.0 (all three cards) | Weights: Yes (Apache-2.0). Training media terms are separate | [METHOD.md](https://github.com/Q-Future/Q-ReAlign/blob/main/docs/METHOD.md) | [GitHub](https://github.com/Q-Future/Q-ReAlign), [Pro-9B](https://huggingface.co/q-future/Q-ReAlign-Pro-9B), [Lite-4B](https://huggingface.co/q-future/Q-ReAlign-Lite-4B), [Mini-0.8B](https://huggingface.co/q-future/Q-ReAlign-Mini-0.8B) | verified |
| Q-Instruct | 2023 / CVPR 2024 | Low-level perception Q&A and description (images) | mPLUG-Owl2 and others | S-Lab License 1.0 (NC) | MIT ([mPLUG-Owl2 variant card](https://huggingface.co/q-future/q-instruct-mplug-owl2-1031)) | No (NC code) | [arXiv 2311.06783](https://arxiv.org/abs/2311.06783) | [GitHub](https://github.com/Q-Future/Q-Instruct) | verified |
| Co-Instruct | 2024 / ECCV 2024 Oral | Open-ended multi-image quality comparison | mPLUG-Owl2 (8.2B stored) | S-Lab License 1.0 (NC) | Not stated ([card](https://huggingface.co/q-future/co-instruct)) | No (NC code) | [arXiv 2402.16641](https://arxiv.org/abs/2402.16641) | [GitHub](https://github.com/Q-Future/Co-Instruct) | verified |
| VQA² | 2024 / arXiv | Image + video quality scoring and interpretation | LLaVA-OneVision (Qwen2-7B LLM) | Apache-2.0 | Apache-2.0 ([UGC scorer](https://huggingface.co/q-future/VQA-UGC-Scorer-llava_qwen), [assistant](https://huggingface.co/q-future/VQA-Assistant-llava-qwen-enhanced)) | Yes for code/weights; data terms separate | [arXiv 2411.03795](https://arxiv.org/abs/2411.03795) | [GitHub](https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment) | verified |
| Q-Eval-Score (open version) | 2025 / CVPR 2025 Oral | AIGC image + video quality and text alignment | Not checked | Not stated (no LICENSE in repo) | CC BY-NC 4.0 ([card](https://huggingface.co/AGI-Eval/Q-Eval-Score)) | No (NC) | [arXiv 2503.02357](https://arxiv.org/abs/2503.02357) | [GitHub](https://github.com/zzc-1998/Q-Eval) | verified |
| UniPercept | 2025 / ICML 2026 Spotlight | IAA + IQA + structure/texture (images only) | InternVL3-8B | Not stated (no LICENSE in repo) | Apache-2.0, gated click-through on HF | Weights: Yes per card. Code: Unclear | [arXiv 2512.21675](https://arxiv.org/abs/2512.21675) | [GitHub](https://github.com/thunderbolt215/UniPercept), [HF](https://huggingface.co/Thunderbolt215215/UniPercept) | verified |
| VITAL-Series | 2025 / CVPR 2026 | IQA + VQA scoring and interpretation | InternVL3-8B + SlowFast motion features | Not stated (no LICENSE in repo) | MIT ([Base-8B](https://huggingface.co/JZHWS/VITAL-Base-8B), [Assistant-8B](https://huggingface.co/JZHWS/VITAL-Assistant-8B)) | Weights: Yes per card. Code: Unclear | [arXiv 2511.17962](https://arxiv.org/abs/2511.17962) | [GitHub](https://github.com/jzhws/VITAL-Series) | verified |
| OmniQuality-R | 2025 / arXiv | Technical quality + aesthetics + text-image alignment (images) | Qwen2.5-VL-7B (from `config.json`) | Not stated | Not stated ([yeeeeeyy/VisualScore](https://huggingface.co/yeeeeeyy/VisualScore)) | Unclear | [arXiv 2510.10609](https://arxiv.org/abs/2510.10609) | [GitHub](https://github.com/yeppp27/VisualScore) | verified |
| TATAR | 2026 / arXiv | IQA + IAA | Qwen2.5-VL-7B | README says "To be released" | Not released | Unclear | [arXiv 2603.19779](https://arxiv.org/abs/2603.19779) | [GitHub](https://github.com/yinwen2019/TATAR) (README only) | verified |
| LLaVA-Assessor-GIGA-7B | 2026 / arXiv | Image + video quality scoring and interpretation | LLaVA-style, 8.08B stored | Apache-2.0 | Apache-2.0 ([card](https://huggingface.co/JZHWS/Llava-Assessor-GIGA-7B)) | Yes for code/weights; data terms separate | [arXiv 2609.26205](https://arxiv.org/abs/2609.26205) | [GitHub](https://github.com/jzhws/LLaVA-Assessor) | verified (license); numbers not extracted |
| Qwen-Image-Bench judge (Q-Judger) | 2026 / arXiv | T2I judge over 5 rubric dimensions, incl. quality and aesthetics | Qwen3.6-27B (27.36B stored) | Apache-2.0 (card) | Apache-2.0 | Yes per card | [arXiv 2605.28091](https://arxiv.org/abs/2605.28091) | [Qwen/Qwen-Image-Bench](https://huggingface.co/Qwen/Qwen-Image-Bench) | verified |

Notes:

- Q-Instruct and Co-Instruct are describers and comparers, not MOS scorers. They matter because their data (Q-Instruct-200K, Co-Instruct-562K) feeds later unified models. See [../datasets/iqa.md](../datasets/iqa.md).
- Q-Judger reports Spearman rho 0.92 between its rankings and expert rankings of 18 generators. That is a generator-level ranking, not per-image MOS correlation. Use it as a teacher or critic, not as a ONE-ALIGN competitor.
- Q-Eval-Score original weights are not public. The open version was re-trained on public data only.
- UniQA ([arXiv 2406.01069](https://arxiv.org/abs/2406.01069)) is a CLIP-based IQA + IAA model, not an LMM. It is out of scope here. Status: not checked.

### 1.2 Q-ReAlign checkpoints

All three variants train on the same ONE-ALIGN mix: KonIQ + SPAQ + KADID + AGIQA-20K + AVA + LSVQ ([README](https://github.com/Q-Future/Q-ReAlign)). Training is full-parameter BF16 SFT via ms-swift, with the vision tower and projector trainable (model cards).

| Variant | Stored params (safetensors) | Stored bytes | Raw base model params | Vision tower (card) |
|---|---:|---:|---:|---|
| [Mini-0.8B](https://huggingface.co/q-future/Q-ReAlign-Mini-0.8B) | 1,107,265,600 | 2.21 GB (1 file) | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B): 873,438,784 | depth 12, hidden 768, patch 16, merge 2 |
| [Lite-4B](https://huggingface.co/q-future/Q-ReAlign-Lite-4B) | 5,174,964,736 | 10.35 GB (3 shards) | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B): 4,659,865,088 | depth 24, hidden 1024, patch 16, merge 2 |
| [Pro-9B](https://huggingface.co/q-future/Q-ReAlign-Pro-9B) | 9,409,813,744 | 18.82 GB (4 shards; index `total_size` 18,819,627,488) | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B): 9,653,104,368 | depth 27, hidden 1152, patch 16, merge 2 |

- The Pro index file declares `total_parameters: 1469680`. That is wrong. The HF badge shows this small number. Use the per-dtype count (9.41B BF16) or the byte total.
- Stored counts differ from the raw bases in both directions. Mini and Lite are larger; Pro is smaller. The exact source revision and any added or removed heads are not documented. Audit before claiming the same architecture.
- Scoring: softmax over the next-token logits of `" excellent"`, `" good"`, `" fair"`, `" poor"`, `" bad"` (with leading space), weights `[1.0, 0.75, 0.5, 0.25, 0.0]`, output in [0, 1]. Video: 8 frames by default.
- Q-ReAlign is available in IQA-PyTorch as `qrealign`, `qrealign-lite`, `qrealign-pro` ([README](https://github.com/Q-Future/Q-ReAlign)).

### 1.3 Q-ReAlign results (converted to PLCC / SRCC)

Source: [Q-ReAlign README](https://github.com/Q-Future/Q-ReAlign) and model cards. The source order is SRCC / PLCC; we swapped each pair. Protocol: trained on the six-dataset ONE-ALIGN mix; the card says "full evaluation sets". The cards do not say which LIVE set or which AGI set is used. LIVE is not in the training mix, so it is a cross-dataset test. Author-reported.

| Model | KonIQ | SPAQ | KADID | AGI | LIVE | AVA | LSVQ | Avg. |
|---|---|---|---|---|---|---|---|---|
| "Q-Align" row in Q-ReAlign table | 0.944 / 0.942 | 0.933 / 0.932 | 0.920 / 0.912 | 0.781 / 0.738 | 0.870 / 0.897 | 0.796 / 0.798 | 0.866 / 0.867 | 0.873 / 0.869 |
| Q-ReAlign Mini (0.8B) | 0.938 / 0.935 | 0.933 / 0.931 | 0.907 / 0.903 | 0.848 / 0.811 | 0.873 / 0.907 | 0.794 / 0.797 | 0.869 / 0.869 | 0.880 / 0.879 |
| Q-ReAlign Lite (4B) | 0.941 / 0.943 | 0.934 / 0.932 | 0.931 / 0.928 | 0.871 / 0.829 | 0.862 / 0.899 | 0.804 / 0.814 | 0.879 / 0.880 | 0.889 / 0.889 |
| Q-ReAlign Pro (9B) | 0.952 / 0.950 | 0.937 / 0.935 | 0.939 / 0.934 | 0.885 / 0.843 | 0.876 / 0.902 | 0.828 / 0.832 | 0.884 / 0.883 | 0.900 / 0.896 |

### 1.4 OneAlign published results (converted to PLCC / SRCC)

Source: [q-future/one-align card](https://huggingface.co/q-future/one-align) (source order SRCC / PLCC). Protocol: OneAlign trained on KonIQ + SPAQ + KADID (IQA), AVA (IAA), LSVQ (VQA). "Unseen" = cross-dataset. Author-reported.

| OneAlign | KonIQ | SPAQ | KADID | LIVE-C (unseen) | LIVE (unseen) | CSIQ (unseen) | AGIQA (unseen) | AVA | LSVQ test | LSVQ 1080p | KoNViD | MaxWell test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PLCC / SRCC | 0.950 / 0.941 | 0.935 / 0.932 | 0.942 / 0.941 | 0.894 / 0.881 | 0.856 / 0.887 | 0.906 / 0.881 | 0.838 / 0.801 | 0.819 / 0.823 | 0.886 / 0.886 | 0.837 / 0.803 | 0.888 / 0.876 | 0.786 / 0.781 |

Important: the "Q-Align" row in the Q-ReAlign table does not match OneAlign's own card. KADID is 0.920 / 0.912 there versus 0.942 / 0.941 on the OneAlign card. AVA is 0.796 / 0.798 versus 0.819 / 0.823. The Q-ReAlign README does not say which Q-Align checkpoint or eval code produced its row. OneAlign did not train on AGIQA-20K, and Q-ReAlign did, so the AGI columns are not comparable. Re-run OneAlign and Q-ReAlign on the same frozen manifests before claiming a margin. See [../evaluation/protocol.md](../evaluation/protocol.md).

### 1.5 Other unified models: published numbers (PLCC / SRCC)

Each block is its own protocol. Do not rank across blocks.

**UniPercept** ([paper Table 1](https://arxiv.org/html/2512.21675v1)). Trained on ArtiMuse-10K + KonIQ-10K + ISTA-10K (visual rating), after ~800K-sample domain-adaptive pre-training and task-aligned RL. Other sets are cross-domain.

| ArtiMuse-10K | AVA | TAD66K | FLICKR-AES | KonIQ | SPAQ | KADID | PIPAL | ISTA-10K |
|---|---|---|---|---|---|---|---|---|
| 0.738 / 0.746 | 0.577 / 0.589 | 0.346 / 0.336 | 0.681 / 0.688 | 0.949 / 0.940 | 0.895 / 0.904 | 0.870 / 0.872 | 0.594 / 0.581 | 0.767 / 0.778 |

**TATAR** ([paper Table 1](https://arxiv.org/html/2603.19779v1)). Qwen2.5-VL-7B, SFT + GRPO. Trained on KonIQ train (6,189 kept) + ArtiMuse-10K train (7,726 kept). Other sets are cross-domain. Source reports percent; converted.

| ArtiMuse-10K | AVA | TAD66K | FLICKR-AES | KonIQ | SPAQ | KADID | PIPAL |
|---|---|---|---|---|---|---|---|
| 0.595 / 0.586 | 0.524 / 0.518 | 0.349 / 0.334 | 0.633 / 0.604 | 0.942 / 0.941 | 0.894 / 0.897 | 0.727 / 0.731 | 0.498 / 0.501 |

**VITAL-Base-8B** ([paper image and video tables](https://arxiv.org/html/2511.17962)). Vision-encoder-centred pre-training on ~4.5M machine-annotated pairs. The machine labels come from specialist models trained on KonIQ train (images) and LSVQ train (videos). The paper marks some columns as OOD; we did not re-check which.

| KonIQ | SPAQ | LIVE-C | AGIQA | KADID | LSVQ test | LSVQ 1080p | KoNViD-1k |
|---|---|---|---|---|---|---|---|
| 0.931 / 0.931 | 0.886 / 0.884 | 0.866 / 0.851 | 0.811 / 0.736 | 0.708 / 0.759 | 0.879 / 0.883 | 0.815 / 0.786 | 0.881 / 0.878 |

**Q-Eval-Score, open version** ([Q-Eval README](https://github.com/zzc-1998/Q-Eval)). Re-trained on the public part of Q-Eval-100K; authors re-tested it. AIGC content only.

| Image quality | Video quality | Image alignment | Video alignment |
|---|---|---|---|
| 0.760 / 0.747 | 0.700 / 0.673 | 0.797 / 0.826 | 0.613 / 0.614 |

Metric-order warning: the UniPercept and TATAR tables label pairs as SRCC / PLCC. But their DeQA-Score and Q-Align baseline rows (KonIQ 0.953 / 0.941 and 0.941 / 0.940) are the exact pairs from [DeQA-Score Table 3](https://arxiv.org/html/2501.11561), which uses PLCC / SRCC. So baseline rows in those tables may be in the wrong order. Their own-model rows are converted as labelled. Treat all small SRCC-vs-PLCC gaps in these tables with care.

### 1.6 Use in this project

- Frozen references: Q-ReAlign Pro, Lite, and Mini; OneAlign. They share the five-word scoring contract, so they are the cleanest controls.
- IQA + IAA challenger: UniPercept (weights available). TATAR has no weights.
- IQA + VQA challengers: VITAL-Base-8B, VQA² scorer, LLaVA-Assessor-GIGA-7B.
- AIGC image/video: Q-Eval-Score open version is NC-licensed. Use it for research comparison only.
- None of these models reports results on the full seven-set Q-ReAlign table. A fair comparison needs one shared manifest. See [../evaluation/protocol.md](../evaluation/protocol.md).

## Part 2: Candidate VLM backbones

### 2.1 Backbone table

Stored params are the `safetensors` totals from the Hugging Face API, queried 2026-09-27. They include vision, audio, and embedding parts. BF16 weights = 2 bytes x params, weights only. VRAM columns are planning estimates from the dossier, not measurements. They assume microbatch 1, gradient checkpointing, bounded visual tokens, and a frozen or partly trainable vision tower.

| Model | Release | Stored params | BF16 weights | Weights license (HF card) | Vision input notes | Planning VRAM, LoRA | Quality-task evidence | Status |
|---|---|---:|---:|---|---|---|---|---|
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 2026-03 (HF repo 2026-02-28) | 0.873B | 1.75 GB | Apache-2.0 | Native image and video; same family as Q-ReAlign | 24 GB | Q-ReAlign Mini avg 0.880 / 0.879 (Section 1.3) | verified |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 2026-03-02 | 4.660B | 9.32 GB | Apache-2.0 | Native image and video; hybrid Gated DeltaNet / full attention decoder | 48 GB | Q-ReAlign Lite avg 0.889 / 0.889. [MR-IQA-2](https://huggingface.co/RobinY99/MR-IQA-2) uses a Qwen3.5-4B actor (from dossier) | verified |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 2026-03-02 | 9.653B | 19.31 GB | Apache-2.0 | As above; vision depth 27, hidden 1152, patch 16 | 48 GB pilot; 80-96 GB with vision gradients | Q-ReAlign Pro avg 0.900 / 0.896. Z-Reward distils Qwen3.5-27B into a 9B student (from dossier) | verified |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 2026-08-14 | 27.781B | 55.56 GB | Apache-2.0 | Native image and video; config still `qwen3_5` family (from dossier) | 80-96 GB constrained LoRA; 48 GB only with QLoRA, needs profiling | None for quality. NeMo recipes show general VLM fine-tuning only (from dossier) | verified |
| [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) | 2026-04-02 (from dossier; HF repo 2026-03-02) | 7.996B | 15.99 GB | Apache-2.0 | ~150M vision encoder; variable aspect ratio; token budgets 70 / 140 / 280 / 560 / 1120; video as frame sequences; ~300M audio encoder | 48 GB pilot; 80-96 GB for high token budgets or video | BrightRate-LM LoRA on BrightVQ split 0, SDR input: 0.8805 / 0.8624 ([adapter](https://huggingface.co/shreshthsaini/brightrate-study-gemma4-e4b-sdr)) | verified |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | 2026-04-02 (from dossier) | 5.123B | 10.25 GB | Apache-2.0 | As E4B | 24-48 GB | Same study: 0.8849 / 0.8817 ([adapter](https://huggingface.co/shreshthsaini/brightrate-study-gemma4-e2b-sdr)) | verified |
| [Gemma 4 12B Unified](https://huggingface.co/google/gemma-4-12B-it) | 2026-06-03 | 11.960B | 23.92 GB | Apache-2.0 | Encoder-free: raw image patches are projected straight into the decoder; no vision tower to unfreeze | 48 GB constrained; 80-96 GB preferred | Same study: SDR 0.7821 / 0.7763 ([adapter](https://huggingface.co/shreshthsaini/brightrate-study-gemma4-12b-sdr)). Native-PQ adapter card: 0.0251 / 0.0100 (failed run) | verified |
| [MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) | 2026-05-11 | 1.300B | 2.60 GB | Apache-2.0 (weights and code) | SigLIP2-400M + Qwen3.5-0.8B; mixed 4x / 16x visual token compression (default 16x); video at 1 FPS up to 128 frames, then uniform sampling | 24 GB LoRA; 48 GB for full or partial tuning | None found for IQA / IAA / VQA | verified |
| [InternVL3.5-4B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF) | 2025-08-26 (from dossier) | 4.732B | 9.46 GB | Apache-2.0 | InternViT-300M + Qwen3 LLM; dynamic tiling with thumbnail | 48 GB | None fine-tuned. UniPercept and VITAL build on InternVL3-8B, not 3.5 | verified |
| [InternVL3.5-8B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) | 2025-08-26 (from dossier) | 8.528B | 17.06 GB | Apache-2.0 | As 4B | 48 GB | Zero-shot on UniPercept-Bench IAA Q&A: 28.18% vs InternVL3-8B 62.60% (accuracy, [UniPercept](https://arxiv.org/html/2512.21675v1)) | verified |
| [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) | 2025-12-08 (from dossier) | 10.293B | 20.59 GB | MIT | Images and video; native Transformers support | 48 GB | None found | verified |

Row notes:

- Qwen3.5-0.8B is not in the dossier list. We added it because Q-ReAlign Mini uses it and MiniCPM-V-4.6 uses it as the LLM.
- Release dates marked "from dossier" follow official announcements in the dossier. HF repo creation dates can come before public release.
- Planning VRAM for full fine-tuning is much higher. At about 18 bytes per parameter before activations: MiniCPM-V-4.6 ~23 GB, Gemma 4 E4B ~144 GB, Qwen3.5-9B ~174 GB, Qwen3.8-27B ~500 GB (from dossier; arithmetic, not measured).

### 2.2 BrightRate-LM study: one task, one split

Source: [BrightRate-LM](https://github.com/shreshthsaini/BrightRate-LM) adapter cards on Hugging Face. Task: HDR video quality on BrightVQ. Protocol: content-separated split 0, 420 test videos, rank-16 LoRA, five quality words, same training settings. SDR-proxy input unless stated. Author-reported, submitted study. It is not a ONE-ALIGN result.

| Backbone | PLCC / SRCC | Adapter |
|---|---|---|
| Gemma 4 31B | 0.8843 / 0.8818 | [card](https://huggingface.co/shreshthsaini/brightrate-study-gemma4-31b-sdr) |
| Gemma 4 E2B | 0.8849 / 0.8817 | [card](https://huggingface.co/shreshthsaini/brightrate-study-gemma4-e2b-sdr) |
| Gemma 4 E4B | 0.8805 / 0.8624 | [card](https://huggingface.co/shreshthsaini/brightrate-study-gemma4-e4b-sdr) |
| Qwen3-VL-8B | 0.8729 / 0.8586 | [card](https://huggingface.co/shreshthsaini/brightrate-study-qwen3vl-8b-sdr) |
| Gemma 4 26B (MoE) | 0.8639 / 0.8568 | [card](https://huggingface.co/shreshthsaini/brightrate-study-gemma4-26b-sdr) |
| Qwen3-VL-4B | 0.8579 / 0.8418 | [card](https://huggingface.co/shreshthsaini/brightrate-study-qwen3vl-4b-sdr) |
| Gemma 4 12B Unified | 0.7821 / 0.7763 | [card](https://huggingface.co/shreshthsaini/brightrate-study-gemma4-12b-sdr) |

What this shows: the five-word LoRA recipe transfers to Gemma 4. Bigger is not always better here: E2B matches 31B, and the encoder-free 12B is lowest. Input format matters: the dossier reports a statistics-matched native-PQ 12B run at SRCC 0.8383, but the released native-PQ adapter card shows 0.0100. We could not confirm the 0.8383 run on a card. One split is not enough to rank backbones.

### 2.3 Licenses: what to watch

- Gemma 4 is Apache-2.0. The HF card `license_link` points to a Gemma 4 page that serves the Apache 2.0 text. This is a change: [Gemma 3](https://huggingface.co/google/gemma-3-4b-it) uses the `gemma` license with manual gating. Do not copy Gemma 3 terms to Gemma 4, or the reverse.
- Qwen3.5, Qwen3.8-27B, MiniCPM-V-4.6, and InternVL3.5 weights are Apache-2.0 per their cards. GLM-4.6V-Flash is MIT.
- A permissive backbone license does not clear the training data. KonIQ, AVA, LSVQ, and others have their own terms. See [../datasets/iqa.md](../datasets/iqa.md), [../datasets/iaa.md](../datasets/iaa.md), and [../datasets/vqa.md](../datasets/vqa.md).

### 2.4 Porting caveats

- **MiniCPM-V vs MiniCPM5-2B.** [MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) is the vision model (image-text-to-text, 1.30B). [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) is a text-only LLM (text-generation, 2.52B stored, HF repo 2026-09-06). It cannot score images. Do not mix them up.
- **Stored vs advertised size.** Gemma 4 E4B is "4.5B effective" but stores 8.0B params. E2B stores 5.1B. Plan memory on stored params.
- **Label tokens.** Q-ReAlign's scorer takes the first token id of each quality word. We ran each tokenizer on the five words (tokenizer only, not in chat context, 2026-09-27):

| Tokenizer | With leading space (`" excellent"` ...) | Without space |
|---|---|---|
| Qwen3.5-9B, MiniCPM-V-4.6, InternVL3.5-8B, GLM-4.6V-Flash | all 5 words = 1 token | `excellent` and `poor` = 2 tokens |
| Gemma 4 E4B | all 5 words = 1 token | all 5 words = 1 token |

  So the leading space is required for Qwen-style tokenizers. Without it, "first token" silently scores a word prefix. For each new backbone, re-compute ids in the real prompt context. Check that the scored position is right after the stem (for example "The quality of the image is"). Disable thinking mode or prefill empty reasoning tags so the next token is the quality word.
- **Vision preprocessing.** Low token budgets and 16x compression can erase blur, noise, and compression cues. Tiling can change composition for aesthetics. Keep the original 8 uniform frames and resize policy first, then change one factor at a time.
- **Capacity mismatch.** Q-ReAlign is full SFT with a trainable vision tower. A frozen-vision LoRA run has less capacity. Report the training mode with every result.

### 2.5 Status of the backbone question

No newer backbone (Gemma 4, Qwen3.8, MiniCPM-V-4.6, InternVL3.5, GLM-4.6V) has a published result on the matched ONE-ALIGN setting (same six training sets, same seven test sets, same scorer). The only direct quality evidence for Gemma 4 is the single-task BrightRate-LM study above. Q-ReAlign Pro (Qwen3.5-9B) remains the reference to beat.

Suggested first matrix (from dossier): Qwen3.5-4B (cost control) vs Gemma 4 E4B (new architecture), same data, splits, scorer, and visual budget. Then MiniCPM-V-4.6 as a low-cost student. Add Qwen3.8-27B only if smaller models plateau. Details: [../training/experiment-plan.md](../training/experiment-plan.md).

Excluded from the shortlist (from dossier): Qwen3.6-35B-A3B, Gemma 4 26B-A4B, and Qwen3.8-Flash-Next. Their active-parameter counts hide large stored weights.
