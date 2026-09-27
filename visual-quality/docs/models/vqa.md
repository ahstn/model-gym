# VQA models (video quality)

This file lists video quality assessment (VQA) models: natural user-generated video (UGC) scorers, LMM and RL-based video scorers, AI-generated video (AIGC) evaluators, and HDR video scorers. It gives release links, licences, and published PLCC / SRCC grouped by training protocol. Unified IQA/IAA/VQA assessors (Q-Align ONE-ALIGN, Q-ReAlign, LLaVA-Assessor) are in [unified-and-backbones.md](unified-and-backbones.md).

Last verified: 2026-09-27

## How to read this file

- Metric pairs are always `PLCC / SRCC`. Most VQA papers print SRCC first. We converted them.
- All numbers are author-reported. We did not run any model.
- Each table uses one protocol. Do not rank rows from different tables against each other.
- "As quoted" means the number comes from a later paper, not the model's own paper. Later papers often copy old numbers. A copied number keeps the protocol of the paper it came from.
- LSVQ train has about 28K videos. LSVQ-test has 7,182 and LSVQ-1080p has 3,573 ([VQAThinker Table 1](https://arxiv.org/abs/2508.06051)).
- `Code` and `Weights` are separate licences. `Commercial use?` covers only these two licences. Video dataset terms are separate and are often non-commercial. See [../datasets/vqa.md](../datasets/vqa.md).
- `Not stated` means we found no licence text. It does not mean "free to use".
- Metric rules, split rules, and calibration are in [../evaluation/protocol.md](../evaluation/protocol.md).

## 1. Model catalog

### 1.1 Natural UGC video: classic and deep specialists

| Model | Year / venue | Idea | Paper | Release | Code | Weights | Commercial use? | Status |
|---|---|---|---|---|---|---|---|---|
| VSFA | 2019 ACM MM | ResNet-50 frame features + GRU + temporal memory pooling | [arXiv 1908.00375](https://arxiv.org/abs/1908.00375) | [lidq92/VSFA](https://github.com/lidq92/VSFA) (`models/VSFA.pt`, KoNViD split 9) | MIT | In repo (MIT) | Yes (code/weights only) | verified |
| BVQA (Li et al.) | 2022 TCSVT | Quality-aware image pre-training + SlowFast motion features | [arXiv 2108.08505](https://arxiv.org/abs/2108.08505) | [zwx8981/TCSVT-2022-BVQA](https://github.com/zwx8981/TCSVT-2022-BVQA) (sample weights on Drive / Baidu) | Apache-2.0 | Not stated | Unclear | verified |
| SimpleVQA | 2022 ACM MM | ResNet-50 spatial + SlowFast motion, end-to-end | [arXiv 2204.14047](https://arxiv.org/abs/2204.14047) | [sunwei925/SimpleVQA](https://github.com/sunwei925/SimpleVQA) (weights on Drive) | Apache-2.0 | Not stated | Unclear | verified |
| FAST-VQA / FasterVQA | 2022 ECCV / 2023 TPAMI (journal per search result) | Grid mini-patch fragments + Video Swin; FasterVQA uses 3D fragments (69G vs 279G MACs in README) | [arXiv 2207.02595](https://arxiv.org/abs/2207.02595), [arXiv 2210.05357](https://arxiv.org/abs/2210.05357) | [VQAssessment/FAST-VQA-and-FasterVQA](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA) (GitHub releases) | S-Lab License 1.0 (non-commercial) | GitHub release under the same repo | No (NC) | verified |
| DOVER / DOVER-Mobile | 2023 ICCV | FAST-VQA technical branch + aesthetic branch | [arXiv 2211.04894](https://arxiv.org/abs/2211.04894) | [VQAssessment/DOVER](https://github.com/VQAssessment/DOVER); [teowu/DOVER](https://huggingface.co/teowu/DOVER) | S-Lab License 1.0 (non-commercial) | HF card says MIT; GitHub release is under the S-Lab repo | Unclear (conflict) | verified |
| MinimalisticVQA | 2024 TPAMI (as quoted) | Minimal spatial (ResNet / Swin) + optional SlowFast models to probe datasets | [arXiv 2307.13981](https://arxiv.org/abs/2307.13981) | [sunwei925/MinimalisticVQA](https://github.com/sunwei925/MinimalisticVQA) (LSVQ weights on Dropbox / Drive) | Apache-2.0 | Not stated | Unclear | verified |
| KSVQE | 2024 CVPR | CLIP quality-aware region selection + distortion adapter; for short-form video | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_KVQ_Kwai_Video_Quality_Assessment_for_Short-form_Videos_CVPR_2024_paper.html), [arXiv 2402.07220](https://arxiv.org/abs/2402.07220) | [lixinustc/KVQ-Challenge-CVPR-NTIRE2024](https://github.com/lixinustc/KVQ-Challenge-CVPR-NTIRE2024) (training code; only a backbone checkpoint linked) | Not stated (no licence file) | Trained KSVQE weights not found | Unclear | verified |
| COVER | 2024 CVPRW (AIS 2024 winner) | Semantic (CLIP), aesthetic, and technical branches with cross-gating | [AIS 2024 report arXiv 2404.16205](https://arxiv.org/abs/2404.16205) | [vztu/COVER](https://github.com/taco-group/COVER) (`Model/COVER.pth`) | MIT | In repo (MIT) | Yes (code/weights only) | verified |
| DPC-VQA | 2026 arXiv | Frozen MLLM base score + small residual calibrator (few-shot) | [arXiv 2604.12813](https://arxiv.org/abs/2604.12813) | No code link in paper | - | - | - | verified |

### 1.2 LMM and RL-based video scorers

| Model | Year / venue | Idea | Paper | Release | Code | Weights | Commercial use? | Status |
|---|---|---|---|---|---|---|---|---|
| Q-Align (video) / ONE-ALIGN | 2024 ICML | mPLUG-Owl2 at 1 fps; 5 rating words | [arXiv 2312.17090](https://arxiv.org/abs/2312.17090) | [Q-Future/Q-Align](https://github.com/Q-Future/Q-Align); [q-future/one-align](https://huggingface.co/q-future/one-align) | S-Lab License 1.0 (non-commercial) | MIT (HF card) | Unclear | verified |
| LMM-VQA | 2024 arXiv (TCSVT version per repo) | ViT + SlowFast tokens into Llama-3; LSVQ instruction pairs | [arXiv 2408.14008](https://arxiv.org/abs/2408.14008) | [Sueqk/LMM-VQA](https://github.com/Sueqk/LMM-VQA) | MIT | Not found | - | verified |
| VQA² (VQA2-Scorer) | 2025 ACM MM | Video quality instruction data (VQA²-ID, 157K); Scorer and Assistant models | [arXiv 2411.03795](https://arxiv.org/abs/2411.03795) | [Q-Future/VQA²](https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment); [VQA-UGC-Scorer](https://huggingface.co/q-future/VQA-UGC-Scorer-llava_qwen) | Apache-2.0 | Apache-2.0 | Yes (code/weights only) | verified |
| FineVQ | 2025 CVPR | InternVL2-8B + LoRA; one model for 6 quality dimensions + QA | [arXiv 2412.19238](https://arxiv.org/abs/2412.19238) | [IntMeGroup/FineVQ](https://github.com/IntMeGroup/FineVQ); [FineVQ_LSVQ](https://huggingface.co/IntMeGroup/FineVQ_LSVQ), [FineVQ_score](https://huggingface.co/IntMeGroup/FineVQ_score) | Not stated (no licence file) | Apache-2.0 (HF cards) | Unclear | verified |
| Q-Insight (video, retrained) | 2025 NeurIPS (spotlight) | Qwen2.5-VL-7B + GRPO for image quality; VQAThinker authors retrained it on LSVQ | [arXiv 2503.22679](https://arxiv.org/abs/2503.22679) | [bytedance/Q-Insight](https://github.com/bytedance/Q-Insight); [ByteDance/Q-Insight](https://huggingface.co/ByteDance/Q-Insight) | Apache-2.0 | Apache-2.0 | Yes (code/weights only) | verified |
| VQ-Insight | 2026 AAAI (oral) | Image warm-up, then RL with temporal-modeling reward; natural and AIGC video | [arXiv 2506.18564](https://arxiv.org/abs/2506.18564) | [xuanyuzhang21/VQ-Insight](https://github.com/xuanyuzhang21/VQ-Insight); weights in [ByteDance/Q-Insight](https://huggingface.co/ByteDance/Q-Insight/tree/main) subfolders (`vqinsight-naturalvideo`, `vqinsight-aigcvideo`, `vqinsight-comp`) | Not stated in VQ-Insight repo; Q-Insight family repo is Apache-2.0 | Apache-2.0 (HF card) | Unclear | verified |
| VQAThinker | 2026 AAAI | InternVL3-8B + GRPO; bell-shaped regression, pairwise ranking, and temporal consistency rewards | [arXiv 2508.06051](https://arxiv.org/abs/2508.06051) | [clh124/VQAThinker](https://github.com/clh124/VQAThinker); [InternVL3-VQAThinker-8B](https://huggingface.co/kkkkkklinhan/InternVL3-VQAThinker-8B) | Not stated | Not stated | Unclear | verified |
| VersusQ | 2026 arXiv | Qwen3-VL-4B + frozen SlowFast; pairwise margin reasoning; MC-GRPO; least-squares score recovery | [arXiv 2605.21130](https://arxiv.org/abs/2605.21130) | No code or weights found | - | - | - | verified |
| MDS-VQA | 2026 CVPR (highlight) | Failure predictor picks unlabeled target videos to label (5% budget) | [arXiv 2603.11525](https://arxiv.org/abs/2603.11525) | [MDS-VQA code](https://github.com/Multimedia-Analytics-Laboratory/MDS-VQA); [hollow404/MDS-VQA-Active-Finetuning](https://huggingface.co/hollow404/MDS-VQA-Active-Finetuning) | Apache-2.0 | Apache-2.0 | Yes (code/weights only) | verified (release); numbers in [../training/data-mixture-evidence.md](../training/data-mixture-evidence.md) |

VisualQuality-R1 (an image RL scorer) appears below only as a retrained video baseline. See [iqa.md](iqa.md).

### 1.3 AI-generated video (AIGC) evaluators

| Model | Year / venue | Idea | Paper | Release | Code | Weights | Commercial use? | Status |
|---|---|---|---|---|---|---|---|---|
| T2VQA | 2024 ACM MM | BLIP text-video alignment + Video Swin fidelity + LLM head; built T2VQA-DB (10K videos) | [arXiv 2403.11956](https://arxiv.org/abs/2403.11956) | [QMME/T2VQA](https://github.com/QMME/T2VQA) (training code; no trained T2VQA weights linked) | Not stated | Not found | - | verified |
| VideoScore (v1, v1.1) | 2024 EMNLP | MLLM scorer for 5 aspects (visual quality, temporal, dynamic degree, text alignment, factual); trained on VideoFeedback | [arXiv 2406.15252](https://arxiv.org/abs/2406.15252) | [TIGER-AI-Lab/VideoScore](https://github.com/TIGER-AI-Lab/VideoScore); [VideoScore](https://huggingface.co/TIGER-Lab/VideoScore), [VideoScore-v1.1](https://huggingface.co/TIGER-Lab/VideoScore-v1.1) | MIT | v1: Apache-2.0; v1.1: MIT | Yes (code/weights only) | verified |
| AIGV-Assessor | 2025 CVPR | InternVL2-8B + spatiotemporal projectors + regression + pairwise stage; built AIGVQA-DB | [arXiv 2411.17221](https://arxiv.org/abs/2411.17221) | [wangjiarui153/AIGV-Assessor](https://github.com/wangjiarui153/AIGV-Assessor); [static_quality](https://huggingface.co/IntMeGroup/AIGV-Assessor-static_quality) and 3 other dimension checkpoints | Not stated (no licence file) | Apache-2.0 (HF cards) | Unclear | verified |
| VideoAlign / VideoReward | 2025 NeurIPS | Multi-dimension reward model for text-to-video preference | [arXiv 2501.13918](https://arxiv.org/abs/2501.13918) | [KlingAIResearch/VideoAlign](https://github.com/KlingAIResearch/VideoAlign); [KlingTeam/VideoReward](https://huggingface.co/KlingTeam/VideoReward) | MIT | Apache-2.0 | Yes (code/weights only) | verified (release); pairwise accuracy, no MOS numbers here |
| VideoScore2 | 2025 arXiv | Qwen2.5-VL-7B; think-then-score; SFT + GRPO on VideoFeedback2 (27,168 videos); visual, alignment, physics scores | [arXiv 2509.22799](https://arxiv.org/abs/2509.22799) | [TIGER-AI-Lab/VideoScore2](https://github.com/TIGER-AI-Lab/VideoScore2); [TIGER-Lab/VideoScore2](https://huggingface.co/TIGER-Lab/VideoScore2) | MIT | Apache-2.0 | Yes (code/weights only) | verified |
| VQ-Insight (AIGC) | 2026 AAAI | See 1.2 | [arXiv 2506.18564](https://arxiv.org/abs/2506.18564) | `vqinsight-aigcvideo` in [ByteDance/Q-Insight](https://huggingface.co/ByteDance/Q-Insight/tree/main) | See 1.2 | Apache-2.0 | Unclear | verified |
| RefVQA | 2026 arXiv | Reference-aware graph model; retrieves reference videos for comparison | [arXiv 2604.17074](https://arxiv.org/abs/2604.17074) | No code link found | - | - | - | verified |

Also seen: Video Inspector / Holmes-I2V (CVPR 2026 Findings, [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Somers_Video_Inspector_An_Agentic-RL_Framework_and_Benchmark_for_Human-Aligned_Generative_CVPRF_2026_paper.pdf)). Not checked beyond the search result; unverified. Benchmarks such as VBench and EvalCrafter are in [../evaluation/benchmarks.md](../evaluation/benchmarks.md).

### 1.4 HDR video

| Model | Year / venue | Idea | Paper | Release | Code | Weights | Commercial use? | Status |
|---|---|---|---|---|---|---|---|---|
| HIDRO-VQA | 2024 WACV | Self-supervised HDR-aware features | [arXiv 2311.11059](https://arxiv.org/abs/2311.11059) | [shreshthsaini/HIDRO-VQA](https://github.com/shreshthsaini/HIDRO-VQA) | MIT | Not stated | Unclear | verified |
| BrightRate | 2026 WACV (oral) | CONTRIQUE + CLIP + HDR luminance transform + temporal difference features | [CVF PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Saini_BrightRate_Quality_Assessment_for_User-Generated_HDR_Videos_WACV_2026_paper.pdf) | [shreshthsaini/BrightVQ](https://github.com/shreshthsaini/BrightVQ) | LICENSE file: CC BY-NC-SA 4.0; README badge: CC BY-NC 4.0 | Not stated | No (NC) | verified |
| BrightRate-LM | 2026 (journal submission, not peer-reviewed yet) | Qwen2.5-VL-7B LoRA on multi-exposure frame stacks; 0–100 score + defect text; 16 study adapters (Qwen2.5-VL, Qwen3-VL, Gemma 4) | [repo README](https://github.com/shreshthsaini/BrightRate-LM) (no arXiv yet) | [BrightRate-LM](https://github.com/shreshthsaini/BrightRate-LM); [brightrate-lm-7b-multiexposure](https://huggingface.co/shreshthsaini/brightrate-lm-7b-multiexposure) | MIT | MIT (HF cards; check base-model terms) | Yes for adapters; base-model terms separate | verified |
| HDR-Q | 2026 CVPR | HDR-aware SigLIP-2 encoder + HAPO RL (HDR-aware policy optimization); Ovis2.5 base per a search summary (unverified) | [arXiv 2603.00938](https://arxiv.org/abs/2603.00938) | [Beyond8Bits](https://github.com/shreshthsaini/Beyond8Bits) (dataset repo) | Metadata/site: CC BY-4.0; model code not found | Not found (HF search for "HDR-Q" returned no model) | - | verified |

## 2. Protocol V1: trained on LSVQ train, cross-dataset tests

All rows train on LSVQ train (about 28K videos) unless noted. LSVQ-test and LSVQ-1080p are intra-dataset. KoNViD-1k, LIVE-VQC, and YouTube-UGC are cross-dataset (no training on them).

### 2.1 VQAThinker Table 1 (one source, consistent columns)

Source: [VQAThinker Table 1](https://arxiv.org/abs/2508.06051). Q-Insight and VisualQuality-R1 were retrained on LSVQ by the VQAThinker authors. Q-Align uses a "fused (287K)" multi-task training set, and VQA² uses VQA²-ID (157K), so those two rows are not LSVQ-only.

| Model | Training data | LSVQ-test | LSVQ-1080p | KoNViD-1k | LIVE-VQC | YouTube-UGC |
|---|---|---|---|---|---|---|
| SimpleVQA | LSVQ | 0.861 / 0.864 | 0.801 / 0.756 | 0.860 / 0.861 | 0.799 / 0.762 | 0.808 / 0.808 |
| FAST-VQA | LSVQ | 0.880 / 0.880 | 0.813 / 0.781 | 0.854 / 0.859 | 0.845 / 0.826 | 0.747 / 0.730 |
| DOVER | LSVQ | 0.866 / 0.878 | 0.813 / 0.782 | 0.869 / 0.874 | 0.840 / 0.817 | 0.781 / 0.771 |
| MinimalisticVQA | LSVQ | 0.882 / 0.885 | 0.828 / 0.792 | 0.859 / 0.862 | 0.821 / 0.775 | 0.821 / 0.826 |
| Q-Align | fused 287K | 0.884 / 0.886 | 0.822 / 0.761 | 0.878 / 0.876 | 0.819 / 0.783 | 0.846 / 0.834 |
| VQA²-Scorer | VQA²-ID 157K | 0.872 / 0.878 | 0.821 / 0.794 | 0.880 / 0.881 | 0.830 / 0.785 | 0.823 / 0.811 |
| Q-Insight (retrained) | LSVQ | 0.639 / 0.644 | 0.648 / 0.601 | 0.753 / 0.751 | 0.708 / 0.624 | 0.591 / 0.560 |
| VisualQuality-R1 (retrained) | LSVQ | 0.796 / 0.795 | 0.744 / 0.716 | 0.792 / 0.784 | 0.781 / 0.732 | 0.730 / 0.717 |
| VQ-Insight | LSVQ + KonIQ (38K) | 0.876 / 0.875 | 0.823 / 0.786 | 0.884 / 0.875 | 0.835 / 0.790 | - |
| VQAThinker | LSVQ | 0.880 / 0.883 | 0.834 / 0.798 | 0.884 / 0.881 | 0.847 / 0.808 | 0.863 / 0.860 |

The VQ-Insight row matches [VQ-Insight Table 3](https://arxiv.org/abs/2506.18564).

### 2.2 Other LSVQ-trained results (separate sources)

| Model | LSVQ-test | LSVQ-1080p | KoNViD-1k | LIVE-VQC | YouTube-UGC | Source |
|---|---|---|---|---|---|---|
| VersusQ | 0.886 / 0.884 | 0.852 / 0.824 | 0.884 / 0.879 | 0.848 / 0.801 | - | [VersusQ Table 1](https://arxiv.org/abs/2605.21130) (baseline rows copied from VQAThinker) |
| LMM-VQA (with multi-task) | 0.919 / 0.916 | 0.899 / 0.891 | 0.902 / 0.901 | 0.805 / 0.767 | 0.761 / 0.748 | [LMM-VQA Tables III–IV](https://arxiv.org/abs/2408.14008) |
| LMM-VQA (without multi-task) | 0.914 / 0.913 | 0.898 / 0.879 | 0.876 / 0.875 | 0.863 / 0.831 | 0.877 / 0.858 | [LMM-VQA Tables III–IV](https://arxiv.org/abs/2408.14008) |
| DOVER | 0.889 / - | 0.830 / - | 0.883 / - | 0.854 / - | - | [DOVER README](https://github.com/VQAssessment/DOVER) (PLCC only) |
| DOVER-Mobile | 0.867 / - | 0.802 / - | 0.853 / - | 0.835 / - | - | [DOVER README](https://github.com/VQAssessment/DOVER) (PLCC only) |
| FAST-VQA-B | 0.877 / - | 0.814 / - | 0.855 / - | 0.844 / - | - | [FAST-VQA README](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA) (PLCC only) |
| FasterVQA | 0.874 / - | 0.811 / - | 0.864 / - | 0.837 / - | - | [FAST-VQA README](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA) (PLCC only) |

### 2.3 Q-Align paper Table 5 (LSVQ train; MaxWell test is cross-dataset)

Source: [Q-Align Table 5 and Table 7](https://arxiv.org/abs/2312.17090). Baseline rows are quoted from earlier papers.

| Model | LSVQ-test | LSVQ-1080p | KoNViD-1k | MaxWell-test |
|---|---|---|---|---|
| TLVQM | 0.774 / 0.772 | 0.616 / 0.589 | 0.724 / 0.732 | - |
| VSFA | 0.796 / 0.801 | 0.704 / 0.675 | 0.794 / 0.784 | - |
| PVQ | 0.828 / 0.827 | 0.739 / 0.711 | 0.795 / 0.791 | 0.634 / 0.618 |
| BVQA | 0.854 / 0.852 | 0.788 / 0.772 | 0.830 / 0.839 | 0.673 / 0.675 |
| SimpleVQA | 0.861 / 0.867 | 0.803 / 0.764 | 0.834 / 0.840 | 0.715 / 0.720 |
| FAST-VQA | 0.877 / 0.876 | 0.814 / 0.779 | 0.855 / 0.859 | 0.728 / 0.720 |
| Q-Align (1 fps) | 0.882 / 0.883 | 0.830 / 0.797 | 0.877 / 0.865 | 0.782 / 0.780 |
| DOVER (ensemble) | 0.887 / 0.886 | 0.830 / 0.795 | 0.884 / 0.883 | 0.755 / 0.748 |
| Q-Align + FAST-VQA (ensemble) | 0.899 / 0.899 | 0.850 / 0.818 | 0.897 / 0.895 | 0.784 / 0.779 |
| ONE-ALIGN (IQA + IAA + VQA data) | 0.886 / 0.886 | 0.837 / 0.803 | 0.888 / 0.876 | 0.786 / 0.781 |

### 2.4 Out-of-distribution sets (LSVQ-trained)

Source: [VQAThinker Table 1, OOD part](https://arxiv.org/abs/2508.06051). VersusQ row from [VersusQ Table 1](https://arxiv.org/abs/2605.21130).

| Model | LIVE-YT-Gaming | CGVDS | LIVE-YT-HFR | Waterloo-IVC-4K | VDPVE |
|---|---|---|---|---|---|
| SimpleVQA | 0.728 / 0.657 | 0.809 / 0.773 | 0.502 / 0.416 | 0.425 / 0.379 | 0.647 / 0.643 |
| FAST-VQA | 0.677 / 0.631 | 0.747 / 0.725 | 0.415 / 0.326 | 0.363 / 0.327 | 0.620 / 0.611 |
| DOVER | 0.728 / 0.647 | 0.747 / 0.694 | 0.465 / 0.360 | 0.418 / 0.368 | 0.631 / 0.627 |
| MinimalisticVQA | 0.746 / 0.686 | 0.816 / 0.797 | 0.388 / 0.301 | 0.502 / 0.459 | 0.641 / 0.639 |
| Q-Align (fused 287K) | 0.681 / 0.611 | 0.798 / 0.756 | 0.342 / 0.329 | 0.497 / 0.414 | 0.649 / 0.639 |
| VQA²-Scorer | 0.698 / 0.613 | 0.741 / 0.656 | 0.413 / 0.332 | 0.474 / 0.415 | 0.692 / 0.684 |
| Q-Insight (retrained) | 0.326 / 0.310 | 0.384 / 0.372 | 0.268 / 0.256 | 0.206 / 0.218 | 0.564 / 0.547 |
| VisualQuality-R1 (retrained) | 0.548 / 0.472 | 0.574 / 0.493 | 0.347 / 0.340 | 0.298 / 0.227 | 0.637 / 0.622 |
| VQAThinker | 0.806 / 0.767 | 0.845 / 0.856 | 0.610 / 0.528 | 0.624 / 0.573 | 0.716 / 0.706 |
| VersusQ | 0.814 / 0.774 | - | - | 0.670 / 0.587 | 0.745 / 0.723 |

Finding: frame-rate (LIVE-YT-HFR) and compression-only (Waterloo-IVC-4K) sets stay hard for every model. LSVQ training does not cover these distortions.

## 3. Protocol V2: dataset-specific training (train and test on the same dataset)

### 3.1 KSVQE paper Table 1 (KVQ and in-the-wild sets)

Source: [KVQ / KSVQE Table 1](https://arxiv.org/abs/2402.07220). KVQ numbers are trained on KVQ. Other columns are mostly copied from original papers (`N/A` = not in the original paper). DOVER* excludes the aesthetic branch.

| Model | KVQ | KoNViD-1k | YouTube-UGC | LIVE-VQC |
|---|---|---|---|---|
| TLVQM | 0.509 / 0.490 | 0.768 / 0.773 | 0.659 / 0.669 | 0.802 / 0.798 |
| RAPIQUE | 0.717 / 0.740 | 0.817 / 0.803 | 0.768 / 0.759 | 0.786 / 0.754 |
| VSFA | 0.765 / 0.762 | 0.775 / 0.773 | 0.743 / 0.724 | 0.795 / 0.773 |
| GSTVQA | 0.781 / 0.786 | 0.825 / 0.814 | - | 0.796 / 0.788 |
| PVQ | 0.801 / 0.794 | 0.786 / 0.791 | - | 0.837 / 0.827 |
| SimpleVQA | 0.847 / 0.840 | 0.860 / 0.856 | 0.856 / 0.847 | - |
| FAST-VQA | 0.834 / 0.832 | 0.892 / 0.891 | 0.852 / 0.855 | 0.862 / 0.849 |
| DOVER* | 0.837 / 0.833 | 0.910 / 0.908 | 0.851 / 0.841 | 0.875 / 0.844 |
| KSVQE | 0.869 / 0.867 | 0.921 / 0.922 | 0.912 / 0.900 | 0.883 / 0.861 |

### 3.2 FineVQ paper Table 3 (compiled; mixed origins)

Source: [FineVQ Table 3](https://arxiv.org/abs/2412.19238). For small datasets the authors use 4:1 train/test splits with multi-round cross-validation. Many baseline cells repeat the KSVQE table above, so they are copied, not rerun.

| Model | LIVE-YT-Gaming | KoNViD-1k | YouTube-UGC | LIVE-VQC | LSVQ-test | LSVQ-1080p |
|---|---|---|---|---|---|---|
| SimpleVQA | 0.866 / 0.861 | 0.860 / 0.856 | 0.856 / 0.847 | - | 0.861 / 0.867 | 0.803 / 0.764 |
| FAST-VQA | 0.880 / 0.869 | 0.892 / 0.891 | 0.852 / 0.855 | 0.862 / 0.849 | 0.877 / 0.876 | 0.814 / 0.779 |
| DOVER | 0.868 / 0.852 | 0.910 / 0.908 | 0.851 / 0.841 | 0.875 / 0.844 | 0.878 / 0.877 | 0.812 / 0.778 |
| MinimalisticVQA | 0.888 / 0.857 | 0.890 / 0.889 | 0.891 / 0.890 | 0.854 / 0.842 | 0.879 / 0.881 | 0.820 / 0.781 |
| KSVQE | - | 0.921 / 0.922 | 0.912 / 0.900 | 0.883 / 0.861 | 0.888 / 0.886 | 0.823 / 0.790 |
| FineVQ | 0.926 / 0.912 | 0.910 / 0.915 | 0.914 / 0.910 | 0.895 / 0.895 | 0.900 / 0.900 | 0.857 / 0.828 |

### 3.3 FAST-VQA README fine-tune results (LSVQ pre-train, then per-dataset fine-tune)

Source: [FAST-VQA README](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA).

| Model | KoNViD-1k | CVD2014 | LIVE-Qualcomm | LIVE-VQC | YouTube-UGC |
|---|---|---|---|---|---|
| FAST-VQA-B | 0.892 / 0.891 | 0.903 / 0.891 | 0.851 / 0.819 | 0.862 / 0.849 | 0.852 / 0.855 |
| FasterVQA | 0.898 / 0.895 | 0.904 / 0.896 | 0.843 / 0.826 | 0.858 / 0.843 | 0.859 / 0.863 |

### 3.4 COVER on YouTube-UGC (AIS 2024 challenge setting)

Source: [COVER README](https://github.com/taco-group/COVER). The README marks some rows with `*` (kept below) but we did not find what the star means. Training details for the AIS 2024 setting are in the [challenge report](https://arxiv.org/abs/2404.16205).

| Model | YouTube-UGC |
|---|---|
| FAST-VQA* | 0.867 / 0.862 |
| DOVER* | 0.875 / 0.876 |
| Q-Align (AIS 2024 entry) | 0.912 / 0.908 |
| COVER | 0.912 / 0.914 (headline table); 0.917 / 0.914 (detail table) |

The README gives two PLCC values for COVER (0.9122 and 0.9165). We could not tell which is final.

## 4. Protocol V3: FineVD same-split retrain (FineVQ Table 1)

Source: [FineVQ Table 1](https://arxiv.org/abs/2412.19238). FineVD is split 4:1:1 (train / val / test). The FineVQ authors retrained every DNN baseline on the same split, with a separate model per dimension. FineVQ uses one model for all dimensions. Overall-quality column only; 4 decimals as in the source.

| Model | FineVD overall PLCC / SRCC |
|---|---|
| TLVQM | 0.6546 / 0.6537 |
| RAPIQUE | 0.6501 / 0.6379 |
| VIDEVAL | 0.7307 / 0.7310 |
| VSFA | 0.7929 / 0.7730 |
| GSTVQA | 0.7825 / 0.7834 |
| SimpleVQA | 0.8358 / 0.8311 |
| FAST-VQA | 0.8474 / 0.8348 |
| DOVER | 0.8393 / 0.8422 |
| FineVQ | 0.8891 / 0.8834 |

Cross-dataset with FineVD ([FineVQ Tables 4–5](https://arxiv.org/abs/2412.19238)):

| Direction | SimpleVQA | FAST-VQA | FineVQ |
|---|---|---|---|
| KoNViD-1k → FineVD | 0.6330 / 0.5947 | 0.5155 / 0.5155 | 0.7020 / 0.6475 |
| YouTube-UGC → FineVD | 0.6265 / 0.6050 | 0.6454 / 0.5990 | 0.6520 / 0.6261 |
| LIVE-VQC → FineVD | 0.4753 / 0.4631 | 0.5883 / 0.5730 | 0.6550 / 0.6125 |
| FineVD → KoNViD-1k | 0.3402 / 0.4818 | 0.6490 / 0.6793 | 0.7312 / 0.7171 |
| FineVD → YouTube-UGC | 0.7231 / 0.7421 | 0.5651 / 0.5607 | 0.7880 / 0.7797 |
| FineVD → LIVE-VQC | 0.6137 / 0.5922 | 0.6817 / 0.6767 | 0.7620 / 0.7417 |

## 5. AI-generated video results

### 5.1 AIGVQA-DB (AIGV-Assessor Table 3; each model trained on AIGVQA-DB)

Source: [AIGV-Assessor CVPR 2025 paper, Table 3](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_AIGV-Assessor_Benchmarking_and_Evaluating_the_Perceptual_Quality_of_Text-to-Video_Generation_CVPR_2025_paper.pdf). 4 decimals as in the source.

| Model | Static quality | Temporal smoothness | Text-video correspondence |
|---|---|---|---|
| MUSIQ | 0.8044 / 0.7880 | 0.6920 / 0.7199 | 0.4093 / 0.4125 |
| LIQE | 0.8691 / 0.8776 | 0.7720 / 0.7935 | 0.3639 / 0.3862 |
| FAST-VQA | 0.8644 / 0.8738 | 0.9134 / 0.9036 | 0.6704 / 0.6875 |
| DOVER | 0.8895 / 0.8907 | 0.9195 / 0.9063 | 0.6802 / 0.6783 |
| Q-Align | 0.8383 / 0.8516 | 0.7025 / 0.8116 | 0.5647 / 0.5542 |
| AIGV-Assessor | 0.9190 / 0.9162 | 0.9216 / 0.9232 | 0.7697 / 0.7500 |

The SimpleVQA row in that table has KRCC above SRCC (0.8489 vs 0.8355), which is not possible for the same ranking. We left it out.

### 5.2 T2VQA-DB (trained on T2VQA-DB)

| Model | T2VQA-DB | Source |
|---|---|---|
| SimpleVQA | 0.6338 / 0.6275 | [T2VQA Table 2](https://arxiv.org/abs/2403.11956) |
| FAST-VQA | 0.7295 / 0.7173 | [T2VQA Table 2](https://arxiv.org/abs/2403.11956) |
| BVQA | 0.7486 / 0.7390 | [T2VQA Table 2](https://arxiv.org/abs/2403.11956) |
| Q-Align | 0.7768 / 0.7601 | [T2VQA Table 2](https://arxiv.org/abs/2403.11956) |
| DOVER | 0.7693 / 0.7609 | [T2VQA Table 2](https://arxiv.org/abs/2403.11956) |
| T2VQA | 0.8066 / 0.7965 | [T2VQA Table 2](https://arxiv.org/abs/2403.11956) |
| AIGV-Assessor | 0.8222 / 0.8131 | [AIGV-Assessor Table 5](https://arxiv.org/abs/2411.17221) |
| RefVQA | 0.835 / 0.826 | [RefVQA Table 1](https://arxiv.org/abs/2604.17074) |

Zero-shot transfer from T2VQA-DB to Sora videos is weak for all models (T2VQA 0.3124 / 0.6485; AIGV-Assessor 0.3318 / 0.6612; [AIGV-Assessor Table 5](https://arxiv.org/abs/2411.17221)).

### 5.3 VideoScore family (own benchmarks; not MOS PLCC / SRCC tables)

| Model | Benchmark | Metric | Visual quality | Average | Source |
|---|---|---|---|---|---|
| VideoScore (gen) | VideoFeedback-test (in-domain) | SRCC x 100 | 86.2 | 77.1 (5 aspects) | [VideoScore Table 4](https://arxiv.org/abs/2406.15252) |
| VideoScore2 | VideoScore-Bench-v2 (in-domain) | PLCC x 100 | 60.13 | 60.37 (3 aspects) | [VideoScore2 Table 5](https://arxiv.org/abs/2509.22799) |
| VideoScore-v1.1 | VideoScore-Bench-v2 | PLCC x 100 | 49.00 | 42.30 | [VideoScore2 Table 5](https://arxiv.org/abs/2509.22799) |
| DOVER | VideoScore-Bench-v2 | PLCC x 100 | 50.24 | 38.69 | [VideoScore2 Table 5](https://arxiv.org/abs/2509.22799) |
| Q-Align | VideoScore-Bench-v2 | PLCC x 100 | 54.71 | 42.17 | [VideoScore2 Table 5](https://arxiv.org/abs/2509.22799) |

VideoScore2 also reports 44.35 exact-match accuracy on VideoScore-Bench-v2. Its SRCC is not in that table.

## 6. HDR video results

### 6.1 BrightVQ README (SRCC only)

Source: [BrightVQ README](https://github.com/shreshthsaini/BrightVQ). The README gives SRCC only, so the PLCC slot is `-`.

| Model | BrightVQ | LIVE-HDR | SFV+HDR |
|---|---|---|---|
| CONTRIQUE | - / 0.7081 | - / 0.8170 | - / 0.5901 |
| DOVER | - / 0.7745 | - / 0.6303 | - / 0.6001 |
| FAST-VQA | - / 0.8094 | - / 0.5182 | - / 0.7130 |
| HIDRO-VQA | - / 0.8526 | - / 0.8793 | - / 0.7003 |
| BrightRate | - / 0.8887 | - / 0.8907 | - / 0.7328 |

### 6.2 BrightRate-LM (author-reported, submitted study)

Source: [BrightRate-LM README](https://github.com/shreshthsaini/BrightRate-LM) and HF cards. BrightRate-LM is the mean of 5 content-separated 80/20 splits. BrightRate is the published median of 100 splits. These are different protocols.

| Model | Input | BrightVQ PLCC / SRCC | Protocol |
|---|---|---|---|
| BrightRate (published) | HDR-aware features | 0.8970 / 0.8887 | 100-split median |
| BrightRate-LM, Qwen2.5-VL-7B | Multi-exposure stack | 0.9107 / 0.9052 | mean of 5 content-separated splits |
| BrightRate-LM, Qwen2.5-VL-7B | Zero-shot on Beyond8Bits test (8,281 clips) | - / 0.8958 | no retraining |

Gemma 4 study, same SDR tone-mapped input, split 0 only (420 test videos), LoRA rank 16, 2 epochs:

| Adapter | Base | BrightVQ split-0 PLCC / SRCC | Source |
|---|---|---|---|
| brightrate-study-gemma4-e4b-sdr | Gemma 4 E4B (16-layer vision encoder) | 0.8805 / 0.8624 | [HF card](https://huggingface.co/shreshthsaini/brightrate-study-gemma4-e4b-sdr) |
| brightrate-study-gemma4-12b-sdr | Gemma 4 12B (encoder-free, single projection) | 0.7821 / 0.7763 | [HF card](https://huggingface.co/shreshthsaini/brightrate-study-gemma4-12b-sdr) |

Findings from the README: the smaller E4B beats the encoder-free 12B on the same input. Multi-exposure input helps every tested Qwen model; a 3B multi-exposure model (SRCC 0.8875) beats an 8B tone-mapped model (SRCC 0.8586). With matched PQ statistics, the 12B reaches SRCC 0.8383 on native PQ. So input representation and vision encoder matter more than parameter count here.

## 7. Notes for the Q-ReAlign comparison panel

- Natural video baselines with public weights: DOVER, FAST-VQA / FasterVQA, MinimalisticVQA, FineVQ-LSVQ, VQ-Insight natural video, VQAThinker, VQA²-Scorer, ONE-ALIGN. Re-score all of them on one locked LSVQ-test / LSVQ-1080p manifest plus KoNViD-1k, LIVE-VQC, and YouTube-UGC as cross-dataset tests.
- Use Section 2.1 as the reference style: one training set, one set of test lists, all baselines in one table.
- Report the OOD sets in Section 2.4 too. They show the largest gaps between models.
- AIGC video needs its own panel (AIGV-Assessor, VideoScore2, VQ-Insight AIGC). Hold out prompts and generators. Pairwise preference accuracy is not a substitute for MOS PLCC / SRCC.
- HDR results depend on decoding and tone mapping. Keep them out of SDR leaderboards.
- Licences: FAST-VQA, DOVER code, and Q-Align code are S-Lab non-commercial. FineVQ, VQAThinker, KSVQE, T2VQA, and AIGV-Assessor code have no licence file. BrightRate is CC BY-NC.
