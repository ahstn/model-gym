# IQA models (no-reference, technical quality)

This file lists no-reference image quality assessment (NR-IQA) models: classic specialists, LMM-based scorers, and AI-generated image (AIGC) quality models. It gives release links, licences, and published PLCC / SRCC grouped by training protocol. Unified IQA/IAA/VQA assessors (Q-Align ONE-ALIGN, Q-ReAlign, VITAL, UniPercept) are in [unified-and-backbones.md](unified-and-backbones.md).

Last verified: 2026-09-27

## How to read this file

- Metric pairs are always `PLCC / SRCC`. Many sources print SRCC first. We converted them.
- All numbers are author-reported unless a row says otherwise. We did not run any model.
- Each table uses one protocol. Do not rank rows from different tables against each other.
- "As quoted" means the number comes from a later paper, not the model's own paper.
- `Code` and `Weights` are separate licences. `Commercial use?` covers only these two licences. Training-data terms are separate and are often non-commercial. See [../datasets/iqa.md](../datasets/iqa.md).
- `Not stated` means we found no licence text. It does not mean "free to use".
- `Status: verified` means we checked the paper, repo, or model card on the date above. `from dossier` means we did not re-check it.
- Metric rules, split rules, and calibration are in [../evaluation/protocol.md](../evaluation/protocol.md).

## 1. Model catalog

### 1.1 Classic and compact specialists

| Model | Year / venue | Idea | Paper | Release | Code | Weights | Commercial use? | Status |
|---|---|---|---|---|---|---|---|---|
| HyperIQA | 2020 CVPR | Content-adaptive hyper network | [CVF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf) | [SSL92/hyperIQA](https://github.com/SSL92/hyperIQA); pyiqa `hyperiqa` (retrained by pyiqa) | MIT | Not stated (Google Drive) | Unclear | verified |
| DBCNN | 2019 arXiv / 2020 TCSVT | Two-stream bilinear CNN | [arXiv 1907.02665](https://arxiv.org/abs/1907.02665) | [zwx8981/DBCNN-PyTorch](https://github.com/zwx8981/DBCNN-PyTorch); pyiqa `dbcnn` (retrained) | MIT | Not stated | Unclear | verified |
| PaQ-2-PiQ | 2020 CVPR | Patch and picture quality; built FLIVE | [arXiv 1912.10088](https://arxiv.org/abs/1912.10088) | [baidut/PaQ-2-PiQ](https://github.com/baidut/PaQ-2-PiQ); pyiqa `paq2piq` | Not stated (repo has a CC BY-NC 4.0 file; scope unclear) | Not stated | Unclear | verified |
| MUSIQ | 2021 ICCV | Multi-scale transformer at native aspect ratio | [arXiv 2108.05997](https://arxiv.org/abs/2108.05997) | [google-research/musiq](https://github.com/google-research/google-research/tree/master/musiq) (KonIQ, SPAQ, PaQ2PiQ, AVA ckpts); pyiqa `musiq` | Apache-2.0 | Not stated (GCS bucket) | Unclear | verified |
| TReS | 2022 WACV | Transformer, relative ranking, self-consistency | [arXiv 2108.06858](https://arxiv.org/abs/2108.06858) | [isalirezag/TReS](https://github.com/isalirezag/TReS); pyiqa `tres-koniq`, `tres-flive` | CMU academic non-commercial licence | Not stated (Google Drive) | No (NC) | verified |
| MANIQA | 2022 CVPRW (NTIRE) | Multi-dimension attention on ViT | [arXiv 2204.08958](https://arxiv.org/abs/2204.08958) | [IIGROUP/MANIQA releases](https://github.com/IIGROUP/MANIQA/releases) (KonIQ, KADID, PIPAL ckpts); pyiqa `maniqa` | Apache-2.0 | Not stated (GitHub releases) | Unclear | verified |
| CLIP-IQA / CLIP-IQA+ | 2023 AAAI | CLIP antonym prompts; `+` learns prompts on KonIQ | [arXiv 2207.12396](https://arxiv.org/abs/2207.12396) | [IceClear/CLIP-IQA](https://github.com/IceClear/CLIP-IQA); pyiqa `clipiqa`, `clipiqa+` | S-Lab License 1.0 (non-commercial) | pyiqa copy: CC BY-NC-SA 4.0 | No (NC) | verified |
| CONTRIQUE | 2021 arXiv / 2022 TIP | Contrastive self-supervised features + linear regressor | [arXiv 2110.13266](https://arxiv.org/abs/2110.13266) | [pavancm/CONTRIQUE](https://github.com/pavancm/CONTRIQUE) | Not stated (no licence file) | Not stated (Box / Drive) | Unclear | verified |
| Re-IQA | 2023 CVPR | Separate content and quality encoders (MoCo) + linear regressor | [arXiv 2304.00451](https://arxiv.org/abs/2304.00451) | [avinabsaha/ReIQA](https://github.com/avinabsaha/ReIQA) | MIT | Not stated | Unclear | verified |
| LIQE | 2023 CVPR | CLIP multitask: quality, scene, distortion | [arXiv 2303.14968](https://arxiv.org/abs/2303.14968) | [zwx8981/LIQE](https://github.com/zwx8981/LIQE); pyiqa `liqe` (KonIQ), `liqe_mix` (multi-dataset) | MIT | Not stated (Drive / Baidu); pyiqa copy CC BY-NC-SA 4.0 | Unclear | verified |
| QPT | 2023 CVPR | Quality-aware contrastive pre-training (ResNet-50) | [arXiv 2303.00521](https://arxiv.org/abs/2303.00521) | No official code found | - | - | - | verified (no release found) |
| TOPIQ-NR | 2023 arXiv / 2024 TIP | Top-down attention from semantics to distortions | [arXiv 2308.03060](https://arxiv.org/abs/2308.03060) | pyiqa `topiq_nr` (KonIQ), `topiq_nr-flive`, `topiq_nr-spaq`; [weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights) | PolyForm Noncommercial 1.0.0 (pyiqa) | CC BY-NC-SA 4.0 | No (NC) | verified |
| ARNIQA | 2024 WACV | Self-supervised distortion manifold + linear regressor | [arXiv 2310.14918](https://arxiv.org/abs/2310.14918) | [miccunifi/ARNIQA](https://github.com/miccunifi/ARNIQA) (torch.hub) | Apache-2.0 | Not stated separately | Unclear | verified |
| QualiCLIP / QualiCLIP+ | 2024 arXiv | Opinion-unaware CLIP ranking; `+` adds MOS supervision | [arXiv 2403.11176](https://arxiv.org/abs/2403.11176) | [miccunifi/QualiCLIP](https://github.com/miccunifi/QualiCLIP); pyiqa `qualiclip`, `qualiclip+` | CC BY-NC 4.0 | Not stated (GitHub release); pyiqa copy CC BY-NC-SA 4.0 | No (NC) | verified (licence); pyiqa names from dossier |
| LoDa | 2024 CVPR | Local distortion adapters on a frozen ViT | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Boosting_Image_Quality_Assessment_through_Efficient_Transformer_Adaptation_with_Local_CVPR_2024_paper.html) | [NeosXu/LoDa](https://github.com/NeosXu/LoDa) | Apache-2.0 | Not released (README checklist open) | - | verified |
| SigLIP2 + AGM | 2026 WACV | SigLIP2 LoRA (rank 4) + MLP head with gated or sigmoid activation | [arXiv 2509.17374](https://arxiv.org/abs/2509.17374) | [drkkgy/NR_IQA_AGM](https://github.com/drkkgy/NR_IQA_AGM) (7 ckpt runs: CLIVE, KonIQ, cross pairs) | MIT | In repo (MIT); base [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch16-512) Apache-2.0 | Yes | verified |
| ReLIQS | 2026 CVPR | Native-resolution multiscale CLIP patches + learned saliency | [arXiv 2608.01730](https://arxiv.org/abs/2608.01730) | No code link in paper; no checkpoint found | - | - | - | verified |

NIMA (2018) appears in tables below as a baseline. It is mainly an aesthetics model. See [iaa.md](iaa.md).

### 1.2 LMM-based scorers

| Model | Year / venue | Base | Output | Main training data (paper) | Paper | Release | Code | Weights | Commercial use? | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Q-Align (IQA) | 2024 ICML | mPLUG-Owl2 (LLaMA-2 7B) | 5 level words, expected score | KonIQ for Table 3 rows; other variants on cards | [arXiv 2312.17090](https://arxiv.org/abs/2312.17090) | [Q-Future/Q-Align](https://github.com/Q-Future/Q-Align); HF [q-align-iqa](https://huggingface.co/q-future/q-align-iqa), [q-align-quality](https://huggingface.co/q-future/q-align-quality), [one-align](https://huggingface.co/q-future/one-align) | S-Lab License 1.0 (non-commercial) | MIT (HF cards) | Unclear (code NC, weights MIT) | verified |
| Compare2Score | 2024 NeurIPS | mPLUG-Owl2 | Pairwise comparison, then anchored score | KonIQ (DeQA row); six IQA sets (ReLIQS row) | [arXiv 2405.19298](https://arxiv.org/abs/2405.19298) | [Q-Future/Compare2Score](https://github.com/Q-Future/Compare2Score); [HF](https://huggingface.co/q-future/Compare2Score) | MIT | MIT | Yes | verified |
| DepictQA / DepictQA-Wild | 2023 arXiv (v1) / 2024 arXiv (Wild) | Vicuna-7B v1.5 based | Text description and comparison, not a MOS regressor | DQ-495K | [v1](https://arxiv.org/abs/2312.08962), [Wild](https://arxiv.org/abs/2405.18842) | [XPixelGroup/DepictQA](https://github.com/XPixelGroup/DepictQA); [HF DepictQA2-DQ495K](https://huggingface.co/zhiyuanyou/DepictQA2-DQ495K) | Apache-2.0 | Apache-2.0 | Yes | verified (licence); venues from dossier |
| DeQA-Score | 2025 CVPR | mPLUG-Owl2 (LLaMA-2 7B) | Soft score distribution, fidelity loss | Release: KonIQ + SPAQ + KADID (Mix3) | [arXiv 2501.11561](https://arxiv.org/abs/2501.11561) | [zhiyuanyou/DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score); HF [Mix3](https://huggingface.co/zhiyuanyou/DeQA-Score-Mix3), [LoRA-Mix3](https://huggingface.co/zhiyuanyou/DeQA-Score-LoRA-Mix3) | MIT | MIT (base model terms may also apply) | Yes (check base) | verified |
| Q-Insight | 2025 NeurIPS (spotlight) | Qwen2.5-VL-7B, GRPO | Reasoning text + score | About 7k KonIQ + 7k DQ-495K (dossier) | [arXiv 2503.22679](https://arxiv.org/abs/2503.22679) | [bytedance/Q-Insight](https://github.com/bytedance/Q-Insight); [HF](https://huggingface.co/ByteDance/Q-Insight) | Apache-2.0 | Apache-2.0 | Yes | verified |
| VisualQuality-R1 | 2025 NeurIPS (spotlight) | Qwen2.5-VL-7B, GRPO with Thurstone rank reward | Reasoning text + score | KADID (main table); KADID + SPAQ (dagger rows) | [arXiv 2505.14460](https://arxiv.org/abs/2505.14460) | [TianheWu/VisualQuality-R1](https://github.com/TianheWu/VisualQuality-R1); [HF 7B](https://huggingface.co/TianheWu/VisualQuality-R1-7B) | Apache-2.0 | MIT | Yes | verified |
| Q-Ponder | 2025 arXiv | Reasoning cold start, then GRPO | Reasoning + score | See paper | [arXiv 2506.05384](https://arxiv.org/abs/2506.05384) | [vivoCameraResearch/Q-Ponder](https://github.com/vivoCameraResearch/Q-Ponder) | Apache-2.0 | Not released ("coming soon") | - | verified |
| EvoQuality | 2026 ICLR | Qwen2.5-VL-7B, self-evolving voting + GRPO | Reasoning + score | No MOS labels (pseudo-labels from pairwise voting) | [arXiv 2509.25787](https://arxiv.org/abs/2509.25787) | [bytedance/EvoQuality](https://github.com/bytedance/EvoQuality); [HF](https://huggingface.co/ByteDance/EvoQuality) | Apache-2.0 | Apache-2.0 | Yes | verified |
| RALI | 2026 ICLR (oral) | CLIP-based scorer aligned to reasoning text | Score only, no text generation | KonIQ | [arXiv 2510.11369](https://arxiv.org/abs/2510.11369) | [xuanyuzhang21/RALI](https://github.com/xuanyuzhang21/RALI); [HF subfolder](https://huggingface.co/ByteDance/Q-Insight/tree/main/RALI) | Apache-2.0 | Apache-2.0 | Yes | verified |
| Zoom-IQA | 2026 ECCV | Qwen2.5-VL-7B, two-round crop-and-inspect reasoning | Reasoning + score | KonIQ (paper); release is an "enhanced" checkpoint | [arXiv 2601.02918](https://arxiv.org/abs/2601.02918) | [EthanLiang99/Zoom-IQA](https://github.com/EthanLiang99/Zoom-IQA); [HF 7B](https://huggingface.co/Ethanliang99/Zoom-IQA-7B) | Apache-2.0 | Apache-2.0 | Yes | verified |
| Q-Hawkeye | 2026 arXiv | Qwen2.5-VL-7B, GRPO with uncertainty and perception terms | Reasoning + score | KonIQ (README) | [arXiv 2601.22920](https://arxiv.org/abs/2601.22920) | [aba122/Q-Hawkeye](https://github.com/aba122/Q-Hawkeye) | Not stated (no licence file) | Not found | Unclear | verified |
| IQA-T1 | 2026 ECCV | Qwen3-VL-4B, tool-based visual evidence reasoning | Reasoning + score | KonIQ | [arXiv 2607.12375](https://arxiv.org/abs/2607.12375) | [zibuyu-02/IQA-T1](https://github.com/zibuyu-02/IQA-T1); [HF](https://huggingface.co/zibuyu-02/IQA-T1) | MIT | MIT | Yes | verified |
| MR-IQA | 2026 arXiv | Qwen3-VL-2B (per MR-IQA-2 card); margin view of regression and ranking | Score | KonIQ | [arXiv 2606.29760](https://arxiv.org/abs/2606.29760) | [RobinY99/MR-IQA](https://github.com/RobinY99/MR-IQA); [HF](https://huggingface.co/RobinY99/MR-IQA) | MIT | MIT | Yes | verified |
| MR-IQA-2 | 2026 arXiv | Qwen3.5-4B actor; frozen FLUX.2-klein-4B editor and E5 judge used in training | Reasoning + score | KonIQ-10k train split (paper appendix) | [arXiv 2608.18579](https://arxiv.org/abs/2608.18579) | [RobinY99/MR-IQA-2](https://github.com/RobinY99/MR-IQA-2); [HF](https://huggingface.co/RobinY99/MR-IQA-2) (`actor/`, `actor-2`, `actor-3`) | MIT | Apache-2.0 (card) | Yes (check bundled editor terms) | verified |

Notes:
- MR-IQA-2 uses the same backbone family as Q-ReAlign Lite (Qwen3.5-4B). Its card table is for `actor/` at step 1,455. The newer `actor-2` and `actor-3` folders have no reported PLCC / SRCC.
- Zoom-IQA's card says the released checkpoint was trained on 8 H200 GPUs and differs from the paper checkpoint. Paper numbers do not transfer to it automatically.
- RALI uses about 4% of Q-Insight's parameters. On one A100 at batch size 16 it uses 14.7% of Q-Insight's memory and 3.4% of its inference time ([RALI paper](https://arxiv.org/html/2510.11369v2)).
- Reasoning models sample text. Report parse failures and token counts. See [../evaluation/protocol.md](../evaluation/protocol.md).

### 1.3 AI-generated image (AIGC) quality models

| Model | Year / venue | Idea | Paper | Release | Code | Weights | Commercial use? | Status |
|---|---|---|---|---|---|---|---|---|
| MA-AGIQA | 2024 ACM MM | MANIQA features + mPLUG-Owl2 semantic features | [arXiv 2404.17762](https://arxiv.org/abs/2404.17762) | [wangpuyi/MA-AGIQA](https://github.com/wangpuyi/MA-AGIQA) (Drive ckpts) | Apache-2.0 | Not stated | Unclear | verified |
| IPCE | 2024 CVPRW (NTIRE) | CLIP image-prompt correspondence | [CVF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Peng_AIGC_Image_Quality_Assessment_via_Image-Prompt_Correspondence_CVPRW_2024_paper.pdf) | [pf0607/IPCE](https://github.com/pf0607/IPCE) | Not stated (no licence file) | No checkpoint link found | Unclear | verified |
| M3-AGIQA | 2025 arXiv | MLLM, multi-round, three aspects (quality, correspondence, authenticity) | [arXiv 2502.15167](https://arxiv.org/abs/2502.15167) | [strawhatboy/M3-AGIQA](https://github.com/strawhatboy/M3-AGIQA) (Drive ckpts) | MIT | Not stated | Unclear | verified |
| Q-Eval-Score | 2025 CVPR | Unified visual quality + text alignment for T2I and T2V | [arXiv 2503.02357](https://arxiv.org/abs/2503.02357) | [zzc-1998/Q-Eval](https://github.com/zzc-1998/Q-Eval); [HF](https://huggingface.co/AGI-Eval/Q-Eval-Score) | Not stated (no licence file) | CC BY-NC 4.0 | No (NC) | verified |

SigLIP2 + AGM also reports AGIQA-1K and AGIQA-3K (Table D1). Prompt-conditioned preference reward models (HPSv3, HPSv3++) are in [iaa.md](iaa.md).

### 1.4 Other items seen, not catalogued in full

| Item | Why not in main tables | Link | Status |
|---|---|---|---|
| Life-IQA (2026 CVPR) | GCN layer interaction + MoE decoder; no code link found | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Life-IQA_Boosting_Blind_Image_Quality_Assessment_through_GCN-enhanced_Layer_Interaction_CVPR_2026_paper.html) | unverified (numbers not checked) |
| IQARAG (2026 arXiv) | Training-free retrieval prompts for general LMMs | [arXiv 2601.08311](https://arxiv.org/html/2601.08311) | unverified |
| SA-IQA | Interior-scene assessor (Ovis2.5-9B); narrow domain | [HF](https://huggingface.co/AliHome3D/SA-IQA-model) | from dossier |
| PR-IQA | Needs a partial reference view; not no-reference | [HF](https://huggingface.co/kakaomacao/PR-IQA) | from dossier |
| VITAL (2026 CVPR) | Unified image + video LMM | [unified-and-backbones.md](unified-and-backbones.md) | - |

## 2. Benchmark tables by protocol

Abbreviations: LIVE-W = LIVE Challenge = CLIVE. AGIQA = AGIQA-3K unless named. `-` = not reported.

### Protocol A: trained on KonIQ only, tested across datasets

KonIQ is in-domain. All other columns are cross-dataset.

**Table A1. DeQA-Score paper, Table 3** ([source](https://arxiv.org/html/2501.11561#S5.T3)). One paper, one table. Baselines are the DeQA authors' reported rows.

| Model | KonIQ | SPAQ | KADID | PIPAL | LIVE-W | AGIQA-3K | CSIQ | FLIVE |
|---|---|---|---|---|---|---|---|---|
| NIQE (no training) | 0.533 / 0.530 | 0.679 / 0.664 | 0.468 / 0.405 | 0.195 / 0.161 | 0.493 / 0.449 | 0.560 / 0.533 | 0.718 / 0.628 | 0.147 / 0.100 |
| NIMA | 0.896 / 0.859 | 0.838 / 0.856 | 0.532 / 0.535 | 0.390 / 0.399 | 0.814 / 0.771 | 0.715 / 0.654 | 0.695 / 0.649 | 0.561 / 0.467 |
| HyperIQA | 0.917 / 0.906 | 0.791 / 0.788 | 0.506 / 0.468 | 0.410 / 0.403 | 0.772 / 0.749 | 0.702 / 0.640 | 0.752 / 0.717 | 0.485 / 0.383 |
| DBCNN | 0.884 / 0.875 | 0.812 / 0.806 | 0.497 / 0.484 | 0.384 / 0.381 | 0.773 / 0.755 | 0.730 / 0.641 | 0.586 / 0.572 | 0.485 / 0.385 |
| MUSIQ | 0.924 / 0.929 | 0.868 / 0.863 | 0.575 / 0.556 | 0.431 / 0.431 | 0.789 / 0.830 | 0.722 / 0.630 | 0.771 / 0.710 | 0.565 / 0.467 |
| CLIP-IQA+ | 0.909 / 0.895 | 0.866 / 0.864 | 0.653 / 0.654 | 0.427 / 0.419 | 0.832 / 0.805 | 0.736 / 0.685 | 0.772 / 0.719 | 0.427 / 0.316 |
| MANIQA | 0.849 / 0.834 | 0.768 / 0.758 | 0.499 / 0.465 | 0.457 / 0.452 | 0.849 / 0.832 | 0.723 / 0.636 | 0.623 / 0.627 | 0.512 / 0.401 |
| Compare2Score | 0.923 / 0.910 | 0.867 / 0.860 | 0.500 / 0.453 | 0.354 / 0.342 | 0.786 / 0.772 | 0.777 / 0.671 | 0.735 / 0.705 | 0.474 / 0.413 |
| Q-Align | 0.941 / 0.940 | 0.886 / 0.887 | 0.674 / 0.684 | 0.403 / 0.419 | 0.853 / 0.860 | 0.772 / 0.735 | 0.785 / 0.737 | 0.554 / 0.483 |
| DeQA-Score | 0.953 / 0.941 | 0.895 / 0.896 | 0.694 / 0.687 | 0.472 / 0.478 | 0.892 / 0.879 | 0.809 / 0.729 | 0.787 / 0.744 | 0.589 / 0.501 |

**Table A2. Newer models, same KonIQ-only setup, each from its own source.** Rows come from different papers. Splits and test subsets can differ. Use this table to shortlist, not to rank.

| Model | KonIQ | SPAQ | KADID | PIPAL | LIVE-W | AGIQA-3K | CSIQ | Source |
|---|---|---|---|---|---|---|---|---|
| Q-Insight (released, + DQ-495K task) | 0.933 / 0.916 | 0.907 / 0.905 | 0.742 / 0.736 | 0.486 / 0.474 | 0.893 / 0.865 | 0.811 / 0.764 | 0.870 / 0.824 | [Q-Insight T1](https://arxiv.org/html/2503.22679v2) |
| Q-Insight (score-only ablation) | 0.918 / 0.895 | 0.903 / 0.899 | 0.702 / 0.702 | 0.458 / 0.435 | 0.870 / 0.839 | 0.816 / 0.766 | 0.685 / 0.640 | [Q-Insight T3](https://arxiv.org/html/2503.22679v2) |
| RALI | 0.939 / 0.922 | 0.897 / 0.897 | 0.723 / 0.725 | 0.527 / 0.528 | 0.896 / 0.876 | 0.779 / 0.715 | 0.828 / 0.788 | [RALI T2](https://arxiv.org/html/2510.11369v2) |
| VisualQuality-R1 (as quoted by RALI) | 0.923 / 0.908 | 0.891 / 0.892 | 0.712 / 0.711 | 0.441 / 0.438 | 0.874 / 0.849 | 0.822 / 0.767 | 0.712 / 0.662 | [RALI T2](https://arxiv.org/html/2510.11369v2) |
| VisualQuality-R1 (as quoted by Zoom-IQA and IQA-T1) | 0.910 / 0.896 | 0.889 / 0.892 | 0.703 / 0.712 | 0.451 / 0.441 | 0.856 / 0.827 | 0.817 / 0.760 | 0.768 / 0.707 | [Zoom-IQA T1](https://arxiv.org/html/2601.02918v2) |
| Zoom-IQA | 0.938 / 0.922 | 0.902 / 0.900 | 0.701 / 0.700 | 0.468 / 0.465 | 0.887 / 0.870 | 0.816 / 0.765 | 0.797 / 0.754 | [Zoom-IQA T1](https://arxiv.org/html/2601.02918v2) |
| IQA-T1 (Qwen3-VL-4B) | 0.942 / 0.925 | 0.905 / 0.908 | 0.686 / 0.708 | 0.480 / 0.490 | 0.888 / 0.860 | 0.816 / 0.754 | 0.850 / 0.840 | [IQA-T1 T1](https://arxiv.org/html/2607.12375v1) |
| MR-IQA (Qwen3-VL-2B) | 0.949 / 0.931 | 0.892 / 0.897 | 0.672 / 0.683 | - | 0.899 / 0.883 | 0.804 / 0.732 | 0.767 / 0.732 | [MR-IQA-2 card](https://huggingface.co/RobinY99/MR-IQA-2) |
| MR-IQA-2 (`actor/`, step 1,455) | 0.937 / 0.917 | 0.900 / 0.899 | 0.667 / 0.669 | - | 0.893 / 0.863 | 0.809 / 0.739 | 0.824 / 0.785 | [MR-IQA-2 card](https://huggingface.co/RobinY99/MR-IQA-2) |
| ReLIQS (median of 10 random splits) | 0.958 / 0.949 | 0.891 / 0.894 | 0.701 / 0.707 | - | 0.892 / 0.865 | 0.768 / 0.705 | 0.842 / 0.818 | [ReLIQS T1](https://arxiv.org/html/2608.01730v1) |

- ReLIQS also reports FLIVE 0.654 / 0.549 in the same table.
- MR-IQA-2 card averages: 0.838 / 0.812 (MR-IQA-2) vs 0.831 / 0.810 (MR-IQA) over six sets. KonIQ goes down; CSIQ goes up.
- Label-free control. EvoQuality round 2 uses no MOS labels, so it is not in the table. Its own Table 3 reports KonIQ 0.835 / 0.791, SPAQ 0.903 / 0.900, KADID 0.803 / 0.807, PIPAL 0.649 / 0.583, LIVE-W 0.847 / 0.813, AGIQA 0.831 / 0.771, CSIQ 0.865 / 0.839, TID2013 0.674 / 0.611 ([EvoQuality T3](https://arxiv.org/html/2509.25787)). Its baseline rows differ from DeQA Table 3 (see Section 5).

### Protocol B: multi-dataset training

**Table B1. KonIQ + SPAQ + KADID ("Mix3").** SPAQ and KADID are in-domain here. Do not call them zero-shot.

| Model | KonIQ | SPAQ | KADID | PIPAL | LIVE-W | AGIQA-3K | TID2013 | CSIQ | Source |
|---|---|---|---|---|---|---|---|---|---|
| Q-Align | 0.945 / 0.938 | 0.933 / 0.931 | 0.935 / 0.934 | 0.409 / 0.420 | 0.887 / 0.883 | 0.788 / 0.733 | 0.829 / 0.808 | 0.876 / 0.845 | [DeQA T4](https://arxiv.org/html/2501.11561#S5.T4) |
| DeQA-Score (paper) | 0.957 / 0.944 | 0.938 / 0.934 | 0.955 / 0.953 | 0.495 / 0.496 | 0.900 / 0.887 | 0.808 / 0.745 | 0.852 / 0.820 | 0.900 / 0.857 | [DeQA T4](https://arxiv.org/html/2501.11561#S5.T4) |
| DeQA-Score-Mix3 (HF card) | 0.956 / 0.943 | 0.938 / 0.934 | 0.955 / 0.953 | 0.495 / 0.496 | 0.900 / 0.887 | 0.808 / 0.745 | 0.852 / 0.820 | 0.900 / 0.857 | [card](https://huggingface.co/zhiyuanyou/DeQA-Score-Mix3) |
| ReLIQS (median of 10 splits) | 0.954 / 0.946 | 0.936 / 0.933 | 0.955 / 0.953 | - | 0.890 / 0.869 | 0.792 / 0.729 | - | 0.894 / 0.857 | [ReLIQS T2](https://arxiv.org/html/2608.01730v1) |

The effect of adding PIPAL to this mix is in [../training/data-mixture-evidence.md](../training/data-mixture-evidence.md).

**Table B2. KonIQ + SPAQ + KADID + PIPAL, RALI paper Table 3** ([source](https://arxiv.org/html/2510.11369v2)). First four columns are in-domain. RACT is RALI's multi-dataset variant.

| Model | KonIQ | SPAQ | KADID | PIPAL | LIVE-W | AGIQA-3K | CSIQ | TID2013 |
|---|---|---|---|---|---|---|---|---|
| Q-Align | 0.926 / 0.932 | 0.917 / 0.920 | 0.950 / 0.954 | 0.702 / 0.671 | 0.853 / 0.845 | 0.765 / 0.722 | 0.838 / 0.789 | 0.811 / 0.795 |
| DeQA-Score | 0.958 / 0.946 | 0.932 / 0.929 | 0.963 / 0.961 | 0.724 / 0.690 | 0.877 / 0.857 | 0.770 / 0.735 | 0.863 / 0.807 | 0.828 / 0.796 |
| VisualQuality-R1 | 0.899 / 0.881 | 0.918 / 0.914 | 0.918 / 0.920 | 0.603 / 0.588 | 0.852 / 0.834 | 0.812 / 0.753 | 0.859 / 0.772 | 0.799 / 0.764 |
| Q-Insight | 0.899 / 0.871 | 0.913 / 0.907 | 0.757 / 0.765 | 0.579 / 0.559 | 0.867 / 0.830 | 0.805 / 0.757 | 0.768 / 0.720 | 0.743 / 0.651 |
| RACT | 0.928 / 0.907 | 0.922 / 0.918 | 0.919 / 0.916 | 0.642 / 0.626 | 0.881 / 0.846 | 0.813 / 0.763 | 0.892 / 0.838 | 0.844 / 0.817 |

**Table B3. Six-dataset joint training (KonIQ, CLIVE, BID, KADID, CSIQ, LIVE), ReLIQS paper Table 2** ([source](https://arxiv.org/html/2608.01730v1)). All columns are in-domain. This is the LIQE / Compare2Score style protocol. Baseline rows are as quoted by ReLIQS.

| Model | KonIQ | CLIVE | BID | KADID | CSIQ | LIVE |
|---|---|---|---|---|---|---|
| UNIQUE | 0.900 / 0.895 | 0.884 / 0.854 | 0.875 / 0.852 | 0.885 / 0.884 | 0.921 / 0.902 | 0.952 / 0.961 |
| LIQE | 0.908 / 0.919 | 0.910 / 0.904 | 0.900 / 0.875 | 0.931 / 0.930 | 0.939 / 0.936 | 0.951 / 0.970 |
| Q-Align | 0.934 / 0.935 | 0.921 / 0.931 | 0.920 / 0.904 | 0.927 / 0.869 | 0.936 / 0.915 | 0.919 / 0.913 |
| Compare2Score | 0.939 / 0.931 | 0.928 / 0.914 | 0.939 / 0.919 | 0.939 / 0.952 | 0.943 / 0.950 | 0.969 / 0.972 |
| ReLIQS | 0.955 / 0.944 | 0.938 / 0.921 | 0.937 / 0.914 | 0.945 / 0.948 | 0.953 / 0.944 | 0.980 / 0.978 |

### Protocol C: trained on KADID only (synthetic), tested on other sets

**Table C1. VisualQuality-R1 paper, Table 2** ([source](https://arxiv.org/html/2505.14460v2)). Source prints SRCC and PLCC in separate blocks; we paired them. Dagger rows train on KADID + SPAQ, so SPAQ is in-domain for them. Avg is over 8 test sets (4 shown plus deblurring, super-resolution, dehazing).

| Model | BID | CLIVE | KonIQ | SPAQ | AGIQA-3K | Avg (8 sets) |
|---|---|---|---|---|---|---|
| LIQE | 0.680 / 0.677 | 0.726 / 0.719 | 0.652 / 0.684 | 0.814 / 0.815 | 0.653 / 0.653 | 0.709 / 0.717 |
| Q-Align | 0.651 / 0.576 | 0.643 / 0.554 | 0.612 / 0.573 | 0.779 / 0.767 | 0.705 / 0.682 | 0.679 / 0.632 |
| DeQA-Score | 0.743 / 0.702 | 0.795 / 0.743 | 0.703 / 0.677 | 0.858 / 0.852 | 0.790 / 0.738 | 0.772 / 0.731 |
| Q-Insight | 0.796 / 0.784 | 0.795 / 0.761 | 0.829 / 0.806 | 0.872 / 0.872 | 0.810 / 0.749 | 0.803 / 0.766 |
| VisualQuality-R1 | 0.806 / 0.790 | 0.794 / 0.750 | 0.840 / 0.830 | 0.878 / 0.875 | 0.843 / 0.775 | 0.814 / 0.777 |
| Q-Insight (dagger) | 0.818 / 0.806 | 0.837 / 0.804 | 0.809 / 0.812 | 0.912 / 0.907 | 0.705 / 0.657 | 0.793 / 0.759 |
| VisualQuality-R1 (dagger) | 0.820 / 0.811 | 0.844 / 0.811 | 0.870 / 0.855 | 0.917 / 0.913 | 0.820 / 0.754 | 0.831 / 0.791 |

### Protocol D: dataset-specific (train and test on splits of the same dataset)

These are in-domain numbers. They are much higher than cross-dataset numbers. They do not show generalization.

**Table D1. SigLIP2 + AGM paper, Table 5** ([source](https://arxiv.org/html/2509.17374v2)). Three-seed means. Baseline rows are as quoted by the AGM authors. Source order is SRCC, PLCC; we converted.

| Model | CLIVE | KonIQ | FLIVE | SPAQ | AGIQA-3K | AGIQA-1K | KADID |
|---|---|---|---|---|---|---|---|
| DBCNN | 0.869 / 0.851 | 0.884 / 0.875 | 0.551 / 0.545 | 0.915 / 0.911 | - | - | 0.856 / 0.851 |
| HyperIQA | 0.882 / 0.859 | 0.917 / 0.906 | 0.602 / 0.544 | 0.915 / 0.911 | - | - | 0.845 / 0.852 |
| TReS | 0.877 / 0.846 | 0.928 / 0.915 | 0.625 / 0.554 | - | - | - | 0.859 / 0.859 |
| MUSIQ | 0.746 / 0.702 | 0.928 / 0.916 | 0.661 / 0.566 | 0.921 / 0.918 | - | - | 0.872 / 0.875 |
| Re-IQA | 0.854 / 0.840 | 0.923 / 0.914 | 0.733 / 0.645 | 0.925 / 0.918 | 0.845 / 0.785 | 0.670 / 0.614 | 0.885 / 0.872 |
| LoDa | 0.899 / 0.876 | 0.944 / 0.932 | 0.679 / 0.578 | 0.928 / 0.925 | - | - | 0.936 / 0.931 |
| LGDM (diffusion-based; not catalogued) | 0.940 / 0.908 | 0.972 / 0.967 | 0.812 / 0.705 | 0.948 / 0.947 | 0.929 / 0.863 | 0.903 / 0.891 | 0.961 / 0.958 |
| SigLIP2 + AGM, B_Sig | 0.930 / 0.909 | 0.947 / 0.938 | 0.608 / 0.521 | 0.926 / 0.921 | 0.923 / 0.878 | 0.897 / 0.872 | 0.943 / 0.939 |
| SigLIP2 + AGM, B_Gated | 0.912 / 0.887 | 0.962 / 0.953 | 0.647 / 0.556 | 0.932 / 0.928 | 0.919 / 0.867 | 0.892 / 0.873 | 0.973 / 0.970 |

The AGM paper also has a cross-dataset table (SRCC only): B_Gated trained on KonIQ reaches CLIVE SRCC 0.899; B_Sig trained on CLIVE reaches KonIQ SRCC 0.808 (Table 3).

**Table D2. TOPIQ paper, Tables V and VIII** ([source](https://arxiv.org/abs/2308.03060)). Table V: mean of 10 random 8:2 splits. Table VIII: official KonIQ split.

| Model | CLIVE (T-V) | KonIQ (T-V) | FLIVE (T-V) | KonIQ official split (T-VIII) |
|---|---|---|---|---|
| DBCNN | 0.869 / 0.869 | 0.884 / 0.875 | 0.551 / 0.545 | - |
| HyperIQA | 0.882 / 0.859 | 0.917 / 0.906 | 0.602 / 0.544 | - |
| TReS | 0.877 / 0.846 | 0.928 / 0.915 | 0.625 / 0.554 | - |
| MUSIQ | - | 0.928 / 0.916 | 0.739 / 0.646 | 0.937 / 0.924 |
| TOPIQ-NR (ResNet50) | 0.884 / 0.870 | 0.939 / 0.926 | 0.722 / 0.633 | 0.941 / 0.928 |

TOPIQ Table VI (trained on KonIQ, SRCC and PLCC): TOPIQ-NR CLIVE 0.8389 / 0.8206, FLIVE 0.6272 / 0.5796, SPAQ 0.8791 / 0.8758; MUSIQ 0.8295 / 0.7889, 0.5128 / 0.4978, 0.8626 / 0.8676.

**Table D3. ARNIQA paper, Table 1** ([source](https://arxiv.org/html/2310.14918)). Self-supervised encoder + linear regressor per dataset. Source order is SRCC, PLCC; we converted.

| Model | LIVE | CSIQ | TID2013 | KADID | FLIVE | SPAQ |
|---|---|---|---|---|---|---|
| CONTRIQUE | 0.961 / 0.960 | 0.955 / 0.942 | 0.857 / 0.843 | 0.937 / 0.934 | 0.641 / 0.580 | 0.919 / 0.914 |
| Re-IQA | 0.971 / 0.970 | 0.960 / 0.947 | 0.861 / 0.804 | 0.885 / 0.872 | 0.733 / 0.645 | 0.925 / 0.918 |
| ARNIQA | 0.970 / 0.966 | 0.973 / 0.962 | 0.901 / 0.880 | 0.912 / 0.908 | 0.671 / 0.595 | 0.910 / 0.905 |

### Protocol E: IQA-PyTorch toolbox benchmark (default checkpoints)

**Table E1. pyiqa `tests/NR_benchmark_results.csv`** ([source](https://github.com/chaofengc/IQA-PyTorch/blob/main/tests/NR_benchmark_results.csv), read 2026-09-27). Default NR weights are trained on KonIQ unless the name says otherwise. PLCC has no nonlinear correction. Full image, single input. Source keeps 4 decimals. A blank cell in the CSV (usually a training set) is shown as `-`.

| pyiqa metric | LIVE-C | KonIQ (official test) | TID2013 | FLIVE | SPAQ |
|---|---|---|---|---|---|
| `topiq_nr` | 0.8261 / 0.8106 | 0.9436 / 0.9299 | 0.5625 / 0.4452 | 0.6289 / 0.5819 | 0.8744 / 0.8704 |
| `musiq-koniq` | 0.8295 / 0.7889 | 0.8958 / 0.8654 | 0.6814 / 0.5750 | 0.5128 / 0.4978 | 0.8626 / 0.8676 |
| `maniqa` | 0.8262 / 0.8399 | 0.9133 / 0.8934 | 0.4570 / 0.4515 | 0.4416 / 0.4489 | 0.8140 / 0.8166 |
| `clipiqa+` | 0.8312 / 0.8045 | 0.8454 / 0.8026 | 0.7010 / 0.6318 | 0.5973 / 0.5746 | - |
| `clipiqa+_rn50_512` | 0.8181 / 0.8180 | 0.9012 / 0.8847 | 0.6577 / 0.5949 | 0.5766 / 0.5654 | - |
| `dbcnn` (retrained by pyiqa) | 0.7740 / 0.7562 | 0.9197 / 0.9034 | 0.5141 / 0.3855 | 0.6151 / 0.5764 | 0.8549 / 0.8473 |
| `hyperiqa` (retrained by pyiqa) | 0.7779 / 0.7546 | 0.9233 / 0.9040 | 0.5627 / 0.4537 | 0.3656 / 0.3719 | - |
| `tres-koniq` | 0.8118 / 0.7771 | - | - | 0.5130 / 0.4919 | 0.8624 / 0.8619 |
| `paq2piq` | 0.7542 / 0.7188 | 0.7062 / 0.6430 | 0.5776 / 0.4011 | 0.6906 / 0.6457 | 0.7750 / 0.8289 |
| `clipiqa` (zero-shot) | 0.6883 / 0.6955 | 0.7211 / 0.6572 | 0.6471 / 0.5786 | 0.5028 / 0.4674 | - |
| `qalign` | 0.8942 / 0.8814 | - | 0.8529 / 0.8313 | - | - |

The CSV has no rows for `liqe`, `qualiclip`, or `arniqa`. The `qalign` checkpoint's training data is not stated in the CSV.

### Protocol F: AIGC images, dataset-specific

**Table F1. M3-AGIQA paper, Table 2** ([source](https://arxiv.org/html/2502.15167)). Train and test on each dataset. Source order is SRCC, PLCC and keeps 4 decimals. "Quality" and "Correspondence" are separate MOS targets. Only quality should be compared with technical-quality MOS.

| Model | AGIQA-3K quality | AGIQA-3K correspondence | AIGIQA-20K quality |
|---|---|---|---|
| HyperIQA | 0.8975 / 0.8526 | 0.8471 / 0.7437 | 0.8584 / 0.8171 |
| MA-AGIQA | 0.9136 / 0.8709 | 0.8785 / 0.7721 | 0.9006 / 0.8619 |
| IPCE | 0.9246 / 0.8841 | 0.8725 / 0.7697 | 0.9274 / 0.9076 |
| M3-AGIQA | 0.9317 / 0.9045 | 0.9142 / 0.8523 | 0.9292 / 0.8988 |

MA-AGIQA's own paper reports AGIQA-3K 0.9273 / 0.8939 and AIGCQA-20K 0.9050 / 0.8644 ([MA-AGIQA T1](https://arxiv.org/html/2404.17762)). Its 20K number differs from the M3-AGIQA quote.

## 3. IQA-PyTorch (pyiqa): the practical toolbox

| Item | Value | Source |
|---|---|---|
| Install | `pip install pyiqa` (latest on PyPI: 0.1.16) | [README](https://github.com/chaofengc/IQA-PyTorch) |
| Code licence | PolyForm Noncommercial 1.0.0; NTU S-Lab License for some components | [LICENSE](https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE) |
| Weights licence | CC BY-NC-SA 4.0 | [HF weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights) |
| Commercial use? | No (NC), for both code and hosted weights | - |
| Weight origin | Official author weights where available. `cnniqa`, `dbcnn`, `hyperiqa`, `metaiqa`, `wadiqam_nr` are retrained by pyiqa and may differ from papers | README |
| Benchmark rules | Raw PLCC, full image, single input, KonIQ official split | README, Table E1 |
| Dataset mirror | [IQA-Toolbox-Datasets](https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets); README says academic use only and original dataset terms apply | README |

Use pyiqa for research baselines only. Pin the package version and weight file. For a commercial path, re-implement from permissive author code (for example ARNIQA, MANIQA, LIQE) and check each weight file.

## 4. Shortlist for the first comparison

This follows the prior research. Run these next to frozen Q-ReAlign checkpoints on one frozen test manifest.

| Priority | Model | Why | Watch out for |
|---|---|---|---|
| 1 | DeQA-Score-Mix3 | Closest control for soft score distributions vs Q-ReAlign's five-word target; MIT weights | Mix3 has seen KonIQ, SPAQ, KADID. Report those as in-domain. Paper KonIQ-only rows (Table A1) are a different checkpoint |
| 1 | TOPIQ-NR (`topiq_nr`) | Cheap ResNet-50 specialist; tests if a large VLM is worth its cost | Code and weights are non-commercial (pyiqa) |
| 1 | SigLIP2 + AGM (B_Gated and B_Sig) | Compact, recent, MIT; a strong cost baseline | Released runs cover CLIVE and KonIQ only. Retrain on our mix for a fair test. Table D1 is in-domain |
| 2 | Q-Insight and RALI | Reasoning teacher and its cheap CLIP distillation | Extra DQ-495K data; count tokens and latency |
| 2 | MR-IQA-2, IQA-T1, Zoom-IQA | 2026 reasoning scorers; MR-IQA-2 shares Q-ReAlign Lite's Qwen3.5-4B base | Released checkpoints can differ from paper rows; parse failures |
| 3 | LIQE, MUSIQ, CLIP-IQA+, MANIQA, VisualQuality-R1, EvoQuality | Wider coverage | Match checkpoint to training data |

## 5. Known issues in published tables

- Q-Align CSIQ is 0.785 / 0.737 in DeQA Table 3. Q-Insight, RALI, Zoom-IQA, and IQA-T1 copy it as 0.671 / 0.737.
- Zoom-IQA and IQA-T1 quote Q-Insight as KonIQ 0.918 / 0.895. That equals Q-Insight's score-only ablation, not the released model (0.933 / 0.916). Only SPAQ SRCC differs (0.903 vs 0.899).
- RALI Table 2 rows for MANIQA and CLIP-IQA+ are shifted. For example, the MANIQA row shows CLIP-IQA+ PLCC values. Use DeQA Table 3 for those baselines.
- VisualQuality-R1 has two different KonIQ-trained quotes (RALI vs Zoom-IQA). Its own paper trains on KADID (Table C1).
- EvoQuality Table 3 re-reports baselines with different numbers. Example: DeQA-Score CSIQ 0.868 / 0.847 vs 0.787 / 0.744 in DeQA Table 3.
- ReLIQS Table 1 repeats the AGIQA-3K values in the LIVE column for all baseline rows. It also lists BRISQUE KonIQ as 0.702 / 0.715, not DeQA's 0.225 / 0.226.
- pyiqa `musiq-koniq` on the official KonIQ test split is 0.8958 / 0.8654. TOPIQ Table VIII quotes MUSIQ on the same split as 0.937 / 0.924. The toolbox uses a single full-image input; check the input policy before you compare.
- DeQA-Score-Mix3 card KonIQ is 0.956 / 0.943; the paper Mix3 row is 0.957 / 0.944.
