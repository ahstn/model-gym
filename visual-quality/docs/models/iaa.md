# IAA models (image aesthetics)

This file lists image aesthetics assessment (IAA) models: classic CNN and ViT scorers, CLIP-based predictors, and LMM-based aesthetic scorers. It gives release links, licences, and published PLCC / SRCC grouped by training protocol. Unified IQA/IAA/VQA assessors (Q-Align ONE-ALIGN, Q-ReAlign, UniPercept) are in [unified-and-backbones.md](unified-and-backbones.md).

Last verified: 2026-09-27

## How to read this file

- Metric pairs are always `PLCC / SRCC`. Many sources print SRCC first. We converted them.
- All numbers are author-reported. We did not run any model.
- Each table uses one protocol. Do not rank rows from different tables against each other.
- "As quoted" means the number comes from a later paper, not the model's own paper.
- `Code` and `Weights` are separate licences. `Commercial use?` covers only these two licences. AVA, TAD66K, PARA and other image terms are separate. See [../datasets/iaa.md](../datasets/iaa.md).
- `Not stated` means we found no licence text. It does not mean "free to use".
- `Status: verified` means we checked the paper, repo, or model card on the date above. `from dossier` means we did not re-check it.
- Metric rules, split rules, and calibration are in [../evaluation/protocol.md](../evaluation/protocol.md).

## 1. Model catalog

### 1.1 Classic and CLIP-based scorers

| Model | Year / venue | Idea | Paper | Release | Code | Weights | Commercial use? | Status |
|---|---|---|---|---|---|---|---|---|
| NIMA | 2018 TIP | CNN predicts the 10-bin score distribution (EMD loss) | [arXiv 1709.05424](https://arxiv.org/abs/1709.05424) | No official Google code. Ports: [idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment) (MobileNet, weights in repo); pyiqa `nima` | idealo: Apache-2.0; pyiqa: PolyForm Noncommercial 1.0.0 | idealo: in repo (Apache-2.0); pyiqa copy: [CC BY-NC-SA 4.0](https://huggingface.co/chaofengc/IQA-PyTorch-Weights) | idealo: Yes; pyiqa: No (NC) | verified |
| MUSIQ-AVA | 2021 ICCV | Multi-scale transformer at native aspect ratio | [arXiv 2108.05997](https://arxiv.org/abs/2108.05997) | [google-research/musiq](https://github.com/google-research/google-research/tree/master/musiq) (`ava_ckpt.npz`); pyiqa `musiq-ava` port | Apache-2.0 | Not stated (GCS checkpoint); pyiqa copy CC BY-NC-SA 4.0 | Unclear | verified |
| TANet | 2022 IJCAI | Theme-aware network; also built TAD66K | [IJCAI 2022 repo page](https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment) | Official repo (README links data, not weights); third-party port [ddecosmo/TANet-AVA](https://huggingface.co/ddecosmo/TANet-AVA) | Apache-2.0 | Port card: Apache-2.0 (third-party, 2026) | Unclear | verified |
| VILA / VILA-R | 2023 CVPR | CoCa image-text pre-training on AVA comments; rank-based adapter | [arXiv 2303.14302](https://arxiv.org/abs/2303.14302) | [google-research/vila](https://github.com/google-research/google-research/tree/master/vila); VILA-R on TF Hub | Apache-2.0 | Not stated (TF Hub) | Unclear | verified |
| LAION aesthetic predictor v1 | 2022 | Linear head on CLIP embeddings | No paper | [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor) | MIT | In repo (MIT) | Yes | verified |
| LAION improved aesthetic predictor (v2) | 2022 | MLP on CLIP ViT-L/14; trained on SAC + LOGOS + AVA | No paper | [christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) (`sac+logos+ava1-l14-linearMSE.pth`) | Apache-2.0 | In repo (Apache-2.0) | Yes (code/weights only) | verified |
| Aesthetic Predictor V2.5 | 2024 | SigLIP-based 1–10 scorer | No paper | [discus0434/aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5) | AGPL-3.0 | Not stated separately | Unclear (AGPL copyleft) | verified |
| AesMamba | 2024 ACM MM (oral) | Visual state-space model for generic, multi-attribute, and personal IAA | ACM MM 2024 ([repo](https://github.com/AiArt-Gao/AesMamba)) | [AiArt-Gao/AesMamba](https://github.com/AiArt-Gao/AesMamba) (IAA checkpoint on Baidu) | MIT | Not stated (Baidu) | Unclear | verified |
| Charm | 2025 CVPR | Tokenizer that keeps aspect ratio, high-res detail, and multi-scale info for ViT/DINOv2 | [arXiv 2504.02522](https://arxiv.org/abs/2504.02522) | [FBehrad/Charm](https://github.com/FBehrad/Charm); [FatemehBehrad/Charm](https://huggingface.co/FatemehBehrad/Charm) | Apache-2.0 | Apache-2.0 (base DINOv2-small) | Yes (code/weights only) | verified |
| FGAesQ | 2026 CVPR (oral) | Fine-grained IAA on photo series; rank-aware regression | [arXiv 2603.03907](https://arxiv.org/abs/2603.03907) | [yzc-ippl/FG-IAA](https://github.com/yzc-ippl/FG-IAA); [yzc002/FGAesQ](https://huggingface.co/yzc002/FGAesQ) | Not stated (no licence file) | Not stated | Unclear | verified |

### 1.2 LMM-based aesthetic scorers

| Model | Year / venue | Idea | Paper | Release | Code | Weights | Commercial use? | Status |
|---|---|---|---|---|---|---|---|---|
| Q-Align (aesthetic) | 2024 ICML | mPLUG-Owl2; 5 rating words; score = softmax-weighted level | [arXiv 2312.17090](https://arxiv.org/abs/2312.17090) | [Q-Future/Q-Align](https://github.com/Q-Future/Q-Align); [q-future/q-align-aesthetic](https://huggingface.co/q-future/q-align-aesthetic) | S-Lab License 1.0 (non-commercial) | MIT (HF card) | Unclear (code NC, weights MIT) | verified |
| AesExpert | 2024 ACM MM | LLaVA-1.5 tuned on AesMMIT (409K aesthetic instructions) | [arXiv 2404.09624](https://arxiv.org/abs/2404.09624) | [yipoh/AesExpert](https://github.com/yipoh/AesExpert); [qyuan/AesMMIT_LLaVA_v1.5_7b_240325](https://huggingface.co/qyuan/AesMMIT_LLaVA_v1.5_7b_240325) | Apache-2.0 | Not stated (no card licence) | Unclear | verified |
| UNIAA-LLaVA | 2024 arXiv | Unified IAA baseline and benchmark; data built with IDCP pipeline | [arXiv 2404.09619](https://arxiv.org/abs/2404.09619) | [KlingAIResearch/Uniaa](https://github.com/KlingAIResearch/Uniaa); data [zkzhou/UNIAA](https://huggingface.co/datasets/zkzhou/UNIAA) | Not stated (no licence file) | No checkpoint found | - | verified |
| HumanAesExpert | 2025 arXiv | Aesthetics of human images; 1B and 8B models | [arXiv 2503.23907](https://arxiv.org/abs/2503.23907) | [KlingAIResearch/HumanAesExpert](https://github.com/KlingAIResearch/HumanAesExpert); [KlingTeam/HumanAesExpert-8B](https://huggingface.co/KlingTeam/HumanAesExpert-8B) | MIT | MIT | Yes (code/weights only) | verified |
| ArtiMuse / ArtiMuse-AVA | 2026 CVPR | InternVL3-8B; 100 ordered score tokens; 8 expert attributes; per-dataset checkpoints | [arXiv 2507.14533](https://arxiv.org/abs/2507.14533) | [thunderbolt215/ArtiMuse](https://github.com/thunderbolt215/ArtiMuse); [ArtiMuse](https://huggingface.co/Thunderbolt215215/ArtiMuse), [ArtiMuse_AVA](https://huggingface.co/Thunderbolt215215/ArtiMuse_AVA) | Not stated (no licence file) | Apache-2.0 (HF cards; base InternVL3-8B) | Unclear | verified |
| Aes-R1 | 2025 arXiv | Qwen2.5-VL-7B; AesCoT reasoning data (~15K) + RAPO (absolute + ranking reward) | [arXiv 2509.21871](https://arxiv.org/abs/2509.21871) | [ssssmark/Aes-R1](https://huggingface.co/ssssmark/Aes-R1) (GitHub `ssssmark/Aes-R1` returns 404) | No code found | Apache-2.0 (HF card) | Unclear | verified |
| RvTC / RvTC+ | 2026 WACV | Bin-based classification head for regression; `+` adds AVA challenge titles as prompt | [arXiv 2507.14997](https://arxiv.org/abs/2507.14997) | [royhj/rvtc](https://github.com/royhj/rvtc); [rvtc-qwen2vl-2b-image-only](https://huggingface.co/royhj/rvtc-qwen2vl-2b-image-only) | Not stated | Not stated | Unclear | verified |
| ROC4MLLM | 2026 AAAI | Score tokens separate from text tokens; regression loss; mPLUG-Owl2 | [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37726) | [woshidandan/Assessing-Image-Aesthetics-via-MLLMs](https://github.com/woshidandan/Assessing-Image-Aesthetics-via-Multimodal-Large-Language-Models); [Ricardo-M/ROC4MLLM](https://huggingface.co/Ricardo-M/ROC4MLLM) | README badge says Apache-2.0; no licence file | Apache-2.0 (HF card) | Unclear | verified |
| UniPercept | 2026 ICML (spotlight, per repo) | Unified IAA + IQA + structure/texture | [arXiv 2512.21675](https://arxiv.org/abs/2512.21675) | See [unified-and-backbones.md](unified-and-backbones.md) | - | - | - | verified (venue from repo description) |

Also seen, not detailed here (no scoring numbers checked): JoPPO (CVPR 2026, [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_JoPPO_Hierarchical_Photography_Assessment_via_Contrastive_Joint_Conditional_Probabilistic_Reinforcement_CVPR_2026_paper.pdf)), Open World IAA (CVPR 2026 Findings, [CVF page](https://openaccess.thecvf.com/content/CVPR2026F/html/Liao_Open_World_Image_Aesthetic_Assessment_CVPRF_2026_paper.html)), P-MLLM zero-shot personalized IAA ([arXiv 2604.17233](https://arxiv.org/abs/2604.17233)). Status: unverified beyond the search result. RED / RED-20k (relative edit-induced aesthetics data for Aes-R1) is covered in [../training/data-mixture-evidence.md](../training/data-mixture-evidence.md).

## 2. Protocol A: AVA official split, trained on AVA train only

Train on AVA train (about 236K images), test on the AVA test set (about 19.9K images). Dataset-specific. Rows from different papers use the same split but not the same crop, resolution, or score mapping.

| Model | AVA PLCC / SRCC | Source |
|---|---|---|
| NIMA (Inception-v2) | 0.636 / 0.612 | NIMA paper, as quoted in [MUSIQ Table 4](https://arxiv.org/abs/2108.05997), [VILA Table 1](https://arxiv.org/abs/2303.14302), [Q-Align Table 6](https://arxiv.org/abs/2312.17090) |
| NIMA (MobileNet, idealo port) | 0.626 / 0.609 | [idealo README](https://github.com/idealo/image-quality-assessment) |
| MUSIQ | 0.738 / 0.726 | [MUSIQ Table 4](https://arxiv.org/abs/2108.05997) (mean of 10 runs) |
| LAION aesthetic predictor (trained on AVA) | 0.723 / 0.721 | As quoted in [Q-Align Table 6](https://arxiv.org/abs/2312.17090) |
| MaxViT | 0.745 / 0.708 | As quoted in [VILA Table 1](https://arxiv.org/abs/2303.14302) |
| TANet | 0.765 / 0.758 | As quoted in [VILA Table 1](https://arxiv.org/abs/2303.14302) |
| LIQE | 0.763 / 0.776 | As quoted in [Q-Align Table 6](https://arxiv.org/abs/2312.17090) |
| VILA-R (frozen image encoder; extra AVA-Captions pre-training) | 0.774 / 0.774 | [VILA Table 1](https://arxiv.org/abs/2303.14302) |
| VILA-R, image encoder fine-tuned | 0.780 / 0.780 | [VILA Table 3](https://arxiv.org/abs/2303.14302) |
| Charm (DINOv2-small) | 0.779 / 0.777 | [Charm Table 2](https://arxiv.org/abs/2504.02522) |
| Charm (DINOv2-large) | 0.783 / 0.781 | [Charm Table 2](https://arxiv.org/abs/2504.02522) |
| FGAesQ (joint AVA + FGAesthetics training) | 0.781 / 0.770 | [FGAesQ Table 3](https://arxiv.org/abs/2603.03907) |
| AesMamba | 0.760 / 0.751 | As quoted in [ROC4MLLM Table 1](https://ojs.aaai.org/index.php/AAAI/article/view/37726) |
| AesMamba | 0.769 / 0.774 | As quoted in [ArtiMuse Table 3](https://arxiv.org/abs/2507.14533) (values differ from ROC4MLLM quote) |
| EAT | 0.814 / 0.803 | As quoted in [UNIAA Table 8](https://arxiv.org/abs/2404.09619) |
| AesExpert (fine-tuned on AVA by ROC4MLLM authors) | 0.779 / 0.784 | [ROC4MLLM Table 1](https://ojs.aaai.org/index.php/AAAI/article/view/37726) |
| Q-Align (aesthetic) | 0.817 / 0.822 | [Q-Align Table 6](https://arxiv.org/abs/2312.17090) |
| Few-shot Q-Align (10% of AVA train) | 0.775 / 0.776 | [Q-Align Table 6](https://arxiv.org/abs/2312.17090) |
| UNIAA-LLaVA + MOS regression head | 0.838 / 0.840 | [UNIAA Table 8](https://arxiv.org/abs/2404.09619) |
| ArtiMuse-AVA | 0.826 / 0.827 | [ArtiMuse Table 3 / 13](https://arxiv.org/abs/2507.14533) |
| RvTC (image only, mPLUG-Owl2) | 0.831 / 0.833 | [RvTC Table 1](https://arxiv.org/abs/2507.14997) |
| RvTC (image only, Qwen2-VL-2B) | 0.842 / 0.843 | [RvTC prompt table](https://arxiv.org/abs/2507.14997) |
| ROC4MLLM (mPLUG-Owl2) | 0.832 / 0.833 | [ROC4MLLM Table 1](https://ojs.aaai.org/index.php/AAAI/article/view/37726) |

Not comparable to the table above:

- RvTC+ reaches 0.901 / 0.899 on AVA. It feeds the DPChallenge challenge title as a text prompt. That is extra metadata tied to the AVA source. Do not compare it with image-only models ([RvTC](https://arxiv.org/abs/2507.14997)).
- VILA-P zero-shot (ensemble prompts) gets 0.663 / 0.657 with no AVA score labels ([VILA Table 1](https://arxiv.org/abs/2303.14302)).
- The ONE-ALIGN multi-task model (AVA plus IQA and VQA data) gets 0.819 / 0.823 on AVA ([Q-Align Table 7](https://arxiv.org/abs/2312.17090)). See [unified-and-backbones.md](unified-and-backbones.md).

## 3. Protocol B: other IAA datasets, dataset-specific training

Each model is trained and tested on the split of the named dataset. Splits come from each paper. Do not mix rows across papers.

### 3.1 ArtiMuse paper ([arXiv 2507.14533](https://arxiv.org/abs/2507.14533) Table 3)

The main ArtiMuse row equals its single-dataset rows in Table 13, so we read it as one checkpoint per dataset. `†` values are copied from original papers.

| Model | AVA | PARA | TAD66K | FLICKR-AES | ArtiMuse-10K |
|---|---|---|---|---|---|
| VILA | 0.775 / 0.776 | 0.658 / 0.651 | 0.444 / 0.418 | 0.645 / 0.616 | 0.268 / 0.273 |
| TANet † | 0.765 / 0.758 | - | 0.531 / 0.513 | - | - |
| AesMamba † | 0.769 / 0.774 | 0.902 / 0.936 | 0.483 / 0.511 | - | - |
| UNIAA-LLaVA † | 0.704 / 0.713 | 0.895 / 0.864 | 0.425 / 0.411 | 0.751 / 0.724 | - |
| Q-Align | 0.817 / 0.822 | 0.888 / 0.913 | 0.531 / 0.501 | 0.818 / 0.798 | 0.573 / 0.551 |
| ArtiMuse | 0.826 / 0.827 | 0.958 / 0.936 | 0.543 / 0.510 | 0.837 / 0.814 | 0.627 / 0.614 |
| GPT-4o (prompted, zero-shot) | 0.485 / 0.509 | 0.744 / 0.697 | 0.282 / 0.278 | 0.597 / 0.605 | 0.276 / 0.333 |
| Qwen2.5-VL-7B (prompted, zero-shot) | 0.371 / 0.391 | 0.743 / 0.721 | 0.242 / 0.240 | 0.578 / 0.621 | 0.179 / 0.256 |

The UNIAA-LLaVA AVA value here (0.704 / 0.713) is much lower than UNIAA's own MOS-head result (0.838 / 0.840). The two use different output heads. Treat them as different models.

### 3.2 ROC4MLLM paper ([AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37726) Tables 1–3)

Q-Align rows were retrained by the ROC4MLLM authors with official code.

| Model | TAD66K | ICAA17K | PARA | AADB |
|---|---|---|---|---|
| NIMA | 0.405 / 0.390 | 0.815 / 0.809 | 0.862 / 0.877 | 0.711 / 0.700 |
| AesMamba | 0.503 / 0.475 | - | - | - |
| Q-Align (retrained) | 0.536 / 0.506 | 0.781 / 0.761 | 0.940 / 0.923 | 0.770 / 0.762 |
| ROC4MLLM | 0.550 / 0.518 | 0.903 / 0.894 | 0.956 / 0.934 | 0.789 / 0.783 |

### 3.3 Charm paper ([arXiv 2504.02522](https://arxiv.org/abs/2504.02522) Table 1)

DINOv2-small backbone, standard tokenization versus Charm.

| Dataset | DINOv2-small | DINOv2-small + Charm |
|---|---|---|
| AVA | 0.734 / 0.732 | 0.779 / 0.777 |
| AADB | 0.695 / 0.682 | 0.767 / 0.754 |
| TAD66K | 0.429 / 0.401 | 0.488 / 0.458 |
| PARA | 0.904 / 0.855 | 0.938 / 0.905 |
| BAID | 0.428 / 0.342 | 0.439 / 0.368 |

### 3.4 UNIAA paper ([arXiv 2404.09619](https://arxiv.org/abs/2404.09619) Table 8)

| Model | AVA | TAD66K |
|---|---|---|
| NIMA | 0.636 / 0.612 | 0.405 / 0.390 |
| MUSIQ | 0.738 / 0.726 | 0.517 / 0.489 |
| TANet | 0.765 / 0.758 | 0.531 / 0.513 |
| EAT | 0.814 / 0.803 | 0.546 / 0.517 |
| UNIAA-LLaVA + MOS head | 0.838 / 0.840 | 0.553 / 0.521 |

## 4. Protocol C: cross-dataset transfer (the ArtiMuse finding)

Source: [ArtiMuse Table 13](https://arxiv.org/abs/2507.14533). Each model is trained on one dataset only, then tested on all five. Cells outside the training set's own column are cross-dataset results.

| Train set → model | AVA | PARA | TAD66K | FLICKR-AES | ArtiMuse-10K |
|---|---|---|---|---|---|
| AVA → Q-Align | 0.817 / 0.822 | 0.711 / 0.694 | 0.445 / 0.417 | 0.664 / 0.643 | 0.320 / 0.337 |
| AVA → ArtiMuse | 0.826 / 0.827 | 0.725 / 0.697 | 0.451 / 0.419 | 0.676 / 0.647 | 0.376 / 0.395 |
| TAD66K → Q-Align | 0.699 / 0.695 | suspect (see note) | 0.531 / 0.501 | suspect (see note) | 0.304 / 0.317 |
| TAD66K → ArtiMuse | 0.676 / 0.671 | suspect (see note) | 0.543 / 0.510 | suspect (see note) | 0.369 / 0.397 |
| FLICKR-AES → Q-Align | 0.611 / 0.609 | 0.839 / 0.836 | 0.376 / 0.366 | 0.818 / 0.798 | 0.208 / 0.215 |
| FLICKR-AES → ArtiMuse | 0.594 / 0.581 | 0.874 / 0.854 | 0.397 / 0.379 | 0.837 / 0.814 | 0.285 / 0.294 |
| ArtiMuse-10K → Q-Align | 0.386 / 0.398 | 0.395 / 0.346 | 0.197 / 0.194 | 0.123 / 0.137 | 0.573 / 0.551 |
| ArtiMuse-10K → ArtiMuse | 0.385 / 0.397 | 0.461 / 0.446 | 0.232 / 0.230 | 0.334 / 0.349 | 0.627 / 0.614 |

Findings:

- AVA-trained models transfer badly to the expert-rated ArtiMuse-10K test set. Q-Align gets 0.320 / 0.337. ArtiMuse-AVA gets 0.376 / 0.395. So a high AVA score does not mean expert-level aesthetics.
- The reverse also fails. ArtiMuse-10K-trained models score about 0.39 on AVA.
- The ArtiMuse gain over Q-Align is small on AVA (+0.009 PLCC) but larger on ArtiMuse-10K transfer (+0.056 PLCC).
- Likely source errors: in Table 13, the PARA-trained rows repeat their PARA values in the FLICKR-AES column. The TAD66K-trained rows show identical values in the PARA and FLICKR-AES columns (Q-Align 0.667 / 0.688; ArtiMuse 0.677 / 0.719). We left the PARA-trained rows out and marked the TAD66K cells as suspect. Check the paper before reuse.

## 5. Protocol D: low-data mixed training (Aes-R1)

Source: [Aes-R1 Table 1](https://arxiv.org/abs/2509.21871). Trained on a 15K combined set; baselines marked `*` were retrained by the Aes-R1 authors on the same 15K set. Prompted MLLMs are zero-shot. These numbers are far below Protocol A because the training set is small. Do not compare them with Protocol A.

| Model | TAD66K | AVA | FLICKR-AES | PARA | AADB | Mean |
|---|---|---|---|---|---|---|
| GPT-4.1 (zero-shot) | 0.2707 / 0.3298 | 0.4846 / 0.5594 | 0.5909 / 0.6062 | 0.7290 / 0.7203 | 0.5102 / 0.5296 | 0.5171 / 0.5491 |
| NIMA | 0.3885 / 0.3654 | 0.6120 / 0.6361 | 0.5130 / 0.4796 | 0.5868 / 0.5709 | 0.3886 / 0.3904 | 0.4978 / 0.4885 |
| Q-Align * | 0.3498 / 0.3627 | 0.5215 / 0.5407 | 0.6231 / 0.6473 | 0.6181 / 0.6262 | 0.4474 / 0.4508 | 0.5120 / 0.5255 |
| DeQA-Score * (AVA + FLICKR-AES only) | 0.3985 / 0.3885 | 0.5718 / 0.5896 | 0.7057 / 0.6889 | 0.7234 / 0.6796 | 0.4859 / 0.4948 | 0.5771 / 0.5663 |
| Q-Insight * | 0.3980 / 0.3886 | 0.5964 / 0.5898 | 0.7012 / 0.6769 | 0.7745 / 0.7428 | 0.5069 / 0.5184 | 0.5954 / 0.5813 |
| VisualQuality-R1 * | 0.3082 / 0.3915 | 0.4407 / 0.6195 | 0.5524 / 0.6769 | 0.5376 / 0.7507 | 0.3754 / 0.5363 | 0.4429 / 0.5930 |
| Aes-R1 | 0.4513 / 0.4248 | 0.6702 / 0.6619 | 0.7243 / 0.6973 | 0.7842 / 0.7666 | 0.5386 / 0.5423 | 0.6337 / 0.6186 |

Notes:

- The NIMA AVA cell (0.6120 / 0.6361) looks like the original NIMA result (PLCC 0.636, SRCC 0.612) with the two values swapped. Its training set is not stated in the table.
- VisualQuality-R1 has high SRCC but low PLCC. This suggests poor score calibration, not poor ranking.

## 6. Protocol E: fine-grained series versus coarse AVA (FGAesQ)

Source: [FGAesQ Table 3](https://arxiv.org/abs/2603.03907). "Fine-tuned" rows are AVA models fine-tuned on FGAesthetics series rankings.

| Model | AVA PLCC / SRCC | FGAesthetics pair score | FGAesthetics series score |
|---|---|---|---|
| MUSIQ | 0.738 / 0.726 | 0.624 | 0.331 |
| VILA | 0.774 / 0.774 | 0.659 | 0.399 |
| Charm | 0.779 / 0.777 | 0.665 | 0.419 |
| MUSIQ, fine-tuned on FGAesthetics | 0.461 / 0.437 | 0.665 | 0.413 |
| Charm, fine-tuned on FGAesthetics | 0.472 / 0.470 | 0.699 | 0.477 |
| FGAesQ (joint coarse + fine training) | 0.781 / 0.770 | 0.753 | 0.600 |

Pair score = mean of pair accuracy and F1. Series score = mean of series accuracy and series SRCC. Finding: naive fine-tuning on fine-grained rankings drops AVA PLCC by about 0.27–0.33 (NIMA, MUSIQ, VILA, Charm rows in the source). Joint training keeps AVA and improves fine-grained ranking.

## 7. Notes for the Q-ReAlign comparison panel

- Runnable IAA baselines with public weights: MUSIQ-AVA, Q-Align aesthetic (or ONE-ALIGN), ArtiMuse-AVA, Charm, ROC4MLLM, Aes-R1, LAION v2 predictor. Re-score all of them on one locked AVA test manifest before any ranking.
- Add an expert-rated or out-of-domain aesthetic set (ArtiMuse-10K test, TAD66K, PARA) to catch the AVA-only overfit shown in Section 4.
- Licences: Q-Align code and pyiqa ports are non-commercial. ArtiMuse, FGAesQ, UNIAA, and RvTC code have no licence file. Aes-R1 has weights but no code repo.
- LMM scorers that output text scores (Aes-R1, RvTC) need a fixed parser. Log parse failures as part of the result.
