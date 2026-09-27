# Data mixture evidence

This file collects controlled evidence that adding, removing, selecting, or re-labeling training data helps or hurts quality and aesthetics assessors. It covers dataset additions, cross-task mixing, auxiliary annotation data, data selection under a label budget, and label-scale alignment. Dataset facts (size, license, access) are in [datasets/iqa.md](../datasets/iqa.md), [datasets/iaa.md](../datasets/iaa.md), and [datasets/vqa.md](../datasets/vqa.md).

Last verified: 2026-09-27

## How to read this file

- All pairs are `PLCC / SRCC`. Sources in `SRCC / PLCC` order (Q-Align, MDS-VQA, VQA²) were converted.
- Every number is author-reported and comes from one paper's own protocol. None of it is a Q-ReAlign result.
- Status: every table value in this file was re-read in the linked primary source on 2026-09-27 (`verified`), unless a row says `from dossier`.
- Evidence grade: **A** = only the data changes (same model, steps, split). **B** = data plus a new task, loss, or stage. **C** = data changes and model or budget also changes.
- A gain on a held-out split of a dataset you just added is in-domain adaptation. It is not better cross-dataset generalization.
- Bold numbers mark regressions.

## 1. Adding an image-quality dataset

### 1.1 Q-Align: KonIQ, then + SPAQ, then + KADID

Source: [Q-Align Table 4](https://arxiv.org/html/2312.17090) (source order SRCC / PLCC). Model: mPLUG-Owl2, one-hot level labels. Grade A (training set grows, recipe fixed).

| Trained on | KonIQ | SPAQ | KADID | LIVE-C | AGIQA-3K | LIVE | CSIQ |
|---|---|---|---|---|---|---|---|
| KonIQ | 0.941 / 0.940 | 0.886 / 0.887 | 0.674 / 0.684 | 0.853 / 0.860 | 0.772 / 0.735 | 0.838 / 0.867 | 0.759 / 0.700 |
| KonIQ + SPAQ | 0.943 / 0.940 | 0.933 / 0.931 | 0.692 / 0.708 | 0.883 / 0.879 | 0.795 / 0.727 | **0.827 / 0.859** | 0.795 / 0.767 |
| KonIQ + SPAQ + KADID | 0.945 / 0.938 | 0.933 / 0.931 | 0.935 / 0.934 | 0.887 / 0.883 | 0.788 / 0.733 | 0.840 / 0.870 | 0.876 / 0.845 |

Note: adding the synthetic KADID set lifts the synthetic cross-sets (CSIQ). It leaves authentic and AIGC cross-sets almost flat.

### 1.2 DeQA-Score: + KADID, then + PIPAL

Source: [DeQA-Score Table 4](https://arxiv.org/html/2501.11561#S5.T4). Model: mPLUG-Owl2 with soft labels + fidelity loss; the Q-Align rows use the same base model. Grade A. PIPAL is in-domain after it is added; LIVE-Wild, AGIQA-3K, TID2013, and CSIQ are always cross-set.

| # | Trained on | Model | KonIQ | KADID | PIPAL | LIVE-Wild | AGIQA-3K | TID2013 | CSIQ |
|---|---|---|---|---|---|---|---|---|---|
| 0 | KonIQ + SPAQ | DeQA | 0.953 / 0.943 | 0.724 / 0.719 | 0.468 / 0.474 | 0.902 / 0.888 | 0.810 / 0.738 | 0.706 / 0.616 | 0.836 / 0.785 |
| 1 | + KADID | DeQA | 0.957 / 0.944 | 0.955 / 0.953 | 0.495 / 0.496 | 0.900 / 0.887 | 0.808 / 0.745 | 0.852 / 0.820 | 0.900 / 0.857 |
| 2 | + KADID + PIPAL | DeQA | 0.958 / 0.946 | 0.963 / 0.961 | 0.724 / 0.690 | **0.877 / 0.857** | **0.770 / 0.735** | **0.828 / 0.796** | **0.863 / 0.807** |
| 1 | + KADID | Q-Align | 0.945 / 0.938 | 0.935 / 0.934 | 0.409 / 0.420 | 0.887 / 0.883 | 0.788 / 0.733 | 0.829 / 0.808 | 0.876 / 0.845 |
| 2 | + KADID + PIPAL | Q-Align | **0.926 / 0.932** | 0.950 / 0.954 | 0.702 / 0.671 | **0.853 / 0.845** | **0.765 / 0.722** | **0.811 / 0.795** | **0.838 / 0.789** |

Notes:
- Adding KADID (#0 to #1) lifts TID2013 and CSIQ a lot. Other sets barely move.
- Adding PIPAL (#1 to #2) lifts PIPAL but lowers every cross-set. With one-hot labels (Q-Align) it also lowers KonIQ. The paper says PIPAL's model-processed distortions differ from the others.
- The [RALI Table 3](https://arxiv.org/html/2510.11369v2) reuses the #2 mix for other methods. Q-Insight trained on it reaches only 0.757 / 0.765 on in-domain KADID, vs 0.963 / 0.961 for DeQA. RALI's text says Q-Insight "shows severe convergence problem on mixed datasets". Score-regression RL may need ranking rewards to co-train datasets.

### 1.3 VisualQuality-R1: KADID vs KADID + SPAQ

Source: [VisualQuality-R1 Table 2](https://arxiv.org/html/2505.14460). Model: Qwen2.5-VL-7B GRPO. Grade A per model. Mean over 8 sets (BID, CLIVE, KonIQ, SPAQ, deblurring, super-resolution, dehazing, image generation). SPAQ becomes in-domain in the second row of each pair.

| Model | KADID only | KADID + SPAQ |
|---|---|---|
| VisualQuality-R1 (ranking reward) | 0.814 / 0.777 | 0.831 / 0.791 |
| Q-Insight (score reward) | 0.803 / 0.766 | **0.793 / 0.759** |

Note: with a ranking reward, more data helped. With a raw-score reward, the same addition hurt the mean (image generation fell from 0.810 / 0.749 to 0.705 / 0.657). The VisualQuality-R1 authors trained the Q-Insight rows in this table.

## 2. Mixing IQA, IAA, and VQA in one model

### 2.1 Q-Align ONE-ALIGN task mixing

Source: [Q-Align Table 7](https://arxiv.org/html/2312.17090) (source order SRCC / PLCC). Model: mPLUG-Owl2. IQA = KonIQ + SPAQ + KADID; IAA = AVA; VQA = LSVQ. Grade A (data changes only). This is the origin of the Q-ReAlign recipe.

| Trained on | KonIQ | KADID | LIVE-C | AGIQA-3K | CSIQ | AVA | LSVQ test | LSVQ 1080p | KoNViD-1k | MaxWell |
|---|---|---|---|---|---|---|---|---|---|---|
| IQA | 0.945 / 0.938 | 0.935 / 0.934 | 0.887 / 0.883 | 0.788 / 0.733 | 0.876 / 0.845 | 0.228 / 0.208 | 0.757 / 0.755 | 0.718 / 0.680 | 0.806 / 0.799 | 0.694 / 0.682 |
| VQA | 0.788 / 0.731 | 0.651 / 0.659 | 0.727 / 0.715 | 0.834 / 0.780 | 0.814 / 0.755 | 0.323 / 0.289 | 0.882 / 0.883 | 0.830 / 0.797 | 0.877 / 0.865 | 0.782 / 0.780 |
| IAA | 0.603 / 0.574 | 0.547 / 0.536 | 0.636 / 0.685 | 0.792 / 0.750 | 0.596 / 0.527 | 0.817 / 0.822 | 0.600 / 0.624 | 0.511 / 0.515 | 0.681 / 0.717 | 0.648 / 0.659 |
| IQA + VQA | 0.949 / 0.944 | 0.953 / 0.952 | 0.899 / 0.892 | 0.782 / 0.739 | 0.876 / 0.852 | 0.222 / 0.197 | 0.883 / 0.885 | 0.829 / 0.802 | 0.880 / 0.867 | 0.787 / 0.781 |
| IQA + IAA | 0.947 / 0.940 | 0.945 / 0.945 | **0.868 / 0.862** | 0.824 / 0.782 | 0.883 / 0.865 | 0.819 / 0.822 | 0.785 / 0.785 | 0.730 / 0.700 | 0.829 / 0.831 | 0.728 / 0.716 |
| All (OneAlign) | 0.950 / 0.941 | 0.942 / 0.941 | 0.894 / 0.881 | 0.838 / 0.801 | 0.906 / 0.881 | 0.819 / 0.823 | 0.886 / 0.886 | 0.837 / 0.803 | 0.888 / 0.876 | 0.786 / 0.781 |

Notes:
- Mixing all three tasks did not hurt any in-domain task. The largest gains were on cross-sets (AGIQA-3K, CSIQ).
- Adding AVA to IQA lowered LIVE-C. IQA + VQA beats OneAlign on KADID and LIVE-C. "All" is not best everywhere.
- AVA train has 236K images ([Q-Align Table 6](https://arxiv.org/html/2312.17090)). It is the largest set in the mix. Check sampling weights before you add more aesthetic data.

### 2.2 Mixed vs sequential training (VQA²)

Source: [VQA² Tables 8 and 9](https://arxiv.org/html/2411.03795v4). The Table 8 caption does not state the metric order. We assume SRCC / PLCC, as in the paper's Table 2, and convert. Model: VQA² UGC-Scorer and Assistant (7B). Grade A (same data, different schedule).

| Schedule | LIVE-VQC | LSVQ 1080p | LSVQ test | KoNViD-1k | Q-Bench-Video test overall |
|---|---|---|---|---|---|
| Mixed scoring + QA data | 0.836 / 0.789 | 0.831 / 0.761 | 0.873 / 0.883 | 0.892 / 0.895 | 53.75% |
| Sequential | **0.823 / 0.776** | **0.819 / 0.760** | **0.856 / 0.882** | **0.844 / 0.883** | 56.78% |

Note: mixing helped scoring. Sequential training helped question answering. Choose the schedule by the output you ship.

### 2.3 Scoring vs interpreting vs general data (Q-SiT)

Source: [Q-SiT Table VII](https://arxiv.org/html/2503.09197v1). Model: LLaVA-OneVision-7B. D1 = image quality scoring data, D2 = image quality interpreting data, D3 = general image understanding data. Scoring = mean of SRCC and PLCC over IQA sets. Interpreting = LLVisionQA accuracy. Grade A.

| Mix | Scoring | Interpreting |
|---|---|---|
| D2 + D3 (no scoring data) | **0.800** | 0.754 |
| D1 + D3 (no interpreting data) | 0.857 | **0.721** |
| D1 + D2 (no general data) | 0.859 | 0.744 |
| D1 + D2 + D3, naive mix | 0.855 | 0.732 |
| D1 + D2 + D3, tuned ratio | 0.861 | 0.756 |

Note: a naive mix was worse than dropping general data for scoring. The authors found the ratio on a 0.5B model (about 720 A800 GPU-hours) and reused it on 7B.

### 2.4 Video fine-tuning and image skills (general VLMs)

[Temporal Gains, Spatial Costs (arXiv 2603.17541)](https://arxiv.org/abs/2603.17541) reports that video SFT "reliably improves video performance, but often yields limited gains or even degradation on static image benchmarks", tied to the temporal budget. General VLM benchmarks, not quality tasks. Abstract only; verified.

## 3. Auxiliary annotation data

| Data added | Result (PLCC / SRCC unless stated) | Protocol | Grade | Source | Status |
|---|---|---|---|---|---|
| 7K single-distortion images from DQ-495K, as a distortion type/level task | 7-set mean 0.765 / 0.739 to 0.806 / 0.783. CSIQ 0.685 / 0.640 to 0.870 / 0.824. **AGIQA 0.816 / 0.766 to 0.811 / 0.764.** | Q-Insight, Qwen2.5-VL-7B GRPO, score data = KonIQ | B | [Q-Insight Table 3](https://arxiv.org/html/2503.22679v2) | verified |
| DQ data added to Q-Instruct (DepictQA) | Q-Bench distortion questions 55.4% to 66.7%; "what" questions 49.6% to 62.8% (accuracy) | DepictQA model, Q-Instruct only vs Q-Instruct + DQ | A (for QA) | [DepictQA Table XI](https://arxiv.org/html/2405.18842) | verified |
| Comparison-reasoning co-training (DepictQA) | Instant-rating accuracy up on 4 sets (KADID 92.4 / 92.3 to 93.6 / 93.1), **down on LIVE-MD (92.9 / 92.7 to 92.1 / 91.8) and MDID2013 (92.1 / 91.7 to 90.0 / 89.6)** | Pairwise accuracy in two settings, as reported; not MOS correlation | B | [DepictQA Table X](https://arxiv.org/html/2405.18842) | verified |
| Refine-Perception-20K local-distortion stage | Mean of SRCC and PLCC: KADID 0.671 to 0.709; KonIQ 0.916 to 0.931; SPAQ 0.920 to 0.924; AGIQA 0.815 to 0.817 | Refine-IQA, Qwen2.5-VL-7B, extra RL stage | B | [Refine-IQA v2 Table 5](https://arxiv.org/html/2508.03763v2) | verified |
| VQA² Stage-1 distortion pre-training data | LSVQ 1080p 0.812 / 0.751 to 0.847 / 0.782; LIVE-VQC 0.819 / 0.776 to 0.830 / 0.785 | VQA² UGC scorer | B | [VQA² Table 6](https://arxiv.org/html/2411.03795v4) | verified |
| KonIQ++ distortion labels (joint heads) | KonIQ test: ResNet-50 0.9280 / 0.9167 to 0.9293 / 0.9170; ResNeXt-101 0.9428 / 0.9342 to 0.9436 / 0.9350 | Same backbone, 4 extra distortion outputs. The bigger gain (0.9483 / 0.9396) also needs the new side networks. | A (tiny effect) | [KonIQ++ Table 1](https://datasets.vqa.mmsp-kn.de/archives/koniq++/KonIQ++_paper.pdf) | verified |

## 4. Selecting data under a label budget

Source: [MDS-VQA Table 3](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_MDS-VQA_Model-Informed_Data_Selection_for_Video_Quality_Assessment_CVPR_2026_paper.pdf) (CVPR 2026; also [arXiv 2603.11525](https://arxiv.org/html/2603.11525v1)); source order SRCC / PLCC. Base: VisualQuality-R1 (Qwen2.5-VL) trained on YouTube-UGC, then fine-tuned on the full YouTube-UGC train set + 5% of each target pool. Grade A (only the selection rule changes). The target pools come from existing rated datasets, so this simulates a labeling budget. The 6-set mean includes YouTube-UGC (not shown).

| Selection (5% per target) | CGVDS | LIVE-Livestream | YT-SFV SDR | YT-SFV HDR2SDR | AIGVQA-DB | 6-set mean |
|---|---|---|---|---|---|---|
| No adaptation | 0.780 / 0.766 | 0.587 / 0.561 | 0.718 / 0.666 | 0.557 / 0.495 | 0.748 / 0.711 | 0.683 / 0.651 |
| Random | 0.804 / 0.807 | 0.628 / 0.569 | 0.761 / 0.703 | 0.588 / 0.518 | 0.751 / 0.756 | 0.715 / 0.686 |
| FreeSel (best baseline) | 0.849 / 0.832 | 0.646 / 0.627 | 0.787 / 0.719 | 0.590 / 0.498 | 0.785 / 0.789 | 0.742 / 0.713 |
| MDS-VQA (hard + diverse) | 0.875 / 0.874 | 0.654 / 0.632 | 0.794 / 0.731 | 0.595 / **0.507** | **0.769 / 0.769** | 0.749 / 0.722 |

Note: selection beats random on the mean. It does not win every target (HDR2SDR SRCC below random; AIGVQA-DB below FreeSel).

## 5. Label-scale alignment

Source: [MOSAIQ Table 2](https://arxiv.org/pdf/2609.20247) (arXiv, 17 Sep 2026, submitted to IEEE). SRCC only; no PLCC is given. Each cell is `pooled (mean)`: pooled = SRCC over images pooled across the sets (inter-dataset); mean = mean per-dataset SRCC (intra-dataset). Seen = the five training sets (LIVE, CSIQ, KADID, KonIQ, CLIVE), test portion only. Aligned scores come from a common rating study of 1,039 images from 26 datasets. Code and data are promised "upon acceptance" ([project](https://ivc.uwaterloo.ca/projects/unifying_iqa_datasets/)).

| Model | Variant | Seen 5 sets | Unseen sets | Grade vs V1 |
|---|---|---|---|---|
| MonotonicIQA | V1: original method, original labels | 0.7415 (0.9278) | 0.7836 (0.7681) | - |
| MonotonicIQA | V2: aligned labels | 0.7719 (0.9279) | 0.8217 (0.7824) | A |
| MonotonicIQA | V3: aligned labels, plain L1/L2 loss, no per-dataset mapping | 0.7823 (**0.9126**) | 0.8591 (0.7841) | B |
| LIQE | V1: original | 0.7262 (0.9437) | 0.7262 (0.7944) | - |
| LIQE | V2: aligned labels | 0.7878 (**0.9363**) | 0.7885 (**0.7576**) | A |
| HyperIQA | V1: trained on KonIQ only | 0.5561 (0.6722) | 0.5394 (0.6311) | - |
| HyperIQA | V2: five sets, naive per-dataset normalization | 0.6710 (0.8915) | **0.5372 (0.6149)** | C |
| HyperIQA | V3: five sets, aligned labels | 0.7088 (0.9058) | 0.5707 (0.6391) | C (A vs V2) |

Notes:
- Aligned labels raise pooled (inter-dataset) SRCC in every model. For LIQE they lower mean per-dataset SRCC.
- Naive per-dataset normalization (HyperIQA V2) helped on seen sets but not on unseen sets. Aligned labels (V3) beat it on both.
- With aligned labels, a plain loss (MonotonicIQA V3) gave the best unseen pooled SRCC. Special multi-dataset losses may become unnecessary.
- Q-ReAlign maps MOS to five levels with equal-width bins over a stated score range [lo, hi] ([code audit](../evaluation/protocol.md#8-q-realign-baseline-code-audit), row 1). It does not align scales across datasets. That is closer to HyperIQA V2 than to aligned labels.

## 6. Aesthetics data

| Data change | Result (PLCC / SRCC) | Protocol | Grade | Source | Status |
|---|---|---|---|---|---|
| RED-20k (edit pairs, VLM-judged) instead of AesCoT-15k | AVA / Flickr / TAD66K mean 0.6153 / 0.5947 to 0.6628 / 0.6465. AVA 0.6702 / 0.6619 to 0.7506 / 0.7381. Flickr 0.7243 / 0.6973 to 0.7270 / 0.6983. TAD66K 0.4513 / 0.4248 to 0.5107 / 0.5032. | Same Aes-R1 pipeline, data swapped. Cross-domain: no AVA, Flickr, or TAD66K training. 20K vs 15K items; labels from edit models and VLM judges, not humans. | C | [RED-20k v3 Table 3](https://arxiv.org/html/2606.05778v3) | verified |
| HPDv3++ aesthetic pairs (100,463 train pairs; Apache-2.0 card) | Aesthetic preference accuracy 67.2% (Qwen3-VL HPSv3) to 74.7% (stage 1) to 79.1% (full). Not MOS correlation. | HPSv3++ changes data, losses, and stages together | C | [HPSv3++ Table 4](https://arxiv.org/html/2606.14657), [dataset](https://huggingface.co/datasets/Junjun2333/HPDv3-PlusPlus) | verified (table); card from dossier |
| Adding AVA to an IQA mix | See section 2.1: LIVE-C fell, AGIQA-3K and CSIQ rose | Q-Align | A | [Q-Align Table 7](https://arxiv.org/html/2312.17090) | verified |

RED-20k data construction took 2,220 H200 GPU-hours and about 1.3M API calls (four stages). 7B SFT and RL took another 572 H200 GPU-hours ([Table 1](https://arxiv.org/html/2606.05778v3)). No public download or license was found (from dossier; not re-checked today).

## 7. Unlabeled data

| Data | Result (PLCC / SRCC) | Protocol | Grade | Source |
|---|---|---|---|---|
| Round 1: unlabeled KonIQ images, self-voted pairs | Weighted 8-set mean 0.615 / 0.570 (zero-shot base) to 0.751 / 0.709 | EvoQuality, Qwen2.5-VL-7B GRPO, no MOS or distortion labels used | B | [EvoQuality Table 2](https://arxiv.org/html/2509.25787) |
| Round 2: + 10 synthetic variants per KonIQ image (10 of 35 types, 5 levels) | 0.751 / 0.709 to 0.770 / 0.726. KADID 0.784 / 0.782 to 0.803 / 0.807; TID2013 0.624 / 0.587 to 0.674 / 0.611. **AGIQA 0.839 / 0.777 to 0.831 / 0.771.** | Same, second round (data and round both change) | B | [EvoQuality Table 2](https://arxiv.org/html/2509.25787) |

### 7.1 Teacher (synthetic) labels on unlabeled images

Question: can labels from a vision-language model replace human MOS? Evidence so far, author-reported:

| Setting | KonIQ | SPAQ | AGIQA-3K | AIGIQA-20K | Source |
|---|---|---|---|---|---|
| LEAF, label-free: InternVL3.5-8B teacher (frozen, zero-shot) gives 5-level probabilities and pair preferences on the train images; ConvNeXt-Base student | 0.801 / 0.777 | 0.867 / 0.861 | 0.811 / 0.749 | 0.762 / 0.696 | [LEAF section 4.3](https://arxiv.org/html/2601.20689) |
| LEAF + 10% of train MOS for calibration | 0.903 / 0.867 | 0.902 / 0.896 | 0.899 / 0.841 | 0.878 / 0.839 | same |
| LEAF + 30% of train MOS | 0.916 / 0.899 | 0.922 / 0.921 | Not checked | Not checked | [LEAF Table 2](https://arxiv.org/html/2601.20689) |
| Q-Align, fully supervised (reference) | 0.950 / 0.941 (ONE-ALIGN, [Q-Align Table 7](https://arxiv.org/abs/2312.17090), converted from SRCC / PLCC) | – | 0.881 / 0.852 (as quoted in LEAF Table 1) | 0.889 / 0.874 (as quoted) | as linked |

Teacher size matters a little. On AGIQA-3K with 10% MOS: Qwen3-VL-8B 0.873 / 0.835, Qwen3-VL-32B 0.907 / 0.848, InternVL3.5-38B 0.912 / 0.850 ([LEAF Table 4](https://arxiv.org/html/2601.20689)).

Aesthetics is much weaker zero-shot. GPT-4o, prompted: AVA 0.485 / 0.509, TAD66K 0.282 / 0.278 ([models/iaa.md section 3.1](../models/iaa.md#31-artimuse-paper-arxiv-250714533-table-3)). GPT-4.1: AVA 0.4846 / 0.5594 ([Aes-R1 Table 1](https://arxiv.org/abs/2509.21871)).

What this means:

- Zero-shot teacher labels give about 0.75–0.86 SRCC on image quality. That is well below supervised models (about 0.94 on KonIQ). Synthetic labels alone will not match Q-Align accuracy.
- The teacher ranks images well but gets the scale wrong. LEAF fixes the scale with a small human-labeled set. We have no private labels, so the calibration set must be an open MOS set, with its own license terms.
- For aesthetics, a zero-shot teacher is too weak to label a training set.
- A teacher fine-tuned on the research mix would give better labels. [INFERENCE] Its labels likely carry the same license question as its training data ([datasets/iqa.md section 6.1](../datasets/iqa.md#61-license-verdict-for-the-whole-q-realign-training-mix)).
- The LEAF student saw unlabeled images from the same datasets as the tests. Labels on a different image pool (for example PD12M) will likely transfer less well [INFERENCE].

### 7.2 Current API teacher candidates and label costs

The [prompt pack](synthetic-label-prompts.md) defines separate technical-quality and aesthetics ratings, with a joint-call variant for a cost ablation. It documents how these proposed teacher judgments differ from Q-Align's learned label probabilities.

The [teacher benchmark and cost study](synthetic-label-teachers.md) compares the five leading distinct models in the checked MMMU-Pro results, with GPT-6 Luna as an extra low-cost control. It includes verified provider prices and estimates for 1K, 10K, and 250K images. These are general visual-reasoning results and planning estimates, not controlled IQA evidence.

Proposed first test: Gemini 3.8 Flash for bulk labels, Claude Opus 5.5 for a second opinion, and Astra plus Luna as accuracy and cost controls. Select on a human-rated development set before generating the full dataset. Under the study's central token assumptions, 250K Gemini labels plus a 10% Opus audit cost about $1,007 batch before retries. Keep aesthetics separate, and do not equate a generated JSON distribution with Q-Align label-token probabilities.

### 7.3 Filling the low-quality range and adding domain controls

The [low-quality and control dataset study](../datasets/low-quality-and-controls.md) checks real capture faults, distortion sources, and diagram/CG controls through multiple Exa and Hugging Face research rounds. The strongest new sources are LIVE-Meta VI-UGC/VizWiz, RealBlur, and SIDD. PlotQA train, CLEVR train, and owned diagrams/renders supply non-photo controls.

The proposed 10K pilot uses 50% diverse photos, 30% controlled variants, 10% authentic faults, and 10% domain controls. These proportions are an experiment, not established evidence of better IQA. The 250K plan preserves those shares, but the 25K authentic-fault target requires extra mined photos beyond the eligible VI-UGC training pool. Keep originals and derivatives in one split, and reserve current IQA tests.

The [pilot builder](../../README.md) implements that allocation using Megalith-CC0, human defect votes from public VizWiz training images, PlotQA, CLEVR, and owned controls. It adds separate development and test reserves. The broad-photo component is initially a seeded source sample; reviewed quality bands and teacher labels remain pending. Assembly is not evidence of improved model accuracy.

Two corrections matter for the labels. Distortion severity is not an exact perceived-quality rank; mild edits can improve an image. A clean diagram or render can also have high quality and aesthetics. Assign zero to its photo-selection target when the product excludes that domain, while keeping quality ratings separate. MSC helps test low aesthetics, not technical IQA, and its source images include CC BY/CC BY-SA despite the CC0 release label.

## 8. What this means for a Q-ReAlign re-train

1. Every addition in this file helped its own domain and hurt at least one other set in some study. Plan for regressions and report them.
2. Loss choice changes whether more data helps. Ranking or fidelity losses (DeQA, VisualQuality-R1) handled mixed datasets better than raw-score targets (Q-Insight).
3. Cross-task mixing (IQA + IAA + VQA) is supported by Q-Align Table 7. But AVA is large and can pull on authentic IQA (LIVE-C). Tune task weights on a small model first (Q-SiT approach).
4. Candidate additions in order of evidence: PIPAL (A, with known regressions), DQ-495K as a low-weight auxiliary task (B), target-domain video slices chosen by MDS-style selection (A), then MOSAIQ labels when released.
5. Treat RED-20k and HPDv3++ as aesthetic preference sources. Keep them away from technical-quality targets.

## 9. Rules for our own data ablations

- Add one dataset family per run. Keep the base model, start checkpoint, optimizer, visual budget, and total updates fixed.
- Use a random matched-size subset as the control for any selection method.
- Group all versions of one source (crops, distortions, transcodes, generated variants) in the same split. Deduplicate new data against all test sets first. See [split rules](../evaluation/protocol.md#3-split-rules).
- End every arm with the same score-format data so auxiliary QA data does not take over the 5-word interface.
- Report per-dataset rows, regressions, and bootstrap intervals ([protocol](../evaluation/protocol.md#5-confidence-intervals)). A better mean alone is not a better unified model.
