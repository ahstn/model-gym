# Benchmarks and held-out test suites

This file lists diagnostic and understanding benchmarks for image quality, aesthetics, and video quality assessors, plus related challenges. It also lists the held-out MOS test sets that Q-Align, ONE-ALIGN, DeQA-Score, Q-Insight, DOVER, and Q-ReAlign report on.

Last verified: 2026-09-27

Metric rules, split rules, and bootstrap rules are in [protocol.md](protocol.md). Dataset licenses and sizes for MOS datasets are in the dataset docs: [IQA](../datasets/iqa.md), [IAA](../datasets/iaa.md), [VQA](../datasets/vqa.md).

## How to use these benchmarks

- Diagnostic benchmarks test perception, description, comparison, or defect detection. They do not replace MOS correlation. Report them in a separate table.
- Keep them held out. Do not train on them, and do not select checkpoints on their test splits. Use a dev split only where one exists.
- Many benchmarks reuse images or videos from public IQA and VQA datasets. Run the deduplication step in [protocol.md section 3](protocol.md#3-split-rules) before you call a result held out.
- Q-ReAlign is a score-only model. For multiple-choice benchmarks, you need a question-answering prompt path. Report that path and its prompt. Score-only models can still run the pairwise parts (compare two scores).
- Accuracy numbers from different prompt formats or frame counts are not comparable. Record the frame count (Q-Bench-Video uses 16 frames for video models and 8 for image models).
- `License / terms` covers the benchmark annotations. Media rights of the source images or videos are separate. `Not stated` means we found no license text.

## 1. Image quality understanding

| Benchmark | Year / venue | What it tests | Size | Metric | License / terms | Access | Commercial use? | How to use | Status |
|---|---|---|---|---|---|---|---|---|---|
| Q-Bench ([paper](https://arxiv.org/abs/2309.14181), [repo](https://github.com/Q-Future/Q-Bench), [HF A1](https://huggingface.co/datasets/q-future/Q-Bench-HF)) | 2023; ICLR 2024 (spotlight) | A1 low-level perception questions; A2 low-level descriptions; A3 score prediction on IQA datasets (softmax over quality words). | A1 LLVisionQA: 2,990 images, one question each. A2 LLDescribe: 499 images with expert descriptions. | A1 accuracy; A2 GPT-assisted comparison with golden descriptions; A3 PLCC / SRCC. | Code: S-Lab License 1.0 (non-commercial). HF card: Not stated. | Open download (GitHub release, HF). Also in VLMEvalKit. | No (NC) for code; data Unclear. | Run A1 dev and test as a perception check. Skip A3 for MOS claims; use the MOS suite in section 6. | verified |
| Q-Bench+ / Q-Bench2 ([paper](https://arxiv.org/abs/2402.07116), [HF](https://huggingface.co/datasets/q-future/q-bench2)) | 2024; IEEE TPAMI | Perception and description for image pairs (compare two images). | LLVisionQA+: 1,999 pairs. LLDescribe+: 450 pairs. | Accuracy; description scores. | HF card: MIT. Repo code: S-Lab License 1.0. | Open download. | Unclear (MIT card, NC code, third-party images). | Pairwise perception check. Compare with pairwise accuracy from scores. | verified |
| A-Bench ([paper](https://arxiv.org/abs/2406.03070), [repo](https://github.com/Q-Future/A-Bench), [HF](https://huggingface.co/datasets/q-future/A-Bench)) | 2024; ICLR 2025 | Can an LMM judge AI-generated images? P1 semantic understanding; P2 technical quality, aesthetics, generative distortion. | 2,864 AI-generated images from 16 text-to-image models (P1 1,408; P2 1,456 QA). | Accuracy. | HF card: CC BY 4.0. Repo license: Not stated. | Open download; HF-format copy is gated (auto). | Unclear (CC BY 4.0 annotations; generator terms vary). | Use P2 as a held-out AIGC-quality check. P1 tests semantics, not quality. | verified |
| MICBench ([paper](https://arxiv.org/abs/2402.16641), [repo](https://github.com/Q-Future/Co-Instruct), [HF](https://huggingface.co/datasets/q-future/MICBench)) | 2024; ECCV 2024 (oral) | Open-ended quality comparison over groups of 3 or 4 images. | 2,000 multiple-choice questions (dev 1,004; test 996). | Accuracy. | HF card: MIT. Repo code: S-Lab License 1.0. | Open download. | Unclear. | Multi-image comparison check. Do not train on Co-Instruct-562K and then claim MICBench as held out without overlap checks. | verified |
| 2AFC-LMMs ([paper](https://arxiv.org/abs/2402.01162), [repo](https://github.com/h4nwei/2AFC-LMMs)) | 2024; IEEE TCSVT | Two-alternative forced choice between images, coarse to fine. Global scores by MAP estimation from pairs. | Pair subsets drawn from 8 IQA datasets (for example CSIQ, KADID-10k, KonIQ-10k, SPAQ). | Consistency, accuracy, correlation. | Repo license: Not stated. Source datasets keep their own terms. | Open code; you supply the source datasets. | Unclear. | Fine-grained ordering probe (same content, near levels). Useful for hard-label vs soft-label studies. | verified |
| ViDA-UGC-Bench ([paper](https://arxiv.org/abs/2508.12605), [project](https://whycantfindaname.github.io/ViDA-UGC/), [MIPI code](https://github.com/DYEvaLab/ViDA-MIPI-code)) | 2025; MIPI 2025 challenge at ICCV 2025 | Distortion grounding (boxes), low-level perception, reasoning quality description on UGC images. | 476 images; 6,149 QA; 2,567 multiple-choice; 3,106 grounding items. | Accuracy; grounding scores; description scores. | Code: Apache-2.0. Benchmark data: Not stated. | Project page; challenge release. | Unclear. | Local-defect check for crop or native-resolution ablations. | verified |
| SRIQA-Bench ([paper](https://arxiv.org/abs/2503.11221), [repo](https://github.com/ChrisDud0257/AFINE)) | 2025; CVPR 2025 | Quality of super-resolution outputs, where outputs can beat the reference. | 1,000 images: outputs of 10 SR methods; full paired comparisons. | Pairwise accuracy; correlation. | Repo license: custom (GitHub reports NOASSERTION); not read. | Open code and data links. | Unclear. | Stress test for generative restoration. Not a natural-image MOS test. | verified (license text not read) |
| Vista-Bench (Q-Probe, [paper](https://arxiv.org/abs/2601.15356)) | 2026 (arXiv, January) | Fine-grained local degradation in high-resolution IQA. The paper warns about "cropping implies degradation" bias and depth of field read as blur. | Not stated. | PLCC / SRCC (author-reported, from dossier). | Not stated. | Not stated. | Unclear. | Check for native-resolution crop ablations: crops must not lower scores by themselves. | verified (abstract); metric from dossier |

## 2. Aesthetics understanding

| Benchmark | Year / venue | What it tests | Size | Metric | License / terms | Access | Commercial use? | How to use | Status |
|---|---|---|---|---|---|---|---|---|---|
| AesBench ([paper](https://arxiv.org/abs/2401.08276), [repo](https://github.com/yipoh/AesBench), [HF](https://huggingface.co/datasets/qyuan/EAPD_release)) | 2024 | Aesthetic perception, empathy, assessment, and interpretation (AesP, AesE, AesA, AesI). | EAPD: 2,800 images (natural, artistic, AI-generated); 8,400 QA; 2,800 expert interpretations. | Accuracy; interpretation scores. | Code: Apache-2.0. HF data card: Not stated. | Open download. | Unclear (image rights). | Held-out aesthetic understanding check. It does not replace AVA correlation. | verified |
| UNIAA-Bench ([paper](https://arxiv.org/abs/2404.09619), [repo](https://github.com/KlingAIResearch/Uniaa), [HF](https://huggingface.co/datasets/zkzhou/UNIAA)) | 2024 | Aesthetic perception (6 attributes), description, and zero-shot assessment from output logits. In-domain and in-the-wild splits. | Not stated (not reconciled). | Accuracy; description scores; PLCC / SRCC. | Repo and HF card: Not stated. | Open download. | Unclear. | Use the in-the-wild split as a cross-domain aesthetics check. | verified (size not checked) |
| UniPercept-Bench ([HF](https://huggingface.co/datasets/Thunderbolt215215/UniPercept-Bench), [paper](https://arxiv.org/abs/2512.21675)) | 2025 release; ICML 2026 | Aesthetics, technical quality, and structure/texture ratings plus QA. | Not reconciled. | Accuracy; PLCC / SRCC. | Not stated. | HF. | Unclear. | Unified IQA/IAA check; see [unified models](../models/unified-and-backbones.md). | from dossier |

## 3. Video quality understanding

| Benchmark | Year / venue | What it tests | Size | Metric | License / terms | Access | Commercial use? | How to use | Status |
|---|---|---|---|---|---|---|---|---|---|
| Q-Bench-Video ([paper](https://arxiv.org/abs/2409.20063), [repo](https://github.com/Q-Future/Q-Bench-Video), [HF](https://huggingface.co/datasets/zhangzicheng/Q-Bench-Video)) | 2024; CVPR 2025 | Technical, aesthetic, temporal, and AIGC distortions in natural, AIGC, and CG videos. Yes/no, what/how, open-ended, and video-pair questions. | 1,800 videos; 2,378 QA (dev 1,192; test 1,186). | Accuracy; open-ended answers scored by the authors' protocol. | Repo and HF card: Not stated. | Open download. | Unclear. | Main held-out check for temporal understanding. Log frame count. | verified |
| VQA² data ([paper](https://arxiv.org/abs/2411.03795), [repo](https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment)) | 2024; ACM MM 2025 | Not a separate benchmark. Instruction data for video quality scoring and understanding. The authors evaluate understanding on Q-Bench-Video. | 157,755 instruction QA pairs in 3 subsets (UGC, streaming, AIGC). | n/a | Code: Apache-2.0. HF data cards: Not stated. | Open download. | Unclear. | If you train on VQA² data, check its videos against every VQA test set first. | verified |
| LongVQUBench ([paper](https://arxiv.org/abs/2607.01086), [project](https://longvqubench.github.io/), [HF](https://huggingface.co/datasets/Aarna004/LongVQUBench)) | 2026 | Long-video quality: local events (LQU), cross-event reasoning (CQR), global quality (GQU); inserted "needle" distortions. | 1,200+ videos; about 1,500 questions (validation / test). Durations up to about 2 hours (unverified). | Accuracy (multiple choice); open-ended scores. | HF card: CC BY-NC-SA 4.0. | Gated (manual approval). | No (NC). | Check if a frame-sampling policy misses short or late defects. Not a MOS test. | verified |
| Artifact-Bench ([paper](https://arxiv.org/abs/2605.18984), [repo](https://github.com/FrankYang-17/Artifact-Bench), [HF](https://huggingface.co/datasets/DogNeverSleep/Artifact-Bench)) | 2026 | AI-generated video artifacts: real vs AI classification, pairwise realism comparison, fine-grained artifact identification. Photorealistic, animated, and CG videos. | 1,350 videos; 1,100 annotated samples; 30 evaluation aspects. | Accuracy per task; overall score. | Repo README: academic research only; commercial use prohibited. HF card: `other`. | Open download. | No. | Use pairwise realism and artifact parts for generated-video defects. Do not teach a quality scorer that "AI-generated" means "bad"; authenticity is a different target. | verified |

## 4. Generative-media evaluator checks

| Benchmark | Year / venue | What it tests | Size | Metric | License / terms | Access | Commercial use? | How to use | Status |
|---|---|---|---|---|---|---|---|---|---|
| ProtoBias ([paper](https://arxiv.org/abs/2601.04946)) | 2026 | Prototypicality bias: does a metric prefer a typical-looking but wrong image over a correct but less typical one? Animals, Objects, Demography. | 45,400 image pairs (18,397 Animals; 14,019 Demography; 12,984 Objects). | Failure rate; pairwise misranking. | Paper (arXiv) is CC BY-SA 4.0. Data release and terms: Not stated. | Not stated. | Unclear. | Only for a prompt-conditioned evaluator (alignment, preference). Keep it out of the ONE-ALIGN score. | verified (release not found) |

## 5. Challenges

Challenge test labels are often hidden. Final numbers come from a server. Check each competition's data terms before training on its data.

| Challenge | Year | Task | Data | Metric | Access | Status |
|---|---|---|---|---|---|---|
| [AIM 2024 UHD-IQA](https://arxiv.org/abs/2409.16271) ([database](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html), [paper](https://arxiv.org/abs/2406.17472)) | 2024 (ECCV 2024 workshops) | No-reference IQA of 4K photos under a 50 GMACs budget. | UHD-IQA: 6,073 UHD-1 images, expert ratings. | PLCC, SRCC, KRCC, MAE, RMSE, GMACs. | Open download. Images: CC0 (Pixabay) per database page. Annotation terms: Not stated. | verified |
| [NTIRE 2024 AIGC quality](https://arxiv.org/abs/2404.16687) | 2024 (CVPR 2024 workshops) | Image track and video track for AI-generated content. | AIGIQA-20K (20,000 images, 15 models); T2VQA-DB (10,000 videos, 9 models). | PLCC, SRCC. | Challenge release; see dataset docs. | verified |
| [NTIRE 2024 short-form UGC VQA](https://arxiv.org/abs/2404.11313) ([code](https://github.com/lixinustc/KVQ-Challenge-CVPR-NTIRE2024)) | 2024 | Short-form UGC video quality on KVQ. | KVQ: 4,200 videos; split 2,926 / 420 / 854. | PLCC, SRCC, rank accuracy on pairs. | Challenge release; see [VQA datasets](../datasets/vqa.md). | verified |
| [NTIRE 2025 short-form UGC VQA and enhancement](https://arxiv.org/abs/2504.13131) ([code](https://github.com/lixinustc/KVQE-Challenge-CVPR-NTIRE2025)) | 2025 | Track 1 efficient KVQ quality assessment; track 2 diffusion SR (KwaiSR). | KVQ; KwaiSR. | Correlation plus efficiency (track 1). | Challenge release. | verified |
| [NTIRE 2025 XGC quality](https://arxiv.org/abs/2506.02875) | 2025 | UGC video, AI-generated video, and talking-head quality. | FineVD-GC, Q-Eval-Video, THQA. | PLCC, SRCC. | Challenge release. | verified |
| [NTIRE 2025 text-to-image model quality](https://arxiv.org/abs/2505.16314) | 2025 | Image-text alignment; structure distortion masks. | EvalMuse-40K (about 40k images, 20 models); EvalMuse-Structure (10,000 images with masks). | Main score (alignment); mask quality (structure). | Challenge release. | verified |
| [VQualA 2025 visual quality comparison](https://arxiv.org/abs/2509.09190) | 2025 (ICCV 2025 workshops) | Quality comparison by LMMs over single images, pairs, and groups. | Thousands of coarse-to-fine tasks; MICBench-style evaluation. | 2AFC accuracy; multiple-choice accuracy. | Challenge release. | verified |
| [MIPI 2025 detailed IQA](https://openaccess.thecvf.com/content/ICCV2025W/MIPI/papers/Liao_MIPI_2025_Challenge_on_Detailed_Image_Quality_Assessment__Methods_ICCVW_2025_paper.pdf) | 2025 (ICCV 2025 workshops) | Grounding, perception, and description on ViDA-UGC. | ViDA-UGC and ViDA-UGC-Bench (section 1). | Task scores. | Challenge release. | verified |
| [NTIRE 2026 RAIM track 1: professional IQA](https://arxiv.org/abs/2604.12512) ([data](https://github.com/narthchin/RAIM-PIQA), [server](https://www.codabench.org/competitions/12789/)) | 2026 (CVPR 2026 workshops) | Pick the better image in a high-quality pair and explain why. | Pairs of high-quality images; count Not stated. | Selection accuracy; explanation scores. | Repo: MIT (code). Data follows competition rules. | verified |
| [NTIRE 2026 X-AIGC quality](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/html/Liu_NTIRE_2026_X-AIGC_Quality_Assessment_Challenge_Methods_and_Results_CVPRW_2026_paper.html) | 2026 | Track 1 text-to-3D; track 2 image editing. | 3DGCQA-NTIRE (5,004 items, 14 models); IEQA (10,148 edited images, 22 models, 43 tasks). | Correlation-based main score. | Challenge release. | verified |

Use challenge sets as extra cross-dataset tests only where the test labels are public. Otherwise, use them for method ideas.

## 6. Held-out MOS test suites used by the Q-Align line of work

This table shows which MOS test sets each paper reports. `ID` = in-distribution (the model trains on the same dataset's train split). `X` = cross-dataset (no training on it). A blank cell means the paper does not report that set. Sizes, licenses, and access are in the dataset docs.

| Test set | Task | Content | Q-Align IQA (train KonIQ) | ONE-ALIGN (Q-Align Table 7) | DeQA-Score (train KonIQ) | Q-Insight (train KonIQ) | Q-ReAlign README | DOVER | Dataset doc |
|---|---|---|---|---|---|---|---|---|---|
| KonIQ-10k test | IQA | Authentic | ID | ID | ID | ID | ID | | [IQA](../datasets/iqa.md) |
| SPAQ | IQA | Authentic (phone) | X | ID | X | X | ID | | [IQA](../datasets/iqa.md) |
| KADID-10k | IQA | Synthetic | X | ID | X | X | ID | | [IQA](../datasets/iqa.md) |
| LIVE Challenge (LIVE-C, LIVE-Wild) | IQA | Authentic | X | X | X | X | | | [IQA](../datasets/iqa.md) |
| AGIQA-3K | IQA | AI-generated | X | X | X | X | "AGI" (dataset not named) | | [IQA](../datasets/iqa.md) |
| LIVE (IQA) | IQA | Synthetic | X (Table 4) | X | | | X | | [IQA](../datasets/iqa.md) |
| CSIQ | IQA | Synthetic | X (Table 4) | X | X | X | | | [IQA](../datasets/iqa.md) |
| PIPAL | IQA | Restoration outputs | | | X | X | | | [IQA](../datasets/iqa.md) |
| TID2013 | IQA | Synthetic | | | X (Table 4) | | | | [IQA](../datasets/iqa.md) |
| FLIVE | IQA | Authentic | | | X | | | | [IQA](../datasets/iqa.md) |
| AVA test | IAA | Photos | | ID | | | ID | | [IAA](../datasets/iaa.md) |
| LSVQ test | VQA | UGC | | ID | | | ID | ID | [VQA](../datasets/vqa.md) |
| LSVQ 1080p | VQA | UGC, high resolution | | ID (same dataset family) | | | | | [VQA](../datasets/vqa.md) |
| KoNViD-1k | VQA | UGC | | X | | | | reported | [VQA](../datasets/vqa.md) |
| MaxWell test | VQA | UGC | | X | | | | | [VQA](../datasets/vqa.md) |
| LIVE-VQC | VQA | UGC | | | | | | reported | [VQA](../datasets/vqa.md) |
| YouTube-UGC | VQA | UGC | | | | | | reported | [VQA](../datasets/vqa.md) |

Sources: [Q-Align paper](https://arxiv.org/abs/2312.17090) (IQA tables trained on KonIQ, including Table 4 with LIVE and CSIQ; Table 5 VQA trained on LSVQ with KoNViD-1k and MaxWell as cross sets; Table 7 for ONE-ALIGN trained on KonIQ + SPAQ + KADID + AVA + LSVQ); [DeQA-Score Table 3 and Table 4](https://arxiv.org/html/2501.11561); [Q-Insight Table 1](https://arxiv.org/html/2503.22679v2); [Q-ReAlign README](https://github.com/Q-Future/Q-ReAlign#results); [DOVER paper](https://arxiv.org/abs/2211.04894) (LSVQ, KoNViD-1k, LIVE-VQC, YouTube-UGC; DOVER cells say `reported` because the paper lists these sets but we did not map each to cross-dataset or fine-tuned). Status: verified.

Notes:

- The Q-Align, DeQA-Score, and Q-Insight papers train on the KonIQ train split. Their SPAQ and KADID numbers are cross-dataset. In ONE-ALIGN and Q-ReAlign, SPAQ and KADID are in-distribution. Do not rank these numbers together.
- The Q-Align paper and Q-ReAlign README write SRCC / PLCC. DeQA-Score and Q-Insight write PLCC / SRCC. Convert before copying.
- MaxWell contains DIVIDE (from dossier). If you train on DIVIDE-MaxWell, drop the MaxWell test set.
- AGIQA-20K and AGIQA-3K come from related work. Prompt or generator overlap is not verified. If you train on AGIQA-20K (the README recipe), treat AGIQA-3K as a weak cross-dataset test until you check overlap.

## 7. Proposed locked suite for this project

This is a plan, not a result. It keeps each task separate.

| Task | In-distribution (report separately) | Cross-dataset (never tuned on) | Diagnostic (separate table) |
|---|---|---|---|
| IQA | KonIQ-10k test, SPAQ test, KADID-10k test (split by reference image), AGIQA-20K test (split by prompt) | LIVE Challenge, LIVE, CSIQ, PIPAL, AGIQA-3K (after overlap check) | Q-Bench A1, Q-Bench+ pairs, A-Bench P2, 2AFC fine-grained pairs |
| IAA | AVA test | Out-of-domain aesthetics sets from [IAA datasets](../datasets/iaa.md) (for example the public ArtiMuse-10K test data) | AesBench, UNIAA-Bench in-the-wild split |
| VQA | LSVQ test, LSVQ 1080p | KoNViD-1k, LIVE-VQC, MaxWell test (only if no DIVIDE-MaxWell training) | Q-Bench-Video, LongVQUBench (access permitting), Artifact-Bench for generated video |

For every row, apply the rules in [protocol.md](protocol.md): raw PLCC, SRCC, KRCC, `n`, source-level bootstrap intervals, and at least 3 seeds for claimed gains.
