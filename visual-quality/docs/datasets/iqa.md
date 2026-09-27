# Image Quality Assessment (IQA) Datasets

This file lists image datasets for technical quality: synthetic distortions, authentic (in-the-wild) photos, AI-generated images, instruction and description data, and label-alignment work. Aesthetics data is in [iaa.md](iaa.md). Video data is in [vqa.md](vqa.md).

Last verified: 2026-09-27

## How to read this file

- `Labels` gives the label type and, where known, the number of raters or ratings.
- `Raw dist.` tells you if per-image rating counts, raw votes, or a standard deviation (std) are in the release. You need these for soft-label training (see [../training/methods.md](../training/methods.md)).
- `License / terms` is about the annotations and the release as a whole. Photo or media rights are a separate question. A license tag on a mirror or on code does not change the source terms.
- `Not stated` means we found no license or terms on the official page. It does not mean the data is free to use.
- `Commercial use?`: `Yes`, `No (NC)`, or `Unclear`. `Unclear` means you must ask the authors or a lawyer.
- `Status`: `verified` = we checked a primary source on 2026-09-27. `from dossier` = the fact comes from the prior research dossier and was not re-checked.
- `ONE-ALIGN` marks the datasets that the Q-ReAlign README lists for its ONE-ALIGN recipe: KonIQ, SPAQ, KADID, AGIQA-20K (IQA), AVA (IAA) and LSVQ (video). The reference [`configs/onealign.yaml`](https://github.com/Q-Future/Q-ReAlign/blob/main/configs/onealign.yaml) omits AGIQA-20K (from dossier). Record the real manifest when you reproduce it.
- Benchmark results are not in this file. For controlled "add this data" results, see [../training/data-mixture-evidence.md](../training/data-mixture-evidence.md). For split and metric rules, see [../evaluation/protocol.md](../evaluation/protocol.md).

## 1. Synthetic and algorithm-processed distortions

| Dataset | Year / venue | Images | Labels | Raw dist. | Content | Split notes | License / terms | Access | Commercial use? | Release | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [LIVE IQA R2](https://live.ece.utexas.edu/research/quality/subjective.htm) | 2006, TIP | 779 distorted from 29 refs (count unverified) | DMOS, lab study | DMOS; std Not stated | Kodak + licensed photos; JPEG, JP2K, noise, blur, fast-fading | No official split. Split by reference image. | UT Austin notice: use, copy, modify, distribute "for any purpose" with notice and citation. Some source images from Visual Delights Inc. may not be used outside the database. | Open download (free) | Unclear (third-party photo limit) | [Copyright notice](https://www.colorado.edu/lab/live/copyright) | verified |
| CSIQ | 2010, JEI | 866 distorted from 30 refs | DMOS; 5,000 ratings from 35 observers | DMOS; std Not stated | 6 distortion types, 4–5 levels | No official split. Split by reference. | Not stated | Open download | Unclear | [Page](https://s2.smu.edu/~eclarson/csiq.html) | verified |
| TID2013 | 2015, SPIC | 3,000 distorted from 25 refs | MOS 0–9 from 971 observers (524,340 pair comparisons) | MOS; std Not stated on page | Kodak crops; 24 types x 5 levels | No official split. Split by reference. | Not stated on download page | Open download (RAR) | Unclear | [Page](https://www.ponomarenko.info/tid2013.htm) | verified |
| [KADID-10k](https://arxiv.org/pdf/2001.08113.pdf) (ONE-ALIGN) | 2019, QoMEX | 10,125 distorted from 81 refs | DMOS 1–5; 30 degradation category ratings per image (crowd) | Yes: DMOS + variance; raw ratings CSV | 25 types x 5 levels | No official split. Split by reference (81 refs). | "Freely available to the research community." No formal license. | Open download | Unclear | [Page](https://database.mmsp-kn.de/kadid-10k-database.html) | verified |
| KADIS-700k | 2019 | 140,000 refs; 700,000 distorted (you generate them) | None (distortion type and level only) | n/a | Same 25 distortion types; MATLAB generator | Pretraining only. Group by reference. | Same page as KADID; no formal license | Open download (~44.6 GB) | Unclear | [Page](https://database.mmsp-kn.de/kadid-10k-database.html) | verified |
| [PIPAL](https://arxiv.org/abs/2007.12142) | 2020, ECCV | ~29k from 250 refs (116 per ref; 40 distortion types) | Elo-based MOS from 1.13M pairwise judgments | MOS only | Restoration and GAN/SR outputs, plus classic distortions | NTIRE 2021 public train and valid sets. Test labels not public. Split by reference and algorithm. | Not stated | Open (Google Drive) | Unclear | [Repo](https://github.com/HaomingCai/PIPAL-dataset) | verified |
| [DiffIQA](https://arxiv.org/abs/2503.11221) | 2025, CVPR | ~180,000 | Full-reference: worse / similar / better than reference | Per-pair labels | Diffusion-enhanced images | Full-reference task. Not for no-reference training as is. | Non-commercial academic research only. No redistribution. Images on request. | Open link (Google Drive) with agreement | No (NC) | [Repo](https://github.com/ChrisDud0257/AFINE) | verified |
| [JPEG AIC2026](https://arxiv.org/abs/2607.22783) | 2026, arXiv | 9,618 distorted from 70 sources | No human MOS yet. Levels mapped to 0.2–4.0 JND with CVVDP. | n/a | Classic + learned codecs, high-fidelity range | Fine-grained compression test | CC BY-SA 4.0 (DARUS record) | Open download | Yes (attribution, share-alike) | [DARUS](https://doi.org/10.18419/DARUS-6156) | verified |
| [MCIQA-2K](https://arxiv.org/abs/2609.14495) | 2026, arXiv | 2,000 | 3 dims: color smearing, semantic color misalignment, naturalness | Not stated | Colorization outputs from 5 models | Not stated | Not stated | Authors say public; link not checked | Unclear | [arXiv](https://arxiv.org/abs/2609.14495) | verified (abstract only) |
| [AU-IQA](https://arxiv.org/abs/2508.05016) | 2025, ACM MM | 4,800 | MOS | Not stated | AI-enhanced UGC: super-resolution, low-light enhancement, denoising, by 9 models | Not stated. Group by source UGC image. | MIT (repo LICENSE, GitHub API) | Open (download link in repo) | Yes (MIT); source photo rights not checked | [Repo](https://github.com/WNNGGU/AU-IQA-Dataset) | verified (license, size) |

## 2. Authentic (in-the-wild) photos

| Dataset | Year / venue | Images | Labels | Raw dist. | Content | Split notes | License / terms | Access | Commercial use? | Release | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BID | 2011, TIP | 586 (some docs list 582–585) | MOS 0–5 | Not stated | Real blur and camera faults | No official split | Not stated | Official host not verified. Mirrors exist. | Unclear | Not verified | unverified |
| [CLIVE / LIVE Challenge](http://arxiv.org/pdf/1511.02919v1.pdf) | 2015–16, TIP | 1,162 | Continuous-scale MOS; >350,000 scores, ~175 per image, crowd | MOS + variance | Mobile photos, mixed real distortions | No official split. Common: 80/20 random, 10 repeats. | UT Austin notice: use "for any purpose" with notice and citation | Open (form before download) | Unclear (photo rights not stated) | [Page](https://live.ece.utexas.edu/research/ChallengeDB/index.html) | verified |
| [KonIQ-10k](https://arxiv.org/abs/1910.06180) (ONE-ALIGN) | 2018 / 2020, TIP | 10,073 | 5-point ACR; 1.2M ratings from 1,459 crowd workers | Yes: c1–c5 counts, MOS, SD, z-score MOS | YFCC100M photos (paper) | Official split 7,058 / 1,000 / 2,015 (train / val / test) | "Freely available to the research community." No formal data license. Code MIT. | Open download | Unclear | [Page](https://database.mmsp-kn.de/koniq-10k-database.html), [repo](https://github.com/subpic/koniq) | verified |
| KonIQ++ | 2021, BMVC | 10,073 (KonIQ images) | Quality MOS + defect votes: blur, artifacts, color, contrast, other | Yes: counts per defect, SD, votes | Same as KonIQ | Same split as KonIQ | Free for research (page) | Open CSV | Unclear | [Page](http://database.mmsp-kn.de/koniq-image-defects-database.html) | verified |
| [SPAQ](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf) (ONE-ALIGN) | 2020, CVPR | 11,125 | MOS + 5 attributes (brightness, colorfulness, contrast, noise, sharpness), scene labels, EXIF | MOS; raw Not stated | Photos from 66 smartphones | No official split. Papers use random splits. | Release Readme.txt: education and research only; no commercial product without permission (seen in a mirror) | Open (Baidu, Google Drive) | No (without permission) | [Repo](https://github.com/h4nwei/SPAQ) | verified |
| [FLIVE / PaQ-2-PiQ](https://arxiv.org/abs/1912.10088) | 2020, CVPR | ~40,000 images + ~120,000 patches | ~4M ratings from ~8k subjects; image and patch MOS | MOS; raw Not stated | Mixed real photos | Keep patches in the same split as the parent image | Data terms Not stated. PaQ-2-PiQ code shows CC BY-NC-SA 4.0 badge (code only). | Open (download notebook) | Unclear | [Project](https://baidut.github.io/PaQ-2-PiQ/), [data](https://github.com/niu-haoran/FLIVE_Database) | verified |
| KonX | 2023 | 420 sources x 3 resolutions | Resolution-specific MOS + SD | SD yes | KonIQ-derived images | Same source must share split. Overlaps KonIQ. | Free to research community; media terms Not stated | Open download | Unclear | [Page](https://database.mmsp-kn.de/konx.html) | from dossier |
| [PIQ23](https://arxiv.org/abs/2304.05772) | 2023, CVPRW | 5,116 | Pairwise-derived scores for Details, Exposure, Overall | Not stated | Portraits, 50 scenes, 100+ smartphones | Official device split and scene split | Non-commercial research only. No commercial use of images or derived data. Removal on request. | Request form (DXOMARK) | No (NC) | [Repo](https://github.com/DXOMARK-Research/PIQ2023) | verified |
| [HRIQ](https://arxiv.org/abs/2401.16087) | 2024, ICASSP | 1,120 at 2880x2160 (+ 2 smaller versions) | Lab MOS; 175 subjects (~25 ratings per image per dossier) | Not stated | High-res authentic photos | Only the largest size was rated. Smaller copies are not new samples. | Not stated | Open (Google Drive) | Unclear | [Repo](https://github.com/jarikorhonen/hriq) | verified |
| [UHD-IQA](https://arxiv.org/abs/2406.17472) | 2024, AIM/ECCV W | 6,073 at 4K | MOS from 10 expert raters, each twice (20 ratings per image) | MOS only in metadata | High-quality, high-aesthetic Pixabay photos | Official train / val / test in `set` column | Images CC0. Labels: "freely available to the research community", no formal license. | Open download | Images Yes; labels Unclear | [Page](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html) | verified |

## 3. AI-generated images (AGI / AIGC)

These datasets often mix technical quality with prompt alignment. Keep these as separate targets.

| Dataset | Year / venue | Images | Labels | Raw dist. | Content | Split notes | License / terms | Access | Commercial use? | Release | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [AGIQA-1K](https://arxiv.org/abs/2303.12618) | 2023, arXiv | 1,080 | Normalized MOS | MOS only | Diffusion outputs | No official split. Hold out prompts. | MIT (repo) | Open (Baidu, Google Drive) | Yes (MIT); generator terms not checked | [Repo](https://github.com/lcysyzxdxc/AGIQA-1k-Database) | verified |
| [AGIQA-3K](https://arxiv.org/abs/2306.04717) | 2023, arXiv | 2,982 from 6 models | Perception MOS + alignment MOS; 125,244 ratings (21 subjects x 2) | MOS + STD per dimension | GAN, AR, diffusion outputs | No official split. Hold out prompts and models. | MIT (repo) | Open (Quark, Google Drive) | Yes (MIT); generator terms not checked | [Repo](https://github.com/lcysyzxdxc/AGIQA-3k-Database) | verified |
| [AIGCIQA2023](https://arxiv.org/abs/2307.00211) | 2023 | 2,400 (6 models x 100 prompts) | MOS for quality, authenticity, correspondence | MOS only | T2I outputs | No official split | Repo: none. HF card metadata: Apache-2.0. | Open (HF, Baidu, Terabox) | Unclear | [Repo](https://github.com/wangjiarui153/AIGCIQA2023), [HF](https://huggingface.co/datasets/IntMeGroup/AIGCIQA2023) | verified |
| [AIGCIQA2023+](https://arxiv.org/abs/2405.07346) | 2024, arXiv | 2,400 (same images) | 7,200 MOS + 16,800 preference explanations | Not stated | Same as AIGCIQA2023 | Not a new image set | Code MIT; data terms Not stated | Via MINT-IQA repo | Unclear | [Repo](https://github.com/IntMeGroup/MINT-IQA) | from dossier |
| [AIGIQA-20K](https://arxiv.org/abs/2404.03407) (ONE-ALIGN) | 2024, CVPRW (NTIRE) | 20,000 from 15 models | 420,000 ratings from 21 subjects; one score mixing quality and alignment | MOS only | T2I outputs, varied CFG, steps, resolution | Paper: random 7:1:2 = 14,000 / 2,000 / 4,000 | ModelScope record: Apache License 2.0. HF mirror (strawhat) also tags Apache-2.0. | Open (ModelScope) | Yes per record; generator terms not checked | [ModelScope](https://www.modelscope.cn/datasets/lcysyzxdxc/AIGCQA-30K-Image) | verified |
| PKU-AIGIQA-4K | 2024 / ICCVW 2025 | ~4,000 | Subjective quality; T2I and I2I | Not stated | T2I + image-to-image | I2I uses a reference image. Do not mix with no-reference tests. | Repo MIT; media terms Not stated | Via repo links | Unclear | [Repo](https://github.com/jiquan123/AIGIQA4K) | verified (repo) |
| [EvalMuse-40K](https://arxiv.org/abs/2412.18150) | 2024 | 40,000 image-text pairs | Fine-grained image-text alignment | Not stated | T2I outputs | Alignment only. Not a quality label. | HF card: BSD-3-Clause. | Open (HF) | Unclear (generator terms) | [HF](https://huggingface.co/datasets/DY-Evalab/EvalMuse) | verified |
| [Q-Eval-100K](https://arxiv.org/abs/2503.02357) | 2025, CVPR | 60,000 images (+ 40,000 videos) | 960K MOS annotations for visual quality and alignment | Not stated | T2I and T2V outputs from many models | Only training-set annotations are released | CC BY-NC 4.0 (card text) | Open (HF) | No (NC) | [HF](https://huggingface.co/datasets/AGI-Eval-Official/Q-Eval-100K) | verified |
| [AGHI-QA](https://arxiv.org/abs/2504.21308) | 2025 | 4,000 from 400 prompts x 10 models | Quality, correspondence, visible and distorted body-part labels | Not stated | AI-generated human images | Hold out prompts | Not stated | Release not verified | Unclear | Not verified | verified (abstract only) |
| [Q-Real](https://arxiv.org/abs/2511.16908) | 2025 | 3,088 | Realism and plausibility, with grounding | Not stated | AI-generated images | Not stated | Not stated | Authors: release on publication; not found | Unclear | Not released | verified (abstract only) |

RichHF-18K (artifact and misalignment heatmaps on Pick-a-Pic images) is listed in [iaa.md](iaa.md) with the other preference and feedback data.

## 4. Instruction, description, and grounding data

These are not human MOS sets. Most text is written or expanded by a model. They reuse images from the datasets above, so check overlap with your test sets.

| Dataset | Year / venue | Size | Labels | Source images | License / terms | Access | Commercial use? | Release | Status |
|---|---|---|---|---|---|---|---|---|---|
| [Q-Pathway / Q-Instruct](https://arxiv.org/abs/2311.06783) | 2024, CVPR | Q-Pathway: 58K human feedbacks on 18,973 images. Q-Instruct: 200K instruction pairs (GPT-converted). | Free-text low-level descriptions, then QA | Mixed IQA sources | HF card metadata: Apache-2.0. Repo LICENSE: S-Lab 1.0 (non-commercial). README: free for research; commercial use needs prior permission. | Open (HF) | No (without permission) | [HF](https://huggingface.co/datasets/q-future/Q-Instruct-DB), [repo](https://github.com/Q-Future/Q-Instruct) | verified |
| [Co-Instruct-562K](https://arxiv.org/abs/2402.16641) | 2024, ECCV | 562K multi-image QA | LMM-merged descriptions + GPT-4V teacher answers for comparisons | Mixed IQA sources | HF card metadata: MIT. Repo LICENSE: S-Lab 1.0 (non-commercial). | Open (HF) | Unclear (conflict) | [HF](https://huggingface.co/datasets/q-future/Co-Instruct-DB) | verified |
| [DQ-495K / DataDepictQA](https://arxiv.org/abs/2405.18842) | 2024, arXiv | 495K task samples (not 495K rated images) | Ground-truth-informed brief and detailed assessments and comparisons | KADIS700K, BAPPS, PIPAL, KADID10K, DetailDescriptionLAMM | HF card: Apache-2.0. Source datasets keep their own terms. | Open (HF, ModelScope) + rebuild from sources | Unclear | [HF](https://huggingface.co/datasets/zhiyuanyou/DataDepictQA) | verified |
| [QGround-100K](https://arxiv.org/abs/2407.17035) | 2024, ACM MM | 100K triplets | Distortion masks + text; human part and GPT-4V part | KonIQ, SPAQ and others | HF card: CC BY-NC-SA 4.0 (dossier also saw S-Lab 1.0) | Open (HF) | No (NC) | [HF](https://huggingface.co/datasets/chaofengc/QGround-100K) | verified |
| [Q-Insight training data](https://arxiv.org/abs/2503.22679) | 2025, arXiv | ~7,000 KonIQ images (scores) + 7,000 single-distortion DQ-495K images (+1,000 test) | Scores + distortion type and level | KonIQ, DQ-495K | Inherits KonIQ and DQ-495K terms | Rebuild from sources | Unclear | [Repo](https://github.com/bytedance/Q-Insight) | verified |

Repackaged bundles:

- [`q-future/q-align-datasets`](https://huggingface.co/datasets/q-future/q-align-datasets) holds KonIQ, SPAQ, KADID, AVA, LIVE, CLIVE, CSIQ, BID, FLIVE, AGI-CGI and PIQ23 archives under an MIT tag. The MIT tag does not replace source terms. PIQ23 terms, for example, forbid redistribution and commercial use.
- [`chaofengc/IQA-PyTorch-Datasets`](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets) has a CC BY-SA 4.0 tag. That tag also does not replace source terms.

## 5. Label alignment across datasets

| Dataset | Year / venue | Size | Labels | License / terms | Access | Commercial use? | Release | Status |
|---|---|---|---|---|---|---|---|---|
| [MOSAIQ-500K](https://arxiv.org/abs/2609.20247) | 2026-09-17, arXiv | >500,000 images from 23 IQA datasets | Existing MOS mapped to one common scale with monotonic maps, anchored by a new subjective test. Within-dataset rank order is kept. | Not stated; inherits all 23 source terms | Score files and unifying code "To be posted" | Unclear | [Project](https://ivc.uwaterloo.ca/projects/unifying_iqa_datasets/), loader [IQA-Dataset](https://github.com/icbcbicc/IQA-Dataset) | verified |

The MOSAIQ result is in [../training/data-mixture-evidence.md](../training/data-mixture-evidence.md). Keep the calibration anchors out of any test set.

## 6. ONE-ALIGN image parts at a glance

| Dataset | Task | Images | Split to report | Folder in `q-future/q-align-datasets` | Commercial use? |
|---|---|---|---|---|---|
| KonIQ-10k | Authentic IQA | 10,073 | Official 7,058 / 1,000 / 2,015 | `koniq` | Unclear |
| SPAQ | Authentic IQA (smartphone) | 11,125 | No official split. Publish your seed and file list. | `spaq` | No (without permission) |
| KADID-10k | Synthetic IQA | 10,125 | No official split. Split by the 81 references. | `kadid10k` | Unclear |
| AIGIQA-20K | AIGC IQA | 20,000 | Paper 14,000 / 2,000 / 4,000 | Not in the mirror (it has `agi-cgi`). Get it from ModelScope. | Yes per ModelScope record |
| AVA | Aesthetics | ~255,500 | See [iaa.md](iaa.md) | `ava_images` (~34 GB) | Unclear |

LSVQ, the video part, is in [vqa.md](vqa.md). How the recipe mixes these sets is in [../training/methods.md](../training/methods.md) and [../training/experiment-plan.md](../training/experiment-plan.md).

### 6.1 License verdict for the whole Q-ReAlign training mix

Checked 2026-09-27. This is not legal advice. Each row links to the dataset row with the full terms.

| Part | Terms found | Research use | Commercial model |
|---|---|---|---|
| KonIQ-10k | "Freely available to the research community." No formal data license. YFCC100M photos with mixed CC licenses. | Yes | Unclear: needs permission |
| SPAQ | Release readme: education and research only; no commercial product without permission | Yes | No, without permission |
| KADID-10k | "Freely available to the research community." No formal license. | Yes | Unclear: needs permission |
| AGIQA-20K | ModelScope record: Apache-2.0 | Yes | Yes per record; generator terms not checked |
| AVA ([iaa.md](iaa.md)) | No official terms. Photos belong to DPChallenge users. | Yes (common practice) | Unclear: treat as No |
| LSVQ ([vqa.md](vqa.md)) | LIVE notice (UT Austin 2020); request form | Yes | Unclear: needs permission |
| Q-Align label JSONs (`training_sft/train_*.json`, `test_jsons/*.json`), used by `onealign.yaml` | Hosted in the [Q-Align repo](https://github.com/Q-Future/Q-Align), which uses S-Lab License 1.0 (non-commercial) | Yes | No. Rebuild labels from the original MOS files instead. |
| HF mirrors `q-future/q-align-datasets` and `teowu/LSVQ-videos` | Both tagged MIT. They repackage third-party images and videos, including SPAQ and PIQ23 (both NC). | Download convenience only | No. A mirror tag cannot relicense the source data. |

Verdict:

- **Research and evaluation: keep the full mix.** Every part allows research use. We need it to compare with Q-Align and Q-ReAlign on the same data.
- **A model we ship commercially: only AGIQA-20K is clear.** SPAQ and the Q-Align label JSONs are explicit no's. KonIQ, KADID, AVA, and LSVQ need written permission from the owners or legal review.
- **Rebuild labels ourselves.** Map the original MOS files to the 5 levels with our own code (we plan soft labels anyway). This removes the S-Lab dependency. It does not fix the image terms.
- **Distillation does not clean data.** [INFERENCE] A student trained on a research model's outputs likely carries the same question as the research model. Get advice before relying on it.
- **A clean-data track is possible but weaker.** Candidates: AGIQA-20K, TAD66K (Apache-2.0 release), MSC (CC0), UHD-IQA (CC0 images), AU-IQA (MIT), and synthetic distortions we generate ourselves on CC0 images for ranking-only training. [INFERENCE] It will likely score lower than the full mix on the standard tests. Measure the gap before deciding.

### 6.2 Exact size of the Q-Align release

Counted on 2026-09-27 from the label JSONs in the [Q-Align repo](https://github.com/Q-Future/Q-Align/tree/main/playground/data) (`training_sft/`, `test_jsons/`) and the HF file lists.

| Set | Train items | Test file | Test items |
|---|---:|---|---:|
| KonIQ-10k | 7,046 | `test_koniq.json` | 2,010 |
| SPAQ | 8,897 | `test_spaq.json` | 2,224 |
| KADID-10k | 8,106 | `test_kadid.json` | 2,000 |
| AVA | 235,598 | `test_ava.json` | 19,930 |
| LSVQ (videos) | 28,056 | `test_lsvq.json`, `test_lsvq_1080p.json` | 7,186; 3,573 |
| **Total** | **259,647 images + 28,056 videos** (`train_all.json` has 287,597 items, 106 fewer than the sum) | | |
| Cross-dataset tests | – | `agi.json` (AGIQA-3K), `livec.json`, `live.json`, `csiq.json`, `konvid.json`, `maxwell_test.json` | 2,982; 1,169; 982; 750; 1,200; 909 |

- AGIQA-20K is not in the Q-Align release. The Q-ReAlign README adds it (about 14,000 train images per the paper split).
- The HF mirrors hold about 59.8 GB of images (`q-future/q-align-datasets`, 11 archives; AVA is 34.4 GB) and 73.2 GB of LSVQ video (`teowu/LSVQ-videos`). The image mirror also has sets outside the ONE-ALIGN mix (FLIVE, PIQ23, BID, CSIQ, LIVE, LIVE-C, AGIQA).
- 91% of the training images are AVA (aesthetics). Only 24,049 are image-quality items.
- Some test counts differ from the dataset papers (for example LIVE 982 here vs 779 distorted images; CSIQ 750 vs 866). The interval table in [protocol.md section 5](../evaluation/protocol.md#5-confidence-intervals) uses paper sizes. Use the `n` of the file you actually score.

### 6.3 Open image pools for teacher-labeled training

To rebuild a set of this size with teacher labels, we need about 260K images with a wide quality range. Checked 2026-09-27 (HF tags and cards).

| Pool | Size | Terms | Quality range | Fit |
|---|---|---|---|---|
| [PD12M](https://huggingface.co/datasets/Spawning/PD12M) | 12.4M | Package CDLA-Permissive-2.0; images public domain or CC0 (authors say they cannot guarantee every item) | Curated as "highly aesthetic"; skews high | Good for aesthetics and clean references; few bad photos |
| [Megalith-CC0](https://huggingface.co/datasets/Spawning/megalith-cc0) / [Megalith-10m](https://huggingface.co/datasets/madebyollin/megalith-10m) | 2.39M (CC0 subset) / ~10M links | Flickr CC0, PD mark, US Gov, Flickr Commons; card MIT | "Unedited photographs"; amateur Flickr spread | Best match for authentic photo quality (KonIQ is also Flickr / YFCC100M) |
| [CommonCatalog CC-BY](https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by) | ~100M across all license splits (card); CC-BY split size not checked | CC BY per image (attribution needed); YFCC100M source, up to 4K, EXIF and device fields | Amateur photos, wide range | Good; same source family as KonIQ, so dedup against KonIQ and SPAQ tests |
| [DataComp-1B](https://huggingface.co/datasets/mlfoundations/datacomp_1b) | ~1.4B URLs | Metadata CC BY 4.0; images keep their own rights | Web images, all qualities | Wide range, but image rights unclear. Research only. |
| UHD-IQA, MSC, AU-IQA (this file and [iaa.md](iaa.md)) | 6,073; 10,426; 4,800 | UHD-IQA: CC0 images, labels research-only (no formal license). MSC: CC0 data. AU-IQA: MIT. | Rated by humans | Small calibration and test anchors |
| Self-made distortions (KADIS-style, 25 types x 5 levels) on CC0 references | Any | Ours | Full range by construction | Exact within-reference ranking labels for free; no MOS |

Not usable for training a shipped model: ImageNet and SA-1B (non-commercial terms), COCO (mixed Flickr CC incl. NC), ShareGPT4V (CC BY-NC 4.0 tag).

## 7. Label files and scales

Check these before you write a loader. A wrong score direction silently flips the correlation sign.

| Dataset | Label file / columns | Scale and direction | Status |
|---|---|---|---|
| LIVE IQA R2 | DMOS per image | DMOS: higher = worse | verified (type only) |
| CSIQ | DMOS in xlsx | DMOS: higher = worse | verified (type only) |
| TID2013 | MOS per image | 0–9, higher = better | verified |
| KADID-10k | `dmos.csv`: DMOS + variance; raw ratings CSV (~68.5 MB) | 1–5, higher = better (despite the name) | verified |
| CLIVE | MOS and variance files | Continuous scale; MOS higher = better. Range 0–100 is common use (unverified). | verified (files) |
| KonIQ-10k | Scores CSV: c1–c5 counts, c_total, MOS, SD, MOS_zscore | MOS 1–5 (5-point ACR), higher = better | verified |
| KonIQ++ | CSV: qmos, sd, votes, defect counts, degraded.amount | Quality MOS + defect vote counts | verified |
| SPAQ | MOS + attribute scores (xlsx) | Higher = better. Range 0–100 (unverified). | verified (file) |
| FLIVE | `labels_image.csv`, `all_patches.csv` | Image and patch MOS | verified (files) |
| UHD-IQA | Metadata with `quality_mos` and `set` | Higher = better | verified |
| AGIQA-1K | `AIGC_MOS_Zscore.xlsx`: prompt, normalized MOS | Z-score-based MOS | verified |
| AGIQA-3K | Columns 6–7 perception MOS and STD; columns 8–9 alignment MOS and STD | Normalized MOS | verified |
| AIGCIQA2023 | `mosz1`–`mosz3` (quality, authenticity, correspondence) | Z-score-based MOS | verified |
| AIGIQA-20K | MOS per image | Scale Not stated here; check release | verified (split only) |
| Q-Instruct | `cleaned_labels.json`, `qinstruct_qalign.json`, images tar (5.6 GB) | Text QA, no MOS | verified |

## Notes for Q-ReAlign work

- Split by source, not by file. Group distorted images by reference (LIVE, CSIQ, TID2013, KADID, PIPAL). Group patches with their parent image (FLIVE). Group prompts and generators (AGIQA, AIGIQA-20K, Q-Eval-100K).
- Overlap is common. KonIQ++, KonX, QGround, Q-Instruct and the Q-Insight data reuse KonIQ or SPAQ images. DQ-495K reuses KADID and PIPAL. Remove test images from any auxiliary set.
- Scales differ. KADID uses DMOS where high is good. LIVE and CSIQ use DMOS where high is bad. TID2013 uses 0–9. Normalize with care, or use MOSAIQ-style alignment when its files are out.
- Soft labels need raw data. KonIQ (c1–c5 counts), KADID (raw CSV) and AGIQA-3K (STD) support them. Most others give only MOS, or MOS and std.
- AIGIQA-20K mixes quality and alignment in one score. AGIQA-3K, AIGCIQA2023 and Q-Eval-100K keep them apart.
- Licenses that allow commercial use are rare. Only UHD-IQA images (CC0), JPEG AIC2026 (CC BY-SA 4.0), and the MIT or Apache-tagged AIGC sets have clear open terms. Even these do not settle generator or photo rights.
- Diagnostic benchmarks (Q-Bench, A-Bench, UniPercept-Bench) are in [../evaluation/benchmarks.md](../evaluation/benchmarks.md). Model results are in [../models/iqa.md](../models/iqa.md).
