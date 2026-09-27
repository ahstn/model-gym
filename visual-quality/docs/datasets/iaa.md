# Image Aesthetics Assessment (IAA) and Preference Datasets

This file lists aesthetics datasets for photos and art, MLLM instruction data for aesthetics, and pairwise preference data for text-to-image (T2I) outputs. Technical-quality data is in [iqa.md](iqa.md). Video data is in [vqa.md](vqa.md).

Last verified: 2026-09-27

## How to read this file

- `Labels` gives the label type and the number of raters where known.
- `Raw dist.` tells you if per-image vote counts or per-rater scores are in the release. Aesthetics is subjective, so rating spread matters. Soft-label and personalized training need it (see [../training/methods.md](../training/methods.md)).
- `Themes / attributes` lists content, style, or attribute labels. Use them for stratified splits and per-theme tests.
- `License / terms` is for the annotations and release. Photo and artwork rights are a separate issue. A license tag on a Hugging Face (HF) mirror does not change the source terms.
- `Not stated` means we found no terms. It does not mean the data is free to use.
- `Commercial use?`: `Yes`, `No (NC)`, or `Unclear`.
- `Status`: `verified` = primary source checked on 2026-09-27. `from dossier` = from the prior research dossier, not re-checked.
- `ONE-ALIGN`: AVA is the only aesthetics set in the Q-ReAlign ONE-ALIGN recipe. The other parts are listed in [iqa.md](iqa.md) and [vqa.md](vqa.md).
- Model results are in [../models/iaa.md](../models/iaa.md). Data-mixture results are in [../training/data-mixture-evidence.md](../training/data-mixture-evidence.md).

## 1. Photo aesthetics

| Dataset | Year / venue | Images | Labels | Raw dist. | Themes / attributes | Split notes | License / terms | Access | Commercial use? | Release | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [AVA](https://doi.org/10.1109/CVPR.2012.6247954) (ONE-ALIGN) | 2012, CVPR | ~255,500 (exact count varies by mirror) | 1–10 votes from DPChallenge users; ~200+ votes per image | Yes: count per score 1–10 in `AVA.txt` | 2 semantic tags per image (66 tag IDs), challenge ID, 14 style labels on a subset | No single official split. Common test set is ~19.9k images (NIMA-style lists; unverified). State the list you use. | Official terms Not stated. Photos belong to DPChallenge users. | No official host. Unofficial downloaders and mirrors (e.g. 64 7z parts, ~32 GB). | Unclear | [Downloader](https://github.com/imfing/ava_downloader), [HF mirror](https://huggingface.co/datasets/TPagent/AVA) (Apache tag, mirror only) | verified |
| [AADB](https://arxiv.org/abs/1606.01621) | 2016, ECCV | 10,000 | Overall score + 11 attributes; 5 raters per image | Per-rater scores (rater IDs used for ranking in paper) | 11 attributes: balancing, color harmony, content, depth of field, light, motion blur, object, repetition, rule of thirds, symmetry, vivid color | Common split 8,500 / 500 / 1,000 (unverified) | "For research purpose only." Photos from Flickr (CC). Repo mentions patent US20170294010A1. | Open (repo links) | No (research only) | [Repo](https://github.com/aimerykong/deepImageAestheticsAnalysis) | verified |
| [FLICKR-AES](https://openaccess.thecvf.com/content_iccv_2017/html/Ren_Personalized_Image_Aesthetics_ICCV_2017_paper.html) | 2017, ICCV | 40,000 (+ REAL-CUR: 14 personal albums) | 1–5 ratings; 5 AMT workers per image; 210 workers | Yes: per-worker ratings | None beyond worker ID | Split by worker: train 35,263 images / 173 workers; test 4,737 images / 37 workers | "Research purpose only." Photos are Flickr CC. | Open (repo links) | No (research only) | [Repo](https://github.com/alanspike/personalizedImageAesthetics) | verified |
| [PARA](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Personalized_Image_Aesthetics_Assessment_With_Rich_Attributes_CVPR_2022_paper.html) | 2022, CVPR | 31,220 | Aesthetic score + 9 objective and 4 subjective attributes; 438 subjects with profile data | Yes: per-subject ratings | Scene and content attributes, rater personality and background | Paper split used for PIAA; check release files | Page footer forbids commercial use and copying without permission. No other license. | Open (Google Drive, Baidu; password on page) | No (NC) | [Page](https://web.xidian.edu.cn/ldli/en/dataset.html) | verified |
| TAD66K | 2022, IJCAI | ~66,000 | Theme-aware aesthetic scores; >= 1,200 annotations per image | Not stated | 47 themes | Labels grouped by theme in download | HF card and code: Apache-2.0. Photo rights not established. | Open (Google Drive, HF) | Unclear | [Repo](https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment) | verified |
| ICAA17K | 2023, ICCV | 17,726 | Color aesthetics score; ~1,500 opinions per image | Not stated | 30 color combinations | Not stated | Code: Apache-2.0 badge. Dataset terms Not stated. | Open (repo links) | Unclear | [Repo](https://github.com/woshidandan/Image-Color-Aesthetics-and-Quality-Assessment) | verified |
| [FGAesthetics](https://arxiv.org/abs/2603.03907) | 2026, CVPR | 32,217 in 10,028 series | Fine-grained, within-series aesthetic comparison | Not stated | Image series (same scene) | Not stated | Not stated | Repo has inference only; data not found | Unclear | [arXiv](https://arxiv.org/abs/2603.03907) | verified (abstract) |
| [MSC](https://doi.org/10.1038/s41597-026-06816-0) | 2026, Sci Data | 10,426 | 1–5 ratings; 100 ratings per image | Yes: raw rating CSV | Low-semantic objects; includes "ugly" images for full-range coverage | Not stated | OSF data and software: CC0 1.0. Images are public domain, CC0, CC BY, CC BY-SA, or the authors' own. Article text: CC BY-NC-ND. | Open (OSF) | Yes (data CC0; CC BY images need credit) | [OSF](https://doi.org/10.17605/OSF.IO/ZGSVJ) | verified |
| [DEAR](https://arxiv.org/abs/2512.05209) | 2025, arXiv | Card: 30,000 renderings (6 per scene). Paper: 100-image subset. | Pairwise rendering preference | Pair votes | MIT-Adobe FiveK scenes, edit styles | Card has train / test | HF license "other" | Open (HF) | Unclear | [HF](https://huggingface.co/datasets/vsevolodpl/DEAR) | verified |
| [HumanBeauty](https://arxiv.org/abs/2503.23907) | 2025, arXiv | 108K (50K with 12-dim labels + 58K from public sets) | Human-image aesthetics; 12 dimensions | Not stated | People and portrait scenes | Not stated | Code MIT. Data terms Not stated. | HF dataset returns 401 (gated or private) | Unclear | [Repo](https://github.com/KwaiVGI/HumanAesExpert) | verified |
| [RED-20k](https://arxiv.org/abs/2606.05778) | 2026, arXiv (ECCV 2026 per dossier) | ~20K | Reasoning-based aesthetic labels from VLM judges (Qwen3-VL-235B, Qwen2.5-VL-72B, GPT-5) | n/a | OpenImages, DIV2K, UHD-IQA, Unsplash Lite, proprietary data | n/a | Not stated | Not released | Unclear | [arXiv](https://arxiv.org/abs/2606.05778) | verified (paper) |

## 2. Art and design aesthetics

| Dataset | Year / venue | Images | Labels | Raw dist. | Themes / attributes | Split notes | License / terms | Access | Commercial use? | Release | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [BAID](https://arxiv.org/abs/2303.15166) | 2023, CVPR | 60,337 artworks | Score from >360,000 votes | Not stated | Artistic images | Check release | Dataset: CC BY-NC-ND 4.0 | Open (repo links) | No (NC) | [Repo](https://github.com/Dreemurr-T/BAID) | verified |
| [APDDv2](https://arxiv.org/abs/2411.08545) | 2024 | 10,023 | 85,191 annotations: total score + attribute scores and comments | Not stated | Painting categories and attributes | Check release | Conflict: repo README says CC BY-NC-ND 4.0; paper says CC BY 4.0. Use the stricter one. | Open (repo links) | No (NC) until resolved | [Repo](https://github.com/BestiVictory/APDDv2) | verified |
| [LAPIS](https://arxiv.org/abs/2504.07670) | 2025, CVPRW | 11,723 artworks | Per-user aesthetic ratings | Yes: per-user ratings | Title, artist, image attributes; personal rater attributes | For personalized IAA | Not stated | Request form | Unclear | [Repo](https://github.com/Anne-SofieMaerten/LAPIS), [request](https://sites.google.com/view/lapisdataset) | verified |
| [ArtiMuse-10K](https://arxiv.org/abs/2507.14533) | 2025 / CVPR 2026 | 10,000 | 8 aesthetic dimensions + holistic score, with expert text | Not stated | 5 main categories, 15 subcategories (photo, painting, design, AIGC and more) | Only the test set is public (`test.json` with `gt_score`) | HF metadata: Apache-2.0. README: full set by form, non-commercial research only, no redistribution. | Test: open (HF). Full: request form. | No (NC) | [HF](https://huggingface.co/datasets/Thunderbolt215215/ArtiMuse-10K), [form](https://forms.gle/SGn5osMkxdKQArhq8) | verified |

## 3. MLLM instruction and reasoning data for aesthetics

These sets hold model-written or human-written text about aesthetics. Most reuse images from Section 1. Check overlap with your test sets.

| Dataset | Year / venue | Size | Labels | Source images | License / terms | Access | Commercial use? | Release | Status |
|---|---|---|---|---|---|---|---|---|---|
| [AesMMIT](https://arxiv.org/abs/2404.09624) | 2024, ACM MM | 21,904 images; 88K human feedbacks; 409K instructions | Aesthetic critiques turned into QA | Mixed IAA sources | HF card: Apache-2.0. Repo Apache. | Open (HF, 11.5 GB zip) | Unclear (source photos) | [HF](https://huggingface.co/datasets/YiPo/AesMMIT) | verified |
| [UNIAA (IDCP data)](https://arxiv.org/abs/2404.09619) | 2024, arXiv | UNIAA-Bench + IDCP training data | Aesthetic perception, description, assessment | Mixed IAA sources | Not stated (no HF license tag) | Open (HF) | Unclear | [HF](https://huggingface.co/datasets/zkzhou/UNIAA) | verified |
| [AesCoT](https://arxiv.org/abs/2509.21871) | 2025, arXiv (Aes-R1) | Files `AesCoT-3K.jsonl` and `AesCoT-10K.jsonl` | Chain-of-thought aesthetic reasoning | Needs source IAA images | HF card: Apache-2.0. Source images keep their terms. | Open (HF, text only) | Unclear | [HF](https://huggingface.co/datasets/ssssmark/AesCoT) | verified |
| [Impressions](https://huggingface.co/datasets/SALT-NLP/Impressions) | 2023, EMNLP | Image + impression text | Viewer impressions and aesthetic evaluations | Photos and art | HF card: CC BY-SA 4.0 | Open (HF) | Unclear (share-alike; image rights) | [HF](https://huggingface.co/datasets/SALT-NLP/Impressions) | verified |
| [AesCanvas](https://arxiv.org/abs/2608.26713) | 2026, arXiv | 519,136 pairs from 54,300 images; 301 scenarios | Scenario-aware aesthetic preference | Not stated | Not stated | Not verified | Unclear | [arXiv](https://arxiv.org/abs/2608.26713) | verified (abstract) |

AesBench (2,800 images, evaluation only) is in [../evaluation/benchmarks.md](../evaluation/benchmarks.md).

## 4. Preference and feedback data for T2I images

These give pairwise or ranked choices, not MOS. They mix aesthetics, prompt fit and artifacts. Use them for reward models or pairwise losses, not as MOS targets.

| Dataset | Year / venue | Size | Labels | Raw dist. | Content | Split notes | License / terms | Access | Commercial use? | Release | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Pick-a-Pic v1 / v2](https://arxiv.org/abs/2305.01569) | 2023, NeurIPS | v2 mirror: 1,001,352 rows | Web-user pairwise choice (or tie) | Per-pair choice | SD-family outputs for user prompts | Official val / test in the original release | Dataset terms Not stated. Code (PickScore) MIT. | Official HF repos now return 401 (removed). Community mirrors only. | Unclear | [Mirror](https://huggingface.co/datasets/liuhuohuo2/pick-a-pic-v2), [webdataset](https://huggingface.co/datasets/sayakpaul/pickapic_v2_webdataset) | verified |
| [ImageRewardDB](https://arxiv.org/abs/2304.05977) | 2023, NeurIPS | 137K expert comparisons; 62.6K DiffusionDB images | Expert rankings and ratings | Per-group rankings | DiffusionDB prompts and images | Val: 412 prompts / 7.32K pairs. Test: 466 prompts / 7.23K pairs. | HF card: Apache-2.0 | Open (HF) | Yes per card; image terms follow DiffusionDB | [HF](https://huggingface.co/datasets/zai-org/ImageRewardDB) | verified |
| [HPDv2](https://arxiv.org/abs/2306.09341) | 2023, arXiv | 798,090 choices on 430,060 pairs | Human pairwise choice | Per-pair | Outputs from many T2I models + COCO | Official test set with prompts per style | HF card: Apache-2.0 | Open (HF) | Yes per card | [HF](https://huggingface.co/datasets/ymhao/HPDv2) | verified |
| [RichHF-18K](https://arxiv.org/abs/2312.10240) | 2024, CVPR | 17,760 (15,810 / 995 / 955) | Aesthetics, artifact, misalignment, overall scores; artifact and misalignment heatmaps; token labels | Per-image scores | Pick-a-Pic v1 images (not included) | Fixed train / dev / test | No LICENSE file (GitHub license: none). Repo archived. | Open labels; images need Pick-a-Pic | Unclear | [Repo](https://github.com/google-research-datasets/richhf-18k) | verified |
| [HPDv3](https://arxiv.org/abs/2508.03789) | 2025, ICCV | 1.08M text-image pairs; 1.17M comparisons | Pairwise choice; `choice_dist` for some sources | Partial: `choice_dist` null for Discord and HPDv2 pairs | MidJourney (331,955), curated HPDv2 (327,763), real photos (57,759), other generators | Official train / test | HF card: MIT. MidJourney images carry MidJourney terms. | Open (HF) | Unclear (image sources) | [HF](https://huggingface.co/datasets/MizzenAI/HPDv3) | verified |
| [HPDv3++](https://arxiv.org/abs/2606.14657) | 2026, arXiv | ~212K pairs (paper: 117K aesthetics + 95K text fidelity). Card: train_aes 100,463, train_tf 90,908, test_aes 5,720, test_tf 4,465. | Pairwise; kept only when a VLM judge and a reward-model judge agree | No (binary) | Qwen-Image outputs (~268 GB tar) | Separate aesthetics and text-fidelity splits. Preferred image is always `path1`: shuffle before training. | HF card: Apache-2.0 | Open (HF) | Unclear (labels are model-filtered; generator terms) | [HF](https://huggingface.co/datasets/Junjun2333/HPDv3-PlusPlus) | verified |

## 5. Label files and scales

| Dataset | Label file / fields | Scale | Status |
|---|---|---|---|
| AVA | `AVA.txt`: image ID, vote counts for 1–10, 2 semantic tag IDs, challenge ID. Tag and challenge name lists. Style and train/test lists. | 1–10, higher = better. Mean from counts. | verified |
| AADB | Overall score + 11 attribute scores per image | Overall 1–5 (unverified); attributes signed | verified (fields) |
| FLICKR-AES | Per-worker ratings; mean used as ground truth | 1–5 (paper normalizes to [0.2, 1]) | verified |
| PARA | Per-subject aesthetic score, attribute scores, subject profile | Not stated here; check release | verified (fields) |
| MSC | Raw rating CSV (100 ratings per image) | 1–5 | verified |
| ArtiMuse-10K (test) | `test.json` with `gt_score` | Not stated here | verified |
| HPDv3 | Pair records with `choice_dist` (null for Discord and HPDv2 pairs) | Binary choice or vote split | verified |
| HPDv3++ | Pair records; `path1` is always the preferred image; `aes` and `tf` subsets | Binary | verified |
| RichHF-18K | Scores, heatmaps and token labels keyed to Pick-a-Pic v1 images | Scores + masks | verified |

## 6. Which set for which goal

| Goal | Start with | Why |
|---|---|---|
| Reproduce ONE-ALIGN aesthetics | AVA | It is the recipe's only IAA set. It has full vote counts. |
| Test transfer to other photo taste | AADB, TAD66K, PARA | Different raters, attributes and themes from AVA |
| Personalized or rater-aware scores | FLICKR-AES, PARA, LAPIS | Per-rater labels and rater splits |
| Art and design | BAID, APDDv2, ArtiMuse-10K | Art-only content; ArtiMuse also has 8 dimensions |
| Open-license training data | MSC | Data under CC0; images under open licenses |
| Explain scores in text | AesMMIT, AesCoT | Instruction and reasoning text on IAA images |
| T2I reward or pairwise loss | HPDv3, HPDv3++, ImageRewardDB, HPDv2 | Large pairwise sets with open HF tags |
| Split taste from defects | RichHF-18K, HPDv3++ (`aes` vs `tf`) | Separate aesthetics, artifact and alignment labels |

## Notes for Q-ReAlign work

- AVA is the only aesthetics set in ONE-ALIGN. It has vote counts for scores 1–10, so you can train with full distributions, not just the mean.
- Split rules. AVA: state the test list, and keep one challenge from leaking across splits if you test by theme. FLICKR-AES, PARA, LAPIS: split by rater for personalized tests. HPDv3++ and Pick-a-Pic: split by prompt.
- Theme labels (AVA tags, TAD66K themes, ArtiMuse categories, AADB attributes) let you report per-theme scores. See [../evaluation/protocol.md](../evaluation/protocol.md).
- Scales differ. AVA uses 1–10, AADB and FLICKR-AES use 1–5, and preference sets are binary. Do not pool them as one MOS target.
- Only MSC (CC0 data) and a few HF sets with Apache or MIT tags have clear open terms. Most photo sets are research-only or do not state terms. For commercial work, treat AVA, AADB, FLICKR-AES, PARA, BAID, APDDv2, ArtiMuse and PIQ23 ([iqa.md](iqa.md)) as blocked until you get permission.
- Preference sets mix aesthetics with prompt fit. HPDv3++ splits them into `aes` and `tf` subsets. RichHF-18K gives separate aesthetics and artifact scores. Use these to test if a model confuses taste with defects.
- Model and backbone choices are in [../models/unified-and-backbones.md](../models/unified-and-backbones.md). The staged plan is in [../training/experiment-plan.md](../training/experiment-plan.md).
