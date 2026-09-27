# Video Quality Datasets (VQA)

This file lists human-rated video quality datasets: natural and user-generated (UGC) video, streaming and gaming, HDR, AI-generated video, video quality instruction data, and data-selection resources. It gives size, labels, split rules, and license terms. Model results live in [models/vqa.md](../models/vqa.md).

Last verified: 2026-09-27

## Legend

- `#videos`: rated videos, not source clips, unless stated.
- `License / terms`: what the owner states. Labels, code, and video media can have different terms. `Not stated` means we found no terms. It does not mean the data is free to use.
- `Access`: open download, request form, gated, or test-only.
- `Commercial use?`: Yes, No (NC), or Unclear. "LIVE notice" means the UT Austin copyright notice. It allows use "for any purpose" with attribution, but it does not address the rights of the third-party video content. We mark these as Unclear.
- `Status`: `verified` = primary page checked on 2026-09-27. `from dossier` = taken from prior research and not re-checked.
- Correlation numbers use `PLCC / SRCC`.

## 1. Natural and UGC video

| Dataset | Year / venue | #videos | Labels | Content | Split / grouping notes | License / terms | Access | Commercial use? | Status |
|---|---|---:|---|---|---|---|---|---|---|
| **LSVQ** (LIVE-FB Large-Scale Social Video Quality). [paper](https://arxiv.org/abs/2011.13544) · [release](https://github.com/baidut/PatchVQ) | 2020 arXiv; CVPR 2021 | 38,811 labeled videos (~39K per abstract) + ~117K space-time patches | Crowd MOS; ~5.5M ratings; patch MOS | Internet Archive and YFCC100M UGC | Official files: train 28,056 / test 7,182 (`labels_train_test.csv`) + test-1080p 3,573 (`labels_test_1080p.csv`). **In the ONE-ALIGN training mix.** | LIVE notice (UT Austin 2020); cite papers | Request form; data moved to Globus (2026 README update) | Unclear | verified |
| **KoNViD-1k**. [paper](http://datasets.vqa.mmsp-kn.de/archives/papers/Hosu-Konvid-1k.pdf) · [release](https://database.mmsp-kn.de/konvid-1k-database.html) | 2017 QoMEX | 1,200 (8 s each) | Crowd MOS (1-5), per-rater votes | YFCC100M Creative Commons Flickr videos | No official split. Papers use repeated random 80/20 splits. | "Freely available to the research community"; cite 2 refs. Source videos carry their own CC licenses. | Open download (zip; OSF mirror) | Unclear | verified |
| **KonVid-150k** (A and B). [release](https://database.mmsp-kn.de/konvid-150k-vqa-database.html) | 2021 IEEE Access | A: >150,000 (5 ratings each). B: 1,576 (>=89 ratings each) | Crowd MOS; raw votes | YFCC100M public-domain / CC videos | Use B as a clean test set. A has noisy labels. Check overlap with KoNViD-1k and LSVQ (all use YFCC100M). | "Freely available to the research community" | Open download (A videos 176 GB) | Unclear | verified |
| **LIVE-VQC**. [paper](https://arxiv.org/abs/1803.01761) · [release](https://www.colorado.edu/lab/live/live-video-quality-challenge-vqc-database) | 2018 ICIP; 2019 TIP | 585 | Crowd MOS; >205,000 opinions from 4,776 people | Phone and camera capture; 101 devices, 80 users | No official split. Common cross-dataset test. | LIVE notice (UT Austin 2018) | Request form (Microsoft Forms) | Unclear | verified |
| **YouTube-UGC**. [paper](https://arxiv.org/abs/1904.06457) · [MOS paper](https://arxiv.org/abs/2002.12275) · [release](https://media.withyoutube.com/ugc-dataset) | 2019 MMSP; MOS 2020 | ~1,500 clips, 20 s | Crowd MOS (1-5, 100+ raters) for full clip and three 10 s chunks; DMOS for Gaming, Sports, Vlog | 15 YouTube categories incl. HDR, VR, gaming, 4K | No official split. Source set for MDS-VQA. | Videos: CC BY (page). MOS file terms: Not stated | Open download (Google Cloud) | Yes for videos (CC BY, attribution); labels unclear | verified |
| **CVD2014**. [paper](https://doi.org/10.1109/TIP.2016.2562513) · [release](https://zenodo.org/records/2646315) | 2016 TIP | 234 | Lab MOS; raw ratings | 5 scenes, 78 cameras | Scene-level grouping. Small. | Zenodo: CC BY 4.0. University portal notes that single videos may have use limits. | Open download (Zenodo) | Unclear | verified |
| **LIVE-Qualcomm**. [release](https://www.colorado.edu/lab/live/live-qualcomm-mobile-capture-video-quality-database) | 2017 | 208 | Lab MOS (39 subjects) | 8 phones, 6 in-capture distortion types | Group by scene. | LIVE notice (UT Austin 2017) | Request form | Unclear | verified |
| **DIVIDE-MaxWell** (MaxWell contains DIVIDE-3k). [DOVER paper](https://arxiv.org/abs/2211.04894) · [MaxWell paper](https://arxiv.org/abs/2305.12726) · [release note](https://vqassessment.github.io/DOVER/get_divide_dataset/) · [HF](https://huggingface.co/datasets/teowu/DIVIDE-MaxWell) | DIVIDE-3k: ICCV 2023. MaxWell: 2023 | 4,543 (MaxWell). DIVIDE-3k is a subset. | Aesthetic, technical, overall scores; 13 quality-related factors; >2M opinions | In-the-wild UGC | Merged official split: 3,634 train / 909 val. Never use DIVIDE-3k to test a model trained on MaxWell. | Conflict: HF card says Apache-2.0. ExplainableVQA code repo uses S-Lab License 1.0 (non-commercial). NTU data registry lists custom terms (from dossier). | Open download (HF zip + label txt) | Unclear | verified (NTU terms from dossier) |
| **FineVD**. [paper](https://arxiv.org/abs/2412.19238) · [release](https://huggingface.co/datasets/IntMeGroup/FineVD) | CVPR 2025 | 6,104 | MOS for color, noise, artifact, blur, temporal, overall (805,728 ratings, from dossier); text descriptions | Bilibili UGC | FineVQ paper uses 4:1:1 train/val/test (from dossier). Keep all clips of one source in one split. | Not stated | Gated on HF. Card asks you to email name, organization, and purpose. | Unclear | verified |
| **KVQ** (Kwai short-form). [paper](https://arxiv.org/abs/2402.07220) · [2024 challenge](https://github.com/lixinustc/KVQ-Challenge-CVPR-NTIRE2024) · [2025 release](https://github.com/lixinustc/KVQE-Challenge-CVPR-NTIRE2025) | CVPR 2024; data released 2025-05-20 | 4,200 (600 source + 3,600 processed) | Expert scores (1-5, step 0.5); ranked labels for hard pairs | Short-form platform video, 9 scenarios; special effects and processing workflows | Group each source with its 6 processed copies. Challenge split 2,926 / 420 / 854 (from dossier). | Not stated (no LICENSE in either repo) | Open download (Google Drive) | Unclear | verified (split from dossier) |
| **LIVE-ShareChat**. [paper](https://arxiv.org/abs/2401.02794) · [release](https://www.colorado.edu/lab/live/live-sharechat-video-quality-assessment-database) | 2024 | 600 (8 s, portrait) | MOS | Indian social media short video | No official split noted. | LIVE notice (UT Austin 2024) | Request form | Unclear | verified |
| **UltraVQA**. [paper](https://arxiv.org/abs/2602.16856) | 2026 arXiv | Not stated in abstract (~40K per search summary, unverified) | At least 3 raters per video on motion quality, motion amplitude, aesthetic, content, clarity; GPT rationales | UGC | Not stated | Not stated | No release found | Unclear | verified (paper only) |
| **TREND-10K**. [paper](https://arxiv.org/abs/2609.26187) · [HF (uploader not confirmed)](https://huggingface.co/datasets/BOUNCINGBALL1234/TREND-10K) | 2026 arXiv | 10,000 | Technical, aesthetic, and AIGC-trace scores | Trending short video, short drama, AIGC; plus a "static part" from public datasets | Static part reuses public datasets. Dedupe against your test sets. | HF card: MIT. We could not link the uploader to the authors. | Open on HF (unconfirmed) | Unclear | verified |

## 2. Streaming, gaming, and delivery

| Dataset | Year / venue | #videos | Labels | Content | Split / grouping notes | License / terms | Access | Commercial use? | Status |
|---|---|---:|---|---|---|---|---|---|---|
| **LIVE-NFLX-II**. [release](https://www.colorado.edu/lab/live/live-nflx-ii-subjective-video-qoe-database) | 2018 | 420 | Continuous-time and retrospective QoE (65 subjects; 9,750 of each) | 15 contents x 7 network traces x 4 adaptation algorithms; rebuffering | Group by the 15 contents. QoE, not pure picture quality. | LIVE notice (UT Austin 2018) | Request form / Box | Unclear | verified |
| **Waterloo SQoE-III**. [paper](https://www.ece.uwaterloo.ca/~z70wang/publications/TB_QoE_database.pdf) · [release](https://ieee-dataport.org/open-access/waterloo-streaming-quality-experience-database-iii) | 2018 IEEE TBC | 450 (from 20 sources) | QoE 0-100 (34 subjects) | 6 ABR algorithms x 13 network profiles | Group by 20 sources. | CC BY 4.0 (IEEE DataPort license field) | Open (free IEEE account) | Yes | verified |
| **Waterloo SQoE-IV**. [paper](https://arxiv.org/abs/2008.08804) · [release](https://ieee-dataport.org/open-access/waterloo-streaming-quality-experience-database-iv) | 2020 | 1,350 (from 5 sources) | QoE 0-100 | 2 encoders, 9 traces, 5 ABR algorithms, 3 devices | Only 5 sources. Group by source. | CC BY 4.0 (IEEE DataPort license field) | Open (free IEEE account) | Yes | verified |
| **LIVE-APV Livestream** ("LIVE-Livestream"). [release](https://www.colorado.edu/lab/live/live-apv-livestream-video-quality-assessment-database) | TIP 2022 | 367 (97 contents) | Lab MOS (>14,000 opinions, 40 subjects) | 4K live sports; 7 distortion types | Group by content. MDS-VQA target domain. | LIVE notice | Request form | Unclear | verified |
| **LIVE-YT-HFR**. [release](https://www.colorado.edu/lab/live/live-youtube-high-frame-rate-live-yt-hfr-database) | Year not checked | 480 (16 contents, 6 frame rates, 5 compression levels) | Lab MOS | Frame rate plus compression | Group by content. | LIVE notice | Request form | Unclear | verified |
| **LIVE-YT-Gaming**. [paper](https://arxiv.org/abs/2204.00128) · [release](https://live.ece.utexas.edu/research/LIVE-YT-Gaming/index.html) | 2022 | 600 (59 games) | Online MOS (18,600 ratings, 61 subjects) | UGC gameplay | Group by game. | LIVE notice (UT Austin 2021) | Request form | Unclear | verified |
| **LIVE-Meta MCG** (mobile cloud gaming). [paper](https://arxiv.org/abs/2305.17260) · [release](https://www.colorado.edu/lab/live/live-meta-mobile-cloud-gaming-video-quality-assessment-database) | 2023 | 600 (30 sources x 20 ladders) | Lab MOS | Mobile cloud gaming | Group by the 30 sources. | LIVE notice (UT Austin 2023) | Request form | Unclear | verified |
| **GameScope**. [paper](https://arxiv.org/abs/2605.01272) · [release](https://rajeshsureddi.github.io/GameScope/) | ICIP 2026 | 4,048 | Crowd MOS (0-100, ~37 ratings per video); clarity, artifact, immersion attributes | UGC and PS5 gameplay; H.264, H.265, AV1 | Group by game and source clip. | Not stated | Paper says "publicly available"; download path not checked | Unclear | verified |
| **CGVDS** (cloud gaming). [repo](https://github.com/stootaghaj/CGVDS) | ~2020 (unverified) | Not verified (sources disagree) | MOS on several scales | Cloud gaming, H.264 NVENC, 60 fps | MDS-VQA target domain. | Not stated (no LICENSE) | Google Drive | Unclear | unverified |

## 3. HDR video

HDR labels lose meaning if you convert to 8-bit SDR frames. Plan HDR-preserving input before you train on these sets.

| Dataset | Year / venue | #videos | Labels | Content | Split / grouping notes | License / terms | Access | Commercial use? | Status |
|---|---|---:|---|---|---|---|---|---|---|
| **LIVE-HDR**. [release](https://www.colorado.edu/lab/live/live-hdr-video-quality-assessment-database) | ICIP 2022 | 310 (31 contents x 10 ladders) | Lab MOS (>20,000 opinions); two room light conditions | HDR10 professional content | Group by 31 contents. | LIVE notice | Request form | Unclear | verified |
| **YouTube-SFV+HDR**. [paper](https://arxiv.org/abs/2406.05305) · [release](https://media.withyoutube.com/sfv-hdr) | ICIP 2024 | 2,030 SDR + 2,000 HDR (with HDR2SDR versions); 300 HDR rated in a lab | Crowd MOS (SDR, HDR2SDR); lab MOS (HDR) | 5 s portrait short-form, 10 categories | SDR and HDR2SDR MOS come from different tests. Do not pool raw MOS. MDS-VQA target domain. | Videos: CC BY (page). MOS file terms: Not stated | Open download | Yes for videos (CC BY); labels unclear | verified |
| **CHUG**. [paper](https://arxiv.org/abs/2510.09879) · [release](https://www.colorado.edu/lab/live/chug-crowdsourced-user-generated-hdr-video-quality-dataset) | ICIP 2025 | 5,992 (856 sources) | Crowd MOS (211,848 ratings) | HDR UGC, bitrate/resolution ladders | Group by source. Included in Beyond8Bits. | Conflict: LIVE page uses LIVE notice; GitHub repo LICENSE is CC BY-NC-SA 4.0 | Open (S3) + request form | Unclear (treat as NC) | verified |
| **BrightVQ**. [paper](https://openaccess.thecvf.com/content/WACV2026/papers/Saini_BrightRate_Quality_Assessment_for_User-Generated_HDR_Videos_WACV_2026_paper.pdf) · [release](https://github.com/shreshthsaini/BrightVQ) | WACV 2026 | 2,100 (300 sources) | Crowd MOS (73,794 ratings) | HDR UGC | Group by source. Predecessor of Beyond8Bits. | CC BY-NC-SA 4.0 (LICENSE file and HF tag) | Open (S3, HF) | No (NC) | verified |
| **Beyond8Bits** (HDR-UGC-44K). [paper](https://arxiv.org/abs/2603.00938) · [release](https://github.com/shreshthsaini/Beyond8Bits) | CVPR 2026 | Release: 41,419 clips (12,918 crowd + 22,584 Vimeo transcodes + 5,917 references). Paper: ~44K from 6,861 sources. | Crowd MOS 0-100 (~1.46M ratings, ~35 per video, SUREAL) | HDR UGC (crowd + Vimeo) | Released split by source (70/10/20): 28,987 / 4,151 / 8,281. The paper text says 70/20/10. Contains CHUG and BrightVQ, so not independent of them. | Metadata and CSVs: CC BY 4.0. Videos keep original terms; crowd videos are for non-commercial research only. | Open (S3, UT Box) | No (NC) for videos | verified |

## 4. AI-generated video (AIGC)

Hold out prompts and generators, not random clips. Keep technical quality, motion, and prompt alignment as separate targets. Generated videos can also carry the generator's terms of use, which no card below addresses.

| Dataset | Year / venue | #videos | Labels | Content | Split / grouping notes | License / terms | Access | Commercial use? | Status |
|---|---|---:|---|---|---|---|---|---|---|
| **T2VQA-DB**. [paper](https://arxiv.org/abs/2403.11956) · [release](https://github.com/QMME/T2VQA) | 2024 | 10,000 (9 T2V models, 1,000 prompts) | Single MOS (27 subjects) | Early T2V models | Group by prompt. | Not stated (no LICENSE) | Open download (Google Drive) | Unclear | verified |
| **GAIA**. [paper](https://arxiv.org/abs/2406.06087) · [release](https://github.com/zijianchen98/GAIA) | NeurIPS 2024 D&B | 9,180 (18 T2V models) | Action subject, action completeness, action-scene interaction MOS (971,244 ratings) | Action prompts | Group by action prompt. | Repo LICENSE Apache-2.0 (covers repo files: prompts, labels). Video terms Not stated. | Open download (SJTU / Baidu netdisk) | Unclear | verified |
| **LGVQ**. [paper](https://arxiv.org/abs/2407.21408) · [release](https://github.com/zczhang-sjtu/UGVQ) | 2024 | 2,808 (6 models, 468 prompts) | Spatial, temporal, text-video alignment MOS | Structured prompts (foreground, background, motion) | Group by prompt. | Not stated (no LICENSE) | Open download (Google Drive / Baidu) | Unclear | verified |
| **VideoFeedback** (VideoScore data). [paper](https://arxiv.org/abs/2406.15252) · [HF](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback) | 2024 | 37.6K (11 generators) + real videos | 1-4 scores: visual quality, temporal consistency, dynamic degree, text alignment, factual consistency | Generated + some real videos | HF splits: annotated 32,901 train / 680 test; real 4,000 / 80. | HF card: Apache-2.0 | Open (HF) | Yes per card; generator terms not addressed | verified |
| **VideoFeedback2** (VideoScore2 data). [paper](https://arxiv.org/abs/2509.22799) · [HF](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2) | 2025 | 27,168 (22 models, 2,933 prompts) | Visual quality, text alignment, physical / common-sense scores + rationales | Generated | Group by prompt. | HF card: Apache-2.0 | Open (HF) | Yes per card; generator terms not addressed | verified |
| **AIGVQA-DB**. [paper](https://arxiv.org/abs/2411.17221) · [release](https://github.com/wangjiarui153/AIGV-Assessor) · [HF](https://huggingface.co/datasets/IntMeGroup/AIGVQA) | CVPR 2025 | 36,576 (15 models, 1,048 prompts) | Static quality, temporal smoothness, dynamic degree, text-video correspondence; scores + pair ranks (~370K ratings, from dossier) | Generated | Hold out prompts and generator families. Dynamic degree is not quality. MDS-VQA target domain. | Not stated (no LICENSE; HF card has no license) | Open (HF, Baidu) | Unclear | verified |
| **Q-Eval-100K** (video part). [paper](https://arxiv.org/abs/2503.02357) · [HF](https://huggingface.co/datasets/kucklily/Q-Eval-100K) | CVPR 2025 | 40K videos (plus 60K images, see [iqa.md](./iqa.md)) | Visual quality and alignment MOS (960K annotations total) | Text-to-video outputs | Only training-set labels released. | CC BY-NC 4.0 (card text) | Open (HF); test labels withheld | No (NC) | verified |
| **AIGVE-60K** (LOVE). [paper](https://arxiv.org/abs/2505.12098) · [repo](https://github.com/IntMeGroup/LOVE) · [HF](https://huggingface.co/datasets/IntMeGroup/AIGVE-60K) | 2025 arXiv; ICML 2026 | 58,500 (30 models, 3,050 prompts, 20 tasks) | 120K MOS (perception, correspondence) + 60K QA pairs | Generated | Group by prompt and task. | HF card: Apache-2.0 (also mirrored at `anonymousdb/AIGVE-60K`) | Open (HF) | Yes per card; generator terms not addressed | verified |
| **HVEval / HVEval+**. [paper](https://arxiv.org/abs/2607.16742) · [HF (HVEval)](https://huggingface.co/datasets/wsj-sjtu/HVEval) | HVEval+: 2026 arXiv | HVEval+: 20K (24 models, 1K prompts) | 60K MOS (spatial, temporal, text-video) + 60K preference pairs + 20K QA | Human-centric generated video | HF HVEval ships train / test folders. HVEval+ release not confirmed. | HVEval HF card: CC BY-NC-SA 4.0 | Open (HF) for HVEval | No (NC) | verified |
| **VGA-Bench / VGA-BenchV2** (public release). [V2 paper](https://arxiv.org/abs/2608.25452) · [HF](https://huggingface.co/datasets/BestiVictoryLab/VGA-Bench) · [code](https://github.com/BestiVictory/VGA-Bench) | 2026 | Public release: 8,349 videos, 1,016 bilingual prompts. V2 paper corpus: >60K videos from 12 models, 36,000 task-level labels (not all released). | Aesthetic quality (10 attributes); aesthetic tags; generation quality on 31 dimension files (13,440 records, 6,793 videos) | Generated; video aesthetics and generation quality | Join on `video_id`; rows can repeat. Score range differs per question; `-1` means not applicable, not "worst". Missing dimension is not zero. | Data: CC BY-NC 4.0 (HF card). Code repo: Apache-2.0. | Open (HF) | No (NC) for data | verified |

## 5. Video quality instruction and description data

These sets train an assessor to describe or answer questions. They are not independent human MOS. Many reuse videos from the MOS sets above.

| Dataset | Year / venue | Size | Labels | Source videos | Notes | License / terms | Access | Commercial use? | Status |
|---|---|---:|---|---|---|---|---|---|---|
| **VQA² instruction data**. [paper](https://arxiv.org/abs/2411.03795) · [repo](https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment) | 2024 arXiv; ACM MM 2025 | 157,755 QA pairs (paper). HF: stage-2 streaming 2.1K; stage-3 14.3K mix / 11.6K UGC-only | Instruction QA for UGC, streaming, AIGC | Reuses existing VQA sets (exact list unverified) | Check overlap with LSVQ test before use. | Code repo: Apache-2.0. HF dataset cards: Not stated. | Open (HF) | Unclear | verified (source list unverified) |
| **OmniVQA-Chat-400K / OmniVQA-MOS-20K**. [paper](https://arxiv.org/abs/2505.22543) | AAAI 2026 | 400K instruction items; 20K MOS | Machine-built instructions (technical, aesthetic) + human MOS | Not verified | Release not found. | Not stated | No release found | Unclear | verified (paper only) |
| **FineVD QA** | CVPR 2025 | See FineVD | Yes/no and "which" QA on FineVD dimensions | FineVD | Ships with FineVD (gated). | Not stated | Gated | Unclear | verified |

## 6. Data-selection resource: MDS-VQA

[MDS-VQA](https://arxiv.org/abs/2603.11525) (CVPR 2026 Highlight, [code](https://github.com/Multimedia-Analytics-Laboratory/MDS-VQA)) picks unlabeled target videos that are hard for the current model and diverse in content. An auxiliary failure predictor, trained next to the frozen base model, scores difficulty. The method then mixes difficulty with content diversity to pick a small subset for human labels.

- Code: Apache-2.0. The repo ships YouTube-UGC and YouTube-SFV SDR MOS files and split lists. You must fetch videos from the original datasets under their own terms.
- Released weights: [base VQR1-7B-YouTubeUGC](https://huggingface.co/hollow404/VQR1-7B-YouTubeUGC), [failure predictor](https://huggingface.co/hollow404/MDS-VQA-Failure-Predictor), [active fine-tune](https://huggingface.co/hollow404/MDS-VQA-Active-Finetuning), [CLIP features](https://huggingface.co/datasets/hollow404/MDS-VQA_sdr_diversity_features).
- Target domains in the paper: CGVDS, LIVE-Livestream, YouTube-SFV SDR, YouTube-SFV HDR2SDR, AIGVQA-DB. Source domain: YouTube-UGC.

Result (author-reported, 5% labels per target, mean of 6 test sets, PLCC / SRCC): MDS-VQA 0.749 / 0.722 vs random 0.715 / 0.686 vs no adaptation 0.683 / 0.651. MDS wins on the mean but not on every domain: on HDR2SDR its SRCC is below random, and on AIGVQA-DB core-set, FreeSel, ALCS, and NoiseStability all beat it. Full table and protocol: [training/data-mixture-evidence.md](../training/data-mixture-evidence.md#4-selecting-data-under-a-label-budget).

## 7. Overlap and leakage

- MaxWell is a superset of DIVIDE-3k. Treat them as one dataset.
- Beyond8Bits contains CHUG and BrightVQ. Do not report CHUG or BrightVQ as independent tests of a Beyond8Bits-trained model.
- KoNViD-1k, KonVid-150k, and LSVQ all draw from YFCC100M. LSVQ also uses Internet Archive video. [INFERENCE] Some source videos may overlap. Dedupe by Flickr ID and by perceptual hash before you use KoNViD-1k as a cross-dataset test for an LSVQ-trained model.
- KVQ, Waterloo SQoE, LIVE-NFLX-II, LIVE-HDR, CHUG, BrightVQ, Beyond8Bits, and LIVE-Meta MCG make many processed copies from few sources. Split by source, never by clip.
- TREND-10K "static part" and VQA² reuse public datasets. Check them against every test manifest.
- AIGC sets: the same prompt goes to many generators. Split by prompt, and hold out whole generators for a transfer test.
- YouTube-SFV+HDR SDR and HDR2SDR MOS came from separate tests. Their raw scales do not match.

For split and calibration rules see [evaluation/protocol.md](../evaluation/protocol.md). Diagnostic video benchmarks (for example Q-Bench-Video, LongVQUBench) are in [evaluation/benchmarks.md](../evaluation/benchmarks.md).

## 8. Notes for Q-ReAlign

- LSVQ is the only video source in the ONE-ALIGN recipe (KonIQ, SPAQ, KADID, AGIQA-20K, AVA, LSVQ). Keep the official LSVQ test and test-1080p files untouched.
- Clean external video tests with no training overlap: LIVE-VQC, YouTube-UGC, KonVid-150k-B (after dedupe), LIVE-YT-Gaming.
- Candidate video additions with fine-grained or perspective labels: FineVD (gated), DIVIDE-MaxWell (license conflict), KVQ (license not stated). For generated video: AIGVQA-DB (license not stated), AIGVE-60K and VideoFeedback2 (Apache-2.0 cards). HDR only after an HDR-preserving input path.
- NC data (Beyond8Bits videos, BrightVQ, Q-Eval-100K, HVEval, VGA-Bench) can support research runs. Do not ship weights trained on it for commercial use without legal review.
- Staged use and compute: see [training/experiment-plan.md](../training/experiment-plan.md). Models trained on these sets: [models/vqa.md](../models/vqa.md), [models/unified-and-backbones.md](../models/unified-and-backbones.md). Video aesthetics also appears in [datasets/iaa.md](./iaa.md).

## 9. Corrections and open items

Corrections against the prior sources:

- VGA-Bench code repo now has an Apache-2.0 LICENSE ([GitHub](https://github.com/BestiVictory/VGA-Bench)). The dossier said it had no LICENSE.
- BrightVQ: the repo LICENSE and HF tag are CC BY-NC-SA 4.0 ([repo](https://github.com/shreshthsaini/BrightVQ)), not CC BY-NC 4.0 as some summaries say.
- Beyond8Bits: the released split is 70/10/20 by source (28,987 / 4,151 / 8,281), while the paper text says 70/20/10 ([repo](https://github.com/shreshthsaini/Beyond8Bits)).
- Waterloo SQoE-III and SQoE-IV: IEEE DataPort metadata states CC BY 4.0 ([SQoE-IV](https://ieee-dataport.org/open-access/waterloo-streaming-quality-experience-database-iv)).
- DIVIDE-MaxWell: the HF card declares Apache-2.0 ([card](https://huggingface.co/datasets/teowu/DIVIDE-MaxWell)), while the ExplainableVQA repo uses the non-commercial S-Lab License 1.0 ([repo](https://github.com/VQAssessment/ExplainableVQA)). The dossier cited only the NTU custom terms and S-Lab. We mark the terms as a conflict.
- YouTube-UGC: the release page says MOS is "available for all video clips" and links a MOS file ([page](https://media.withyoutube.com/ugc-dataset)). The dossier asked to verify the scored manifest.
- LSVQ access moved to Globus in a 2026 README update ([repo](https://github.com/baidut/PatchVQ)). The official split sizes (28,056 / 7,182 / 3,573) were counted from the released label files.
- LIVE databases now sit on colorado.edu and use a Microsoft Forms access request ([index](https://www.colorado.edu/lab/live/image-video-quality-assessment-live)).
- FineVD: the HF API reports automatic gating, but the card still asks for an email with name, organization, and purpose ([card](https://huggingface.co/datasets/IntMeGroup/FineVD)).
- MDS-VQA Table 3 ([paper](https://arxiv.org/html/2603.11525v1)): on AIGVQA-DB, 0.787 / 0.788 (PLCC / SRCC) is core-set selection. FreeSel is 0.785 / 0.789. The best on that domain is ALCS at 0.792 / 0.789. Four methods beat MDS-VQA (0.769 / 0.769) there.

Left unverified:

- CGVDS size and year (sources disagree).
- KVQ challenge split 2,926 / 420 / 854; FineVD 4:1:1 split and 805,728 ratings; AIGVQA-DB ~370K ratings; DIVIDE-MaxWell NTU custom terms (all from dossier).
- UltraVQA size and release; OmniVQA release; HVEval+ release; TREND-10K HF uploader identity; VQA² source video list.
- GameScope license and exact download path.
