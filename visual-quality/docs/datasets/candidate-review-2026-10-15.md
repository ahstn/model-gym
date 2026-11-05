# Dataset candidate review — 15 October 2026

This review covers the five requested dataset releases and each supported component of [IQA-Dataset](https://github.com/icbcbicc/IQA-Dataset). It compares them with the [acquired pilot](pilot-10k-assembly.md) and the existing dataset catalogue. The ranking is our assessment of usefulness for the next pilot steps. It is not a measured gain in model accuracy.

**The strongest new catalogue omission is CID2013. UHD-IQA and AU-IQA remain useful conditional tests. BIQ2021 has promising human supervision, but its release labels conflict. The deepfake release should be excluded.** None of these findings removes the need for human labels and the split checks in the [pilot preflight](../training/pilot-preflight-2026-09-29.md).

## Ranking the five direct dataset links

Rank includes label validity, new coverage, overlap, source rights, and access readiness. A low rank can reflect a release problem rather than a weak research design. Component datasets from the collection have their own [priority table](iqa-collection-coverage-2026-10-15.md#complete-comparison).

| Rank | Dataset | Best use | Value compared with our current sources | Decision |
|---|---|---|---|---|
| 1 | [UHD-IQA](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html) | Human-rated, high-resolution technical-quality evaluation | Adds subtle detail judgments beyond unlabelled Megalith and severe VizWiz faults. Already catalogued. | Keep for evaluation after metadata access, label terms, view policy, and overlap checks. It is not a severe-fault pool. |
| 2 | [AU-IQA](https://github.com/WNNGGU/AU-IQA-Dataset) | Restoration-artifact diagnostic | Adds learned enhancement outputs beyond our direct blur/noise/JPEG variants. Already catalogued. | Use only with KonIQ parent exclusions and archive checks. It has 400 parents, not 4,800 independent scenes. |
| 3 | [Low Quality Image Dataset](https://www.kaggle.com/datasets/pochihwu/low-quality-image-dataset) | Distortion recipe reference | Three of four operator families repeat our current coverage. Gamma/intensity is the main addition. | Reproduce suitable transforms on cleared parents. Do not acquire 11.12 GB merely to add more unlabelled variants. |
| 4 | [BIQ2021](https://www.kaggle.com/datasets/nisarahmedrana/biq2021) | Potential human-rated capture-fault source | Could add overall quality scores that VizWiz lacks. Its original design is more relevant than the preceding synthetic recipe. | **Hold gold labels.** Identical image bytes have conflicting MOS and reversed rank across releases. Promote only after reconciliation. |
| 5 | [Deepfake Detection Dataset 2026](https://www.kaggle.com/datasets/chuneeb/deepfake-detection-dataset-2026) | No suitable role in the current release | Its procedural avatars add little beyond owned controls; its claimed photorealistic fake provenance is unsupported. | **Exclude this release.** CSV only, incorrect or unsupported origin fields, and source-photo overlap across supplied splits. |

## Comparison with data we already have

“Acquired” means selected into `data/pilot-10k-v1`. “Catalogued” means documented; it does not imply local media or permission to train. The current report and manifest hashes were checked again. The pilot still has 10,000 train, 400 development, and 400 test rows, with `training_ready=false` and no generated IQA/IAA labels.

| Current source | State | Relative priority and reason |
|---|---|---|
| Megalith-CC0 | 5,400 selected photos | Keep as the broad photo pool. Label and audit the existing sample before adding more curated photographs. |
| VizWiz Quality Issues | 1,200 selected photos with defect votes | Still stronger than the Kaggle synthetic set for severe real capture faults. Votes are not MOS. Investigate the already catalogued LIVE-Meta quality labels as a family-aware join; do not count it as an independent photo pool. |
| Own distortions | 3,000 variants from 1,000 training parents | Prefer controlled expansion on these cleared parents to COCO-derived variants. Add unseen-parent holdout diagnostics and verify visible order. A larger transform parameter is not a human score. |
| PlotQA, CLEVR, owned controls | 480 + 480 + 240 selected images | Keep the known domain labels. The deepfake CSV does not supply reliable hard photorealistic controls. A clean render can still have high technical quality and aesthetics. |
| KonIQ, CLIVE, SPAQ, FLIVE | Catalogued human-rated benchmarks | Remain stronger established references for authentic IQA than an unresolved BIQ mapping. Preserve reserved tests and each source's terms. |
| KADID and PIPAL | Catalogued human-rated distortion/restoration sets | Their human judgments are more useful for calibration than unscored operator levels. AU adds newer restoration conditions, with known KonIQ dependence. |
| AVA and MSC | Catalogued aesthetics sources | None of the five requested releases supplies a verified new aesthetics target. High aesthetic selection in UHD does not make its technical MOS an aesthetic score. |

## Release assessments

### UHD-IQA

The [official page](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html) lists 6,073 photos and a 10.7 GB image package. The [paper v2](https://arxiv.org/html/2406.17472v2) specifies 4,269 train, 904 validation, and 900 test images. Images have width 3,840 pixels, preserved aspect ratio, and Lanczos downscaling. They are not all 3840 × 2160. The resized image is the rated view.

Ten experts rated each image twice using a technical-defect rubric. This gives twenty repeated ratings from ten people. Photos were selected from popular, aesthetically strong Pixabay content, then screened for synthetic or heavily edited content. This makes UHD useful for subtle quality errors and resizing tests, but weak for severe faults and low aesthetics.

Images are explicitly CC0. Label terms state research-community availability without a separate formal license. That wording does not establish either unrestricted use or an explicit non-commercial prohibition. The metadata schema lists MOS, split, photographer account, source URL, and original dimensions; raw individual ratings are not established.

The page and archive links were verified. A bounded metadata request failed because the archive host's TLS certificate had expired; no successful archive delivery was established. Split counts therefore come from the paper, not a newly parsed ZIP. Keep the official test reserved, then check photo identities and overlap with other Pixabay-derived sources such as KADID. Shared hosting is an overlap risk, not proof of duplicates.

### AU-IQA

The [paper v2](https://arxiv.org/html/2508.05016v2) describes 4,800 outputs from 400 high-MOS KonIQ parents: three synthetic damage tasks, with four enhancement models per task. Tasks cover super-resolution, low-light enhancement, and denoising. Five trained raters assign 1–5 quality scores. This adds restoration artifacts, not new authentic capture failures or a separate aesthetics target.

The [pinned repository](https://github.com/WNNGGU/AU-IQA-Dataset/tree/dc6ac7feb96304b99a11767f49263d9be98b84fc) contains a README, MIT license, and supplement. The linked Drive page resolves to `AU-IQA.zip`; archive size, contents, raw-vote schema, and parent mapping were not verified. No official AU split was found. [Issue #2](https://github.com/WNNGGU/AU-IQA-Dataset/issues/2) reports missing degraded originals. That is an unresolved user report, not an independently confirmed archive defect.

All twelve descendants of a KonIQ parent must share a split. Exclude parents seen during training, calibration, or teacher selection before claiming unseen-content performance. Use parent-level confidence intervals and report results by task. MIT in the repository covers software and documentation; it does not establish separate rights to the external labels or KonIQ-derived media. These limits keep AU behind UHD for the next general evaluation.

### Low Quality Image Dataset

Kaggle dataset `7424982`, version 1, was updated on 15 May 2025. Its [public metadata](https://www.kaggle.com/api/v1/datasets/view/pochihwu/low-quality-image-dataset) reports 11,118,780,660 file bytes. The card describes 100,000 variants from 5,000 COCO 2017 validation photos: average blur, JPEG compression, gamma/intensity, and Gaussian noise, each at five levels. A file-list page confirms JPEGs with COCO IDs; the full inventory and archive were not downloaded.

No human MOS, rater protocol, or opinion-score distribution was found. Treat the uploader's transform levels as weak supervision. The linked processing repository returned 404, so exact implementations and seeds could not be pinned. Gamma settings can improve an image as well as damage it; do not impose a fixed global quality order.

The MIT badge does not clear the photographs. [COCO's terms](https://cocodataset.org/#termsofuse) separate image copyright from annotation rights. All twenty siblings would need a shared COCO parent ID and exclusions against COCO-derived tests. Our own variants offer better control over rights and provenance, so retain the recipe and defer this media acquisition.

### BIQ2021

The [author paper](https://arxiv.org/abs/2202.03879v2) describes 12,000 images with thirty ratings each: 2,000 ordinary personal captures, 2,000 deliberately faulty camera captures, and 8,000 Unsplash photos. Its target is technical quality. The images were cropped and resized to 512 × 512; these labels do not validate arbitrary native-resolution views.

The complete Kaggle version-4 inventory contains 12,000 JPEGs and `BIQ2021.csv`, totalling 430,126,664 uncompressed bytes. The CSV has `Images`, `MOS`, and `StandardDeviation`. It has no split, source-subset, or scene field. The [original repository](https://github.com/nisarahmedrana/BIQ2021/tree/62c7b40c036b68ee280626409699aa5c1a37fe57) has a 10,000/2,000 train/test allocation, but Kaggle renames the images.

Three downloaded Kaggle JPEGs match Git blob hashes in that pinned original tree. The MOS values were joined by the verified file identities, using Kaggle v4 and the original training CSV:

| Kaggle file | Identical original file | Kaggle MOS | Original MOS |
|---|---|---:|---:|
| `Images (1).jpg` | `ss01 (1882).jpg` | 0.138204682 | 0.7361 |
| `Images (5000).jpg` | `ss03 (1732).jpg` | 0.602092935 | 0.7307 |
| `Images (10000).jpg` | `ss03 (6037).jpg` | 0.681037925 | 0.7464 |

The first two images reverse rank. Both releases use 0–1 MOS, so an increasing scale conversion cannot explain that reversal. An [open author-repository issue](https://github.com/nisarahmedrana/BIQ2021/issues/1) also reports score and rename-mapping problems. The checked paper, card, README, version note, and issue did not explain a new rating study. This proves three conflicts; it does not identify the correct release or map the other 11,997 images. SHA-256, Git blob identities, and annotation hashes are in the [evidence record](evidence/candidate-review-2026-10-15.json).

Hold both releases from trusted gold-label use. First establish an authoritative score-to-image mapping, then recover source subsets and split groups. The Kaggle CC BY 4.0 badge also needs to be reconciled with source-photo rights. In particular, the historical Unsplash subset needs acquisition and training-permission evidence. [Current Unsplash terms](https://unsplash.com/terms) do not establish what earlier grants permitted or whether they were replaced. Keep author-owned camera captures as a separate review group.

### Deepfake Detection Dataset 2026

The [Kaggle version-1 release](https://www.kaggle.com/datasets/chuneeb/deepfake-detection-dataset-2026) hosts one file, `FINAL_DATASET.csv`: 1,549,679 bytes, 6,557 rows, and seventeen columns. It hosts no image media. The card's `v2.0` name differs from the platform version. Direct CSV inspection found 2,790 `REAL` and 3,767 `FAKE` rows, with only `High` or `Medium` categorical quality fields. No human MOS protocol was established.

Every fake row claims `StyleGAN3`, but the URLs have the following hosts:

| Host | Rows | Why it matters |
|---|---:|---|
| `images.unsplash.com` | 2,790 | All labelled real; source-photo identity and rights still need checks. |
| `randomuser.me` | 198 | Labelled StyleGAN3, but the [service](https://randomuser.me/copyright) identifies UI Faces as its photo source. |
| `i.pravatar.cc` | 70 | Placeholder avatars, without a generation record in this release. |
| `api.dicebear.com` | 2,499 | [Procedural SVG assembly](https://www.dicebear.com/introduction/), not evidence of StyleGAN3 generation. |
| `api.multiavatar.com` | 1,000 | [Procedural avatars](https://multiavatar.com/), also labelled StyleGAN3. |

The 2,790 Unsplash rows reduce to 2,349 distinct URL paths/photo IDs. There are 397 repeated-photo groups, of which **194 cross train/test/validation**. These are source-ID overlaps; image response bytes were not compared. The host and overlap counts were independently recomputed from the retained CSV.

The CC0 badge on this CSV does not relicense third-party linked images or avatar styles. Existing PlotQA, CLEVR, and owned controls have better-established identities and labels for our task. Exclude this release from teacher training, gold evaluation, and origin classification. If procedural avatars are needed, create them from a pinned source with correct style terms and provenance.

## Collection coverage

The [full component audit](iqa-collection-coverage-2026-10-15.md) checks the collection at commit `bfc87eeb6b6a9862c13da2a2d765ea7c136f6789`. There are **31 supported loader entries for 30 distinct media datasets**. The two CID:IQ entries rate the same images at different viewing distances. Struck-out KADIS-700k and IVC entries are excluded.

At the start of this review, fourteen entries had inventory rows, BAPPS had only a source mention, and sixteen entries lacked inventory rows. Those sixteen represent fifteen distinct datasets. LIVE-MD and MDID2013 were already mentioned in training prose, so they were not wholly unknown.

| Priority among omissions | Datasets | Reason |
|---|---|---|
| First new compact camera test | **CID2013** | 474 unique images, 79 devices, MOS and raw attribute ratings. More useful for human camera-quality evidence than another unscored synthetic pool. Retain its detailed database notice and use grouped scenes/devices. |
| Targeted tests | **CID:IQ; LIVE-MD; MDID2016; MDID2013; MDIVL** | Viewing-distance effects or compound distortions missing from the current single-operator pilot. They are not broad new scene pools. Some original sources or rights remain unresolved. |
| Conditional high value | **BIQ2021** | Reconsider only after its label mapping and permissions are resolved. It is distinct from BID, which we already list. |
| Task-specific | **GFIQA; PieAPP; BAPPS** | Portrait quality or reference-relative patch fidelity. BAPPS was mentioned already; PieAPP has non-commercial terms and Waterloo parent overlap. These targets are not interchangeable with photo MOS. |
| Defer | **A57; TID2008; SDIVL; Toyama; PDAP-HDDS; VCLFER** | Mostly older, narrow, or redundant coverage. Some source pages and terms remain unavailable. |

This review now records all components, including the omissions. It does not acquire them. AVA and AADB were already documented under aesthetics. The collection supplies no new reliable diagram, screenshot, or photorealistic-origin control set.

## Stored evidence and next actions

Research records are saved here and in the [component audit](iqa-collection-coverage-2026-10-15.md). The [JSON evidence record](evidence/candidate-review-2026-10-15.json) pins releases, counts, and file hashes. Sixteen public metadata/CSV/sample files, totalling 9,714,737 bytes, are retained under `visual-quality/data/sources/candidate-review-2026-10-15/`. That ignored directory includes three small BIQ verification JPEGs, not a training acquisition. No bulk image archive was downloaded and no pilot manifest or label was changed.

1. Complete the existing human-gold, domain-label, split, and delivered-view work before pilot training.
2. Assess CID2013 as the next compact camera-quality test. Check the already shortlisted LIVE-Meta labels against the VizWiz family before collecting another large photo pool.
3. Resolve UHD metadata access and retain its official test. Inspect AU metadata and parent IDs before adding restoration tests.
4. Reproduce useful exposure and compound transforms on cleared parents, with verified visible ranking and held-out families.
5. Keep BIQ gold labels on hold and exclude the deepfake CSV. Do not convert dataset classes, transform levels, or origin labels into technical or aesthetic MOS.

The two submitted paper URLs identify one unique paper. Its bibliography, method notes, and limits are in the [paper library](../papers.md).
