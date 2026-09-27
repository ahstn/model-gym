# Low-quality photos and non-photo controls

Checked: 2026-09-27. Status: source research and proposed sampling plan. No images were downloaded or scored.

The source note, *2026-09-27 Synth Data Generation*, identifies the main gap: large photo pools do not ensure a useful quality spread. **Use authentic capture faults as well as controlled distortions. Add diagrams and rendered images as a separate photo-suitability task.** This extends the [mixture evidence](../training/data-mixture-evidence.md) and the [quality and aesthetics prompt pack](../training/synthetic-label-prompts.md).

The strongest additions are LIVE-Meta VI-UGC and VizWiz for real faults, RealBlur and SIDD for specific capture defects, and PlotQA plus CLEVR for non-photo controls. Keep Megalith-CC0 as the large source pool. Its low-quality yield still needs measurement.

## 1. Define what must score zero

For this proposal, the selection target is a camera photograph of a real scene or object. A standalone chart, diagram, document, interface, or known generated image is outside that target. This is a proposed product rule. If the product accepts broader visual content, change the rule before creating labels.

Keep these outputs separate:

| Output | Meaning | Treatment of a clean diagram or render |
|---|---|---|
| `photo_domain_eligible` | Whether the item meets the photo selection rule | `false` for known excluded content |
| `technical_quality` | Visible capture, encoding, or processing defects | Can be high; never force it low from source type alone |
| `aesthetics` | Visual appeal and composition | Can be high; no automatic penalty for being a diagram or CG |
| `recognizability` | Whether useful content can be identified | Separate from beauty and sharpness |
| `source_origin` | Camera, render, generative model, mixed, or unknown | Use source records; appearance alone cannot prove origin |

An excluded item can receive **zero for photo selection**. That zero is a policy label, not human MOS. Preserve the existing 1–5 quality and aesthetics scales. Domain-only rows have null quality/aesthetics labels and per-task training masks; controls with valid ratings retain them. If a rating does not apply, mask that task instead of inventing a zero rating. Store domain labels in a separate manifest; this proposal does not change the current prompt response schema.

An optional downstream score can be `25 × (technical_quality - 1)` for eligible photos and `0` for known exclusions. Unknown eligibility or a missing quality rating gives no selection score. This formula is a product choice, not a new IQA metric. Keep the exclusion reason so a domain rejection is distinguishable from a very poor photo.

Use boundary examples: photos of paper charts, screens, toys, real geometric objects, abstract textures, and realistic CG. Under the proposed rule, a photo whose main content is a reproduced chart is excluded; a street scene with a sign is eligible. Known photorealistic CG is excluded by provenance even if its quality is excellent. Do not claim that a pixel-only model can detect every such case.

## 2. Authentic capture faults

These sets provide real defects that simple blur, noise, and JPEG operators cannot fully reproduce. Dataset membership is not a low-quality label. Select examples by their ratings, defect votes, and an independent audit.

| Source | Size and labels | How it fills the gap | Access, terms, and limits |
|---|---|---|---|
| **[LIVE-Meta VI-UGC](https://live.ece.utexas.edu/research/LIVE-VI-UGC/index.html)**; [paper](https://arxiv.org/abs/2305.08066) | 39,660 full images plus 39,660 dependent patches. Human quality scores and focus, motion, exposure, and noise labels. About 23.9K train images. | First choice for a rated authentic low-quality pool. Sample across MOS and defect type. | Posted notice permits any-purpose use with notice and attribution. Form access was not tested. The notice says “videos” on this image release; retain and check the bundled terms. Paper cleaning removes 43 constant images, so confirm the delivered count. |
| **[VizWiz-QualityIssues](https://vizwiz.org/tasks-and-datasets/image-quality-issues/)** | 39,181 images: 23,431 train, 7,750 validation, 8,000 test. Five-worker votes for blur, brightness, darkness, framing, obstruction, rotation, and unrecognizability. **No MOS.** | Best selector for severe capture failures and unrecognizable content. Join labels to exact image IDs. | The [caption image release](https://vizwiz.org/tasks-and-datasets/image-captioning/) states CC BY 4.0; confirm the QualityIssues annotation terms at ingestion. The [HF VQA mirror](https://huggingface.co/datasets/HuggingFaceM4/VizWiz) is a different task. |
| **[RealBlur](https://github.com/rimchang/RealBlur)**; [author HF release](https://huggingface.co/datasets/rimchang/RealBlur) | 3,758 blurry/sharp training pairs from 182 scenes; 980 test pairs from 50 scenes. No MOS. | Real camera blur with a sharp comparison. Useful for relative sensitivity checks. | Official dataset CC BY 4.0. Keep scene groups intact. RealBlur-J and RealBlur-R are related representations, not extra independent scenes. |
| **[SIDD](https://www.eecs.yorku.ca/~kamel/sidd/)** | About 24K noisy training frames, but only 10 physical scenes. Medium has 320 noisy/clean pairs across 160 capture instances. No MOS. | Real sensor noise under different phones and conditions. Start with provided sRGB pairs. | Official page explicitly applies MIT to data and code. Cap repeats per physical scene; a large frame count does not create scene diversity. |
| **[LIVE Challenge / CLIVE](https://live.ece.utexas.edu/research/ChallengeDB/)** | 1,162 photos; over 350K human ratings, MOS and variance. | Compact check of mixed authentic defects. | Any-purpose notice with attribution and notice retention. Keep it outside training because it is already part of our standard IQA evaluation. |
| **[KonIQ++](http://database.mmsp-kn.de/koniq-image-defects-database.html)** | New quality and defect labels on the same 10,073 KonIQ images. Adds no unique images. | Helps identify blur, artifacts, color, and contrast faults in an existing source. | Research-community release; original photo terms remain. Keep the existing KonIQ split and its reserved tests. |

**LIVE-Meta and VizWiz overlap.** Treat them as one source family with complementary labels, not as roughly 79K independent photos. Their patches, captions, VQA records, and quality records must share a parent ID. Keep the official family splits. Redaction masks are preprocessing metadata, not camera defects. An unanswerable question is also not proof of poor image quality.

### Useful but conditional sources

| Source | Contribution | Why it is not in the default pool |
|---|---|---|
| [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) | 7,363 low-light images, 10 lighting conditions; object labels, no MOS | Original repository asks users to contact the author for commercial use. The [checked HF mirror](https://huggingface.co/datasets/dronefreak/ExDark) has 7,344 resized images; use original bytes for quality work. Night scenes are not inherently poor. |
| [SIDL](https://sidl-benchmark.github.io/) | 1,588 dirty/clean pairs from 300 scenes; physical lens contamination | Research and education terms; commercial use and derived releases need written permission. No MOS. |
| [SID](https://github.com/cchen156/Learning-to-See-in-the-Dark), [LOL](https://daooshee.github.io/BMVC2018website/) | Real low-light exposure pairs | No MOS. SID needs a documented RAW rendering pipeline; media terms need confirmation. LOL's checked project page did not establish data terms. |

## 3. Large photo pools and low-aesthetic examples

“Raw Flickr” here means ordinary photos before our added distortions. It does not mean camera RAW files or a verified sample of bad images.

| Source | Evidence | Decision |
|---|---|---|
| **[Megalith-CC0](https://huggingface.co/datasets/Spawning/megalith-cc0)** | HF Viewer reports **2,385,784 rows**, `partial: false`, in its [size response](https://datasets-server.huggingface.co/size?dataset=Spawning%2Fmegalith-cc0). Images are hosted separately from the metadata. | Main scale source. Preserve CC0 provenance and original image bytes. Measure the low-quality fraction before estimating acquisition volume. |
| [Megalith-10m](https://huggingface.co/datasets/madebyollin/megalith-10m) | About 10M Flickr links. The author removed about 2M candidates using account review, watermark, edit, content, and size filters. Minimum size is 256×256. Source categories include CC0 and several public-domain markings. | Useful context for selection bias. It is a filtered pool, not an untouched camera-photo stream. Do not assume every item has the CC0 subset's terms. |
| **[CommonCatalog CC-BY](https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by)** | YFCC-derived photos with source URL, creator, license fields, dimensions, and device metadata. No MOS. Exact CC-BY split size was not verified. | Good alternative for content diversity. Keep per-image attribution and license version. Deduplicate against KonIQ and all other reserved photo tests. |
| [Flickr-CC collection pipeline](https://github.com/senthilps8/continuous_ssl_problem/blob/main/datasets/DATA_PREPARATION.md) | Author describes unfiltered Flickr/Openverse link collections from 2004–2021, with subsets up to hundreds of millions of links. No quality ratings. | Optional if the filtered pools miss severe failures. Requires license filtering, live-link checks, and a fresh quality audit. Link count is not usable image count. |
| [PD12M](https://huggingface.co/datasets/Spawning/PD12M) | 12.4M public-domain/CC0 images, selected for high aesthetics | Useful positive examples and distortion references. A poor primary choice for the authentic low-quality end. |
| **[MSC](https://www.nature.com/articles/s41597-026-06816-0)**; [OSF release](https://doi.org/10.17605/OSF.IO/ZGSVJ) | 10,426 natural scenes with limited semantic content; 100 aesthetic ratings per image. Includes deliberately beautified and uglified variants. | Useful for **aesthetic spread**, not technical-quality calibration. Keep each original and its variants together. |

MSC directly addresses beauty bias, but its narrow scene content cannot replace ordinary photos. The paper lists public-domain, CC0, CC BY, CC BY-SA, and author-owned sources. The release's CC0 label does not remove source-image obligations. Check provenance before selecting a deployable subset. Its fitted aesthetic parameter can range from 0.5 to 5.5; the underlying votes are 1–5. Preserve raw votes and distinguish their mean from the fitted value.

For the large photo pool, retain an untouched random sample to estimate prevalence. Mine additional candidates with several signals: human defect votes where present, exposure/clipping statistics, blur and noise estimates, resolution, and teacher disagreement. These are selectors, not ground-truth scores. Cap each photographer and near-duplicate cluster. Audit all selected quality bands so one teacher's blind spots do not define the whole dataset.

## 4. Controlled distortions and generated-image defects

| Source | Scale and supervision | Recommended role |
|---|---|---|
| **Own variants using the [DepictQA construction design](https://github.com/XPixelGroup/DepictQA/tree/main/build_datasets)** | 35 operators in 12 families, five configured levels, and selected two-stage combinations. Code declares Apache 2.0; source images have separate terms. | Best expansion route on owned or verified CC0 photos. Store operator, parameters, seed, parent, and delivered size. Check notices for reused code. |
| **[KADID-10k](https://database.mmsp-kn.de/kadid-10k-database.html)** | 81 references × 25 distortions × 5 levels = 10,125 distorted images, with human ratings. Its `dmos` is higher for better quality. | Human calibration for distortion severity. Split by reference. Research-community access does not establish unrestricted rights; the paper identifies Pixabay sources. Preserve our existing benchmark split. |
| [KADIS-700k](https://database.mmsp-kn.de/kadid-10k-database.html) | 140K references; recipe creates 700K variants. No human MOS. | Construction reference, or conditional source after rights checks. Distortion levels are weak supervision. |
| [DQ-495K / DataDepictQA](https://huggingface.co/datasets/zhiyuanyou/DataDepictQA) | About 495K instruction records from several source datasets, not 495K independent images | Reuse construction methods. An Apache-tagged annotation release does not relicense KADIS, KADID, PIPAL, or BAPPS images. |
| [PIPAL](https://www.jasongt.com/projectpages/pipal.html) | 250 reference patches, 29K variants; restoration artifacts and human pair judgments | Restoration-artifact evaluation. Severe distortions were constrained, so it is not the main source for extreme failures. Image rights remain unresolved here. |
| [Waterloo Exploration](https://kedema.org/project/exploration/index.html) | 4,744 references, 94,880 distorted images; no per-image human MOS | Independent distortion-consistency test, subject to image terms. Do not convert consistency labels into absolute ratings. |
| **[AGIQA-3K](https://github.com/lcysyzxdxc/AGIQA-3k-Database)**; [HF](https://huggingface.co/datasets/strawhat/agiqa-3k) | 2,982 generated images; separate perception and prompt-alignment MOS. Official database license is MIT. | Strong low/high CG quality check. Keep it held out because our standard Q-Align comparison already uses it. |
| [AIGIQA-20K](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Li_AIGIQA-20K_A_Large_Database_for_AI-Generated_Image_Quality_Assessment_CVPRW_2024_paper.html) | 20K generated images; 420K ratings. Judgment combines perception and prompt alignment. | Larger optional CG pool. Its composite MOS is not our pure technical target. The [HF entry](https://huggingface.co/datasets/strawhat/aigciqa-20k) is metadata; images are linked through a differently named [ModelScope release](https://www.modelscope.cn/datasets/lcysyzxdxc/AIGCQA-30K-Image), which declares Apache 2.0. Confirm archive identity and reserved splits. |

Start with focus/motion blur, noise, compression, exposure, color/white balance, resampling, and sharpening/denoising artifacts. Add plausible two-stage combinations after checking individual effects. Keep clean references and subtle defects as well as severe damage.

**A larger distortion parameter does not guarantee worse perceived quality.** Blur may remove noise; brightening may fix underexposure. Three unrelated operators do not define a rank order. For a ranking study, make several levels of one operator per reference, then verify the visible ordering. Ties and reversals are valid outcomes. Score the image at the teacher's actual viewing resolution because resizing can hide defects. Do not map severity directly to a 1–5 rating or force the most severe setting to zero.

## 5. Diagram and computer-generated controls

| Source | Verified scale | Terms and use |
|---|---|---|
| **[PlotQA](https://github.com/NiteshMethani/PlotQA)**; [HF](https://huggingface.co/datasets/achang/plot_qa) | 224,377 unique plots: **157,070 train**, 33,650 validation, 33,657 test. The 28.9M figure counts QA pairs. | Upstream data CC BY 4.0; code MIT. First choice for chart controls. Sample unique training images and retain attribution. |
| **[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/)**; [HF data license](https://huggingface.co/datasets/laion/clevr-webdataset/commit/4cd54020e035d3c08a7b52d9229c0c80f667fb10) | **70K train**, 15K validation, 15K test images of rendered objects | Data CC BY 4.0; generation code BSD. First choice for obvious CG. A sharp render is a useful high-quality, photo-ineligible example. |
| **Owned diagrams, mock interfaces, and renders** | Set size by distinct templates, scenes, and seeds | Generate flowcharts, tables, circuit-style drawings, documents, and renders with owned geometry or suitable CC0 assets. Include both polished and damaged versions. Reserve whole templates and scenes for testing. |
| [MOVi via Kubric](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md) | About 9,750 training scenes per variant, each with 24 frames at 256×256 | Useful harder CG boundary. Start with one frame per scene. Code Apache 2.0; asset sources include GSO CC BY 4.0 and Poly Haven CC0. Pin media and asset terms before use. |
| [FigureQA](https://www.microsoft.com/en-us/research/project/figureqa-dataset/highlights/) | 100K training plots; four validation/test subsets of 20K each | Original image license not established in this pass. A [current HF annotation release](https://huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3/blob/7656391d4d4cb11ec3722b34f10d499435de0460/figureqa/README.md) explicitly requires media from elsewhere. Prefer PlotQA until resolved. |
| [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) | 335,703 training document pages | Annotation license CDLA-Permissive; page images retain their PMC article terms. Select by article license and group by article. Conditional document controls. |
| [RICO](https://interactionmining.org/rico); [HF](https://huggingface.co/datasets/Voxel51/rico) | More than 66K mobile screens; HF reports 66,261 samples | HF CC BY 4.0 card also flags third-party UI rights and commercial redistribution limits. Prefer owned mock interfaces for the default pool. |
| [AI2D](https://prior.allenai.org/projects/diagram-understanding); [HF](https://huggingface.co/datasets/lmms-lab-encoder/ai2d) | Official page: 4,903 diagrams; another registry reports 4,817. Common HF mirror: 3,088 **test QA rows**. | Hold outside training. [AWS](https://registry.opendata.aws/allenai-diagrams/) and mirror licenses conflict; a [secondary provenance card](https://huggingface.co/datasets/yafitzdev/opsis-v1) reports restrictive archive terms that were not independently extracted here. Do not use test QA rows as training diagrams. |

The minimum crossed control design is: clean photo, damaged photo, clean diagram/CG, and damaged diagram/CG. This stops source type from becoming a substitute for quality. Add realistic renders and camera photos of diagram-like objects; PlotQA and CLEVR alone would make domain rejection too easy.

## 6. Proposed 10K pilot and 250K training mix

These are **planning allocations**, not measured optimal ratios or confirmed downloadable totals. The 250K count is image rows, including related variants and domain-only controls. It is not 250K independent scenes or 250K human-rated IQA examples.

| Component | Share | 10K pilot | 250K plan | Sampling rule |
|---|---:|---:|---:|---|
| Broad real photos | 50% | 5,000 | 125,000 | Megalith-CC0 first; CommonCatalog CC-BY if needed. Reserve 20% of this component as a random sample; select the rest across reviewed low, middle, and high quality bands. Include distortion parents here. |
| Controlled damaged photos | 30% | 3,000 | 75,000 | Three variants per 1,000 / 25,000 parent photos already counted above. Vary operators across parents; use local severity series for rank tests. |
| Authentic fault examples | 10% | 1,000 | 25,000 | Unique eligible VI-UGC/VizWiz training images plus newly mined licensed photos. Use RealBlur/SIDD as small scene-capped slices. Do not fill a quota with repeats. |
| Domain controls | 10% | 1,000 | 25,000 | 40% PlotQA train, 40% CLEVR train, 20% owned diagrams/UI/renders: 400/400/200 in the pilot; 10K/10K/5K at scale. |
| **Total** | **100%** | **10,000** | **250,000** | Quality losses exclude domain-only labels; photo selection uses its own target. |

The authentic-fault target exceeds the roughly 23.9K VI-UGC training pool even before filtering. It therefore needs additional verified images. Do not use validation/test images or patches to close that gap. If supply is insufficient, publish the shortfall and adjust the mix before scaling. The control allocation also assumes the selected benchmark training partitions remain outside our final test manifest.

For RealBlur and SIDD, count selected defective images toward the authentic-fault quota. If clean partners are included, count those as separate image rows in the broad-photo component. A pair is not two defective examples.

MSC is an optional aesthetic calibration/ablation source after its image terms are checked. It replaces an equal-sized part of the photo component; it is not an extra uncounted block. Keep an independent aesthetics test. Do not use MSC aesthetic scores to calibrate technical quality.

### Pilot decisions

1. Freeze source eligibility, the viewing policy, and the [existing rating prompts](../training/synthetic-label-prompts.md). Keep source names, MOS, and intended distortion severity hidden from image-only teacher ratings.
2. Allocate original families to training, development, and final test before generating variants. Pilot counts above are candidate training rows. Development and test reserves are additional and must not be reused to reach 250K.
3. Audit at least 400 development images, stratified across sources, score bands, and boundary controls, with three independent raters. This is a proposed first audit, not a power calculation. Use it to examine teacher failures and adjust the sampling rules. Freeze a separate final test before tuning.
4. Measure authentic low-quality recall, false rejection of good photos, domain rejection, and confusion between quality and aesthetics. Report the quality histogram by source, not just one pooled correlation. Use human judgments to check selected low-score examples.
5. Compare matched-size training arms: diverse photos; photos plus distortions; those plus authentic faults; then the domain task. Keep model, steps, image budget, and test data fixed. If synthetic defects help only synthetic tests, reduce their share.
6. Scale only after confirming usable image counts, actual low-quality yield, label consistency, and rights for the selected files. Keep strong night photos, intentional motion, shallow depth of field, and high-quality CG among the controls.

### Manifest and split rules

Record source URL/version, image ID, creator and license evidence, source-byte and decoded-image hashes, original dimensions, delivered dimensions, parent image, scene/capture group, and transform parameters. Record domain labels separately from human MOS, human vote counts, teacher ratings, and pair preferences. Keep each label's scale and direction.

Deduplicate by source ID, decoded hash, and near-duplicate checks against all reserved IQA, aesthetics, and visual-reasoning tests. Keep the reserved KonIQ and KADID test images, the whole CLIVE and AGIQA-3K evaluation sets, and any reserved chart/diagram tests out of synthetic-label training. Do not count QA rows, patches, repeated exposures, or adjacent frames as independent scenes. Provider failures, refusals, and unreadable files are pipeline failures, not score-zero training data.

## 7. Research record and limits

Research used Exa and Hugging Face in discovery, primary-source verification, and contradiction-checking rounds. Four parallel tracks covered photo pools/aesthetics, authentic defects, synthetic distortions, and domain controls. Together they made 38 Exa searches requesting 218 result slots. These include duplicates and are not 218 unique fully reviewed sources.

Hugging Face `dataset_search` was disabled by server configuration. Repository inspection worked over multiple rounds; public Hub queries and cards supplied further checks. Original releases took precedence when mirror rows, license tags, or counts disagreed. The complete Megalith-CC0 metadata count was checked with the HF Dataset Viewer API. It does not prove that all image URLs are reachable.

No bulk image downloads, access-form submissions, paid model inference, dataset uploads, or training runs occurred. Sizes and labels above come from publisher releases unless stated otherwise. Archive contents, per-file rights records, duplicate rates, and obtainable low-quality fractions still need validation during ingestion.
