# Image dataset inventory and adoption review

Checked 29 October 2026. The largest gaps are human aesthetic supervision and explicit labels for partially AI-edited images. The existing technical-quality panels already contain human scores, while the 10,800-image pilot still needs trusted technical and aesthetic labels. Prioritize a human aesthetics acquisition and a small SID-Set edit-detection panel before expanding the unlabelled photo pool.

AI origin means whether an image is generated or edited with generative AI, as confirmed for this review. It is separate from content safety and from whether a moderation service previously processed the image. Recommendations below concern potential adoption; this review does not import image collections or enable training.

**Existing selected data**

Counts describe selected source images or explicitly identified derivatives, not entire upstream releases. Every primary manifest image path was present on this host. This was a manifest and file-presence audit, not a fresh full-media checksum or decode audit. [Inventory evidence](evidence/inventory-and-adoption-2026-10-29/inventory.json) records the manifest paths and counts.

| Dataset | Selected count | Current role and labels |
|---|---:|---|
| Megalith-CC0 | 5,400 | Broad photo pool. Preserved source-model aesthetic scores are not human ratings. |
| VizWiz Quality Issues | 1,200 | Authentic capture failures with human defect votes; no overall technical or aesthetic MOS. |
| Own blur/noise/JPEG variants | 3,000 | Three derivatives of each of 1,000 training parents. Operator severity is not a human quality score. |
| PlotQA | 480 | Chart/domain controls. |
| CLEVR | 480 | Conventional rendered-scene controls. |
| Owned drawings/controls | 240 | Domain controls. These three control sources do not establish generative-AI origin. |
| KonIQ-10k | 350 | Human technical MOS; 150 train, 100 development, 100 test. |
| KADID-10k | 160 | Human reference-relative quality judgments; 40/40/80, from 80 reference parents. |
| UHD-IQA | 60 | Human technical-quality MOS. Original ratings used much larger views than our 512-pixel delivered panel. |
| AGIQA-3K | 6 | Generated-origin examples with perception/alignment scores; these are not independent technical and aesthetic targets. |
| OpenFake v2 | 60 | 30 DOCCI source-real photographs and 30 generated images. |
| SynthCLIC | 320 | 160 CLIC source-real and 160 generated images, grouped by source family. |
| Synthbuster | 260 | Distinct generated images across three acquired panels, representing 140 reference families. |
| RAISE | 100 | Distinct source-real photographs across two panels. |
| VISION | 160 | Native camera photographs for metadata/provenance tests. |
| MS COCOAI / Defactify | 1,600 | 800 source-real and 800 generated; 600 calibration and 1,000 validation. Source-real includes edited/composited examples. |

The first six rows form the **10,800-image pilot: 10,000 train, 400 development, 400 test**. All primary technical/aesthetic scalar targets remain null and their training masks remain disabled. Later label trials keep assistant/model observations separately. The pilot also has acquisition spares; those are not additional selected training rows. See the [assembly record](pilot-10k-assembly.md), [quality panel sources](../../../visual-triage/tracks/quality_v2/SOURCES.md), [KADID sources](../../../visual-triage/tracks/quality/SOURCES.md), and [UHD panel](../../../visual-triage/benchmark_data/quality-research.md).

**No broad human-rated aesthetics dataset is acquired in the active manifests.** TAD66K is a documented training choice, despite the older catalogue calling it “our training set”; no acquisition manifest or local media was found. AVA, AADB, MSC, PARA and the other [aesthetics catalogue](iaa.md) entries remain research/planning entries. The [27 October dual-score trial](../../experiments/dual-score-trial-2026-10-27/README.md) uses 50 pilot-derived images and assistant assessments, with no human gold labels.

For origin detection, the [OpenFake panel](../../../visual-triage/benchmark_data/ai-research.md), [SynthCLIC panel](../../../visual-triage/tracks/ai_detection_v2/SOURCES.md), [native metadata panel](../../../visual-triage/tracks/metadata_v3/README.md), and [MS COCOAI calibration](../../experiments/jev14-calibration-2026-10-18/README.md) supply source-based generated/real labels. No dedicated supervised partially AI-edited class was found. Metadata stripping, conventional distortions and source-real composites do not supply that class. Reruns and resized views reuse these images and must not be added as new acquisitions.

**Priority additions**

These priorities reflect task fit, trustworthy labels, added coverage, release availability and integration constraints. They are recommendations, not measured model improvements. “Known” means already catalogued but unacquired; “new” means absent from the inspected research catalogue; “release update” means a known paper now has additional release evidence.

| Priority | Dataset and status | What it adds | Adoption decision and limits |
|---|---|---|---|
| 1 | [SID-Set](https://huggingface.co/datasets/saberzl/SID_Set), **new** | Explicit real / fully synthetic / tampered labels and manipulation masks. Public Hub splits contain **210,000 train + 30,000 validation**. The 300,000 headline includes a separately obtained 60,000-image test. | First origin addition: start with a balanced, parent-disjoint sample containing all three classes. The [SIDA paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_SIDA_Social_Media_Image_Deepfake_Detection_Localization_and_Explanation_with_CVPR_2025_paper.pdf) describes FLUX-generated images and object replacement/partial modification. Check source identity across classes and inherited OpenImages/COCO/Flickr30k terms; the release declares CC BY 4.0. |
| 1 | [MSC](https://www.nature.com/articles/s41597-026-06816-0), **known** | **10,426** natural-scene/texture images, **100 human aesthetic ratings per image**, with deliberate low- and high-aesthetic variants. | Best focused addition for the missing aesthetic range. Group originals and variants. Its low-semantic-content design excludes many subjects, so pair it with broader photo coverage. The existing source audit records release CC0 with per-image public-domain, CC0, CC BY and CC BY-SA terms. |
| 1 | [TAD66K](https://huggingface.co/datasets/Shuai1995/TAD66K_for_Image_Aesthetics_Assessment), **known/planned** | Broad human aesthetic means across **66,327 images and 47 themes**; existing verified label lists contain 52,248 train / 14,079 test. | Execute the existing research plan after source-rights and overlap checks. The release is tagged Apache-2.0; individual web-photo rights remain unresolved. Keep an independent aesthetics benchmark held out. |
| 2 | [CID2013](https://zenodo.org/records/2647033), **known** | **474 unique images**, 79 devices, technical MOS and raw sharpness/graininess/lightness/saturation ratings. | Compact camera-quality transfer test. Preserve the database copyright notice and group repeated scenes/devices. The 480 figure counts evaluations, not unique photos. Its dynamic-reference human protocol differs from isolated single-image scoring. |
| 2 | [FGAesthetics](https://huggingface.co/datasets/yzc002/FGAesthetics/tree/main), **release update** | Human within-series aesthetic rankings, useful for distinguishing similar shots, crops and generated variants. Paper totals are 32,217 images / 10,028 series. | Newly actionable for inspection: about **19.65 GB** of media archives plus label tables are present. A bounded label-archive check found **44,225 pairwise rows** across nine tables. Confirm schema and permissions before ingestion; no dataset licence was found. Keep whole series in one split. The one-row Hub viewer is only a teaser, not the archive inventory. |
| 2 | [MMP-2k](https://github.com/Future-IQA/MMP-2k), **new** | **2,000 macro photos**, 17 retained human ratings per image, and human defect annotations with templated reports. | Useful targeted test of subject focus versus intentional shallow depth of field. Narrow macro domain; CC BY-NC-SA 4.0. Keep it separate from broad photo MOS and from human-written free-text critiques. |
| 2 | [TIQA](https://huggingface.co/datasets/deepfakesMSU/TIQA_crops_and_images), **new** | Generated-text quality: **1,440 full images** with overall/text MOS, plus 120,197 text crops, of which **9,978 have human MOS** and **110,219 have OCR-derived proxy labels**. | Add only if text-heavy images matter. The official organisation release was updated 28 October and includes full-image rating schema and archives. Preserve human/proxy separation and prompt/image groups. Its CC BY-NC 4.0 front matter still contains a licence-confirmation TODO. |
| 2 | [Community Forensics Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval), **known** | **51,836 rows**, 21 generator models and multiple real-source datasets. | Broader external origin test after deduplication against existing families. Research/evaluation role; CC BY-NC-SA 4.0 and inherited model/media terms. Do not assume every released real/generated row is an exact matched pair. |
| 3 | [OpenSDID](https://github.com/iamwangyabin/OpenSDI), **new** | Global generation and local edits with spatial supervision. Paper describes 300,000; current author Hub releases report **200,824 train + 100,000 test** rows. | Alternative/edit stress test after SID-Set. Real images come from Megalith-10m, creating overlap risk with our Megalith pilot. Reconcile actual row counts and CC BY-SA 4.0 plus “academic use” wording before adoption. |

MSC and TAD66K address different aesthetic gaps: controlled coverage of low aesthetics and broad theme coverage. FGAesthetics adds relative ranking rather than an interchangeable absolute MOS target. CID2013 and MMP-2k are targeted diagnostics; neither is a reason to discard the existing KonIQ/KADID/UHD panels.

**Candidates to defer or treat as updates**

- [BBC PAIR](https://bbc-pair.datasets.bbctest01.uk/) has highly relevant paired real, fully generated and partially edited images, but access is limited to approved academic/public research institutions. Its terms exclude independent researchers and commercial organisations. Do not treat public search visibility as availability to this project.
- [REPID](https://huggingface.co/datasets/vsevolodpl/REPID) is the renamed/expanded DEAR work already catalogued. It supports rendering preferences; do not count it as a wholly independent photo pool. Image terms differ from annotation terms.
- [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) is already acquired in a small pinned panel. A fresh generator-disjoint sample could refresh coverage, but would be an expansion of the same source. Pin current files and generator records. Its `prompt_kind` distinguishes generation prompts from recaptioning; captions do not prove exact matched generation. The live raw card declares CC BY-NC 4.0; stale indexed cards disagree.
- Keep **BIQ2021 gold labels on hold** because identical image bytes have conflicting MOS across releases. Continue excluding the previously audited “Deepfake Detection Dataset 2026” CSV. See the [15 October evidence](candidate-review-2026-10-15.md).
- Instruction or explanation corpora such as GR-IQA and SID-Set descriptions can supply auxiliary supervision, but their generated prose does not become human quality or aesthetic gold.

**Integration order**

1. Preserve existing final-test partitions. Acquire a bounded human-aesthetics sample from the existing MSC/TAD66K shortlist, keeping independent evaluation data aside. Keep per-source scales and rating distributions instead of pooling raw scores.
2. Add a bounded SID-Set sample with explicit `camera/source-real`, `fully_generated`, and `partially_ai_edited` classes; retain an `unknown` state for uncertain provenance. Use mask/edited-area strata and source-family splits.
3. Add CID2013 for external camera-quality checks. Use MMP-2k and TIQA only for their specific failure modes. Inspect FGAesthetics labels and terms before choosing a series-ranking experiment.
4. Freeze source revision, parent IDs, byte hashes and delivered-view policy for any acquisition. Deduplicate against existing train, calibration and final panels. Preserve native bytes where forensic traces matter, and never convert origin labels, distortion strength or model scores into aesthetic/technical human targets.

**Research coverage**

Completed **five rounds with Exa and five with Hugging Face**: broad discovery; targeted expansion; primary release/card validation; challenges to novelty, provenance and access; final checks on the leading existing shortlist entries. The four Exa discovery/verification rounds used 15 searches with 95 requested result slots; those are neither unique sources nor 95 verified datasets. The fifth round reread the MSC and CID2013 primary pages.

The Hugging Face app's `dataset_search` returned “disabled by server configuration.” Its `hub_repo_details` worked. Discovery therefore used the public Hugging Face Hub API, with app metadata/card checks and bounded label inspection. The fifth round rechecked TAD66K and the official TIQA release through the app. No full image dataset was downloaded. The FGAesthetics label archive was only 201,749 bytes; its pairwise row counts were checked, while its exact column semantics remain to be confirmed.

The [research evidence directory](evidence/inventory-and-adoption-2026-10-29/) contains the local inventory, search-round log and compact source assessments. The existing [IQA](iqa.md), [IAA](iaa.md), and [earlier AI-source research](../../../visual-triage/research/expansion-2026-10-11/ai-sources.md) retain the wider catalogue; a catalogue entry alone is not acquisition evidence.
