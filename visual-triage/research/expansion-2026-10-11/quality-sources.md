# Quality source decision, 11 October 2026

Use **KonIQ-10k at its native 1024 × 768 view** for the next technical-quality experiment. It has authentic camera and processing defects, human absolute quality ratings, and no clean reference shown beside the image. The previous KADID test used a clean reference for human raters. This distinction matters when the model must judge one image.

## Research and source checks

Exa supplied five successful searches in one quality workstream, with 40 requested results. An earlier request for ten results failed schema validation and returned no sources. Searches covered dataset discovery, rating protocols, image rights, access restrictions and individual photo attribution. Results were narrowed against the [requested IQA catalogue](https://github.com/chaofengc/Awesome-Image-Quality-Assessment), author pages, papers and primary repository metadata. The catalogue's MIT licence does not apply to the datasets it lists.

The Hugging Face dataset-search plugin was tested again. It returned `MCP tool dataset_search was not returned by tools/list`. Official Hub APIs remained available. The [IQA-PyTorch collection](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/tree/f97c4f5dbdbdea7039bc9b80db808313347b1bef) is pinned to revision `f97c4f5dbdbdea7039bc9b80db808313347b1bef`. Its card states that original dataset terms remain in force. Its general CC BY-SA card is not sufficient proof of an individual dataset's rights. Dataset Viewer `/splits` returned HTTP 500, so no viewer previews are used.

| Candidate | Human labels and native view | Rights and access | Decision |
|---|---|---|---|
| KonIQ-10k | 10,073 authentic images; screened ACR ratings, 1–5; native study images are 1024 × 768 | Primary metadata is CC BY 4.0, with individual photo CC exceptions. The native ZIP is public through valid HTTPS. Use the authors' CC research declaration; preserve available individual licence records. | Main panel |
| LIVE Challenge | 1,162 authentic images; continuous MOS and standard deviation; average 175 raters | Current author page requires an access request form. Broad copyright permission exists, but a mirror must not bypass that form. | External panel blocked by access requirement |
| SPAQ | 11,125 smartphone images; 0–100 MOS; PNG images have a 512-pixel short side | Author repository provides Drive and Baidu links, but no explicit dataset licence was found. The paper reports 3,453 images from repeated scenes. | Needs explicit rights and scene grouping |
| FLIVE | Large authentic full-image and patch database | Collection mirror exists. Primary acquisition and image rights need further verification. Related scenes and patches need grouping. | Reserve candidate; not acquired |
| AGIQA-3K | 2,982 generated images; perception and alignment MOS | Authors explicitly distribute the database under MIT. Perception combines technical defects, AI artefacts and aesthetics; the reported normalisation does not establish the technical ACR anchors. | Separate research question; not a main technical MOS target |

Primary evidence: [KonIQ author page](https://database.mmsp-kn.de/koniq-10k-database.html), [KonIQ paper](https://arxiv.org/abs/1910.06180), [KonIQ primary deposit](https://doi.org/10.18419/darus-2435), [LIVE current author page](https://www.colorado.edu/lab/live/live-wild-image-quality-challenge-database), [SPAQ author repository](https://github.com/h4nwei/SPAQ), [SPAQ paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf), [AGIQA author repository](https://github.com/lcysyzxdxc/AGIQA-3k-Database), [AGIQA study protocol](https://arxiv.org/html/2306.04717v2).

## KonIQ labels and rights

The primary score table has 10,073 rows. Its MD5 is `0bc91efacab484f87f7aaf22bd0fd9b4`, which matches the primary deposit. Preserve the raw `MOS`, standard deviation, screened rating count, rating distribution and normalised MOS. The target is the raw 1–5 `MOS`, with an identity mapping. Normalised `MOS_zscore` is not the target. There is no human aesthetic target in this experiment.

The complete source has 208 bad, 1,452 poor, 4,249 fair, 4,161 good and three excellent images. These counts use the fixed intervals `[1,1.8)`, `[1.8,2.6)`, `[2.6,3.4)`, `[3.4,4.2)` and `[4.2,5]`. Five equal bands are therefore impossible. The proposed panel has half low-quality cases and half controls, with coverage of bad, poor, fair and good images. This sample does not estimate production precision or the natural score distribution.

The primary native archive is file `72289`, version 1.0 of DOI `10.18419/darus-2435`. Its size is 5,608,799,013 bytes and its advertised MD5 is `85f076d7a43b2e306f7080814f69af3e`. Secure HTTP range extraction works. Only selected members are downloaded, so the full archive checksum is not independently verified. Each acquired image must pass its ZIP member size and CRC check. Preserve native JPEG bytes; do not use the 512 × 384 archive or Dataset Viewer previews.

A public Flickr lookup pass retained matching `ImageObject` records, authors, canonical pages, licence URLs, page SHA-256 hashes and acquisition times. Flickr then returned HTTP 429, so requests stopped. Of 350 selected images, 45 have current licence and author confirmation. The remaining 305 rely on the authors' published selection of CC licences that permit editing and redistribution; their current individual rights were not verified. Records with known incompatible current licences are excluded. This use is a noncommercial research evaluation. Attribution and share-alike terms remain attached to each image. Do not publish the JPEG files as an unqualified CC BY dataset.

## Frozen panel and safeguards

Freeze 150 training, 100 development and 100 final images from the official author partitions. Keep each known photographer's selected images in one split. Preserve photo IDs for native content groups and author URLs when available. Complete photographer isolation cannot be claimed for 305 images with unavailable attribution. The builder freezes the selection before model calls. The model receives the exact study image pixels, although its viewing process differs from a human browser display.

Check all images against the prior 150 images and the previous 280 track images. Compare encoded bytes, decoded RGB pixels and dHash distances up to four. Stop for any exact match or potential scene overlap. These checks cannot prove absence of all transformed scene overlap or unknown pretraining exposure. Never reuse prior final images as new final cases.

The previous KADID panel used 80 of 81 pristine parents. More variants cannot supply a large external set with new content. This pass uses KonIQ for both development and final testing. Transfer beyond the KonIQ source remains untested because a suitable external native-MOS source was not acquired.

Source snapshots are under `evidence/quality-*` in this research folder and `tracks/quality_v2/evidence/`. The executable acquisition contract is `tracks/quality_v2/build_dataset.py`. No model calls are part of acquisition.
