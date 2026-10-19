# AI-origin benchmark sources

Research date: 2026-10-11. This benchmark is a separate task from quality and aesthetics. Its labels describe image origin. Low quality, illustration style, and conventional computer graphics do not establish AI origin.

## Selected source

Use 60 new images from the **core/test** split of [OpenFake v2](https://huggingface.co/datasets/ComplexDataLab/OpenFake/blob/3247b942f18d39d1fd476b1077852af033d35d8c/README.md), pinned at revision `3247b942f18d39d1fd476b1077852af033d35d8c`:

- 30 real photographs whose published source is DOCCI.
- 30 generated images: five from each of the six model labels `flux.2-klein-9b`, `gpt-image-1.5`, `midjourney-7`, `z-image-turbo`, `ernie-image-turbo`, and `recraft-v3`.

The first candidate list included Nano Banana Pro, Ideogram 2.0, and Seedream 5.0. The first 1,000 rows had only two, three, and four examples respectively, so those strata were replaced before any model scoring.

The generator names and release dates are dataset metadata, not independently verified provider release records. OpenFake records the generator and prompt for generated images. We use that provenance as the reference label. We do not use a classifier score or this assistant's visual impression as the origin label.

The [DOCCI project](https://google.github.io/docci/) and its [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07764.pdf) state that an author and his family took its images. Most use an iPhone camera. Some contain staged objects, toys, signs, murals, screens, or manually blurred personal details. Those images remain camera photographs. This provides stronger real-image provenance than an unrestricted web pool or a source label such as “real” without capture details.

The OpenFake card reports an out-of-distribution test split with generators and real-image sources distinct from its training split. Our sample is **not an exact scene-paired sample**. It uses one published benchmark with controlled class counts. Real-source identity, subject mix, camera processing, and generator differences can still create shortcuts. The use of only DOCCI for real photographs also limits generalization to other cameras and settings.

## Acquisition and split safeguards

Run `uv run python benchmark_data/ai_builder.py` from `visual-triage`. Use `--verify` for offline hash checks. Use `--rebuild-from-cache` to restore the exact frozen delivered images from the existing source cache without network access or resampling. All source and delivered hashes are checked before writes. A fresh acquisition rerun must match the complete frozen sample records before it can replace any delivered files; it preserves the existing manifest. The builder reads 1,000 row metadata records through the [Hugging Face Dataset Viewer](https://huggingface.co/docs/dataset-viewer/rows), then downloads selected image assets only. The `/filter` endpoint returned HTTP 500 while the dataset index was loading, so the builder uses bounded `/rows` pagination. It does not download full parquet shards.

The sample order comes from SHA-256 ranks with fixed seed `20261011`. There is no inspection of model outcomes during selection. It excludes explicit-content keywords in source prompts or captions to keep the origin benchmark separate from a sexual-content benchmark. It excludes exact delivered duplicates and images with dHash distance at most five from an accepted image or the original 30-image panel. These checks reduce duplicates; they do not prove independence of scenes.

After selection, a seeded shuffle assigns blind IDs from `ai-000` to `ai-059`. IDs do not encode the class or generator. Contact sheets contain at most six images and show only those IDs. The manifest stores reference labels and provenance separately; inference must send only the image and task prompt.

The delivered view uses EXIF orientation, RGB, the full frame, a maximum edge of 512 pixels without upscaling, and JPEG quality 92 with no chroma subsampling. Acquisition can use a Dataset Viewer image that was re-encoded upstream. The manifest stores the published original hash, the actual acquired hash, whether they match, and the delivered hash as separate fields. Do not describe a mismatch as verified original-byte identity.

`group_id` hashes the exact normalized source caption or prompt. It can join exact text matches across real and generated rows. It does not identify every related scene. Keep repeated groups in one evaluation fold, and report the remaining scene-dependence limit.

## Licenses and limits

OpenFake v2 declares **CC BY-NC 4.0** and notes additional non-commercial terms for outputs from proprietary generators. [DOCCI](https://huggingface.co/datasets/google/docci) declares **CC BY 4.0**, attributed to Google LLC. This panel is for non-commercial evaluation. Keep image assets and cached source metadata in ignored `data/` paths; commit source IDs, hashes, reference labels, original analysis, and reproduction code. Do not treat this selection as a cleared source for commercial training or unrestricted media redistribution.

Sixty images are enough to expose failures and compare a fixed prompt under a small test budget. They cannot validate a deployment rejection threshold, false-positive rate for rare content, or performance on an unseen generator. Downsampling and JPEG delivery can remove forensic traces. Record this exact view when comparing scores.

## Alternatives checked

| Source | Useful property | Reason it was not selected for this panel |
| --- | --- | --- |
| [GenImage](https://genimage-dataset.github.io/) | Broad ImageNet classes and eight known generators; real and generated labels | Older generator coverage and large archive distribution; ImageNet source terms need separate handling. |
| [Synthbuster](https://zenodo.org/records/10066460) plus [RAISE](https://loki.disi.unitn.it/RAISE/) | Camera-native real images and generated images based on related prompts | Strong future paired test, but older generators and separate real-image acquisition. The Hugging Face `synthbuster-plus` mirror has little provenance or license text beyond its schema. |
| [SuSy](https://huggingface.co/datasets/HPAI-BSC/SuSy-Dataset) | COCO real photos with several generated sources | Older generators and mixed underlying licenses. Its COCO card summary is not a substitute for individual image terms. |
| [AGIQA-3K](https://huggingface.co/datasets/strawhat/agiqa-3k) | Known generated images and human quality ratings; used in the original panel | No real-image class, and it would repeat the older six-generator scope. |
| `Shanmuk4622/ai-detection-dataset-v3` | Search index describes matched real/AI pairs | Public Hub lookup failed with an unavailable/private result; no acquisition or provenance verification. |
| OpenFake Reddit or LAION real rows | Additional source variety | Reddit source class is weak provenance; some LAION “real” rows are illustrations or posters. DOCCI gives a clear camera-capture basis. |

## Research record

Used Exa for five searches across benchmark design, paired sources, source access, and real-camera provenance: 25 requested result slots, before deduplication. This is a query-volume count, not 25 verified sources. Read primary dataset cards and project/paper pages for the selected source and key alternatives.

Used the Hugging Face plugin `hub_repo_details` to inspect AGIQA-3K, SuSy, Synthbuster Plus, and OpenFake Tiny. The plugin's `dataset_search` call returned unavailable. Continued with the Hugging Face Dataset Viewer skill and public Hub/Viewer APIs. The final pinned manifest is `evidence/benchmarks/ai.json`.

## Acquisition checks

The frozen panel contains 60 distinct delivered hashes and 60 distinct exact-caption groups. The minimum dHash distance is 15 within this panel and 16 against the original 30 images. No final candidate triggered the keyword or duplicate exclusion rule. The 60 delivered files total 5,396,135 bytes.

Only **2 of 60** Dataset Viewer assets match the published upstream SHA-256. The other 58 must be treated as viewer representations, not verified original files. Both hashes remain in the manifest. This is acceptable for a fixed delivered-view comparison, but it prevents claims about native-byte forensic performance. Acquisition also cached two unused Nano Banana candidate assets before the insufficient stratum count was detected; these are excluded from the 60-image panel and remain ignored.
