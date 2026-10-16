# Native AI-image sources for metadata validation

The original Synthbuster release is the strongest source available for this pass. It keeps the original generator bytes, with shared real-reference IDs and common prompts. Its terms permit non-commercial research. A small test of native files found large metadata differences between generators. It did not show balanced camera fields across real and generated classes.

Research used four Exa queries with 40 requested results across two rounds. This counts requested result slots, not unique verified sources. The Hugging Face tool could inspect Community Forensics, GenImage++ and the original GenImage repository. Dataset search failed with unavailable-tool errors for GenImage and UniFD. The official public Hub APIs supplied search results and metadata for pinned releases instead. Image bytes came from the authors' archives, not Dataset Viewer images.

| Candidate | Primary evidence and native access | Decision |
| --- | --- | --- |
| [Synthbuster original release](https://zenodo.org/records/10066460) | Author ZIP has 9,000 images from nine generators, shared RAISE IDs and `prompts.csv`. Explicit CC-BY-NC-SA 4.0. Author readme states that images were not resampled or recompressed. ZIP member CRC provides a local integrity check. | Select Firefly, DALL-E 2, DALL-E 3 and Midjourney v5. Preserve unchanged member bytes. |
| [GenImage++](https://huggingface.co/datasets/Lunahera/genimagepp) | Author release, revision `4160fbc103730771d3641139d3b995a3c2dbc719`; native `.tar.zst` archives, prompts and CC-BY-NC 4.0 card. Several large archives have multiple parts. | Feasible later source for newer generators. Camera EXIF and C2PA coverage were not measured in this pass. |
| [Community Forensics](https://huggingface.co/datasets/OwensLab/CommunityForensics) | Author card covers 2.7 million generated images and model-specific terms. Native image bytes may be embedded in parquet, but original provider-export equivalence is not established by the container alone. | Do not assume that embedded bytes preserve original provider metadata. Earlier native parquet pilot files had no EXIF; that result does not cover this release. |
| [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) | Author release links 1,000 real and 1,000 generated images per diffusion domain through Drive, plus older CNNDetection data. | Underlying real-source terms and native-camera preservation need separate checks. Repository MIT terms do not grant rights to every source image. |
| [DIRE / DiffusionForensics](https://github.com/ZhendongWang6/DIRE) | Original author release links Baidu and RecDrive. Folder structure separates images, reconstructions and DIRE representations. | Reconstruction residuals are not native photographs. Native camera and C2PA coverage were not verified. |
| [DF40](https://github.com/YZY-stack/DF40) | Explicit CC-BY-NC 4.0; most public image data uses frame extraction and face cropping. Original video access is separate. Entire-face synthesis and editing files do not need that preprocessing. | Face/video benchmark with a different target. Processed frames do not support a native camera-metadata comparison. |
| [SDAIE](https://github.com/Ekko-zn/SDAIE) | Author camera-metadata study trains its EXIF task on real photographs only. Drive links provide EXIF training photos and detection sets. | Supports the need to distinguish camera-supervised visual features from metadata supplied during detection. It does not establish camera EXIF on generated images. |

## Measured native pilot

Twenty original members were extracted: four each from Firefly, DALL-E 2, DALL-E 3, Midjourney v5 and SDXL. None has any of the four tested camera fields: ISO, exposure time, F-number or focal length. The pilot is recorded in `visual-triage/data/metadata-v3/research/ai/native-pilot.json`.

| Generator | Native decoded format | EXIF present | Camera fields present | C2PA marker present |
| --- | --- | ---: | ---: | ---: |
| Firefly | JPEG, despite archive `.png` names | 0/4 | 0/4 | 4/4 |
| DALL-E 2 | PNG | 4/4 | 0/4 | 0/4 |
| DALL-E 3 | PNG | 0/4 | 0/4 | 0/4 |
| Midjourney v5 | PNG | 0/4 | 0/4 | 0/4 |
| SDXL | PNG | 0/4 | 0/4 | 0/4 |

C2PA and JUMBF markers do not prove a valid signature or trusted signer. Native Firefly files were sent for separate C2PA checks. Detect PNG and JPEG format from the bytes; file extensions can be wrong. Later candidate filenames use the decoded native format. Their bytes stay unchanged.

The Firefly prompt table contains controls called `exif mode`, manual aperture, shutter speed and FOV. Its generation URL includes those controls. They are inputs to image rendering, not measured camera settings. Do not present them as factual camera EXIF. DALL-E 2 files contain EXIF, but this does not prove that the fields record optical capture.

## Fresh grouped acquisition

The frozen panel has 40 fresh prompt families. Four generated versions per family give 160 AI images. Its source manifest is `visual-triage/data/metadata-v3/research/ai/source.json`, SHA-256 `3f34c17e6d742e078b5d54dd1a109bdb11bc262f49a4083af65f24ea35604c71`. Eligibility SHA-256 is `249c2197f24b6bed15520ec54b2e63a971d8b95a80c4932ef10973918580e2ea`. All 155 previously recorded RAISE source IDs and their prompt hashes were excluded. Ten families go to training, ten to development and twenty to final testing. Each generator thus supplies 10 training, 10 development and 20 final images. Families are selected in a fixed source-ID hash order. Replacements keep each family slot and split.

Selection excludes all RAISE source IDs from prior downloads, reviews and frozen datasets. It also excludes canonical prompt hashes shared with those families. All four generator variants stay together. Prompt hashes prevent the same prompt from crossing local splits under different source IDs. These are fresh prompts from an older generator release. This is not an external benchmark of new generators.

Review checks all four variants in each family for photographic medium before inference. Keep photos of physical sculptures and monuments. Exclude images dominated by flat paintings, illustrations, screens or visible renders. Also exclude non-photographic text/logo designs and unsafe sexual content. Keep photographic defects and borderline photo processing. If any member fails, reject the whole family and use the next unused family. Detector outputs did not guide selection. Review checked 63 complete families (252 images), excluded 23 families and kept 40. Accepted family slots and splits stayed fixed during replacement. The final source bytes total 351,616,931 bytes. Native hash, decoded RGB hash, archive CRC, prior-ID and prior-prompt checks passed. The panel builder separately checks approximate visual overlap and pHash matches across splits.

The [acquisition script](../../tracks/metadata_v3/acquire_ai.py) reads the original ZIP central directory. It downloads only the compressed byte ranges for selected members. Before keeping bytes, it checks local headers, native length and ZIP CRC. Its record includes native and decoded RGB hashes, source groups and canonical prompt hashes. It also keeps provenance labels, format, dimensions, metadata coverage and licence details. Zenodo publishes an MD5 for the whole archive. This was not verified because the full 12.4 GB archive was not downloaded.

```sh
visual-triage/.venv/bin/python visual-triage/tracks/metadata_v3/acquire_ai.py --prepare
# Complete family-level photographic eligibility before freezing the source panel.
visual-triage/.venv/bin/python visual-triage/tracks/metadata_v3/acquire_ai.py
```

The native comparison must report how camera-field coverage differs between sources. This is an observed source difference. A separate controlled test can remove or swap metadata. Label each change, and do not describe inserted fields as native measurements. Fully generated images and AI edits to camera photos need different origin labels. Edits can keep EXIF from the first capture even when some pixels are generated.

## Frozen metadata coverage

| Generator | Images | Native format | EXIF present | Four camera fields present | C2PA marker present |
| --- | ---: | --- | ---: | ---: | ---: |
| Firefly | 40 | JPEG | 0 | 0 | 40 |
| DALL-E 2 | 40 | PNG | 40 | 0 | 0 |
| DALL-E 3 | 40 | PNG | 0 | 0 | 1 |
| Midjourney v5 | 40 | PNG | 0 | 0 | 0 |

Each generator supplies 10 training, 10 development and 20 final images. All four variants of a source ID share one split. Keep C2PA marker counts separate from the parser's signature and trust results. Native AI camera fields were not balanced. The native comparison measures observed source differences. Label controlled metadata removal or swaps as separate interventions.
