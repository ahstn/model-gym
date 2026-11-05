# Image context for quality and AI review

Research checked on 11 October 2026. No paid model calls were made for this work. The context extractor is tested locally; improvements to Decider or Luna remain untested.

## Decision

Start with an ablation that supplies statistics from the delivered image. Keep native dimensions, EXIF availability and file properties in the audit record. Do not supply those fields to the AI detector. Add camera settings only in a separate arm with balanced coverage across classes and sources.

EXIF can carry useful acquisition information. However, its presence, device identity and resolution can also reveal a dataset instead of image quality or origin. Metadata must support visible evidence. It must not determine the label by itself.

## Primary evidence

| Source | Verified contribution | Consequence for this experiment |
| --- | --- | --- |
| [SPAQ, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.html) and [author implementation](https://github.com/h4nwei/SPAQ) | The study collected human quality and attribute ratings for 11,125 pictures from 66 smartphones. It recorded EXIF and included a multi-task variant that uses EXIF tags. | This supports research on camera information for authentic photography. It does not establish that adding EXIF text to Decider improves quality scores. The retrieved material does not verify a specific prompt-time improvement. |
| [EXIF As Language, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_EXIF_As_Language_Learning_Cross-Modal_Associations_Between_Images_and_Camera_CVPR_2023_paper.html) | The authors learned associations between image patches and camera metadata. The learned representation supports forensics and camera calibration. Splice localization can use image features without test-time metadata. | EXIF supervision and supplying EXIF in a prompt are different interventions. Camera traces can help forensic models, but this is not direct evidence for whole-image AI detection. |
| [Image Splice Detection via Learned Self-Consistency, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Jacob_Huh_Fighting_Fake_News_ECCV_2018_paper.html) | The authors used EXIF supervision to learn whether patches could share one imaging pipeline. The target was manipulated and spliced content. | Patch inconsistency is a forensic signal. Missing EXIF is not the method, and it is not evidence that an image was generated. |
| [Fake or JPEG?, 2024 preprint](https://arxiv.org/html/2403.17608) | Controlled GenImage experiments showed that JPEG compression and image dimensions can become detection shortcuts. Removing these biases changed cross-generator results and compression robustness. | Balance the delivered encoding and geometry. Audit native size and EXIF availability by class before a camera arm. A gain that disappears after source balancing is not sufficient evidence for deployment. |
| [NIQE, author paper](https://live.ece.utexas.edu/publications/2013/mittal2013.pdf) and [BRISQUE, author paper](https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf) | These methods model natural scene statistics. NIQE compares features with a natural-image model; BRISQUE learns a mapping to human quality ratings. | Simple statistics are descriptive features, not replacements for these fitted quality models. Compare a frozen blind IQA baseline with the metadata ablation when resources permit. Neither method is an AI-origin certificate. |

The requested [Awesome Image Quality Assessment list](https://github.com/chaofengc/Awesome-Image-Quality-Assessment) was checked as a discovery source. It links no-reference, aesthetic, spatial and generated-image quality work. The list is not validation evidence for a selected method.

Exa supplied five search queries across acquisition-aware IQA, EXIF forensics, detection shortcuts, classical blind metrics and a targeted EXIF-quality follow-up. These calls requested 25 results successfully. Four initial schema errors requested another 20 results but returned no sources. Search matches were screened; the conclusions above use author papers and repositories. The Hugging Face paper connector returned `UNAVAILABLE`; this work makes no claim that HF search succeeded. Saved excerpts may be truncated and are not full-paper archives. See [the evidence directory](evidence/) and [search log](../../ARCHIVE.md).

## What the extractor measures

[image_context.py](../../tracks/image_context.py) uses Pillow and the standard library. It introduces no dependency. It accepts a delivered image and an optional, explicitly supplied native source. Input bytes are limited to 32 MiB per file, and decoded images to 20 million pixels. Multiframe delivered images are unavailable rather than silently reduced to the first frame.

Native source data contain encoded dimensions, decode status and EXIF availability: present, absent or unknown. These are audit fields. Optional camera fields use a fixed numeric allowlist: ISO, exposure seconds, f-number and focal length in millimetres. Missing or invalid camera values are null. Make, model, software, timestamps, serials, GPS, descriptions, paths, labels and dataset identifiers are excluded.

Delivered statistics use EXIF orientation correction, RGB conversion and a BOX downsample to a maximum edge of 256 pixels. Small images are not enlarged. Transparent images use a stated white compositing policy. ICC profiles are not applied. These choices describe this measurement view; they are not assumptions about the native sensor or the exact display pipeline. Use opaque, normalized benchmark inputs for the primary arm.

| Feature | Meaning | Interpretation limit |
| --- | --- | --- |
| Width, height and measurement dimensions | Geometry after orientation correction, and geometry used for statistics | Resolution is not a quality score. Measurements from a resized view are not native measurements. |
| Encoded RGB luma p05, p50 and p95 | Quantiles of `0.2126 R + 0.7152 G + 0.0722 B`, normalized to 0–1 | This is encoded luma, not calibrated physical luminance. A dark scene can be correct. |
| Luma p95 minus p05 | A robust tonal spread | This is not a monitor contrast ratio or a calibrated perceptual contrast score. |
| Endpoint fractions at 1/255 and 254/255 | Proportion of measurement pixels near black or white | Bright backgrounds and silhouettes can be intentional. Resizing can hide small clipped areas. The values do not establish sensor clipping. |
| RGB mean and standard deviation | Channel distribution in the measurement view | Scene content and colour style affect the values. They do not establish a colour defect. |
| Mean absolute adjacent luma difference | Average local intensity change across horizontal and vertical neighbours | Texture, edges, noise and blur all affect it. It is not a calibrated sharpness or noise estimate. A one-pixel view returns null. |

Entropy and a noise estimate are omitted. Their interpretation would need more assumptions, and a textured scene can resemble noise in a simple residual metric. No scalar feature is converted into a quality or origin decision.

## Interface and frozen records

```python
from tracks.image_context import (
    canonical_context_json,
    extract_image_context,
    model_context,
    validate_model_context,
)

audit = extract_image_context(delivered_bytes, native_source=native_bytes)
request_context = model_context(audit, mode="pixel_stats")
validate_model_context(request_context)
frozen_json = canonical_context_json(request_context)
```

`model_context` supplies a strict whitelist. Its default includes delivered statistics, dimensions, measurement constants and fixed guidance. It excludes every native audit field. Camera mode adds only the four numeric camera settings or null. The extractor requires `include_camera=True` to read these settings; `camera_fields` can narrow the explicit allowlist.

Validation rejects unknown nested fields, free text in numeric fields, non-finite values, invalid policies and oversized JSON. It permits only the fixed schema, guidance and measurement strings. The JSON limit is 2,500 UTF-8 bytes. Invalid types raise `TypeError`; invalid fields, policies or ranges raise `ValueError`.

Freeze the audit record, the exact model context and its canonical SHA-256 before evaluation. Keep these in a derived context manifest that preserves the original labels, group IDs and splits but has its own manifest hash. Hash the exact dictionary with the benchmark's existing digest function. Freeze the Pillow version and extractor source hash as audit provenance. During a run, verify and use the frozen context; do not recompute it silently. Keep file hashes and identifiers out of the model context.

## Ablation needed before adoption

Use the same images, labels, splits, score anchors and delivered policy in all arms. Compare image-only, image plus delivered statistics, and a statistics-only diagnostic. The diagnostic tests whether features alone predict labels or expose source imbalance. It is not a production detector. Run a blind IQA baseline separately so its fitted score is not confused with raw descriptive features.

For camera settings, first audit coverage and distributions by source, device and class. Run camera-only and image-plus-camera arms only with balanced availability. Include an EXIF-removal intervention on the same images: removing metadata must not cause an AI label. Use source-grouped and device-grouped validation, and reserve new camera sources and generators for final tests. Missing camera values must remain unknown; do not impute origin labels.

For quality, report human-score MAE, rank correlation, threshold recall and calibration. For AI detection, report Brier error, false-positive rates and paired uncertainty. Inspect results by defect, source and transform. Before future adoption, select context arms on development data, then test once on untouched final groups. This pass uses a pre-frozen paired final diagnostic; its results cannot select a new context variant. Neither the previous KADID panel nor this pixel-statistics test can establish an EXIF-camera gain. The normalised AI panel must not regain native source cues through context.

The fixed guidance explicitly says that resolution is not quality, missing EXIF does not imply AI, camera settings are unverified claims, and pixel statistics do not prove blur, noise, exposure failure or AI origin. These warnings reduce incorrect interpretation; only the ablation can establish whether the context helps.

Local validation passed 22 focused tests plus Ruff checks. Tests cover deterministic JSON, resource limits, tiny and corrupt images, nested camera tags, GPS and serial exclusion, malformed EXIF, orientation, alpha, animation, source/view separation and context injection. No runner, protocol or prompt was changed by this module work.

Archived source snapshots and raw records require [archive restoration](../../ARCHIVE.md). The guide identifies the original paths and hashes.
