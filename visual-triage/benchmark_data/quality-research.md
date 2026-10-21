# Quality and aesthetics benchmark selection

Built on 2026-10-11. The panel uses **60 new photos from the official UHD-IQA test split**. It has native human image-quality scores. It has **no native human aesthetic score**. The assistant's aesthetic assessment is a separate model reference, not human gold.

## Why this source

[UHD-IQA](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html) was already listed in the project's IQA research. Its primary release states that all photos are CC0 and were published on Pixabay before 2019. Ten expert raters judged each photo twice, giving 20 ratings per image. The official test split contains 900 photos. A fixed seed selects six photos from each native-MOS decile, then shuffles their neutral IDs. Selection uses no test-model outputs.

This is a limited test. UHD-IQA favors stock photos with strong aesthetics. Its lowest score decile is relative to that curated population. It does not replace a broad bad-photo or defect benchmark. Human ratings were collected at width 3,840 pixels. The experiment sends full-frame views with a maximum edge of 512 pixels. This can hide noise and blur. Native-MOS correlations therefore measure transfer across views, not agreement with human labels for the exact delivered image. The [paper](https://arxiv.org/html/2406.17472) documents the rating setup and curation.

The native MOS uses a normalized 1–100 slider, stored on a 0.01–1 scale. The manifest preserves it. A second field uses `1 + 4 * (MOS - 0.01) / 0.99` for a 1–5 display. This is an affine scale conversion, not a fitted calibration. The human score covers overall image quality, so it is also not a pure measure of technical defects. No aesthetic target is invented.

## Acquisition and provenance

The [Hugging Face UHD-IQA-JPG mirror](https://huggingface.co/datasets/V-NovaLtd/UHD-IQA-JPG) is pinned at `49423354643300e9eb5a9f090ded4c6c697c41f0` and declares CC0. It contains recompressed JPEG files. The builder instead reads original image members from the [official archive](https://datasets.vqa.mmsp-kn.de/archives/UHD-IQA/UHD-IQA-database.zip) through HTTP byte ranges. It does not download the 10.7 GB archive as a whole.

The builder pins the archive ETag, size, metadata digest, member CRC, original image SHA-256, and delivered image SHA-256. The archive has no immutable release URL. These checks detect changes on rerun; they do not turn its ETag into a cryptographic signature. The manifest retains the original Pixabay URL and author account for each image. Media and source caches stay in the ignored `data/` directory.

The official archive server had an expired HTTPS certificate during acquisition. Only the public, credential-free data requests used a certificate exception. The script defaults to verification and requires `--allow-expired-source-certificate` to reproduce that exception. No system trust settings were changed. This limits transport verification of the first download. The primary dataset page and Hugging Face metadata were read with normal certificate checks.

Pinned source checks:

| Item | Value |
| --- | --- |
| Official archive ETag | `"67093f54-27d6c14ac"` |
| Official archive bytes | `10694169772` |
| Metadata ZIP SHA-256 | `a020225f5972776c66249eac316e0dd5e605e90e9addd5fd4f0ca4df52cd0850` |
| Metadata CSV SHA-256 | `90a46279b4532689efb7959f44c53ffdd902d43870d6532de87291909ccd2940` |
| Selection seed | `20261011` |

Run from `visual-triage`:

```sh
uv run --locked python benchmark_data/quality_builder.py --allow-expired-source-certificate
```

The script preserves the full frame and aspect ratio, applies EXIF orientation, converts to RGB, and uses JPEG quality 92 with no chroma subsampling. It does not upscale. It checks exact delivered hashes against the original 30 cases. A dHash distance of at most two marks candidate near duplicates for review. This screen cannot prove that all scenes are independent. Contact sheets contain six images each at their full delivered size, with only neutral IDs.

## Alternatives checked

| Source | Human targets | Decision |
| --- | --- | --- |
| [PARA](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Personalized_Image_Aesthetics_Assessment_With_Rich_Attributes_CVPR_2022_paper.html) | Separate quality and aesthetics scores, both 1–5 | Best target match, but excluded because its archived terms restrict transfer and commercial use of data and research outputs. No PARA images were acquired or sent to OpenRouter. |
| [MSC](https://www.nature.com/articles/s41597-026-06816-0) | Aesthetics; 100 votes per image | No native technical-quality target. Narrow natural-scene content and edited variants. Source photos have mixed rights despite its release tag. |
| [KonIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html) | Quality; 5-point ACR | Better general defect spread, but the primary page grants research access without an explicit blanket media license. |
| UHD-IQA | Human image quality at 4K | Selected for explicit CC0 photo rights and accessible original test images; narrower quality spread and no aesthetic gold are disclosed above. |

PARA metadata was checked through the [author's page](https://web.xidian.edu.cn/ldli/en/dataset.html) and the [Hugging Face mirror](https://huggingface.co/datasets/chitradrishti/PARA) at `ef3b3e2cf9ccbccc3599821c60c08ee8a2d9c7c3`. Its test CSV has 3,000 rows. The archived README is more restrictive than the web footer. The archive password is public on the author page. Only its metadata and license text were downloaded, into the ignored source cache. The user selected the clearer-license fallback.

## Research coverage

Exa discovery used four searches with five requested result slots each, plus two fetch calls covering four pages. These 20 search slots include repeated sources; they are not 20 unique validated papers. Primary claims were checked against the papers, author pages, archive metadata, and release terms. Hugging Face plugin dataset search failed twice because its backend tool was unavailable. Direct Hub and Dataset Viewer APIs then verified the PARA and UHD-IQA mirrors, revision IDs, license metadata, split metadata, and the PARA preview failure.

## Verification

The locked project runtime reproduced all 60 delivered JPEGs from the frozen source cache. Original hashes, delivered hashes, RGB mode, maximum dimensions, test membership, selection order, and six images per decile passed checks. The JPEGs total 3,895,956 bytes; the largest is 181,141 bytes. The manifest SHA-256 is `5d6ee801963328710f1cc37aef475507dae2fe855799d11c41b0e8dc027c117d`.

A separate agent reproduced all 60 JPEGs and the exact CSV selection. It found no encoded or decoded duplicates against the new AI-detection panel or the original 30 cases. With the same Lanczos dHash calculation on all panels, the minimum Hamming distances were 14 to the AI panel, 9 to the original cases, and 12 within the quality panel. These values rule out the chosen distance-at-most-two candidate threshold; they do not prove scene independence.

The held-out status applies to this experiment's prompt development. We cannot verify whether the foundation models saw these public photos or labels in pretraining.
