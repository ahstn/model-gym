# Matched-photo source decision

Use 60 caption-matched Synthbuster pairs: one real RAISE photograph and one generated photograph per source ID. The panel contains 120 images: 30 for training, 30 for development and 60 for the final test. DALL-E 3, Midjourney v5 and Stable Diffusion XL each contribute 20 generated images. Each source family occurs in one split only.

This is a controlled follow-up to the earlier OpenFake panel. Both classes now depict photographs derived from the same scene descriptions. The images are not pixel-aligned, and their composition or lighting can still differ. The [original Synthbuster release](https://zenodo.org/records/10066460) recommends matching generated images with their source RAISE photographs. Its authors edited the prompts to favour photorealism.

## Labels, rights and acquisition

The [author-linked Hugging Face mirror](https://huggingface.co/datasets/marco-willi/synthbuster-plus/tree/dbfb72f1ee96e953ee5cff80c58832fe89e1d2b5) supplies binary labels, generator names and shared `image_id` values. These are provenance labels, not assistant opinions. The pinned revision is `dbfb72f1ee96e953ee5cff80c58832fe89e1d2b5`. The [CLIP-Cues source repository](https://github.com/marco-willi/clip-cues) links this mirror and describes its matched real/synthetic pairs.

The original Synthbuster images use CC-BY-NC-SA 4.0. [RAISE permits non-commercial research and education](http://loki.disi.unitn.it/RAISE/download.html) and asks users to cite Dang-Nguyen et al., *RAISE: A Raw Images Dataset for Digital Image Forensics* (2015). Cite Quentin Bammey, *Synthbuster: Towards Detection of Diffusion Model Generated Images* (2023), for the synthetic data. The mirror has no separate licence field. This panel uses only three generators from the original licensed release; later Synthbuster+ additions are excluded. Image binaries stay outside version control.

The builder reads all 2,800 rows from the mirror's native test split. These contain 200 source families with one real photo and 13 generated variants each. It ranks complete families with a fixed hash seed, then assigns 60 initial pair slots: 20 per generator, with five training, five development and ten test pairs. Each excluded pair is replaced by the next unused source ID in hash order, with the same generator and split. This local split does not preserve the mirror's train/test purpose; all our rows originate from its native test set.

An independent dataset agent screened medium eligibility before any model calls. The same rule applies to both pair members: exclude clear paintings, illustrations, renders, and photos dominated by artwork or screen content. Keep imperfect photo-like images, visible generation errors, blur, noise and strong photographic processing. Borderline photographic cases remain eligible. The agent reviewed 135 candidate pairs from 111 source families: 61 were rejected, 60 selected and 14 accepted but unused after deterministic replacement. The [eligibility ledger](eligibility.json) records every decision. These are assistant medium judgements, separate from the source's binary provenance labels.

The [first provisional panel](../../ARCHIVE.md) is preserved for audit and was never scored. It contained 26 pairs that failed the medium rule. The acquisition agent could inspect all candidates, including final-test pixels, solely to apply this rule. It did not see detector outputs or design prompts. It withheld test-image details from the prompt authors. This controls the known medium confound, but subjective eligibility and source selection still limit how widely the results apply.

Each image receives the same view transform: EXIF orientation, RGB, aspect-preserving resize, a 512 x 512 centre crop, then JPEG quality 92 with no chroma subsampling. This removes output size, aspect ratio and metadata as direct class cues. Cropping can remove useful context. Dataset Viewer assets may already be re-encoded, and source compression histories remain different. Results apply to visible-photo triage, not native-file forensic detection.

The manifest records source and delivered hashes, decoded RGB hashes, labels, source IDs, generator, split and pairing basis. Verification checks exact encoded and decoded duplicates against the prior 150 images. This does not exclude all related scenes or overlap with model pretraining. Mapping, revision or label conflicts stop acquisition before model scoring. Test images must not guide prompt changes.

## Alternatives considered

| Dataset | Evidence | Decision |
|---|---|---|
| [Synthbuster](https://zenodo.org/records/10066460) | Real source photos, caption-matched synthetic photos, nine generators, explicit research-compatible licence | Selected original-generator subset through the pinned HF mirror. Older generators limit conclusions about current systems. |
| [SynthCLIC](https://huggingface.co/datasets/marco-willi/synthclic) | High-quality CLIC photos paired with recent diffusion outputs; [paper](https://arxiv.org/abs/2602.12381) studies semantic bias | Useful next source. The card lists only `cc`, with no precise licence; permissions need a clearer record. |
| [DDA-COCO](https://huggingface.co/datasets/Junwei-Xi/DDA-COCO) | Pixel and frequency alignment of COCO images with VAE reconstructions; [paper](https://arxiv.org/abs/2505.14359) | Strong forensic stress set, but its reconstructed positives answer a different question from fully generated photographs. |
| [DRCT-2M](https://github.com/beibuwandeluori/DRCT) | COCO-based generated images from 16 diffusion variants; release on ModelScope | Useful broader research set. The searched HF release was empty; acquisition and licence review were less direct. |
| [Chameleon](https://arxiv.org/abs/2406.19435) | Generated photos selected because humans mistook them for real photos | Valuable hard external test, but not an exact caption-matched real/fake design for this pass. |

Research used three Exa search rounds, requesting 30 results in total, followed by primary-source reads. `sources_reviewed: 30` counts requested search results, not unique or verified papers. The Hugging Face plugin search tool returned an unavailable-tool error on all three attempted searches. Public Hub and Dataset Viewer HTTP APIs supplied the metadata and image assets instead; no private account data was used.

## Reproduce

From the repository root, using the existing visual-triage environment:

```sh
visual-triage/.venv/bin/python visual-triage/tracks/ai_detection/build_dataset.py
visual-triage/.venv/bin/python visual-triage/tracks/ai_detection/build_dataset.py --verify
```

Acquisition resumes from the ignored source cache. An existing manifest is frozen: rebuilding must reproduce its complete records before delivered files can change. Offline verification needs the prior 150 delivered images for duplicate checks.

Archived source snapshots and raw records require [archive restoration](../../ARCHIVE.md). The guide identifies the original paths and hashes.
