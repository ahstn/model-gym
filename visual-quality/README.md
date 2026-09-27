# Visual-quality pilot builder

This package acquires public training images and assembles the [proposed visual-quality pilot](docs/datasets/low-quality-and-controls.md). It preserves source records, creates blur/noise/JPEG variants, and writes separate training, development, and test manifests. It does not call a teacher model or assign quality scores.

The [2026-09-27 assembly record](docs/datasets/pilot-10k-assembly.md) contains the completed 10,800-image build counts, source revisions, manifest hashes, and verification results.

## Run

Run commands from this directory. Python 3.13 and `uv` are required.

```sh
uv sync --locked
uv run visual-pilot build --config configs/pilot-10k.json --root data
uv run visual-pilot verify --config configs/pilot-10k.json --root data
```

The full build uses network access and several tens of GB of local storage. Image files and source caches are ignored by Git. The dependency versions are fixed in `uv.lock`. Do not place a second build with a different seed or acquisition quota in the same data root.

Acquisition can run one source at a time. Repeat a command to resume its cached downloads.

```sh
uv run visual-pilot acquire --source megalith --root data
uv run visual-pilot acquire --source vizwiz --root data
uv run visual-pilot acquire --source plotqa --source clevr --root data
uv run visual-pilot acquire --source owned_controls --root data
uv run visual-pilot assemble --root data
```

`assemble` needs the completed source records written by `acquire`. It validates images, excludes exact duplicates, joins near-duplicate photo families, assigns splits, generates variants, and publishes manifests only when the quotas and split checks pass. Download failures are reported by each source adapter; they are never replaced with placeholder images.

## Contents and source policy

| Component | Train | Development | Final test | Acquisition |
|---|---:|---:|---:|---|
| Broad photos | 5,000 | 200 | 200 | Seeded sample across the pinned Megalith-CC0 metadata release |
| Authentic faults | 1,000 | 100 | 100 | Official VizWiz train images with at least 3 of 5 votes for a named defect |
| PlotQA charts | 400 | 40 | 40 | Unique training images; record the pinned mirror and delivery format |
| CLEVR renders | 400 | 40 | 40 | Original training PNG members from the pinned archive |
| Owned controls | 200 | 20 | 20 | Five types of drawings, grouped by template |
| Distorted photos | 3,000 | 0 | 0 | Three variants of each of 1,000 photos already in train |
| **Total** | **10,000** | **400** | **400** | Development and test are additional to the pilot training count |

Source spares cover corrupt files, duplicates, and whole-family allocation. Unused spares remain in the acquisition cache and do not appear in training. The development and final-test images come from upstream training partitions; upstream validation and test partitions remain untouched.

Megalith selection spans the source pool. It applies resource limits of 32 million pixels and 48 MiB per source image. It checks the explicit CC0 license on each metadata row. These limits affect coverage and must be retained in the dataset description. The source's aesthetic model score is preserved as source metadata; it is not a human rating or a technical-quality label. Photographer IDs are absent from this release, so photographer-level separation cannot be verified.

Megalith's metadata hash does not match the delivered bytes for every item. Many mismatches have swapped source dimensions and no remaining EXIF rotation, which suggests upstream orientation correction; that transform is not documented by the publisher. The builder records both hashes and the dimension comparison, and pins the acquired bytes with SHA-256. For an exact replay, retain the byte cache: a metadata revision alone does not freeze mutable image URLs.

VizWiz selection balances the dominant human-voted defects and retains all source votes. It uses the official quality annotation repository and the official image archive. The image and annotation terms are recorded separately. This pilot does not depend on the LIVE-Meta access form, and it does not add LIVE-Meta and VizWiz counts together.

The broad-photo component is a seeded candidate sample. It has not yet been divided into human-reviewed low, middle, and high quality bands. Authentic defect examples and synthetic damage supply the initial quality spread. The independent audit in the research plan remains necessary before training.

## Generated views and labels

The generator preserves each original photo. For derivatives, it applies EXIF orientation, converts to RGB, and reduces the long edge to at most 1536 pixels while preserving aspect ratio. All three variants of one parent use the same base view. The manifest records source and delivered dimensions, resizing, operator, parameters, seed, and software versions.

This first version uses Gaussian blur, Gaussian noise, and JPEG compression. Three configured levels do not establish a human preference order. The resized view can also hide source defects, so do not compare its score with a full-resolution original without a common viewing policy.

All quality and aesthetics fields start as null, with their training masks disabled. Human defect votes and existing source-model outputs stay in `source_labels`. Domain controls have `photo_domain_eligible=false`; ordinary photos remain pending an eligibility audit. No failed request or unreadable image becomes a low-quality label. The assembled data is ready for review and labeling, not for an unqualified training run.

## Output files

Outputs are in `data/pilot-10k-v1/`:

- `dataset.json`: seed, source revisions, path policy, and manifest names.
- `train.jsonl`, `development.jsonl`, `test.jsonl`: image records and task masks.
- `config.json`: frozen source quotas and seed.
- `software.json`: Python and dependency versions, module hashes, and dependency-lock hash.
- `report.json`: counts, manifest hashes, validation results, and pending audits.
- `invalid-images.jsonl`, `excluded-images.jsonl`: rejected input records and reasons.
- `near-duplicate-candidates.jsonl`: conservative perceptual matches grouped into one split.

Resolve `image_path` and relative license evidence paths against `image_root` in `dataset.json`, relative to that file's directory. Source caches and license evidence remain under `data/sources/`. Keep them with the pilot. File hashes describe acquired bytes; decoded hashes describe RGB pixels after orientation correction. The dHash check is a duplicate-candidate heuristic, not proof that two images share a source.

For a known external evaluation set, supply an exclusion manifest:

```sh
uv run visual-pilot assemble --root data --exclude /path/to/evaluation-exclusions.jsonl
```

Each exclusion record can contain `sha256`, `decoded_sha256`, a globally unique `id`, or both `source` and `source_id`. These exact matches are removed before splitting. The builder keeps upstream train boundaries, but that alone cannot prove absence of overlap with every IQA or visual-reasoning benchmark. `report.json` keeps the full external benchmark audit pending.

## Verification

```sh
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run visual-pilot verify --root data
```

Tests cover source selection, bounded archive reads, cached downloads, corruption handling, duplicate removal, family splits, generated-image reproducibility, and source-label isolation. The final verification command reads all listed files, checks their byte hashes, and rejects family or pixel leakage across splits.
