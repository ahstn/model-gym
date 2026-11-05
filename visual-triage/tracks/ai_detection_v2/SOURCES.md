# Extended matched-photo panel

The acquisition target is 400 images: 160 training, 80 development, 80 final and 80 external. Every split has equal real and generated classes. This panel extends the Decider AI detection track. It does not reuse the prior Luna results as Decider evidence.

[SynthCLIC](https://huggingface.co/datasets/marco-willi/synthclic/tree/feff43b543929916bac097e30217cc3fdc09c32d) supplies the main panel at revision `feff43b543929916bac097e30217cc3fdc09c32d`. Its 2,163 CLIC2020 photographs each have four generated counterparts. [The author paper](https://arxiv.org/abs/2602.12381) describes captions from Gemini and outputs from SD3 Medium, FLUX.1-dev, FLUX.1-schnell and Imagen 3. Shared `image_id` values define provenance pairs. These are semantic matches, not pixel-aligned reconstructions.

Training and development use SD3 Medium and the two FLUX models from native training groups. Final images use Imagen 3 from native test groups. A fixed hash seed ranks complete source families within a bounded native cohort, and each family can occur in one local split only. The main candidate cohort contains the first 100 professional and first 100 mobile native training families and the first 125 native test families. The external cohort contains all 200 native test families, minus all 111 previously reviewed family IDs. These cohorts reflect native row-group locality, so this is not a uniform sample of the full release. Photographer and capture-session isolation are not established. This tests generator transfer as well as new image groups. The external panel uses 40 unused native test families from [the author-linked Synthbuster mirror](https://github.com/marco-willi/clip-cues), revision `dbfb72f1ee96e953ee5cff80c58832fe89e1d2b5`. Only SDXL from the original licensed Synthbuster release is selected. This external test contains one generator and a new real-photo source. The earlier plan to use three generators and native training families was changed before any model inference to keep native retrieval below 3 GB. All source IDs in the previous manifest, provisional acquisition and full medium-review ledger are excluded.

## Licence chain

The SynthCLIC card provides only `license: cc`. This generic tag is not treated as a complete grant. The paper states that the dataset release is subject to the original photographs' licences. Official CLIC archive ZIP directory entries identify each source photo as mobile or professional. Each manifest row records that subset and its applicable terms.

- [Mobile CLIC2020 terms](https://data.vision.ee.ethz.ch/cvl/clic/LICENSE_mobile_2020.txt) provide CC0 1.0.
- [Professional CLIC2020 terms](https://data.vision.ee.ethz.ch/cvl/clic/LICENSE_professional_2020.txt) provide the Unsplash licence, which permits download, modification and use. It excludes a competing image service.
- [The official CLIC archive](https://archive.compression.cc/2021/tasks/index.html) links professional CLIC2020 data and its licence. [An organiser statement](https://groups.google.com/g/clic-2020/c/pVCa3pH0nQk) confirms mobile CC0 and professional Unsplash terms.
- Original [Synthbuster](https://zenodo.org/records/10066460) outputs use CC-BY-NC-SA 4.0. [RAISE](http://loki.disi.unitn.it/RAISE/download.html) permits non-commercial research and education.

Use is restricted to this non-commercial research pass. Image binaries remain outside version control. The generic SynthCLIC card leaves more licence ambiguity than the original Synthbuster release; the source chain and this limitation remain explicit. Cite Willi et al., *Synthetic Image Detection with CLIP: Understanding and Assessing Predictive Cues*; the CLIC2020 organisers; Bammey, *Synthbuster*; and Dang-Nguyen et al., *RAISE*.

## Acquisition controls

The same medium rule applies to both pair members. Keep photographs of physical sculptures and monuments. Exclude flat paintings, illustrations, screens or visible renders that dominate the image. Keep imperfect photographic images, visible generator defects, blur, noise and strong processing. A whole pair is excluded if either image fails. Before inference, reviewers examined 206 candidate pairs: 162 main pairs and 44 external pairs. They excluded two main pairs and four external pairs. Each rejected slot received the next unused eligible hash-ranked family with its generator and split fixed. Previously accepted pairs stayed in their original slots. All 200 selected pairs have positive eligibility decisions. Acquisition uses no detector output, and held-out appearance details remain separate from prompt authors.

The delivered view uses the previous policy: EXIF orientation, RGB, a 512 × 512 centre crop, then JPEG quality 92 with no chroma subsampling. Both classes receive this transform. Source bytes, delivered bytes and decoded RGB values receive SHA-256 hashes. Verification excludes encoded and decoded duplicates against the original 150 images, KADID panel and first Synthbuster panel. Cropping can remove useful context; original compression histories can remain different.

Native bytes embedded in pinned parquet files remain in the ignored cache. Retrieval first reads metadata column ranges, then downloads only selected native image row groups. The initial selected row groups contain 2.600 GB. These bytes preserve embedded source EXIF when present; the dataset may itself have processed images before release. Partial catalogue responses from the earlier Viewer attempt remain separate and are not used as selected image bytes. The frozen native audit found that all 400 embedded source files are PNG and none contains EXIF. Camera-setting fields are therefore unavailable for this AI panel. [Audit evidence](../../ARCHIVE.md) contains only presence, dimensions, format and hashes. Inference uses delivered pixels and excludes source metadata, GPS, identifiers, file names and pair partners.

See [the research memo](../../research/expansion-2026-10-11/ai-sources.md) for deferred alternatives. The public HF plugin remained unavailable on a fresh attempt. Viewer rate limits delayed catalogue acquisition, so official pinned parquet files supplied the complete metadata and native image bytes. The parquet reader uses pyarrow 26.0.0 as an explicit acquisition-only dependency. Its cache uses sparse files, metadata ranges and row-group hashes to avoid a full release download.

From the `visual-triage` directory:

```sh
uv run --locked --with pyarrow==26.0.0 python tracks/ai_detection_v2/build_dataset.py --prepare-candidates
# Complete medium eligibility decisions before acquisition can freeze the manifest.
uv run --locked --with pyarrow==26.0.0 python tracks/ai_detection_v2/build_dataset.py
uv run --locked python tracks/ai_detection_v2/build_dataset.py --verify
```

Archived source snapshots and raw records require [archive restoration](../../ARCHIVE.md). The guide identifies the original paths and hashes.
