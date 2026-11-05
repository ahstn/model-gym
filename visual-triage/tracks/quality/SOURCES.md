# Quality track: source and sample

Use **KADID-10k** for this low-quality stress test. Its human labels cover severe defects, and the model receives the exact 512 × 384 PNG pixels used in the rating study. This removes the resizing problem from the earlier UHD-IQA pass. It tests added distortions of photos; it does not establish accuracy on naturally poor photos or on aesthetics.

The [authors' dataset page](https://database.mmsp-kn.de/kadid-10k-database.html) describes 10,125 distorted images from 81 source photos, 25 defect types and five severity levels. Each image has 30 human degradation ratings. The [authors' study protocol](https://ar5iv.labs.arxiv.org/html/2001.08113) used a pristine reference beside each test image. Decider receives only the test image, so the rating context still differs.

## Labels and split

The native DMOS scale runs from 1 to 5, with higher scores meaning less visible degradation. We preserve that value, its source string and the reported variance. The anchors are severe annoyance at 1, annoyance at 2, slight annoyance at 3, visible but acceptable degradation at 4, and no visible degradation at 5. We do not invert, fit or rescale scores. The source has no human aesthetics label.

The sample contains 160 images from 80 source photos. Seed `20261011` fixes the source groups and image selection before model results. Every group contributes one low-quality image and one control. Groups remain within one split:

| Split | Images | Source groups | DMOS < 2.6 | DMOS ≥ 2.6 |
|---|---:|---:|---:|---:|
| Train | 40 | 20 | 20 | 20 |
| Development | 40 | 20 | 20 | 20 |
| Final test | 80 | 40 | 40 | 40 |

The five score bands are `[1,1.8)`, `[1.8,2.6)`, `[2.6,3.4)`, `[3.4,4.2)` and `[4.2,5]`. The full sample has 40, 40, 40, 20 and 20 images in those bands. Low-quality classification uses the fixed rule **DMOS < 2.6**, including for model predictions. These class boundaries are our evaluation rule, not a source-provided binary label. The test has 40 independent content groups, not 80 independent scenes. It covers all 25 distortion types; train and development each cover 21, with all 25 present across those two splits. See the [coverage counts](../../ARCHIVE.md). It tests unseen content with known distortion families.

## Rights and acquisition

The [authors' OSF repository](https://osf.io/xkqjh/) links a **CC BY 4.0** licence through its [public metadata](https://api.osf.io/v2/nodes/xkqjh/). The original paper says the source photos came from Pixabay. We retain the authors' attribution and source links. This research use relies on the published licence; it is not a claim of separate author approval for OpenRouter.

Citation: Hanhe Lin, Vlad Hosu and Dietmar Saupe. *KADID-10k: A Large-scale Artificially Distorted IQA Database*. QoMEX, 2019. [DOI](https://doi.org/10.1109/QoMEX.2019.8743252).

We fetch only selected PNG members from the [Hugging Face mirror](https://huggingface.co/datasets/myzhao1999/kadid10k/tree/251f5882d7c6f5072b37dae3f16a0ba6babf6c75), pinned to commit `251f5882d7c6f5072b37dae3f16a0ba6babf6c75`. The score file hash matches the [authors' OSF checksum](https://api.osf.io/v2/files/5eafe61069d3e100cade8671/). All 160 fetched images match the authors' individual OSF SHA-256 checksums. The builder also checks ZIP size and CRC values. The full 3.07 GB archive is not downloaded, so its advertised archive hash is recorded but not independently verified. No image is resized or re-encoded. Native images stay under ignored `data/tracks/quality/`.

The source archive host has an expired TLS certificate. The pinned Hugging Face mirror avoids a certificate exception. Its Dataset Viewer is not used because its JPEG previews would change the rated pixels. Hugging Face connector searches failed with `tool not returned by tools/list`; public Hub APIs supplied the revision and file metadata instead.

## Research decision

Exa covered source selection, licensing and rating protocol in three rounds. The audit count is 30 requested results, including two failed five-result requests, and 20 returned results. [KonIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html) remains a useful next authentic-distortion test. It was not selected here because human ratings used 1024 × 768 images, and its photos have individual Creative Commons terms recorded through YFCC100M. KADID offers a smaller, controlled first test with a direct human target.

Pretrained exposure to this public dataset is unknown. Exact byte and decoded-pixel checks found no duplicates among the prior 150 images. No prior image pair had a dHash distance of four or less. These checks cannot rule out transformed scene overlap or model training exposure. Score error, rank correlation and low-quality classification answer different questions and should remain separate in the results.

Rebuild from `visual-triage/`: `uv run --locked python tracks/quality/build_dataset.py`. The builder stops if the frozen selection, source labels or image hashes change. See [selection](selection.json), [manifest](dataset.json) and [source evidence](evidence/).

Archived source snapshots and raw records require [archive restoration](../../ARCHIVE.md). The guide identifies the original paths and hashes.
