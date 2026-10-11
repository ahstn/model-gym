# KonIQ quality expansion

This panel uses human technical-quality scores for authentic photographic defects. The model receives unchanged 1024 × 768 JPEG images from the [primary KonIQ-10k deposit](https://doi.org/10.18419/darus-2435), version 1.0. Human raters used these study images without a pristine reference. No model outputs guide selection.

The target is the raw five-point ACR mean, preserved as `quality_mos_1to5`. Higher values mean better quality. The source anchors are bad, poor, fair, good and excellent. We preserve rating counts, distributions, standard deviation and the separate normalised MOS for audit. The normalised score is not the target. There is no human aesthetics target.

| Split | Images | Bad | Poor | Fair | Good | Low quality |
|---|---:|---:|---:|---:|---:|---:|
| Training | 150 | 38 | 37 | 38 | 37 | 75 |
| Development | 100 | 20 | 30 | 25 | 25 | 50 |
| Final test | 100 | 25 | 25 | 25 | 25 | 50 |

The intervals are `[1,1.8)`, `[1.8,2.6)`, `[2.6,3.4)`, `[3.4,4.2)` and `[4.2,5]`. Low quality is strictly below 2.6. The complete source contains 208 bad, 1,452 poor, 4,249 fair, 4,161 good and three excellent images. The panel has equal low and control classes; it does not have five equal score bands or a production score distribution. The official development partition has only 21 bad images, so we use 20 bad and 30 poor cases there.

We retain the [authors' official partitions](https://github.com/subpic/koniq/blob/1c271570ac83aeb6aa8e7ff25449f3299b0d2e53/metadata/koniq10k_distributions_sets.csv), pinned to commit `1c271570ac83aeb6aa8e7ff25449f3299b0d2e53`. That CSV contains a normalised MOS; only its partition field is used. Raw labels come from the primary deposit. The selection seed is `20261012`. The complete selection is frozen before inference.

The native Flickr photo ID defines the content group. All 350 selected group IDs are distinct. Current attribution identifies photographers for 45 selected images. We keep each known photographer in one split. Current attribution is unavailable for the remaining 305 images, so complete photographer isolation cannot be claimed. Official random partitions and image similarity checks do not eliminate every possible related scene.

## Rights and acquisition

The [primary deposit's terms](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2FDARUS-2435) licence metadata under CC BY 4.0, with exceptions for the photos' individual Creative Commons licences. The [authors' paper](https://arxiv.org/abs/1910.06180) states that they selected licences that allow editing and redistribution. The [dataset page](https://database.mmsp-kn.de/koniq-10k-database.html) makes the dataset available to the research community. This panel relies on that published declaration for noncommercial research, as requested by the user.

Individual public Flickr `ImageObject` records confirm CC licences and attribution for 45 selected images. The evidence retains each canonical photo page, author, licence URL, acquisition time and source page hash. During the broader lookup pass, Flickr returned HTTP 429. We stopped requests and preserved those failures. We excluded records with a known incompatible current licence. For the remaining selected images, the source authors' CC declaration supplies the research rights basis; their current individual licence and attribution are not independently confirmed. Do not publish image bytes with a blanket CC BY statement. Retain the source citation and each available licence record.

Citation: Vlad Hosu, Hanhe Lin, Tamas Sziranyi and Dietmar Saupe. *KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment*. IEEE Transactions on Image Processing, 29, 4041–4056, 2020. [DOI](https://doi.org/10.1109/TIP.2020.2967829).

The primary score table MD5 is `0bc91efacab484f87f7aaf22bd0fd9b4`, independently checked against the deposit. The native archive is public file `72289`, 5,608,799,013 bytes, advertised MD5 `85f076d7a43b2e306f7080814f69af3e`. Selected members use valid HTTPS range requests and must match ZIP sizes and CRC values. The full archive is not downloaded, so its full MD5 is not independently verified. Manifest SHA-256 hashes freeze each downloaded member and its decoded RGB pixels. No resize, crop or re-encoding is applied.

The Hugging Face search plugin was unavailable. Public Hub APIs verified the [IQA-PyTorch mirror](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/tree/f97c4f5dbdbdea7039bc9b80db808313347b1bef), but acquisition uses the primary archive. Viewer previews are not used.

## Validation limits

The builder compares image bytes, decoded RGB pixels and dHash against 430 prior records: 150 earlier images and 280 images from the previous two tracks. It stops for exact matches or dHash distance of four or less pending review. These checks cannot rule out every transformed scene or unknown model pretraining exposure. Prior final images are not eligible for this final panel.

The larger panel supports more development comparisons, but it does not determine how many GEPA proposals will remain useful. Keep the final set sealed until the search ends. Repeated use of the same development labels can still cause overfitting. LIVE Challenge remains blocked by its current access form, SPAQ needs explicit rights and scene groups, and AGIQA's normalised mixed-perception score is unsuitable for this technical ACR target. A separate native-MOS external test is still required for cross-dataset transfer evidence.

Rebuild from `visual-triage/`: `uv run --locked python tracks/quality_v2/build_dataset.py`. Selected image bytes are stored under ignored `data/tracks/quality_v2/`. See [selection](selection.json), [manifest](dataset.json), [source research](../../research/expansion-2026-10-11/quality-sources.md) and `evidence/`.
