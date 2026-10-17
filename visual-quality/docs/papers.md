# Visual quality paper library

This file records papers selected for further study. It does not replace the detailed [training methods](training/methods.md) or [mixture evidence](training/data-mixture-evidence.md). Use the arXiv ID or DOI as the unique key. Record a new version under the same entry.

## Liao et al. — Image Quality Assessment With Compressed Sampling

| Field | Record |
|---|---|
| Library key | `liao2024compressed` |
| Authors, as listed | Ronghua Liao; Chen Hui; Lang Yuan; Haiqi Zhu; Feng Jiang |
| Identifier | [arXiv:2404.17170](https://arxiv.org/abs/2404.17170); [DOI](https://doi.org/10.48550/arXiv.2404.17170) |
| Version checked | [v2](https://arxiv.org/html/2404.17170v2), revised 11 September 2024; first submitted 26 April 2024 |
| User link | [alphaXiv](https://www.alphaxiv.org/abs/2404.17170) |
| Status | Paper of interest; methods and experiment sections checked on 15 October 2026; no reproduction |
| Topics | No-reference technical IQA; compressed sampling; efficient image representation |
| Priority | Later student-model experiment, after the pilot label and evaluation gates |
| Duplicate handling | The request supplied this same URL twice. This is one unique paper; no second title was inferred. |

**Method.** CL-IQA learns block-based compressed measurements, then maps them into transformer features. CS-IQA removes the adaptive embedding module. A dual branch predicts local scores and their weights. The sampling ratio controls measurements within an image; it is not a claim that fewer training images or fewer human labels are needed. See [method section II](https://arxiv.org/html/2404.17170v2#S2).

**Evidence.** The paper evaluates LIVE, CSIQ, TID2013, and KADID-10K, with MSE training and a reported random 80/20 split. CL-IQA averages five 224 × 224 test crops. Table I reports CSIQ PLCC/SRCC of 0.971/0.965 at 10% sampling and 0.983/0.975 at 50%. Table II shows large losses when fixed-ratio models receive a different sampling ratio. These are author results, not our measurements. See [experiments](https://arxiv.org/html/2404.17170v2#S3).

**Use in this project.** Test compressed measurements only if image processing remains a measured cost after the compact scalar baseline. Compare with resizing and fixed native-detail crops on the same images. Measure latency, memory, and quality error. Keep the sampling ratio in the model and view specification.

**Limits.** The reported experiments do not establish authentic-photo, aesthetics, UHD-IQA, or synthetic-teacher performance. The experiment text does not clearly establish reference-disjoint splits or a separate validation allocation. Verify both before reproducing the result. No official code or weight release was established in this read. This paper supplies neither new training images nor human quality labels.

```bibtex
@misc{liao2024compressed,
  title = {Image Quality Assessment With Compressed Sampling},
  author = {Ronghua Liao and Chen Hui and Lang Yuan and Haiqi Zhu and Feng Jiang},
  year = {2024},
  eprint = {2404.17170},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2404.17170},
  url = {https://arxiv.org/abs/2404.17170}
}
```
