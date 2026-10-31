# Q-ReAlign Mini hosted Space: seven-image comparison

The third-party [VOIDER/Q-ReAlign-0.8B Space](https://huggingface.co/spaces/VOIDER/Q-ReAlign-0.8B) scored the seven selected images with separate quality and aesthetics calls. Its source loads `q-future/Q-ReAlign-Mini-0.8B` and resizes the longest edge to 768 pixels. Its reported native [0,1] values are converted to [1,5] using `1 + 4 * score`; native values and responses are preserved in [comparison.csv](comparison.csv) and `space-results/`. Its reported values have four decimal places, so conversions inherit that rounding.

Each cell below shows **technical / aesthetic**. The first six photos are held out in the earlier comparison. T2000 is a known poster and is not part of the photo accuracy metrics. The Space has no photography eligibility gate; its poster scores are raw model outputs and are not replaced with the 1.00/1.00 dataset-policy floor.

| Image | Reference | Q-ReAlign | Luna | DeepSeek | Mean | Haiku | Jev | Liquid |
|---|---|---|---|---|---|---|---|---|
| T0028 | 1.30 / 1.70 | 1.97 / 2.35 | 2.00 / 2.80 | 1.10 / 1.60 | 1.55 / 2.20 | 1.40 / 2.10 | 1.43 / 1.53 | 1.25 / 1.55 |
| T0192 | 2.10 / 2.90 | 3.48 / 2.88 | 3.00 / 3.10 | 2.90 / 3.00 | 2.95 / 3.05 | 2.50 / 3.20 | 2.68 / 3.02 | 2.71 / 2.83 |
| T0714 | 2.60 / 2.50 | 3.58 / 2.74 | 3.00 / 2.90 | 3.00 / 2.80 | 3.00 / 2.85 | 2.50 / 2.70 | 3.02 / 3.15 | 3.04 / 2.73 |
| T0965 | 3.50 / 3.80 | 4.72 / 3.09 | 3.80 / 4.00 | 3.70 / 3.60 | 3.75 / 3.80 | 4.20 / 4.00 | 3.42 / 3.59 | 3.52 / 3.55 |
| T0914 | 4.10 / 3.10 | 4.82 / 2.53 | 4.00 / 3.40 | 3.60 / 3.20 | 3.80 / 3.30 | 3.30 / 2.90 | 4.01 / 2.99 | 3.49 / 2.93 |
| T0724 | 4.50 / 3.90 | 4.92 / 3.21 | 4.40 / 4.00 | 3.70 / 3.10 | 4.05 / 3.55 | 4.00 / 3.50 | 3.98 / 3.54 | 3.94 / 3.44 |
| T2000 (poster) | 1.00 / 1.00 | 4.27 / 3.56 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

The following six-photo errors are descriptive against prior assistant references; they are not human accuracy and are too small or selectively sampled for a reliable model ranking. Inputs are not fully matched: the Space resizes to 768 pixels while the earlier assessors saw up to 1,536-pixel views. No inference about general superiority follows from these six cases.

| Model | Technical MAE | Aesthetic MAE | Mean MAE |
|---|---:|---:|---:|
| Jev | 0.303 | 0.270 | 0.287 |
| Liquid | 0.382 | 0.222 | 0.302 |
| Luna + DeepSeek | 0.417 | 0.258 | 0.338 |
| Haiku | 0.433 | 0.283 | 0.358 |
| DeepSeek | 0.483 | 0.267 | 0.375 |
| Luna | 0.417 | 0.383 | 0.400 |
| Q-ReAlign Mini | 0.896 | 0.479 | 0.687 |

The underlying scope rule, references, teachers and full corpus are documented in [the earlier report](../photo-scoring-2026-10-31/README.md). The Q-ReAlign Space model revision, image hashes, per-call status and elapsed times are stored with each response. This is Q-ReAlign Mini, not the preferred Q-Align OneAlign checkpoint. Results are experimental and unapproved for training.
