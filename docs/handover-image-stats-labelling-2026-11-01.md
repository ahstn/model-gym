# Handover: using image-stats in data labelling

From the image-stats session, 1 November 2026. It is for the photo-scoring labelling session (the Luna/DeepSeek 2,000-image run and the Jev/Liquid/Haiku decision-model comparison) as it resumes labelling.

It covers three uses of the metrics:

1. Quality control: checking label decisions.
2. Strengthening label decisions by passing the metrics to the graders.
3. Context for the trained model.

## Where things stand

- **Branch:** `codex/consolidate-model-gym-2026-10-21`. Everything is pushed, and the branch has no PR.

| Commit | What it did |
|---|---|
| `41b23b8` | Added the `image-stats/` package |
| `3b528b1` | Review and verify fixes: noise-aware gate, MPO JPEG metadata, phase-agnostic blockiness, a parity test against visual-quality views |
| `56952f4` | Mug control set, from a separate session; see [mug control-set evidence](../image-stats/data/controls/mug-2026-10-31/README.md) |
| `7c68b02` | Two new evidence views: native-resolution `detail_crop` and `gamma` |

- **Nothing is wired into visual-quality yet.** No label, prompt or manifest has changed. Running image-stats and using its output is the labelling session's call.
- **Algorithm version:** `image-stats-v2`. A `Reference` (corpus percentiles) records the version and `max_edge` it was built with, and `describe` refuses a mismatch.
- **Labels:** still as described in [photo-scoring-2026-10-31/README.md](https://github.com/ahstn/model-gym/tree/ddcbc4d422b4d4b931bf01b2f00b7938935052fe/visual-quality/experiments/photo-scoring-2026-10-31/README.md). None are approved for training. The reference scores are independent assistant judgments, not human gold.

## What image-stats provides

Full detail is in [image-stats/README.md](../image-stats/README.md). In short:

- **`analyse`** computes 33 deterministic metrics on the same pixels as the frozen views. The decoding policy is EXIF transpose, ICC to sRGB, alpha composited on white, then a LANCZOS resize to 1536. `mise run parity` checks it pixel for pixel against `visual_quality.views._render`. It runs in about 0.2 s per image on a laptop CPU. It needs no GPU, no network and no non-commercial weights.
- **`read_metadata`** reads facts from the original file: format, bytes per megapixel, JPEG quality estimate, chroma subsampling and camera EXIF. It never reads capture time, GPS or serial numbers. With a source path, `analyse` reads the metadata from the original file and adds `metadata.source_blockiness`. Resizing destroys the 8×8 JPEG grid, so this must come from the original file.
- **Flags:** `uniform_frame` (the only gating flag), `near_black`, `near_white`, `shadow_clipping`, `highlight_clipping`, `graphic_like`, `monochrome` and `blocking`.
- **`calibrate` and `describe`** rank each metric against a corpus `Reference` and render a 12-line plain-language block for a prompt.
- **`evidence`** writes six inspection images:
  - `sharpness`: a heatmap that outlines where the crop came from
  - `detail_crop`: a 512 px square at native resolution around the sharpest tile
  - `gamma`: a view brightened or darkened so the median brightness becomes mid-grey
  - `noise_residual`, `clipping` and `spectrum`

## Evidence on the current 72 trial views

Measured on 1 November against `visual-quality/experiments/photo-scoring-2026-10-31/images/` and `scores-72.csv`. The percentiles use a reference built from the same 72 images (in-sample). There are only 72 images, and the reference scores are assistant judgments, so treat all of this as direction, not proof.

**Spearman ρ between metric and technical score:**

| Metric | Ref | Luna | DeepSeek | Jev | Liquid | Haiku |
|---|---:|---:|---:|---:|---:|---:|
| `sharpness_peak` | +0.69 | +0.72 | +0.67 | +0.70 | +0.71 | +0.64 |
| `tenengrad` | +0.62 | +0.68 | +0.63 | +0.65 | +0.65 | +0.60 |
| `local_contrast` | +0.60 | +0.63 | +0.64 | +0.65 | +0.66 | +0.57 |
| `luma_entropy` | +0.56 | +0.58 | +0.58 | +0.60 | +0.59 | +0.48 |
| `top_colour_share` | −0.48 | −0.50 | −0.50 | −0.58 | −0.52 | −0.49 |
| `noise_sigma` | 0.00 | +0.01 | +0.09 | +0.10 | +0.05 | +0.10 |

`sharpness_peak` against Ref aesthetics is +0.62. Aesthetic and technical scores move together on this set.

**Every grader overrates soft images.** Grader error (grader minus Ref) correlates most strongly with `sharpness_peak`. It ranges from ρ −0.27 (Jev) to −0.48 (Luna, Haiku). The table splits images at the 25th sharpness percentile. Below it are 18 images (Ref technical mean 1.73); above it are the other 54 (mean 3.53).

| Grader | Bias, soft quarter | MAE, soft quarter | Bias, rest | MAE, rest |
|---|---:|---:|---:|---:|
| Luna | +0.58 | 0.58 | +0.22 | 0.32 |
| DeepSeek | +0.53 | 0.58 | +0.21 | 0.45 |
| Luna/DeepSeek mean | +0.56 | 0.56 | +0.21 | 0.36 |
| Jev 1.14 | +0.33 | 0.42 | +0.14 | 0.33 |
| Liquid d1.1 | +0.20 | 0.32 | +0.00 | 0.36 |
| Haiku 5.5 | +0.31 | 0.34 | −0.01 | 0.39 |

These figures cover all 72 images, including the 24 development cases used for prompt selection. They will differ slightly from the 48-case held-out table in the photo-scoring README.

**Flags on the 72:**

| Flag | Cases |
|---|---|
| `uniform_frame` (gate) | T0571 (black) and T0984 (red), both Ref 1.00 and both reference-"uncertain" |
| `blocking` | T0340 (Ref 2.3) and T0587 (Ref 1.1) |
| `shadow_clipping` | T0571, T0587 and T0857 (Ref 2.5) |
| `highlight_clipping` | T0549 (Ref 1.4) |
| `monochrome` | T0571, T0958 (Ref 3.3) and T1843 (Ref 3.6) |
| `graphic_like` | T0984 |

**The gate catches a real grader failure.** T0984, the red frame, scored 4.00 from DeepSeek, 2.40 from Luna, 2.15 from Jev, 1.60 from Haiku and 1.41 from Liquid. Ref scored it 1.00.

**Stats alone carry signal, but far less than the graders.** A leave-one-out linear fit on four percentile features (`sharpness_peak`, `local_contrast`, `luma_entropy` and `top_colour_share`) gives:

| Target | Stats-only MAE | Predicting the mean |
|---|---:|---:|
| Ref technical | 0.61 | 1.02 |
| Ref aesthetics | 0.57 | 0.82 |

The graders reach 0.32–0.48. Stats can correct or flag a score, but cannot replace a grader.

## 1. Quality control: checking label decisions

Use the stats to choose which labels a person reviews, never to overwrite a label automatically.

**a. The gate.** When `analysis.gate == "uniform_frame"`, apply the 1.00/1.00 policy floor and route the case to scope review. Keep the grader's native scores alongside the floor. This is a separate rule from the decision models' `excluded_probability >= 0.50` floor, so record which rule fired. Per the photo-scoring README, never floor an image only because it is blurry, dark, or contains a sign or artwork. `uniform_frame` is deliberately narrower than that: it requires almost no edges and almost no tile contrast once the noise level is subtracted.

**b. The soft-overrate check.** Send a case to review when `sharpness_peak` is at or below the 25th corpus percentile and the grader's technical score is 2.0 or higher. On the 72:

| Grader | Flagged | MAE of flagged | Errors ≥ 0.75 caught |
|---|---:|---:|---:|
| Luna/DeepSeek mean | 11 | 0.74 | 4 of 11 |
| Jev | 9 | 0.66 | 4 of 9 |
| Liquid | 7 | 0.46 | 2 of 4 |

Raising the threshold to 2.5 flags fewer cases with higher error per flag, but catches fewer. Expect precision to be useful and recall to be modest. Large errors also come from noise, scope and content, not only softness.
- **Example:** T0677 (Ref 1.7) was scored 2.80 by both Luna and DeepSeek. Its `sharpness_peak` is at the 5th percentile.
- **Watch for deliberate blur:** shallow depth of field reads as soft (finding 2 in the mug handover). A reviewer should confirm the blur is a defect before lowering a score.

**c. Secondary cues.** These need more cases before they become rules; for now, use them to choose what to review.
- **Noise:** T0231 has luma and chroma noise at the 97th and 98th percentiles. Ref gave it 2.4, but every grader gave 3.2–4.0, and the sharpness check misses it because the image is sharp (93rd percentile). `noise_sigma` has no rank correlation overall, so treat high noise together with a high score as a reason to review, not as evidence.
- **Clipping and blocking flags:** check that the grader's evidence text mentions the defect. For blocking, prefer `source_blockiness` from the original file to the view value.
- **`monochrome` on a high-scoring image** (T0958, T1843) is usually deliberate black and white. It is not a defect.

**d. Distortion-ladder sanity.** The 2,000-image manifest has 120 controlled families. Each has a parent and severities 1–3, with `operator` set to `gaussian_blur`, `gaussian_noise` or `jpeg_compression`.
- **First, check the stimuli.** Confirm that each variant really degrades:
  - Blur families: `sharpness_peak` and `tenengrad` fall as severity rises.
  - Noise families: `noise_sigma` and `chroma_noise_sigma` rise.
  - JPEG families: `blockiness` rises, or `read_metadata` shows the JPEG quality falling.

  A family whose stats don't move is a bad stimulus. Exclude it from grader monotonicity counts rather than blaming the grader.
- **Then judge the graders.** The current counts are 94 of 120 non-increasing for Luna, 78 for DeepSeek and 85 for the mean. Recount them on stimulus-verified families only.
- **Add a holistic-override rate,** after PACE ([arXiv 2609.22942](https://arxiv.org/abs/2609.22942)): the share of severity-3 variants whose technical score is at or above their parent's. It measures technical scores that follow overall appeal, and it needs no new labels.

**e. Choosing what to review.** When choosing images for human or reference review, stratify by sharpness, brightness and noise percentile bands rather than sampling at random. Over-sample cases where the graders disagree with each other, or where the stats disagree with the graders. A random sample from 1,640 families will under-represent the soft and dark tail, where the errors are.

**What not to do:**
- **Don't compare noise, blockiness or `high_frequency_ratio` across different encoding histories.** Re-encoding to WebP cut measured noise 7–24× in the mug set. Stratify by format and source when the pool mixes Megalith, VizWiz and generated images.
- **Don't build the reference from the 72 alone.** Its percentiles are coarse and in-sample.

## 2. Strengthening label decisions: metrics in the grader prompt

The prompt block exists (`image-stats describe`), but it has never been tested with a grader. Measure it before adopting it.

**Trial design.** Use the 48 held-out cases. If budget allows, add a fresh reviewed sample that includes the 2,000-run soft and dark tail. Run each grader in three arms:

| Arm | Prompt | Purpose |
|---|---|---|
| A | Current prompt, unchanged | Baseline |
| B | Current prompt plus the image's own stats block | Effect of stats |
| C | Current prompt plus the stats block of a different image from the same sample | Anchoring control |

- **Why arm C:** if scores follow the swapped stats, the grader is reading the text rather than the image. "Of Chains and Gains" ([arXiv 2610.33203](https://arxiv.org/abs/2610.33203)) shows that RL-trained quality models exploit text cues that correlate with the target score.
- **What to report:** technical and aesthetic MAE and bias overall, and in the soft quarter, where we expect the gain. Use the paired bootstrap from `visual-quality/docs/evaluation/protocol.md`.
- **Decision models:** changing the Jev and Liquid prompts invalidates their prompt selection. Re-select on the 24 development cases before scoring held-out cases.

**Before the trial, fix or drop the `focus_area` line.** It is measured relative to the image's own sharpest tile, so a uniformly soft frame reads as sharp everywhere. T0677 shows "Frame area close to peak detail: 89%" next to "Detail in the sharpest region: low (5th percentile)", which could mislead a grader. Either remove it from `PROMPT_METRICS` in `image-stats/src/image_stats/describe.py`, or reword the label and bump `ALGORITHM_VERSION`.

**Other recommendations:**
- **Technical task only, at first.** The metrics are technical. Putting them in the aesthetics prompt risks turning aesthetics into a sharpness proxy, which is the holistic bias PACE describes.
- **Build the reference from the labelling pool.** Use the 2,000 views, at `max_edge` 1536, with one fixed `Reference` per study. Record its path and hash in the run record.
- **Gate before the call.** Skip the grader for `uniform_frame` cases. That saves cost, and the T0984 result shows graders can't be trusted on them.
- **Evidence images are a separate trial from the text block.** They cost image tokens and only help graders that accept several images.
  - `gamma`: most useful when `luma_mean` is low or `shadow_clip` is high. On T0587 it reveals fingers over the lens and a phone that are invisible in the view.
  - `detail_crop`: most useful when the original is larger than 1536 px.

  The approach follows Tool-IQA ([arXiv 2606.16082](https://arxiv.org/abs/2606.16082)) and IQA-T1. Test the text block and the images as separate arms, so their effects can be told apart.

## 3. Context for the trained model

These options are in order of evidence and cost. Whatever the student sees in training, the inference API must compute in exactly the same way: the same `ALGORITHM_VERSION`, the same `Reference` and the same `max_edge`.

**a. The gate, at inference.** This is a deterministic check with no learned component, and it costs about 0.2 s of CPU. It returns the policy floor without calling the model. It is ready now.

**b. Evidence views as extra image inputs.** The compact student (MiniCPM-V-4.6, at 4× or 16× visual compression) loses fine detail by design. `detail_crop` supplies native pixels from the sharpest region, which is the information Q-Probe and Zoom-IQA recover by zooming. If the student uses it at inference, train with it, and compare against a run without it at the same token budget.

**c. Numeric stats in a small correction head.** Feed the student's own score, plus image-stats percentiles, into a small residual head that predicts a correction. This is the MetaRefine pattern ([arXiv 2610.40487](https://arxiv.org/abs/2610.40487)).
- **Evidence:** MetaRefine's own ablation shows that image size and aspect ratio add almost nothing (−0.03% PLCC). Its gains come from rich features. Our stats are richer than image size, and the soft-image bias above is exactly what such a head could correct. This is still untested.
- **Training:** fit the head on folds that the student was not trained on, so the head does not learn the student's training-set errors.

**d. Stats as text in the student prompt.** Only adopt this if arm B beats arms A and C for the graders, and the inference API renders the identical block. Otherwise the student can learn the shortcut that arm C is designed to detect.

**e. Data curation.** Use the stats to stratify training batches and the split checks by softness, exposure and encoding, and to keep the soft and dark tail represented. It is cheap and carries no risk at inference.

## Commands

Run these from `image-stats/` with `uv run`. Each one takes seconds for 72 images and a few minutes for 2,000 with `--workers 8`.

The 72 trial views:

```sh
uv run image-stats analyse ../visual-quality/experiments/photo-scoring-2026-10-31/images \
  --output out/photo-72/stats.jsonl --workers 8
```

The 2,000 views. Paths in `manifest-2000.json` are absolute paths on the machine under `/home/ahstn/git/model-gym/`, so run this there.

```sh
jq -c '.cases[]' ../visual-quality/experiments/photo-scoring-2026-10-31/manifest-2000.json > out/cases-2000.jsonl
uv run image-stats analyse --manifest out/cases-2000.jsonl --root / --id-key case_id \
  --path-key image_path --source-key source_path --output out/photo-2000/stats.jsonl --workers 8
uv run image-stats calibrate out/photo-2000/stats.jsonl --output out/photo-2000/reference.json --name photo-2000
uv run image-stats describe out/photo-2000/stats.jsonl --reference out/photo-2000/reference.json \
  --output out/photo-2000/prompts.jsonl
uv run image-stats evidence <view.png> --output-dir out/evidence
```

With `--source-key source_path`, the record's `metadata` is read from the original file and includes `source_blockiness`. Records with an `error` key are images that could not be read, and `analyse` exits with status 1 when there are any.

Join the output to `scores-72.csv` or `scores-2000.csv` on `case_id`. When analysing a directory, the record `id` is the file name, such as `T0028.png`; in manifest mode it is `case_id`. The soft-overrate check in Python:

```python
from image_stats import Reference

reference = Reference.load("out/photo-2000/reference.json")


def soft_overrate(record: dict, technical_score: float) -> bool:
    peak = record["analysis"]["metrics"]["sharpness_peak"]
    return reference.percentile("sharpness_peak", peak) <= 25 and technical_score >= 2.0
```

## Open decisions for the user

1. Adopt the gate and the soft-overrate check for labelling QC now, or wait for the 2,000-image recount.
2. Run the three-arm prompt trial (A, B and C), and decide which graders it covers.
3. Choose how to handle `focus_area` in the prompt block: drop it or reword it.
4. Choose where the evidence images go: grader trials, the student, or neither.

## Sources

- **Papers:**
  - Tool-IQA ([2606.16082](https://arxiv.org/abs/2606.16082))
  - Of Chains and Gains ([2610.33203](https://arxiv.org/abs/2610.33203))
  - PACE ([2609.22942](https://arxiv.org/abs/2609.22942))
  - MetaRefine ([2610.40487](https://arxiv.org/abs/2610.40487))
  - QualiRAG ([2601.18195](https://arxiv.org/abs/2601.18195)), training-free metadata context for graders
  - JLD ([2610.05967](https://arxiv.org/abs/2610.05967)), a candidate for checking distortion-ladder step sizes
- **Survey:** these came from a review of [Awesome-Image-Quality-Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment) on 31 October. The papers already covered in `visual-quality/docs/models/iqa.md` were not repeated.
