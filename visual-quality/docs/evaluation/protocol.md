# Evaluation protocol

This file defines how we measure image quality (IQA), image aesthetics (IAA), and video quality (VQA) models. It sets the metrics, split rules, model selection, confidence intervals, calibration, and reporting rules. It ends with an audit of the Q-ReAlign baseline code.

Last verified: 2026-09-27

Related files: [benchmarks](benchmarks.md), [experiment plan](../training/experiment-plan.md), [unified models and backbones](../models/unified-and-backbones.md), [IQA datasets](../datasets/iqa.md), [IAA datasets](../datasets/iaa.md), [VQA datasets](../datasets/vqa.md).

## 1. Metrics

Write correlation pairs as `PLCC / SRCC`, in that order, with 3 decimals (keep 4 if the source uses 4). Some sources (for example the Q-ReAlign README and the Q-Align paper) use `SRCC / PLCC`. Convert them before you copy a number.

| Metric | Definition (short) | What it shows | Notes |
|---|---|---|---|
| PLCC | Pearson linear correlation between predicted scores and MOS. | Linear agreement. | Sensitive to scale and outliers. State if it is raw or mapped (see section 2). |
| SRCC (SROCC) | Pearson correlation of the ranks. | Rank (monotonic) agreement. | Not changed by any monotonic mapping. Main ranking metric. |
| KRCC (KROCC) | Kendall tau: (concordant pairs - discordant pairs) / all pairs. Use tau-b when ties exist. | Pairwise order agreement. | Lower in value than SRCC. Easy to read as pair probability: P(correct order) is about (1 + tau) / 2 when there are no ties. |
| RMSE | Root mean square error between predictions and MOS on one stated scale. | Absolute error. | Only valid after you map predictions to the MOS scale. Say which scale and which mapping. |
| MAE | Mean absolute error on one stated scale. | Absolute error, less outlier weight. | Same rules as RMSE. |
| Pairwise accuracy | Share of test pairs where the model orders the two items like humans. | Direct 2AFC agreement. | Use when labels are pairs (preference data, KVQ rank labels, RAIM pairs). Drop or report tied pairs separately. |
| Outlier ratio | Share of predictions outside a confidence band (for example 2 x standard error of the MOS). | Consistency. | Used in VQEG and ITU-T P.1401 style reports. Needs per-item MOS standard error. |
| Distribution metrics | EMD, JS divergence, or NLL between predicted and human rating histograms. | Match of the full rating distribution. | Only where raw rating counts exist (for example AVA vote histograms). See section 7. |

Notes:

- SRCC and PLCC are agreement metrics. They are not accuracy. A high SRCC on one dataset does not mean correct absolute scores.
- Compute metrics per dataset. Never pool items from datasets with different MOS scales into one correlation.
- Report `n` (the number of scored items) next to each result. The Q-ReAlign scorer skips items that fail and only logs a warning (see section 8).

## 2. Raw PLCC vs logistic-mapped PLCC

Many IQA and VQA papers fit a monotonic nonlinear function from predictions to MOS before they compute PLCC and RMSE. This is the VQEG convention. It removes a nonlinear link between model output and the rating scale.

| Form | Function | Source |
|---|---|---|
| 3-parameter logistic | `MOS_p = b1 / (1 + exp(-b2 * (x - b3)))` | Used in the [VQEG FR-TV Phase II final report (2003)](http://atc.umh.es/gatcom/bin/oqam/Referencias/VQEG/VQEGII_Final_Report_Aug2003.pdf), section 4.7. The test plan asked for a 5-parameter model. The 5- and 4-parameter fits did not converge, so VQEG used 3 parameters for all models. |
| 4-parameter logistic | `MOS_p = (b1 - b2) / (1 + exp(-(x - b3) / abs(b4))) + b2` | Common form in IQA papers. The VQEG Phase II report notes that the T1A1 analysis report asked for 4-parameter models. |
| 5-parameter logistic | `MOS_p = b1 * (0.5 - 1 / (1 + exp(b2 * (x - b3)))) + b4 * x + b5` | [Sheikh, Sabir, Bovik, IEEE TIP 2006](https://live.ece.utexas.edu/publications/2006/hrs-transIP-06.pdf): logistic plus a linear term, fitted before CC and RMSE. |
| Current standard | Monotonic mappings (linear, 3rd-order, logistic), RMSE, epsilon-insensitive RMSE, significance tests | [ITU-T P.1401 (01/2020)](https://www.itu.int/rec/T-REC-P.1401-202001-I). Status: page and summary checked, full text not read. |

How large is the effect? The VQEG Phase II report says the logistic step raised PLCC by 0.002 to 0.015 for the strong models, and by 0.098 for one model. It helped weak models more. So a mapped PLCC can hide a poorly scaled model.

Rules:

1. Always report raw PLCC. The Q-ReAlign scorer reports raw PLCC (`scipy.stats.pearsonr` on raw scores, [scorer.py L137](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/scorer.py#L137)).
2. If you also report mapped PLCC, label it `mapped (test-fit)` or `mapped (val-fit)`. Name the function. Fit one mapping per dataset.
3. A test-fit mapping uses test labels. It is a reporting convention, not a deployable model. A deployable mapping must be fitted on train or validation data only.
4. Never compare a raw PLCC from one paper with a mapped PLCC from another paper in one ranking.
5. SRCC and KRCC do not change under a monotonic mapping. Use them when papers differ on mapping.
6. Record fit failures. A logistic fit that does not converge gives a wrong PLCC. Keep the start values and the solver in the log.

## 3. Split rules

The goal: no source content appears in both training and test. A model must not see a different version of a test image, video, or prompt.

| Rule | Why | How |
|---|---|---|
| Split by source content | Synthetic datasets (KADID-10k, LIVE, CSIQ, TID2013, PIPAL) apply many distortions to a small set of reference images. A random split puts the same scene in train and test. | Group all distorted versions by reference ID. Split the groups. |
| Keep derivatives with the parent | Patches (FLIVE / PaQ-2-PiQ), crops, exposure renders, transcodes, and processed versions (KVQ: 600 source videos and 3,600 processed versions) share content. | Assign every derivative to the split of its parent item. |
| Keep frames with the video | Adjacent frames are near copies. | Split by video ID. Never split or bootstrap frames. |
| Hold out generators and prompts for AI-generated media | A model can learn a generator's look or a prompt's content instead of quality. | Make at least one test split with unseen generators and one with unseen prompts. Report both. Watch for near-duplicate prompts. |
| Deduplicate instruction and auxiliary data | Instruction sets (Q-Instruct, Co-Instruct, DQ-495K / DataDepictQA, VQA² data) and benchmarks often reuse images from public IQA and VQA datasets. A different dataset name does not mean different content. | Hash (exact) and perceptual-hash (near-duplicate) all training media against all validation and test media. Remove matches from training. Log the counts. |
| Check dataset families | MaxWell contains DIVIDE (from dossier). AGIQA-20K and AGIQA-3K come from related work; overlap of prompts or generators is not verified. | Before you train on one family member, remove the held-out member's items, or drop that test set. |
| Use official splits when they exist | Published numbers use them (for example LSVQ test and LSVQ 1080p; KonIQ train/test). | Pin the split file and its hash. If you change the split, say so and do not compare with papers. |
| Keep cross-dataset tests untouched | Cross-dataset (out-of-distribution) sets show generalization. | Never train, tune, or select a checkpoint on them. |

## 4. Model selection, seeds, and repeated runs

- Make three manifests per dataset: train, validation, test. Keep the test manifest locked.
- Select the checkpoint, learning rate, loss weights, temperature, and score mapping on validation data only.
- Run the full locked test set once per final configuration. Do not use a 200-item smoke subset as a result.
- Run at least 3 seeds for each configuration that you want to claim is better. Report the mean, the standard deviation, and each seed.
- Some datasets are small (for example KoNViD-1k, LIVE-VQC, CSIQ). For dataset-specific training, papers often repeat random content splits (for example 10 splits) and report the median. If you compare with such papers, use the same number of splits and the same statistic.
- If a method adapts on unlabeled test inputs, label the result `transductive`.

## 5. Confidence intervals

Use a paired bootstrap over independent source units.

1. Define the unit: the reference image for synthetic IQA, the source video for processed video sets, the prompt (or prompt group) for AI-generated media, and the image or video for authentic sets.
2. Draw units with replacement. Take all items of each drawn unit.
3. Compute the metric for model A and model B on the same draw. Store the difference.
4. Repeat 1,000 to 10,000 times. Report the 2.5 and 97.5 percentiles of each metric and of the difference.
5. Call a gain real only if the 95% interval of the difference excludes 0 on the locked test set, and the gain holds across seeds.

Do not bootstrap frames, patches, or distorted versions as if they were independent. That makes intervals too narrow.

Expected interval size. Approximate 95% half-width of one model's SRCC, from the Fisher z method with the Spearman variance factor 1.06. We computed these; they are not from a paper. They assume independent items, so they are too narrow for sets with few references (LIVE, CSIQ, KADID). A paired difference between two models is usually tighter, because both models see the same items.

| Test set | n | Typical SRCC | ± half-width |
|---|---:|---:|---:|
| KonIQ test | 2,015 | 0.94 | 0.005 |
| AVA test | ~19,930 | 0.82 | 0.005 |
| LSVQ test | 7,182 | 0.88 | 0.005 |
| LSVQ-1080p | 3,573 | 0.80 | 0.012 |
| AGIQA-3K | 2,982 | 0.80 | 0.013 |
| LIVE-C | 1,162 | 0.88 | 0.013 |
| KoNViD-1k | 1,200 | 0.87 | 0.014 |
| LIVE | 779 | 0.89 | 0.015 |
| CSIQ | 866 | 0.85 | 0.019 |
| LIVE-VQC | 585 | 0.80 | 0.030 |

So a gain of 0.005 on one small cross set means nothing on its own. Pool cross sets per task, and decide on the paired interval.

## 6. Reporting rules

- Report IQA, IAA, and VQA in separate tables. Aesthetic appeal, technical quality, and video quality are different targets.
- A macro average across tasks is a summary only. Show the per-dataset rows next to it. Mark every regression.
- Keep in-distribution and cross-dataset columns apart. The Q-ReAlign `_avg_srcc` mixes them ([scorer.py L174](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/scorer.py#L174)).
- For each number, state: training data, test split, `n`, raw or mapped PLCC, score direction, and source (our run, or paper table with link).
- Mark numbers copied from papers or model cards as `author-reported`.
- For each model, state the input policy: resolution, crops, frame count and sampling, decode path, cache format, and visual-token count.
- Report cost: GPU type, wall time, visual tokens per item, and throughput.

Result row template:

| Model | Train data | Test set (split) | n | PLCC (raw) | SRCC | KRCC | 95% CI (SRCC) | Seeds | Source |
|---|---|---|---|---|---|---|---|---|---|
| name + revision | datasets + versions | dataset (official test / cross) | 0 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 3 | our run / paper link |

## 7. Calibration

Two kinds of calibration matter.

| Kind | Question | Method | Metric |
|---|---|---|---|
| Score calibration | Is the predicted number on the MOS scale? | Fit a monotonic map (linear or logistic) from model score to MOS on train or validation data. Fit one map per dataset. Apply it unchanged to test. | RMSE, MAE on the stated scale. |
| Distribution calibration | Does the predicted level distribution match the human rating distribution? | Compare the 5 level probabilities with the human histogram, binned the same way. Temperature-scale on validation only. | EMD, JS divergence, NLL; reliability plot. |
| Confidence calibration | Does model confidence match its error rate? | Bin items by confidence (for example entropy of the 5 level probabilities). | Expected calibration error for the top level; error vs confidence plot. |

Rules:

- Min-max normalization does not make MOS scales equal across datasets.
- The 5 level probabilities are not human disagreement by default. Treat them as model uncertainty until you validate them against raw rating histograms.
- Human disagreement (rater spread), model uncertainty, and domain shift are different things. Do not report one as another.
- If a dataset gives only MOS (no raw ratings), do not invent a rating spread for distribution metrics.

## 8. Q-ReAlign baseline code audit

Scope: code at [Q-Future/Q-ReAlign commit f5fd748](https://github.com/Q-Future/Q-ReAlign/tree/f5fd748399ca26e2655b210a609bdcff35953dff) (2026-06-24, latest on `main` today). The earlier audit is in the dossier file `baseline-and-evaluation.md` (26 September 2026). These are code observations, not results of a training run. Status `verified` means we read the file at this commit today.

| # | Finding | Evidence | Impact | Control | Status |
|---|---|---|---|---|---|
| 1 | Hard 5-bin labels. MOS is mapped to one of 5 words by equal-width bins over [lo, hi]. DMOS sets are flipped. Released Q-Align JSON answers are used as-is. | [levels.py L44-L60](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/levels.py#L44-L60): `idx = min(k - 1, int(t * k))`. | Items near a bin edge get different targets. Rater spread is lost. | Compare with soft labels (see [training methods](../training/methods.md)). | verified |
| 2 | Video training frames: 8 frames from `np.linspace` over the whole clip, long side resized to 448, saved as JPEG `quality=90`. The README calls 8 frames a "faithful Q-Align default", but the Q-Align paper says it samples 1 frame per second. | [frames.py L54-L63](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/frames.py#L54-L63); `resize_long: 448` in [onealign.yaml L32](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/configs/onealign.yaml#L32); [Q-Align paper](https://arxiv.org/abs/2312.17090) section 4.1 ("one frame per second"). | Downscaling can remove noise and blur cues. JPEG adds artifacts. 8 sparse frames can miss flicker and short defects. Frame policy differs from the original paper for clips that are not 8 s long. | Lossless cache (PNG) at the same size; then native-resolution crops; then short contiguous clips. Log fps and frame count. | verified |
| 3 | Cached vs uncached eval pixels differ. On the first eval pass, `sample_frames` returns resized frames that were never JPEG-encoded. It saves them as JPEG q90. Later passes load the JPEGs. | [scorer.py L48-L53, L67-L73](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/scorer.py#L48-L73). | Two eval runs of the same checkpoint can see different pixels. Training frames are JPEG, first-pass eval frames are not. | One decode path for train and eval. Lossless cache. Check that cached and uncached scores match. | verified (effect size not measured) |
| 4 | Eval frame cache key is the file name only (`.mp4` removed). The training cache uses parent folder + name. | [scorer.py L48-L50](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/scorer.py#L48-L50) vs [frames.py L31](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/frames.py#L31). | [INFERENCE] Two test videos with the same file name in different folders can share cached frames. | Key the cache by full path hash. | verified (collision risk inferred) |
| 5 | Still images are packed byte-for-byte, with no resize. | [cache.py L9-L11](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/cache.py#L9-L11). | The 448 limit applies to video frames only. The model processor may still resize images. | Record processor limits (max pixels, tokens) for each backbone. | verified |
| 6 | First-token label scoring. For each level word the scorer takes the first token of `" word"` (or `"word"` if empty). It does not check that tokens are single or distinct. | [scorer.py L31-L41](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/scorer.py#L31-L41). | A tokenizer that splits a word, or maps two words to one first token, breaks the 5-class score. | Assert 5 distinct single tokens at the real answer position. Else score full token sequences. | verified |
| 7 | Tokenizer check (our run today). With the space prefix, all 5 words (excellent, good, fair, poor, bad) are one distinct token each in the `tokenizer.json` of Q-ReAlign-Pro-9B, Gemma-4-E4B-it, and MiniCPM-V-4.6. Without the space, some split: Qwen `excellent` and `poor` (2 tokens each); MiniCPM `ex`+`cellent`, `po`+`or`. | Checked with the `tokenizers` library on files from [Q-ReAlign-Pro-9B](https://huggingface.co/q-future/Q-ReAlign-Pro-9B), [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), [MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6). | The space-prefixed path works for these three. The chat template must end so that the next token carries the space. | Re-run the check inside the real chat template for each backbone. | verified |
| 8 | Raw PLCC only; failed items are dropped. `evaluate` uses `pearsonr` and `spearmanr` on raw scores. Items that raise an error are skipped with a warning. | [scorer.py L122-L138](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/scorer.py#L122-L138). | Results are raw PLCC (good). `n` can be smaller than the test set without an error. | Fail the run if `n` is not the full test size. | verified |
| 9 | Checkpoint selection uses test-named sets. `eval.sets: [test_koniq, test_lsvq, live]`, `in_training: true`, `keep_best_n: 1`. The callback keeps the best checkpoint by average SRCC over these sets. In contrast, the Q-Align paper says it reports the final weights after training. | [onealign.yaml L87-L91](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/configs/onealign.yaml#L87-L91); [callback.py](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/callback.py); [Q-Align paper](https://arxiv.org/abs/2312.17090) section 4.1. | Copying this config turns the KonIQ test, LSVQ test, and the LIVE cross-dataset set into model-selection data. This concerns the example config. It does not prove leakage in the released runs. | Add separate validation manifests. Select on them only, or report the final checkpoint. | verified |
| 10 | `limit: 200` also applies to standalone eval. The CLI `--limit` defaults to `None`, and then the scorer uses `cfg.eval.limit`. The limit takes the first 200 records in manifest order, not a random sample. | [onealign.yaml L88](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/configs/onealign.yaml#L88); [scorer.py L123, L147](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/scorer.py#L147); [cli.py L128](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/cli.py#L128). | `qalign eval --config configs/onealign.yaml` scores only 200 items per set. The subset can be biased if the manifest is sorted. | Always pass `--limit 0` for final runs. Check `n`. | verified |
| 11 | Config mix omits AGIQA-20K. `mix: [koniq, spaq, kadid, ava, lsvq]`. The README says all three sizes train on KonIQ + SPAQ + KADID + AGIQA-20K + AVA + LSVQ. The original Q-Align ONE-ALIGN (paper Table 7) also used 5 sets: KonIQ + SPAQ + KADID + AVA + LSVQ. | [onealign.yaml L67](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/configs/onealign.yaml#L67); [README](https://github.com/Q-Future/Q-ReAlign#results); [Q-Align paper](https://arxiv.org/abs/2312.17090) Table 7. | The public config matches the original ONE-ALIGN, not the README's Q-ReAlign recipe. The exact released training manifest is not public. | Treat the README recipe as the claim. Rebuild the 6-set mix yourself and log it. | verified |
| 12 | README result table labels. Order is SRCC / PLCC. The `AGI` column does not name the dataset (a footnote mentions "AIGC10K"). `LIVE` is the synthetic LIVE set in the config (`dmos: true`). The Q-Align row does not match the ONE-ALIGN row of the Q-Align paper (for example KonIQ SRCC / PLCC 0.942 / 0.944 in the README vs 0.941 / 0.950 in paper Table 7). | [README](https://github.com/Q-Future/Q-ReAlign#results); [Q-Align paper](https://arxiv.org/abs/2312.17090) Table 7. | Test set identity and baseline provenance are Not stated. KonIQ, SPAQ, KADID, AVA, and LSVQ are in-distribution for this recipe. Only LIVE (and AGI if it is not AGIQA-20K) are cross-dataset. | Re-evaluate the released checkpoints and Q-Align yourself on named, pinned test files. | verified |
| 13 | Author-reported headline: Pro (9B) average PLCC / SRCC 0.900 / 0.896 vs Q-Align 0.873 / 0.869, over 7 sets. | [README](https://github.com/Q-Future/Q-ReAlign#results) (converted from SRCC / PLCC). | Mixed in-distribution and cross-dataset average. Not an independent replication. | Per-dataset rows in [unified models](../models/unified-and-backbones.md). | verified (author-reported) |
| 14 | No image size limit at inference. `open_image` loads the full image. The config sets no `max_pixels`. The Qwen3.5 processor allows up to 16,777,216 pixels, with 16 px patches merged 2x2 (about one visual token per 32x32 px). | [scorer.py L26-L28](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/scorer.py#L26-L28); [Qwen3.5-0.8B preprocessor_config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/preprocessor_config.json). | [INFERENCE] A 1024x768 KonIQ image gives about 770 visual tokens, vs 64 in Q-Align. A full-size SPAQ photo could exceed `max_length: 8192`. ms-swift may apply its own pixel cap; not checked. | Log visual tokens per item. Sweep a pixel cap and plot accuracy vs tokens. | verified (token counts inferred) |
| 15 | Attention kernel is set for training only. `train.py` passes `--attn_impl flash_attn`. `model.load` calls `get_model_processor(path, model_type=...)` without it. `flash-attn` is commented out in `requirements.txt`. | [train.py L102](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/train.py#L102); [model.py L26-L27](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/model.py#L26-L27); [requirements.txt](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/requirements.txt). | [INFERENCE] Eval and inference may run on the library default kernel. Qwen3.5 linear-attention layers may also fall back to a slow path if their fast kernels are missing. | Record the attention kernel and any "fast path not available" warning in every timing run. | verified (runtime effect not measured) |
| 16 | One item per forward pass. `score_record` collates a single input and reads `logits[0, -1]`. It does not use `logits_to_keep`, so the LM head runs on every position. | [scorer.py L109-L112](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/qalign/scorer.py#L109-L112). | The repo code has no batching. The README speed chart shows batch sizes up to 14, so it came from another harness (Not stated). | Build our own batched scorer; compute logits only at the last position. | verified |

### License status

| Item | Code | Weights | Commercial use? | Status |
|---|---|---|---|---|
| [Q-ReAlign repo](https://github.com/Q-Future/Q-ReAlign) | No `LICENSE` file (raw `LICENSE` returns 404; GitHub API license is null). [pyproject.toml](https://github.com/Q-Future/Q-ReAlign/blob/f5fd748399ca26e2655b210a609bdcff35953dff/pyproject.toml) declares `license = { text = "MIT" }`. | n/a | Unclear: MIT is declared only in package metadata. Ask the authors before redistribution. | verified |
| [Q-ReAlign-Pro-9B](https://huggingface.co/q-future/Q-ReAlign-Pro-9B), [Lite-4B](https://huggingface.co/q-future/Q-ReAlign-Lite-4B), [Mini-0.8B](https://huggingface.co/q-future/Q-ReAlign-Mini-0.8B) | n/a | Model cards: `apache-2.0`; `base_model: Qwen/Qwen3.5-VL`. | Yes for the weights under Apache-2.0. Training-data terms (KonIQ, SPAQ, KADID, AGIQA-20K, AVA, LSVQ) are separate; see dataset docs. | verified |
| [Q-Align repo](https://github.com/Q-Future/Q-Align) (original method) | S-Lab License 1.0 (non-commercial redistribution and use). | See [unified models](../models/unified-and-backbones.md). | No (NC) for code. | verified |

Correction vs dossier: the dossier said only that no `LICENSE` file exists at the root. The package metadata does declare MIT.

### Reproduction checklist

1. Pin the Q-ReAlign commit, the checkpoint revision, the tokenizer, and the processor config.
2. Build train, validation, and test manifests with source-level splits (section 3). Hash them.
3. Set `eval.sets` to validation manifests only. Keep `limit: 200` for in-training smoke checks only.
4. Run final eval with `--limit 0`. Assert `n` equals the test size.
5. Use one lossless frame cache for train and eval. Score each test set twice (cold and warm cache) and confirm equal scores.
6. Assert 5 distinct single-token labels at the answer position for each backbone.
7. Report raw PLCC, SRCC, KRCC, `n`, and a source-level bootstrap interval for each dataset.

## 9. Efficiency measurement

No shared benchmark exists for scorer cost. Papers use different GPUs, image sets, and input sizes. To claim "cheaper or faster than Q-Align", run every model on one harness.

### Published reference numbers (not comparable to each other)

| Model | Stored params | Visual input | GPU | Peak images/s (batch) | Batch-1 latency | Source |
|---|---:|---|---|---|---|---|
| Q-Align (mPLUG-Owl2) | ~8.2B | 448 px; visual abstractor cuts 1,024 tokens to 64 | RTX 3090 | 22.9 (bs 64) | 101 ms | [Q-Align Tables 9–10](https://arxiv.org/html/2312.17090v1) |
| Q-Align, video | ~8.2B | 1 fps | RTX 3090 | 4.2 five-second videos/s; 1.9 twelve-second videos/s (bs 1) | 236–514 ms | [Q-Align Table 10](https://arxiv.org/html/2312.17090v1) |
| Q-ReAlign Mini | 1.11B | Qwen3.5 processor (resize policy not stated) | RTX 4090 / H200 | 26.7 (bs 4) / 61.1 (bs 14) | Not stated | [README speed chart](https://github.com/Q-Future/Q-ReAlign/blob/main/assets/Speed.png), measured on SPAQ |
| Q-ReAlign Lite | 5.17B | As above | RTX 4090 / H200 | 9.6 (bs 2) / 25.7 (bs 4) | Not stated | Same chart |
| Q-ReAlign Pro | 9.41B | As above | RTX 4090 / H200 | 6.4 (bs 1) / 17.6 (bs 3) | Not stated | Same chart |
| RALI vs Q-Insight | ~4% of Q-Insight params | CLIP-based | A100 | 3.4% of Q-Insight inference time (bs 16) | – | [RALI](https://arxiv.org/html/2510.11369v2) |

What this shows: on a faster GPU, Q-ReAlign Lite and Pro reach less than half of Q-Align's image throughput. Only Mini is faster. [INFERENCE] The likely cause is the visual-token count: Q-Align sends 64 tokens per image, while the Qwen3.5 processor sends many more for large SPAQ photos. So the visual-token budget is the main cost lever for our model. It also affects accuracy, because native-resolution detail helps quality scoring (ReLIQS).

Key finding: **Q-ReAlign Lite and Pro are slower than the original Q-Align, even on a faster GPU** (RTX 4090 vs RTX 3090). Q-ReAlign gains accuracy at a cost in throughput. For our goal (beat Q-Align on cost and latency first), Lite and Pro are accuracy references, not cost references.

### Inference configuration to investigate

Before we blame the backbone, check how Q-ReAlign runs. Audit rows 14–16 in section 8 show the settings. Test each factor alone, on the harness below, and record accuracy and speed together.

| Factor | Current Q-ReAlign code | What to try |
|---|---|---|
| Image size / visual tokens | No cap (row 14) | Pixel caps that give about 64, 256, 576, and 1,024 tokens; also a native-resolution crop policy |
| Attention kernel | Not set at inference (row 15) | SDPA vs FlashAttention; confirm Qwen3.5 linear-attention fast kernels load |
| Batching | 1 item per pass (row 16) | Padded batches, sorted by token count |
| LM head | Full sequence | Last position only |
| Serving stack | Plain Transformers forward | vLLM or SGLang with prefix caching (the prompt is the same for every item) |
| Precision | BF16 (assumed) | FP8 or INT8 weights; check the score drift on the T1 sets |
| Video frames | 8 frames, long side 448 | Fewer frames, or token merging across frames |

Goal: find the best speed each backbone can reach at a fixed accuracy. Then compare backbones.

### Harness rules

- One machine, one GPU type, one software stack (driver, CUDA, PyTorch, attention kernel). Record all of them.
- Same inputs for all models: a fixed 1,000-image mix (KonIQ, SPAQ full-size, UHD-IQA) and a fixed 200-video mix (LSVQ, LSVQ-1080p). Include decode and preprocessing in the time.
- Report: visual tokens per item, batch-1 latency (p50 and p95), peak throughput and its batch size, peak VRAM, and cost per 1M images at a stated GPU price.
- Warm up before timing. Use BF16 unless the model needs another dtype. Record whether you used compiled or served inference (for example vLLM).
- Put accuracy and cost in the same table. A model is only "better" if it is on or above the accuracy-vs-cost line of Q-Align and Q-ReAlign.
