# Visual quality docs

This folder stores research notes for image quality (IQA), image aesthetics (IAA), and video quality (VQA) models. It covers existing models, datasets, training methods, and evaluation. The goal is to re-train or improve [Q-ReAlign](https://github.com/Q-Future/Q-ReAlign) and compare it fairly with other assessors.

Last verified: 2026-09-27

## Index

| Area | File | Contents |
|---|---|---|
| Models | [models/iqa.md](models/iqa.md) | No-reference IQA models, incl. AI-generated image quality. PLCC / SRCC tables by protocol. |
| Models | [models/iaa.md](models/iaa.md) | Aesthetics models. PLCC / SRCC on AVA, TAD66K, PARA, ArtiMuse-10K. |
| Models | [models/vqa.md](models/vqa.md) | Video quality models: natural UGC, AI-generated video, HDR. |
| Models | [models/unified-and-backbones.md](models/unified-and-backbones.md) | Unified LMM assessors (Q-Align, Q-ReAlign, UniPercept) and VLM backbones to fine-tune. |
| Datasets | [datasets/iqa.md](datasets/iqa.md) | IQA datasets: synthetic, authentic, AIGC, instruction data. |
| Datasets | [datasets/iaa.md](datasets/iaa.md) | Aesthetics and preference datasets. |
| Datasets | [datasets/vqa.md](datasets/vqa.md) | Video quality datasets: UGC, streaming, HDR, AIGC video. |
| Training | [training/methods.md](training/methods.md) | Training-method papers and what to borrow. |
| Training | [training/data-mixture-evidence.md](training/data-mixture-evidence.md) | Controlled evidence that adding or choosing data helps or hurts. |
| Training | [training/experiment-plan.md](training/experiment-plan.md) | Staged experiment plan and compute assumptions. |
| Evaluation | [evaluation/protocol.md](evaluation/protocol.md) | Metrics, split rules, calibration, and the Q-ReAlign code audit. |
| Evaluation | [evaluation/benchmarks.md](evaluation/benchmarks.md) | Diagnostic benchmarks and the standard held-out test suite. |

## Quick summary

- **Baseline.** Q-ReAlign Mini/Lite/Pro (Qwen3.5-based). The README lists the mix KonIQ, SPAQ, KADID, AGIQA-20K, AVA, and LSVQ, but the public `onealign.yaml` omits AGIQA-20K. It scores with five quality words and a probability-weighted mean. See the [code audit](evaluation/protocol.md).
- **Backbones to test first.** Qwen3.5-4B (control) and Gemma 4 E4B (newer architecture). Then MiniCPM-V-4.6 as a small student. No newer backbone has a matched unified ONE-ALIGN win yet.
- **Speed.** Q-ReAlign Lite and Pro are slower than the original Q-Align, even on a faster GPU. Only Mini is faster. The Q-ReAlign code sets no image-size cap, scores one item per pass, and does not set the attention kernel at inference. Investigate the inference configuration before comparing backbones. See [protocol.md section 9](evaluation/protocol.md#9-efficiency-measurement).
- **Training changes to test first.** Soft score distributions (DeQA-Score), native-resolution crops (ReLIQS), contiguous frames for video (VQ-Insight), and task-balanced batches (TATAR). Reasoning and RL come later.
- **Competitors for the first comparison.** DeQA-Score-Mix3, TOPIQ-NR, SigLIP2+AGM (IQA); ArtiMuse-AVA, MUSIQ-AVA (IAA); DOVER, FineVQ-LSVQ (VQA).

## License watch

Found during the 2026-09-27 checks. Details and links are in each file.

| Item | Finding | Effect |
|---|---|---|
| Q-ReAlign | Weights cards: Apache-2.0. Repo has no LICENSE file, but `pyproject.toml` declares MIT. | Weights allow commercial use. Code license is weakly declared. |
| Gemma 4 | Apache-2.0 (Gemma 3 used the gated Gemma license). | No special Gemma terms for Gemma 4. |
| IQA-PyTorch (pyiqa) | Code: PolyForm Noncommercial 1.0.0. Hosted weights: CC BY-NC-SA 4.0. | TOPIQ-NR, CLIP-IQA+, LIQE via pyiqa are research-only. |
| Q-Align, DOVER, Q-Instruct, Co-Instruct, DIVIDE-MaxWell | Repos use S-Lab License 1.0 (non-commercial). HF cards say MIT or Apache-2.0. | The licenses conflict. Treat as non-commercial until the authors clarify. |
| Datasets | Many are research-only or by request: SPAQ, PARA, PIQ23, DiffIQA, ArtiMuse-10K (full), FineVD, VGA-Bench data (CC BY-NC 4.0), BrightVQ (CC BY-NC-SA 4.0), LongVQUBench (CC BY-NC-SA 4.0). | A model trained on them may inherit use limits. Check before any commercial release. |
| Q-ReAlign training mix | Only AGIQA-20K is clearly permissive. SPAQ is research-only. The Q-Align label JSONs are S-Lab (non-commercial). KonIQ, KADID, AVA, LSVQ have no formal license. MIT tags on HF mirrors do not relicense the source data. | Fine for research and evaluation. A shipped model needs permission or a clean-data mix. See [datasets/iqa.md section 6.1](datasets/iqa.md#61-license-verdict-for-the-whole-q-realign-training-mix). |
| Open exceptions | AIGIQA-20K (Apache-2.0 on ModelScope), UHD-IQA images (CC0), MSC (CC0), Waterloo SQoE-III/IV (CC BY 4.0), JPEG AIC2026 (CC BY-SA 4.0). | Still check media rights for each source. |

## Corrections to the earlier dossiers

- MOSAIQ: the unseen pooled SRCC gain (0.7836 → 0.8217) holds for MonotonicIQA only. LIQE drops with aligned labels. Code and data come "upon acceptance". See [data-mixture-evidence.md](training/data-mixture-evidence.md).
- DeQA + PIPAL: adding PIPAL lowers every cross-dataset test, not only CSIQ and LIVE-Wild.
- BrightRate-LM: Gemma 4 E2B (0.8849 / 0.8817) is slightly better than E4B (0.8805 / 0.8624) on the same SDR split. See [models/unified-and-backbones.md](models/unified-and-backbones.md).
- Q-ReAlign README: its Q-Align baseline row does not match the Q-Align paper or the OneAlign card. The Q-Align paper samples video at 1 fps, not 8 uniform frames.
- Standalone `qalign eval` with `onealign.yaml` scores only the first 200 records per test set unless you pass `--limit 0`.
- Later papers quote Q-Insight's score-only ablation (KonIQ 0.918 / 0.895), not the released model (0.933 / 0.916).

## Conventions

- **Metric order.** Correlation pairs are always `PLCC / SRCC`. The Q-ReAlign model card uses the reverse order; these docs convert it.
- **Author-reported.** Numbers come from papers or model cards. We did not run them. Numbers from different splits, training sets, or fitted mappings do not form one leaderboard.
- **Status.** `verified` means a primary source was checked on the date above. `from dossier` means the number or fact comes from the earlier research notes and was not re-checked.
- **Licenses.** Code, model weights, dataset annotations, and the images or videos themselves can each have different terms. `Not stated` means we found no license. It does not mean the item is free to use.

## Glossary

| Term | Meaning |
|---|---|
| IQA | Image quality assessment: technical quality (blur, noise, compression, exposure). |
| IAA | Image aesthetics assessment: visual appeal, composition, color, and theme. |
| VQA | Video quality assessment. (Not visual question answering in these docs.) |
| NR / FR | No-reference (the model sees only the test image) / full-reference (the model also sees a clean original). |
| MOS | Mean opinion score: the mean of human ratings for one item. |
| SRCC | Spearman rank correlation. It measures agreement of rank order with human scores. |
| PLCC | Pearson linear correlation. It measures linear agreement with human scores. Some papers fit a logistic mapping first. |
| UGC | User-generated content: real photos and videos with authentic distortions. |
| AIGC / AIGV | AI-generated content / AI-generated video. |
| LMM / VLM | Large multimodal model / vision-language model. |
| ONE-ALIGN | Q-Align's unified training mix for IQA, IAA, and VQA. Q-ReAlign re-uses it. |

## Sources

These docs condense two earlier research dossiers (26 September 2026) and add new web research and license checks. The dossiers used Exa search, Hugging Face search, and primary-source reads. No model was trained and no gated dataset was requested.
