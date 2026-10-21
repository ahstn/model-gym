# Follow-up validation: quality and AI detection

Historical three-proposal pass. The later [extended validation](EXTENDED-VALIDATION.md) uses new datasets and 20 GEPA proposals per track. Results below retain their original scope.

**Validation complete, 11 October 2026. Decider needs quality-score calibration; Luna's original prompt remains the cheaper AI-review default.** Both [prompt-engineering tracks](tracks/README.md) now have labelled data, frozen splits, three GEPA trials and per-image results.

## Results

Accuracy uses fixed thresholds: quality `< 2.6`, AI probability `>= 0.5`. Equal class counts make accuracy and balanced accuracy equal. The tasks remain separate.

| Task and prompt | Final images | Accuracy | Positive cases detected | Final cost | Median / P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decider quality, original prompt | 80 | **57.5%** | **6/40 low-quality images** | $0.001662 | 0.911 / 1.258 s |
| Luna AI detection, original prompt | 60 | **85.0%** | 21/30 generated photos | $0.009520 | 3.289 / 4.730 s |
| Luna AI detection, GEPA prompt | 60 | **88.3%** | 24/30 generated photos | $0.014369 | 3.847 / 6.471 s |

**Decider needs calibration before use as a quality teacher.** Human-score MAE was **0.970** on the 1–5 scale, despite a Spearman correlation of **0.781**. Scores were high by 0.878 points on average. It missed 34/40 low-quality cases and falsely flagged 0/40 controls. That zero count still gives an upper 95% false-positive bound of 8.8%.

Large errors include clipped highlights and colour blocks. Score-anchor mismatch and missed defects may contribute; this post-test diagnosis did not guide prompt changes. See the [quality analysis](tracks/quality/README.md), [per-image results](summaries/quality.csv) and [compact metrics](summaries/quality.json).

**Luna's gain remains uncertain.** Brier error fell from 0.1241 to 0.1194; lower is better. The paired 95% interval for improvement was **−0.0326 to +0.0432**, which includes no gain. False flags rose from 0/30 to 1/30 real photos. Cost rose 51%, median latency rose 17%, and the review band grew from 19 to 39 images. Keep the cheaper prompt for now. See the [AI analysis](tracks/ai_detection/README.md) and [per-image results](summaries/ai_detection.csv).

## What prompt optimisation changed

Each track used **GEPA 0.1.4 with three context-only proposals**. Output fields and score anchors stayed fixed. Training supplied feedback; development selected the prompt at a minimum reward gain of 0.005. Final images and labels were excluded from search.

- **Quality:** all three proposals lost on their eight-image training batches. The original prompt was retained and tested once. Search cost was **$0.003937** for 88 task calls and three reflection calls.
- **AI detection:** proposal two lowered development Brier error from **0.1351 to 0.1236**. Search cost was **$0.022273** for 108 task calls and three reflection calls. It favours specific local evidence over generic visual cues. Its final-test gain was smaller.

GEPA reused the existing API requests through a custom adapter. DSPy may help with several model stages; its experimental Jev decision types and ReAnchor calibration merit a later comparison. OpenRouter Decider integration was not tested through DSPy. [Framework research](tracks/OPTIMISATION.md)

Search used text feedback and optimised score error. Cost and latency were recorded separately. Image-based reflection remains untested.

## Why these datasets

| Track | Label source and split | Main limit |
| --- | --- | --- |
| Quality | [KADID-10K](tracks/quality/SOURCES.md): 160 images, 40 train / 40 development / 80 final; pristine parents stay in one split; half of each split is low quality | Human raters saw a clean reference; Decider did not. Defects are synthetic. No human aesthetic labels |
| AI detection | [Synthbuster + RAISE](tracks/ai_detection/SOURCES.md): 60 caption-matched pairs, 30 train / 30 development / 60 final images; each source pair stays in one split | Older generators and imperfect semantic matching limit generalisation |

KADID uses unchanged 512×384 PNGs, checked against the authors' hashes. AI images share one 512×512 crop and encoding policy. An independent acquisition screen excluded clear artwork and renders without detector outputs. Photo eligibility is an assistant judgement; origin labels come from published provenance.

Source terms permit non-commercial research. Exa and official Hugging Face APIs supplied source checks; the HF search connector was unavailable. The prior 150 images remain separate. Pretraining exposure is unknown, and balanced panels do not estimate production precision. NSFW is outside this pass.

## Evidence and next decision

The pass made **402 calls for $0.051761**, including trials and reflection. All **200 final responses** were valid across 140 distinct held-out images. Records retain inputs, hashes, every API attempt, provider cost and client latency. Costs include upstream BYOK usage. AI prompts alternated order by pair; thresholds stayed fixed.

Validation passed: **110 tests**, lint, formatting, and independent checks of final request hashes, selection timing and reported metrics.

The [run guide](tracks/README.md) rebuilds reports offline. Next, test a quality rubric aligned with human ratings or fit score calibration, then validate on new source groups and authentic defects. For AI review, retain both frozen prompts and test the gain on more matched photos from newer generators. Neither track supports automatic rejection yet.

The compact exports are derived results. Restore the [full evidence archive](ARCHIVE.md) to validate raw responses or rebuild reports. CSV phase and panel fields identify each comparison.
