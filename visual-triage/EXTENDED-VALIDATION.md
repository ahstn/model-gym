# Extended Decider validation

**The quality prompt improved, but neither track supports automatic rejection.** Both tracks completed 20 GEPA proposals. The pass acquired 750 new images, with 260 reserved for final tests. All 780 final responses were valid, including the pixel-statistics diagnostic.

## Final results

Lower error is better. Quality uses mean absolute error (MAE) against human scores on the 1–5 scale. AI uses Brier error against source labels. Thresholds stayed fixed: quality below 2.6; AI probability at least 0.5. Each panel has equal positive and control counts.

| Panel | Images | Error: original → GEPA | Accuracy: original → GEPA | Selected prompt: positives detected | Cost per 1,000: original → GEPA | Selected latency: p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KonIQ quality | 100 | 0.568 → **0.443** | 76% → **84%** | 34/50 low-quality photos | $0.0438 → $0.0571 | 1.054 / 1.391 s |
| SynthCLIC AI, unseen Imagen 3 | 80 | 0.3480 → **0.2624** | 55% → **60%** | 9/40 generated photos | $0.0280 → $0.0447 | 0.677 / 0.949 s |
| External AI, SDXL / RAISE | 80 | 0.2309 → **0.1791** | 66.25% → **72.5%** | 19/40 generated photos | $0.0280 → $0.0447 | 0.683 / 0.867 s |

Quality MAE fell 22%, at 30% more inference cost. The paired 95% interval for error reduction was 0.103–0.148 points. The prompt missed 16 low-quality photos and falsely flagged none of 50 controls. Zero observed false flags still permits a 7.1% upper 95% bound.

AI error also fell, but recall remains weak. Each panel falsely flagged one of 40 real photos. Imagen 3 Brier error remains worse than a constant 0.5 probability, which scores 0.25. External AUROC stayed near 0.928 despite lower Brier error. Better probability estimates do not establish reliable rejection. [Quality details](tracks/quality_v2/report/README.md), [AI details](tracks/ai_detection_v2/report/README.md), [external results](tracks/ai_detection_v2/report/external.md).

## Prompts and safeguards

The [quality prompt](tracks/quality_v2/prompts/selected.json) weighs defect severity, affected area and useful detail. The [AI prompt](tracks/ai_detection_v2/prompts/selected.json) weighs local structural evidence and keeps weak evidence near even odds. Beauty and technical quality remain separate from origin.

Quality produced 13 valid proposals from 20 attempts; eight improved their training batch and reached full development evaluation. AI produced 20 valid proposals; 11 reached that stage. Development selected each prompt before final calls. Reflection received only training feedback and up to four training images. Seven invalid quality proposals stayed in the record and used the prior candidate.

Frozen groups, duplicate checks, unseen generators and separate final tests reduce leakage. One search seed and repeated development comparisons cannot rule out overfitting. The [run guide](tracks/OPTIMISATION.md) records the rules and recovery steps.

## Did extra context help?

The fixed diagnostic added delivered resolution, luminance percentiles, colour statistics and local pixel differences. These describe pixels; they are not calibrated blur or contrast-ratio scores.

**EXIF and camera settings were not passed to Decider or used in GEPA.** The request for extra context was only partly evaluated:

| AI detection stage | Decider calls | Context actually sent |
| --- | ---: | --- |
| GEPA training and development | 1,580 | Images only |
| Baseline and selected final tests | 320 | Images only |
| Selected-prompt metadata diagnostic | 160 | Delivered resolution, tonal spread, clipping fractions, RGB statistics and adjacent-pixel differences |

All 2,060 request hashes were reconstructed from the frozen inputs and matched the saved records. The [frozen plan](tracks/ai_detection_v2/ablation-plan.json) sets `camera_context: false`; the [runner](bench_runner.py) rejects camera fields in this mode. The extractor supports ISO, exposure time, aperture and focal length, but that code was not connected to an evaluated camera arm. GEPA did not optimise the pixel-statistics instructions either; they were added after prompt selection.

| Panel | Error: image only → pixel statistics | Additional inference cost |
| --- | ---: | ---: |
| Quality | 0.443 → 0.464 | +29% |
| Imagen 3 AI | 0.2624 → 0.2660 | +55% |
| External AI | 0.1791 → 0.1883 | +54% |

Quality worsened; its paired interval excluded no change. Both AI intervals included no change. Image-only remains the preselected configuration. Pillow detected no EXIF in the 400 downloaded AI benchmark PNG representations, including 200 real photos; this does not describe their original camera files. Delivered images were all 512×512 JPEGs without detected EXIF. A camera-aware trial needs native files with metadata, balanced coverage, metadata-removal controls and fresh final groups. This pass cannot establish whether EXIF helps. [Quality diagnostic](tracks/quality_v2/report/metadata-report.md), [AI diagnostic](tracks/ai_detection_v2/report/metadata-report.md).

## Evidence and next steps

GPT-6.1 Sol agents at medium effort reviewed the requested catalogue using Exa and primary sources. The HF plugin was unavailable; official Hugging Face APIs and pinned files supplied the data checks. Research-only licences were accepted where permitted. The [research index](research/expansion-2026-10-11/README.md) covers KonIQ++, Community Forensics, SPAQ, LIVE Challenge, AGIQA and other methods.

Quality still needs another human-rated source and score calibration; no aesthetic gold exists here. AI needs broader unseen generators and a specialist detector baseline. Individual KonIQ attribution was verified for 45/350 photos; the rest rely on the authors' published research rights. AI uses bounded source cohorts and imperfect scene matching. These tests cannot rank Decider against the earlier Luna panel.

The pass made **3,560 API attempts**. Known reported cost was **$0.203025**. One failed training request has an unknown charge and a **$0.10 budget reservation**, which is not billed cost. [Cost evidence](research/expansion-2026-10-11/cost-recovery.md). Latency measures client requests, excludes acquisition and feature extraction, and includes concurrent track activity.

Per-image records: [quality](summaries/quality_v2.csv), [AI](summaries/ai_detection_v2.csv), [external AI](summaries/ai_detection_v2.csv). Metadata records: [quality](summaries/quality_v2.csv), [AI](summaries/ai_detection_v2.csv). Image bytes remain in ignored local data; manifests and ledgers retain hashes and source records.

Verification passed: 187 tests, lint, formatting and an [independent evidence audit](research/expansion-2026-10-11/evidence/final-audit-summary.json). The audit checked hashes, split access, selection timing, raw responses, metrics and costs. All six original earlier report files rebuild byte for byte from the archive. Retained Markdown links now use the compact exports.

The compact exports are derived results. Restore the [full evidence archive](ARCHIVE.md) to validate raw responses or rebuild reports. CSV phase and panel fields identify each comparison.
