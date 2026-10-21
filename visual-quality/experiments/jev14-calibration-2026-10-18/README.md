# Jev 1.14 threshold calibration

This experiment selects an AI-origin flag threshold on a separate dataset, then checks the frozen rule on an untouched validation set. It keeps the prompt and image delivery used by the earlier Jev comparison. It does not train Jev or convert its scores into calibrated probabilities.

**Outcome:** the selected 0.78 threshold detected 51.8% of generated validation images and falsely flagged 0.6% of source-real images. Its 95% false-positive upper bound is 1.54%, so it did not pass the requested 1% target. Read the [assessment](assessment.md), including the source-label limit found during error review.

Read the [results](results/README.md), [threshold plot](results/calibration-curve.svg), and [frozen selection](selection.json) for the outcome. The [previous run](../jev14-2026-10-18/confidence-analysis.md) explains why this study was needed.

The [calibration operating points](calibration-tradeoffs.md) compare the requested 1% false-positive target with 0.5% and 2% alternatives. These were saved before validation. Only the 1% rule is selected for the formal validation gate.

## Cohort and decision rule

The source is the author's [MS COCOAI dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset/tree/787334f7857fa54f29027a7f09c30e895ad486ef). Calibration uses 300 real and 300 generated images. Independent validation uses 500 real and 500 generated images from different caption families. Both study splits use the publisher's official validation pool. Each caption family supplies one real image and one generated image. The generated sample has equal counts from SD2.1, SDXL, SD3, DALL-E3, and Midjourney-v6.

The publisher's test split repeats its caption and label rows in five image-variant blocks. The first block has reduced brightness. This was found before any model call, so the experiment switched to separate family groups from the official validation pool. The discarded preflight selection remains in ignored acquisition evidence.

Caption families cannot cross the study splits. Selection excludes matches to earlier origin cohorts and checks exact and perceptual image hashes. The [acquisition audit](acquisition-audit.json) records the scope and limits of those checks. Local image files and native Parquet downloads stay in the ignored data directory.

The fixed candidate grid tests `p_ai >= threshold` from 0.50 to 1.00 in steps of 0.01. Each threshold is tested alone and with a rule that requires origin uncertainty to be `low` or `medium`. The selection maximizes AI recall while keeping observed calibration false positives at or below 1%. Ties prefer fewer false positives, then the rule without an uncertainty condition, then the higher threshold.

Validation requires complete valid responses, a one-sided exact 95% false-positive upper bound no greater than 1%, and observed AI recall of at least 70%. The recall floor is a provisional operating assumption. With 500 real validation photos, at most one false flag passes the statistical false-positive test. A failed gate does not permit a new threshold to be chosen from validation results.

## Files and execution

The experiment separates acquisition, inference, statistical analysis, and reporting:

| File | Purpose |
| --- | --- |
| `acquire.py` | Download pinned native files, check source hashes, select families, and make the fixed image views |
| `study.py` | Freeze source and execution hashes; select and lock the rule after calibration |
| `runner.py` | Send bounded requests; preserve native responses and cost; stop on failures |
| `metrics.py` | Select rules and calculate exact binomial bounds |
| `report.py` | Verify saved evidence and generate the report without model calls |
| `manifest.json` | Source labels, family groups, file locations, and image hashes |
| `payload-audit.json` | Public source and request hashes; no image bytes or credentials |

The adapter comes from the existing read-only `visual-triage` worktree at `/Users/ahstn/.codex/worktrees/d187/model-gym/visual-triage`. The protocol pins that code and the unchanged prompt. Acquisition also reads earlier origin cohorts and the local `pilot-10k-v1` manifests to exclude overlaps. These local dependencies are required for a full rebuild.

Use the existing `visual-quality/.venv/bin/python` interpreter. The completed run must not be overwritten or retried. To verify its evidence and regenerate derived results:

```sh
PYTHONDONTWRITEBYTECODE=1 visual-quality/.venv/bin/python visual-quality/experiments/jev14-calibration-2026-10-18/runner.py verify --split validation
PYTHONDONTWRITEBYTECODE=1 visual-quality/.venv/bin/python visual-quality/experiments/jev14-calibration-2026-10-18/report.py
```

For a new experiment, copy the design into a new versioned directory and refresh the public OpenRouter Decisions model catalog. Run acquisition, freeze the protocol, run calibration, freeze selection, then run validation. The runner accepts at most four concurrent requests and starts no more than four per second. Its local budget guard is $0.25, with a conservative reserve for each request in progress. It never retries an unsuccessful or unpriced model call automatically.

The source license is not specified. Public availability does not establish rights for commercial use or image redistribution. The experiment measures the sampled source mixture; it does not establish performance on all real photos or newer image generators. No threshold here authorizes image deletion.
