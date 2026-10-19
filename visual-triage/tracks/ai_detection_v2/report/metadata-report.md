# Frozen pixel statistics ablation

The same development-selected prompt was evaluated with the same delivered images. Pixel statistics were frozen before final calls. Test and external labels did not guide prompt or context selection.

## Test panel

| Arm | Valid / expected | Group mean reward | Error metric | Score association | Balanced accuracy (95% group CI) | Reported arm cost USD | Client latency p50 / p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| image_only | 80 / 80 | 0.7376 | 0.2624 | 0.8550 | 0.6000 [0.5375, 0.6750] | 0.003578 | 0.6774 / 0.9486 |
| pixel_stats | 80 / 80 | 0.7340 | 0.2660 | 0.8800 | 0.6250 [0.5625, 0.7000] | 0.005529 | 0.6766 / 0.8558 |

test: metadata minus image-only reward is -0.0036; paired group bootstrap 95% interval is [-0.0115, 0.0043].

## External panel

| Arm | Valid / expected | Group mean reward | Error metric | Score association | Balanced accuracy (95% group CI) | Reported arm cost USD | Client latency p50 / p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| image_only | 80 / 80 | 0.8209 | 0.1791 | 0.9275 | 0.7250 [0.6500, 0.8000] | 0.003578 | 0.6833 / 0.8665 |
| pixel_stats | 80 / 80 | 0.8117 | 0.1883 | 0.9300 | 0.7250 [0.6500, 0.8125] | 0.005520 | 0.6660 / 1.0049 |

external: metadata minus image-only reward is -0.0092; paired group bootstrap 95% interval is [-0.0204, 0.0026].

Error metric is Brier loss; score association is AUROC.

Missing and invalid attempts retain zero reward. Group bootstrap intervals do not account for source bias. Descriptive pixel measurements are not calibrated quality scores or proof of origin.

Total reported track spending: $0.103077.

See [per-image evidence](../../../summaries/ai_detection_v2.csv) and [JSON results](../../../summaries/ai_detection_v2.json).

The compact exports are derived results. Restore the [full evidence archive](../../../ARCHIVE.md) to validate raw responses or rebuild reports. CSV phase and panel fields identify each comparison.
