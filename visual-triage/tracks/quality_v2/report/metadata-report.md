# Frozen pixel statistics ablation

The same development-selected prompt was evaluated with the same delivered images. Pixel statistics were frozen before final calls. Test and external labels did not guide prompt or context selection.

## Test panel

| Arm | Valid / expected | Group mean reward | Error metric | Score association | Balanced accuracy (95% group CI) | Reported arm cost USD | Client latency p50 / p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| image_only | 100 / 100 | 0.8893 | 0.4429 | 0.9482 | 0.8400 [0.7685, 0.8977] | 0.005706 | 1.0541 / 1.3910 |
| pixel_stats | 100 / 100 | 0.8841 | 0.4638 | 0.9466 | 0.8100 [0.7411, 0.8725] | 0.007334 | 1.0930 / 1.4322 |

test: metadata minus image-only reward is -0.0052; paired group bootstrap 95% interval is [-0.0083, -0.0022].

Error metric is MAE on the native 1..5 quality scale; score association is Spearman correlation.

Missing and invalid attempts retain zero reward. Group bootstrap intervals do not account for source bias. Descriptive pixel measurements are not calibrated quality scores or proof of origin.

Known reported track cost: $0.099948. Unreconciled budget reservation: $0.10. Guard-accounted total: $0.199948. Actual total cost remains unknown; the reservation is not a provider-reported charge.

See [per-image evidence](../../../summaries/quality_v2.csv) and [JSON results](../../../summaries/quality_v2.json).

The compact exports are derived results. Restore the [full evidence archive](../../../ARCHIVE.md) to validate raw responses or rebuild reports. CSV phase and panel fields identify each comparison.
