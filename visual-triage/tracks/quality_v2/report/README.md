# Low-quality detection: final validation

Baseline and selected prompts used the same frozen test views. Prompts were selected on development groups; test labels did not guide selection.

| Prompt | Valid / expected | MAE | Spearman | Balanced accuracy (95% group CI) | Recall | False-positive rate | Reported cost USD | Latency p50 / p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 100 / 100 | 0.5677 | 0.9398 | 0.7600 [0.6837, 0.8333] | 0.5200 | 0.0000 | 0.004382 | 1.0069 / 1.3211 |
| selected | 100 / 100 | 0.4429 | 0.9482 | 0.8400 [0.7685, 0.8977] | 0.6800 | 0.0000 | 0.005706 | 1.0541 / 1.3910 |

Selected minus baseline reward: **0.0312** across 100 source groups; paired group bootstrap 95% interval: [0.0258, 0.0371]. The 0.005 development selection margin does not establish statistical significance.

Accuracy intervals resample whole source groups. The JSON report includes confusion counts, Wilson intervals for recall and false-positive rates within each class, continuous-score intervals, invalid-output penalties, costs from every attempt, and derived per-image results. Full responses and feedback remain in the archive. A zero observed false-positive count does not imply zero future risk.

Low quality means a score strictly below 2.6. Native human MOS uses the original 1..5 direction: higher is better. These authentic photographs come from a bounded source panel; source coverage limits transfer claims. Human aesthetic labels are absent; aesthetic predictions are retained but not scored.

Known reported track cost: $0.099948. Unreconciled budget reservation: $0.10. Guard-accounted total: $0.199948. Actual total cost remains unknown; the reservation is not a provider-reported charge. This includes search, reflection with training images, final inference and metadata calls. Final calls alternate prompt order by source group. BYOK provider inference and OpenRouter charges are separate fields; the spending guard conservatively counts both when separately reported.

See [per-image results](../../../summaries/quality_v2.csv) and [compact machine-readable report](../../../summaries/quality_v2.json).

The compact exports are derived results. Restore the [full evidence archive](../../../ARCHIVE.md) to validate raw responses or rebuild reports. CSV phase and panel fields identify each comparison.
