# Matched photo AI detection: final validation

Baseline and selected prompts used the same frozen test views. Prompts were selected on development groups; test labels did not guide selection.

| Prompt | Valid / expected | Brier | AUROC | Balanced accuracy (95% group CI) | Recall | False-positive rate | Reported cost USD | Latency p50 / p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 80 / 80 | 0.3480 | 0.8344 | 0.5500 [0.5125, 0.6125] | 0.1000 | 0.0000 | 0.002238 | 0.6581 / 0.8286 |
| selected | 80 / 80 | 0.2624 | 0.8550 | 0.6000 [0.5375, 0.6750] | 0.2250 | 0.0250 | 0.003578 | 0.6774 / 0.9486 |

Selected minus baseline reward: **0.0856** across 40 source groups; paired group bootstrap 95% interval: [0.0673, 0.1066]. The 0.005 development selection margin does not establish statistical significance.

Accuracy intervals resample whole source groups. The JSON report includes confusion counts, Wilson intervals for recall and false-positive rates within each class, continuous-score intervals, invalid-output penalties, costs from every attempt, and derived per-image results. Full responses and feedback remain in the archive. A zero observed false-positive count does not imply zero future risk.

AI probability at least 0.5 predicts generated origin. The open review band is (0.2, 0.8). Scene-caption matching controls some content differences but does not produce identical scenes or cameras. Source and generator effects can remain; pixel judgments do not prove provenance.

External validation used the same frozen prompt selection. See [external results](external.md) and [external image results](../../../summaries/ai_detection_v2.csv).

Total reported track spending: $0.103077. This includes search, reflection with training images, final inference and metadata calls. Final calls alternate prompt order by source group. BYOK provider inference and OpenRouter charges are separate fields; the spending guard conservatively counts both when separately reported.

See [per-image results](../../../summaries/ai_detection_v2.csv) and [compact machine-readable report](../../../summaries/ai_detection_v2.json).

The compact exports are derived results. Restore the [full evidence archive](../../../ARCHIVE.md) to validate raw responses or rebuild reports. CSV phase and panel fields identify each comparison.
