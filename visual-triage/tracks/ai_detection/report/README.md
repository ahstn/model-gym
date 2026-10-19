# Matched photo AI detection: final validation

Baseline and selected prompts used the same frozen test views. Prompts were selected on development groups; test labels did not guide selection.

| Prompt | Valid / expected | Brier | AUROC | Balanced accuracy (95% group CI) | Recall | False-positive rate | Reported cost USD | Latency p50 / p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 60 / 60 | 0.1241 | 0.9611 | 0.8500 [0.7667, 0.9333] | 0.7000 | 0.0000 | 0.009520 | 3.2890 / 4.7304 |
| selected | 60 / 60 | 0.1194 | 0.9750 | 0.8833 [0.8000, 0.9500] | 0.8000 | 0.0333 | 0.014368 | 3.8468 / 6.4708 |

Selected minus baseline reward: **0.0047** across 30 source groups; paired group bootstrap 95% interval: [-0.0326, 0.0432]. The 0.005 development selection margin does not establish statistical significance.

Accuracy intervals resample whole source groups. The JSON report includes confusion counts, Wilson intervals for recall and false-positive rates within each class, continuous-score intervals, invalid-output penalties, costs from every attempt, and derived per-image results. Full responses and feedback remain in the archive. A zero observed false-positive count does not imply zero future risk.

AI probability at least 0.5 predicts generated origin. The open review band is (0.2, 0.8). Scene-caption matching controls some content differences but does not produce identical scenes or cameras. Source and generator effects can remain; pixel judgments do not prove provenance.

Total track spending, including optimisation and text reflection: **$0.046161**. Final calls alternate prompt order by source group. BYOK provider inference and OpenRouter charges are separate fields; the spending guard conservatively counts both when separately reported.

See [per-image results](../../../summaries/ai_detection.csv) and [compact machine-readable report](../../../summaries/ai_detection.json).

The compact exports are derived results. Restore the [full evidence archive](../../../ARCHIVE.md) to validate raw responses or rebuild reports. CSV phase and panel fields identify each comparison.
