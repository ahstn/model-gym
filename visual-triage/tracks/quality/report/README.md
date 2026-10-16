# Low-quality detection: final validation

Development selection kept the baseline. It was scored once on the frozen test panel; no independent prompt gain was measured. Test labels did not guide selection.

| Prompt | Valid / expected | MAE | Spearman | Balanced accuracy (95% group CI) | Recall | False-positive rate | Reported cost USD | Latency p50 / p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 80 / 80 | 0.9704 | 0.7806 | 0.5750 [0.5250, 0.6250] | 0.1500 | 0.0000 | 0.001662 | 0.9105 / 1.2578 |

Accuracy intervals resample whole source groups. The JSON report includes confusion counts, Wilson intervals for recall and false-positive rates within each class, continuous-score intervals, invalid-output penalties, costs from every attempt, and derived per-image results. Full responses and feedback remain in the archive. A zero observed false-positive count does not imply zero future risk.

Low quality means a score strictly below 2.6. Human DMOS comes from reference-aware distortion assessment; the VLM judges each image without its reference. Score agreement therefore compares related tasks. These are synthetic defects, so results do not establish performance on all authentic camera failures. Human aesthetic labels are absent; aesthetic predictions are retained but not scored.

Total track spending, including optimisation and text reflection: **$0.005600**. Final calls used one baseline prompt. BYOK provider inference and OpenRouter charges are separate fields; the spending guard conservatively counts both when separately reported.

See [per-image results](../../../summaries/quality.csv) and [compact machine-readable report](../../../summaries/quality.json).

The compact exports are derived results. Restore the [full evidence archive](../../../ARCHIVE.md) to validate raw responses or rebuild reports. CSV phase and panel fields identify each comparison.
