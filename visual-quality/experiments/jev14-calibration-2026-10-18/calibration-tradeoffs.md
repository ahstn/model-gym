# Calibration operating points

These points use calibration responses only. They were saved before any validation request. The 1% rule is the only rule selected for the validation gate. Calibration rates include selection bias.

| Empirical FPR limit | Threshold | Uncertainty condition | Real false flags | AI detected | AI recall |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0.5% | 0.80 | none | 1/300 | 119/300 | 39.7% |
| 1.0% | 0.78 | none | 3/300 | 155/300 | 51.7% |
| 2.0% | 0.76 | none | 5/300 | 180/300 | 60.0% |

A lower threshold can increase AI recall and real-photo false flags together. Reported uncertainty did not improve the selected primary rule. These empirical limits are not confidence bounds or production guarantees.

See [machine-readable values](calibration-tradeoffs.json) and the [held-out result](results/README.md).
