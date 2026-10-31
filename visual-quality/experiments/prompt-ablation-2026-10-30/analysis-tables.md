# Complete prompt-ablation tables

Scores use the 1–5 scale. MAE measures distance from frozen assistant assessments, not human accuracy. Negative deltas mean lower error. Individual additions are exploratory; the combined arm uses intervals adjusted for eight primary comparisons.

## Scores and coverage

| Model | Arm | Technical MAE | Aesthetic MAE | Valid T/A | Bias T/A |
|---|---|---:|---:|---|---|
| deepseek | baseline | 0.326 | 0.232 | 50/50 | 0.066 / -0.044 |
| deepseek | A | 0.366 | 0.206 | 50/50 | -0.026 / -0.018 |
| deepseek | B | 0.378 | 0.250 | 50/50 | 0.062 / -0.086 |
| deepseek | C | 0.340 | 0.228 | 50/50 | 0.088 / 0.008 |
| deepseek | D | 0.336 | 0.240 | 50/50 | 0.068 / -0.036 |
| deepseek | combined | 0.350 | 0.206 | 50/50 | 0.110 / 0.022 |
| luna | baseline | 0.366 | 0.242 | 50/50 | 0.202 / 0.206 |
| luna | A | 0.364 | 0.306 | 50/50 | 0.212 / 0.238 |
| luna | B | 0.358 | 0.256 | 50/50 | 0.182 / 0.200 |
| luna | C | 0.366 | 0.266 | 50/50 | 0.194 / 0.182 |
| luna | D | 0.378 | 0.312 | 50/50 | 0.198 / 0.224 |
| luna | combined | 0.386 | 0.262 | 50/50 | 0.214 / 0.198 |
| gemini | baseline | 0.530 | 0.440 | 50/50 | -0.302 / -0.364 |
| gemini | A | 0.506 | 0.438 | 50/50 | -0.254 / -0.326 |
| gemini | B | 0.492 | 0.414 | 50/50 | -0.248 / -0.314 |
| gemini | C | 0.530 | 0.462 | 50/50 | -0.322 / -0.370 |
| gemini | D | 0.522 | 0.416 | 50/50 | -0.258 / -0.312 |
| gemini | combined | 0.484 | 0.470 | 50/50 | -0.224 / -0.362 |
| haiku | baseline | 0.390 | 0.253 | 50/49 | -0.014 / -0.069 |
| haiku | A | 0.402 | 0.222 | 50/49 | 0.002 / -0.067 |
| haiku | B | 0.436 | 0.233 | 50/49 | -0.008 / -0.073 |
| haiku | C | 0.402 | 0.238 | 50/45 | -0.010 / -0.073 |
| haiku | D | 0.398 | 0.245 | 49/44 | -0.002 / -0.082 |
| haiku | combined | 0.392 | 0.231 | 50/48 | 0.016 / -0.069 |

## Every addition versus its own baseline

| Addition | Model | Task | MAE delta | Interval | Observed effect | Statistical support | Cases worsened ≥0.5 | New critical contradictions |
|---|---|---|---:|---|---|---|---:|---:|
| A | deepseek | technical_quality | +0.040 | [-0.024, +0.110] | negative | uncertain | 3 | 0 |
| A | deepseek | aesthetics | -0.026 | [-0.078, +0.024] | positive | uncertain | 0 | 0 |
| A | luna | technical_quality | -0.002 | [-0.048, +0.042] | positive | uncertain | 0 | 0 |
| A | luna | aesthetics | +0.064 | [+0.022, +0.106] | negative | supports_increased_error | 0 | 0 |
| A | gemini | technical_quality | -0.024 | [-0.064, +0.016] | positive | uncertain | 0 | 0 |
| A | gemini | aesthetics | -0.002 | [-0.048, +0.050] | positive | uncertain | 1 | 0 |
| A | haiku | technical_quality | +0.012 | [-0.044, +0.066] | negative | uncertain | 0 | 0 |
| A | haiku | aesthetics | -0.023 | [-0.052, +0.004] | positive | uncertain | 0 | 0 |
| B | deepseek | technical_quality | +0.052 | [-0.008, +0.114] | negative | uncertain | 4 | 0 |
| B | deepseek | aesthetics | +0.018 | [-0.014, +0.054] | negative | uncertain | 0 | 0 |
| B | luna | technical_quality | -0.008 | [-0.060, +0.050] | positive | uncertain | 0 | 0 |
| B | luna | aesthetics | +0.014 | [-0.022, +0.052] | negative | uncertain | 0 | 0 |
| B | gemini | technical_quality | -0.038 | [-0.094, +0.014] | positive | uncertain | 0 | 0 |
| B | gemini | aesthetics | -0.026 | [-0.052, -0.002] | positive | supports_reduced_error | 0 | 0 |
| B | haiku | technical_quality | +0.046 | [+0.002, +0.090] | negative | supports_increased_error | 0 | 0 |
| B | haiku | aesthetics | -0.013 | [-0.041, +0.016] | positive | uncertain | 0 | 0 |
| C | deepseek | technical_quality | +0.014 | [-0.034, +0.066] | negative | uncertain | 2 | 0 |
| C | deepseek | aesthetics | -0.004 | [-0.040, +0.032] | positive | uncertain | 0 | 0 |
| C | luna | technical_quality | +0.000 | [-0.048, +0.050] | unchanged | uncertain | 0 | 0 |
| C | luna | aesthetics | +0.024 | [-0.020, +0.066] | negative | uncertain | 0 | 0 |
| C | gemini | technical_quality | -0.000 | [-0.036, +0.036] | unchanged | uncertain | 0 | 0 |
| C | gemini | aesthetics | +0.022 | [-0.018, +0.068] | negative | uncertain | 2 | 0 |
| C | haiku | technical_quality | +0.012 | [-0.026, +0.050] | negative | uncertain | 0 | 0 |
| C | haiku | aesthetics | -0.007 | [-0.049, +0.031] | positive | uncertain | 0 | 0 |
| D | deepseek | technical_quality | +0.010 | [-0.038, +0.054] | negative | uncertain | 1 | 0 |
| D | deepseek | aesthetics | +0.008 | [-0.026, +0.042] | negative | uncertain | 0 | 0 |
| D | luna | technical_quality | +0.012 | [-0.030, +0.054] | negative | uncertain | 0 | 0 |
| D | luna | aesthetics | +0.070 | [+0.022, +0.120] | negative | supports_increased_error | 1 | 0 |
| D | gemini | technical_quality | -0.008 | [-0.052, +0.030] | positive | uncertain | 0 | 0 |
| D | gemini | aesthetics | -0.024 | [-0.050, +0.002] | positive | uncertain | 0 | 0 |
| D | haiku | technical_quality | +0.008 | [-0.024, +0.041] | negative | uncertain | 1 | 0 |
| D | haiku | aesthetics | -0.005 | [-0.056, +0.042] | positive | uncertain | 0 | 0 |
| combined | deepseek | technical_quality | +0.024 | [-0.060, +0.116] | negative | uncertain | 3 | 0 |
| combined | deepseek | aesthetics | -0.026 | [-0.076, +0.028] | positive | uncertain | 0 | 0 |
| combined | luna | technical_quality | +0.020 | [-0.046, +0.086] | negative | uncertain | 0 | 0 |
| combined | luna | aesthetics | +0.020 | [-0.036, +0.078] | negative | uncertain | 0 | 0 |
| combined | gemini | technical_quality | -0.046 | [-0.104, +0.008] | positive | uncertain | 0 | 0 |
| combined | gemini | aesthetics | +0.030 | [-0.026, +0.098] | negative | uncertain | 2 | 0 |
| combined | haiku | technical_quality | +0.002 | [-0.048, +0.050] | negative | uncertain | 0 | 0 |
| combined | haiku | aesthetics | -0.019 | [-0.069, +0.029] | positive | uncertain | 0 | 0 |

## Targeted versus other cases

A targets controlled noise/JPEG variants; B targets controlled blur variants. C uses VizWiz as an imperfect proxy for difficult composition. These metadata groups do not establish the visible issue in every image. D and combined apply generally.

| Addition | Model | Task | Target n | Target ΔMAE | Other n | Other ΔMAE | Other noninferiority |
|---|---|---|---:|---:|---:|---:|---|
| A | deepseek | technical_quality | 12 | -0.050 | 38 | 0.068 | not_established |
| A | deepseek | aesthetics | 12 | -0.033 | 38 | -0.024 | supported |
| A | luna | technical_quality | 12 | -0.008 | 38 | 0.000 | supported |
| A | luna | aesthetics | 12 | 0.050 | 38 | 0.068 | not_established |
| A | gemini | technical_quality | 12 | -0.025 | 38 | -0.024 | supported |
| A | gemini | aesthetics | 12 | -0.000 | 38 | -0.003 | supported |
| A | haiku | technical_quality | 12 | 0.050 | 38 | 0.000 | supported |
| A | haiku | aesthetics | 10 | 0.100 | 38 | -0.055 | supported |
| B | deepseek | technical_quality | 6 | 0.033 | 44 | 0.055 | not_established |
| B | deepseek | aesthetics | 6 | 0.050 | 44 | 0.014 | supported |
| B | luna | technical_quality | 6 | -0.017 | 44 | -0.007 | supported |
| B | luna | aesthetics | 6 | 0.067 | 44 | 0.007 | supported |
| B | gemini | technical_quality | 6 | 0.050 | 44 | -0.050 | supported |
| B | gemini | aesthetics | 6 | -0.050 | 44 | -0.023 | supported |
| B | haiku | technical_quality | 6 | -0.000 | 44 | 0.052 | supported |
| B | haiku | aesthetics | 6 | -0.017 | 42 | -0.012 | supported |
| C | deepseek | technical_quality | 14 | 0.021 | 36 | 0.011 | supported |
| C | deepseek | aesthetics | 14 | 0.050 | 36 | -0.025 | supported |
| C | luna | technical_quality | 14 | -0.050 | 36 | 0.019 | supported |
| C | luna | aesthetics | 14 | -0.036 | 36 | 0.047 | supported |
| C | gemini | technical_quality | 14 | -0.114 | 36 | 0.044 | supported |
| C | gemini | aesthetics | 14 | 0.021 | 36 | 0.022 | supported |
| C | haiku | technical_quality | 14 | 0.007 | 36 | 0.014 | supported |
| C | haiku | aesthetics | 14 | -0.029 | 30 | 0.003 | supported |

## Combined-arm regression flags

These overlapping groups are exploratory. A flag means observed MAE worsened by more than 0.10. Sparse groups and wide intervals prevent a general claim of no harm.

- **deepseek, technical_quality:** groups: issue:bright, issue:natural_vizwiz, issue:rotation, role:vizwiz, stratum:vizwiz; cases worsened ≥0.5: P027, P042, P047; invalid increase: 0.
- **deepseek, aesthetics:** groups: none; cases worsened ≥0.5: none; invalid increase: 0.
- **luna, technical_quality:** groups: reference_high:aesthetics; cases worsened ≥0.5: none; invalid increase: 0.
- **luna, aesthetics:** groups: issue:bright, reference_high:aesthetics; cases worsened ≥0.5: none; invalid increase: 0.
- **gemini, technical_quality:** groups: none; cases worsened ≥0.5: none; invalid increase: 0.
- **gemini, aesthetics:** groups: issue:rotation, severity:2; cases worsened ≥0.5: P023, P037; invalid increase: 0.
- **haiku, technical_quality:** groups: issue:dark, issue:source_parent, parents, role:source_parent; cases worsened ≥0.5: none; invalid increase: 0.
- **haiku, aesthetics:** groups: none; cases worsened ≥0.5: none; invalid increase: 1.

## Diagnostic averages

| Ensemble | Arm | Technical MAE | Aesthetic MAE |
|---|---|---:|---:|
| deepseek_luna | baseline | 0.328 | 0.179 |
| deepseek_luna | A | 0.337 | 0.196 |
| deepseek_luna | B | 0.334 | 0.177 |
| deepseek_luna | C | 0.319 | 0.189 |
| deepseek_luna | D | 0.339 | 0.206 |
| deepseek_luna | combined | 0.344 | 0.190 |
| all_four | baseline | 0.343 | 0.204 |
| all_four | A | 0.362 | 0.199 |
| all_four | B | 0.358 | 0.189 |
| all_four | C | 0.352 | 0.201 |
| all_four | D | 0.363 | 0.190 |
| all_four | combined | 0.342 | 0.209 |

These averages are diagnostic, with no weight fitting. Technical and aesthetic dimensions remain separate.

## Controlled technical severity diagnostics

| Model | Arm | Nonincreasing families / complete | Mean parent-to-severe gap |
|---|---|---:|---:|
| deepseek | baseline | 5 / 6 | 1.450 |
| deepseek | A | 5 / 6 | 1.350 |
| deepseek | B | 5 / 6 | 1.417 |
| deepseek | C | 5 / 6 | 1.250 |
| deepseek | D | 4 / 6 | 1.283 |
| deepseek | combined | 5 / 6 | 1.650 |
| luna | baseline | 6 / 6 | 1.450 |
| luna | A | 6 / 6 | 1.417 |
| luna | B | 5 / 6 | 1.383 |
| luna | C | 5 / 6 | 1.600 |
| luna | D | 5 / 6 | 1.467 |
| luna | combined | 6 / 6 | 1.600 |
| gemini | baseline | 4 / 6 | 0.750 |
| gemini | A | 5 / 6 | 0.817 |
| gemini | B | 4 / 6 | 0.750 |
| gemini | C | 5 / 6 | 0.700 |
| gemini | D | 6 / 6 | 0.767 |
| gemini | combined | 4 / 6 | 0.817 |
| haiku | baseline | 4 / 6 | 1.150 |
| haiku | A | 3 / 6 | 1.167 |
| haiku | B | 3 / 6 | 1.033 |
| haiku | C | 5 / 6 | 1.350 |
| haiku | D | 5 / 6 | 1.283 |
| haiku | combined | 5 / 6 | 1.200 |

Source parents are not necessarily pristine. Severity strength is a diagnostic ordering, not perceptual ground truth; aesthetics need not decline monotonically.
