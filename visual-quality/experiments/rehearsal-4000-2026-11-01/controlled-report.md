## Supplemental blind review of controlled degradation

Supplemental descriptive technical-score diagnostics; all 100 controlled families remain in the original analysis. Blind categorical stimulus reviews select additional comparisons, never numeric labels. A comparison is verified only for clear_degradation. Parent comparisons, severe-versus-parent comparisons and fully verified adjacent ladders have separate denominators. Positive mean drop is comparison technical score minus derivative technical score. Native precision is retained; absolute differences <=1e-8 are ties. Exact (1,1) floor pairs are reported separately and can be policy-correct for art/nonphotographic content; neither ties nor every severe score at least as high as its parent is automatically an error. No aesthetic monotonicity is required. This is not PACE ground-truth-threshold Holistic Override Rate.

The audit covers all 100 families (400 images), 600 recorded entries and 500 unique image pairs. V1’s parent and previous comparisons name the same pair. All 100 families remain in the primary diagnostics; no label changed.

All three derivatives are verified against the parent in 73 families; 73 families have all three adjacent comparisons verified; 100 severe-versus-parent comparisons are verified. These are distinct subsets. Review classifications and per-pair reasons are preserved in `controlled-review/audit.json` and `controlled-analysis.json`.

Unique-pair review classifications: clear_degradation: 473; not_clearly_worse: 27; unjudgeable: 0.

### All derivatives versus parent

↓ / = / ↑ count strict decreases, ties and strict increases. Floor counts are included in ties. Drop uses all complete pairs, including ties.

| Model | Operator | Review subset | Complete/expected | Missing | ↓ / = / ↑ | Floor 1 | Mean technical drop |
|---|---|---|---:|---:|---:|---:|---:|
| luna | all | all | 300/300 | 0 | 255 / 31 / 14 | 7 | 0.8613 |
| luna | gaussian_blur | all | 102/102 | 0 | 96 / 5 / 1 | 5 | 1.4225 |
| luna | gaussian_noise | all | 99/99 | 0 | 88 / 8 / 3 | 2 | 0.8303 |
| luna | jpeg_compression | all | 99/99 | 0 | 71 / 18 / 10 | 0 | 0.3141 |
| deepseek | all | all | 300/300 | 0 | 227 / 43 / 30 | 7 | 0.8037 |
| deepseek | gaussian_blur | all | 102/102 | 0 | 93 / 7 / 2 | 6 | 1.6284 |
| deepseek | gaussian_noise | all | 99/99 | 0 | 69 / 18 / 12 | 1 | 0.3697 |
| deepseek | jpeg_compression | all | 99/99 | 0 | 65 / 18 / 16 | 0 | 0.3879 |
| jev | all | all | 300/300 | 0 | 260 / 1 / 39 | 0 | 0.9142 |
| jev | gaussian_blur | all | 102/102 | 0 | 102 / 0 / 0 | 0 | 1.8994 |
| jev | gaussian_noise | all | 99/99 | 0 | 92 / 0 / 7 | 0 | 0.5759 |
| jev | jpeg_compression | all | 99/99 | 0 | 66 / 1 / 32 | 0 | 0.2376 |
| liquid | all | all | 300/300 | 0 | 260 / 0 / 40 | 0 | 0.9161 |
| liquid | gaussian_blur | all | 102/102 | 0 | 102 / 0 / 0 | 0 | 2.0130 |
| liquid | gaussian_noise | all | 99/99 | 0 | 83 / 0 / 16 | 0 | 0.4255 |
| liquid | jpeg_compression | all | 99/99 | 0 | 75 / 0 / 24 | 0 | 0.2765 |
| qrealign_lite | all | all | 300/300 | 0 | 294 / 0 / 6 | 0 | 1.3054 |
| qrealign_lite | gaussian_blur | all | 102/102 | 0 | 102 / 0 / 0 | 0 | 2.1652 |
| qrealign_lite | gaussian_noise | all | 99/99 | 0 | 96 / 0 / 3 | 0 | 0.8650 |
| qrealign_lite | jpeg_compression | all | 99/99 | 0 | 96 / 0 / 3 | 0 | 0.8599 |
| luna_deepseek_mean | all | all | 300/300 | 0 | 260 / 20 / 20 | 6 | 0.8325 |
| luna_deepseek_mean | gaussian_blur | all | 102/102 | 0 | 96 / 5 / 1 | 5 | 1.5255 |
| luna_deepseek_mean | gaussian_noise | all | 99/99 | 0 | 88 / 5 / 6 | 1 | 0.6000 |
| luna_deepseek_mean | jpeg_compression | all | 99/99 | 0 | 76 / 10 / 13 | 0 | 0.3510 |
| luna | all | visually_verified | 273/273 | 0 | 240 / 26 / 7 | 7 | 0.9388 |
| luna | gaussian_blur | visually_verified | 102/102 | 0 | 96 / 5 / 1 | 5 | 1.4225 |
| luna | gaussian_noise | visually_verified | 99/99 | 0 | 88 / 8 / 3 | 2 | 0.8303 |
| luna | jpeg_compression | visually_verified | 72/72 | 0 | 56 / 13 / 3 | 0 | 0.4028 |
| deepseek | all | visually_verified | 273/273 | 0 | 214 / 38 / 21 | 7 | 0.8725 |
| deepseek | gaussian_blur | visually_verified | 102/102 | 0 | 93 / 7 / 2 | 6 | 1.6284 |
| deepseek | gaussian_noise | visually_verified | 99/99 | 0 | 69 / 18 / 12 | 1 | 0.3697 |
| deepseek | jpeg_compression | visually_verified | 72/72 | 0 | 52 / 13 / 7 | 0 | 0.4931 |
| jev | all | visually_verified | 273/273 | 0 | 249 / 1 / 23 | 0 | 1.0055 |
| jev | gaussian_blur | visually_verified | 102/102 | 0 | 102 / 0 / 0 | 0 | 1.8994 |
| jev | gaussian_noise | visually_verified | 99/99 | 0 | 92 / 0 / 7 | 0 | 0.5759 |
| jev | jpeg_compression | visually_verified | 72/72 | 0 | 55 / 1 / 16 | 0 | 0.3299 |
| liquid | all | visually_verified | 273/273 | 0 | 245 / 0 / 28 | 0 | 1.0045 |
| liquid | gaussian_blur | visually_verified | 102/102 | 0 | 102 / 0 / 0 | 0 | 2.0130 |
| liquid | gaussian_noise | visually_verified | 99/99 | 0 | 83 / 0 / 16 | 0 | 0.4255 |
| liquid | jpeg_compression | visually_verified | 72/72 | 0 | 60 / 0 / 12 | 0 | 0.3720 |
| qrealign_lite | all | visually_verified | 273/273 | 0 | 269 / 0 / 4 | 0 | 1.4224 |
| qrealign_lite | gaussian_blur | visually_verified | 102/102 | 0 | 102 / 0 / 0 | 0 | 2.1652 |
| qrealign_lite | gaussian_noise | visually_verified | 99/99 | 0 | 96 / 0 / 3 | 0 | 0.8650 |
| qrealign_lite | jpeg_compression | visually_verified | 72/72 | 0 | 71 / 0 / 1 | 0 | 1.1366 |
| luna_deepseek_mean | all | visually_verified | 273/273 | 0 | 241 / 18 / 14 | 6 | 0.9057 |
| luna_deepseek_mean | gaussian_blur | visually_verified | 102/102 | 0 | 96 / 5 / 1 | 5 | 1.5255 |
| luna_deepseek_mean | gaussian_noise | visually_verified | 99/99 | 0 | 88 / 5 / 6 | 1 | 0.6000 |
| luna_deepseek_mean | jpeg_compression | visually_verified | 72/72 | 0 | 57 / 8 / 7 | 0 | 0.4479 |

### Severe derivative versus parent

↓ / = / ↑ count strict decreases, ties and strict increases. Floor counts are included in ties. Drop uses all complete pairs, including ties.

| Model | Operator | Review subset | Complete/expected | Missing | ↓ / = / ↑ | Floor 1 | Mean technical drop |
|---|---|---|---:|---:|---:|---:|---:|
| luna | all | all | 100/100 | 0 | 94 / 5 / 1 | 2 | 1.2770 |
| luna | gaussian_blur | all | 34/34 | 0 | 32 / 1 / 1 | 1 | 1.7794 |
| luna | gaussian_noise | all | 33/33 | 0 | 31 / 2 / 0 | 1 | 1.3091 |
| luna | jpeg_compression | all | 33/33 | 0 | 31 / 2 / 0 | 0 | 0.7273 |
| deepseek | all | all | 100/100 | 0 | 95 / 4 / 1 | 2 | 1.4330 |
| deepseek | gaussian_blur | all | 34/34 | 0 | 32 / 2 / 0 | 2 | 2.4647 |
| deepseek | gaussian_noise | all | 33/33 | 0 | 31 / 1 / 1 | 0 | 0.7697 |
| deepseek | jpeg_compression | all | 33/33 | 0 | 32 / 1 / 0 | 0 | 1.0333 |
| jev | all | all | 100/100 | 0 | 100 / 0 / 0 | 0 | 1.4602 |
| jev | gaussian_blur | all | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.5712 |
| jev | gaussian_noise | all | 33/33 | 0 | 33 / 0 / 0 | 0 | 1.1109 |
| jev | jpeg_compression | all | 33/33 | 0 | 33 / 0 / 0 | 0 | 0.6648 |
| liquid | all | all | 100/100 | 0 | 98 / 0 / 2 | 0 | 1.3466 |
| liquid | gaussian_blur | all | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.3747 |
| liquid | gaussian_noise | all | 33/33 | 0 | 32 / 0 / 1 | 0 | 0.8996 |
| liquid | jpeg_compression | all | 33/33 | 0 | 32 / 0 / 1 | 0 | 0.7343 |
| qrealign_lite | all | all | 100/100 | 0 | 100 / 0 / 0 | 0 | 2.2624 |
| qrealign_lite | gaussian_blur | all | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.7556 |
| qrealign_lite | gaussian_noise | all | 33/33 | 0 | 33 / 0 / 0 | 0 | 1.8527 |
| qrealign_lite | jpeg_compression | all | 33/33 | 0 | 33 / 0 / 0 | 0 | 2.1640 |
| luna_deepseek_mean | all | all | 100/100 | 0 | 96 / 2 / 2 | 1 | 1.3550 |
| luna_deepseek_mean | gaussian_blur | all | 34/34 | 0 | 32 / 1 / 1 | 1 | 2.1221 |
| luna_deepseek_mean | gaussian_noise | all | 33/33 | 0 | 32 / 0 / 1 | 0 | 1.0394 |
| luna_deepseek_mean | jpeg_compression | all | 33/33 | 0 | 32 / 1 / 0 | 0 | 0.8803 |
| luna | all | visually_verified | 100/100 | 0 | 94 / 5 / 1 | 2 | 1.2770 |
| luna | gaussian_blur | visually_verified | 34/34 | 0 | 32 / 1 / 1 | 1 | 1.7794 |
| luna | gaussian_noise | visually_verified | 33/33 | 0 | 31 / 2 / 0 | 1 | 1.3091 |
| luna | jpeg_compression | visually_verified | 33/33 | 0 | 31 / 2 / 0 | 0 | 0.7273 |
| deepseek | all | visually_verified | 100/100 | 0 | 95 / 4 / 1 | 2 | 1.4330 |
| deepseek | gaussian_blur | visually_verified | 34/34 | 0 | 32 / 2 / 0 | 2 | 2.4647 |
| deepseek | gaussian_noise | visually_verified | 33/33 | 0 | 31 / 1 / 1 | 0 | 0.7697 |
| deepseek | jpeg_compression | visually_verified | 33/33 | 0 | 32 / 1 / 0 | 0 | 1.0333 |
| jev | all | visually_verified | 100/100 | 0 | 100 / 0 / 0 | 0 | 1.4602 |
| jev | gaussian_blur | visually_verified | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.5712 |
| jev | gaussian_noise | visually_verified | 33/33 | 0 | 33 / 0 / 0 | 0 | 1.1109 |
| jev | jpeg_compression | visually_verified | 33/33 | 0 | 33 / 0 / 0 | 0 | 0.6648 |
| liquid | all | visually_verified | 100/100 | 0 | 98 / 0 / 2 | 0 | 1.3466 |
| liquid | gaussian_blur | visually_verified | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.3747 |
| liquid | gaussian_noise | visually_verified | 33/33 | 0 | 32 / 0 / 1 | 0 | 0.8996 |
| liquid | jpeg_compression | visually_verified | 33/33 | 0 | 32 / 0 / 1 | 0 | 0.7343 |
| qrealign_lite | all | visually_verified | 100/100 | 0 | 100 / 0 / 0 | 0 | 2.2624 |
| qrealign_lite | gaussian_blur | visually_verified | 34/34 | 0 | 34 / 0 / 0 | 0 | 2.7556 |
| qrealign_lite | gaussian_noise | visually_verified | 33/33 | 0 | 33 / 0 / 0 | 0 | 1.8527 |
| qrealign_lite | jpeg_compression | visually_verified | 33/33 | 0 | 33 / 0 / 0 | 0 | 2.1640 |
| luna_deepseek_mean | all | visually_verified | 100/100 | 0 | 96 / 2 / 2 | 1 | 1.3550 |
| luna_deepseek_mean | gaussian_blur | visually_verified | 34/34 | 0 | 32 / 1 / 1 | 1 | 2.1221 |
| luna_deepseek_mean | gaussian_noise | visually_verified | 33/33 | 0 | 32 / 0 / 1 | 0 | 1.0394 |
| luna_deepseek_mean | jpeg_compression | visually_verified | 33/33 | 0 | 32 / 1 / 0 | 0 | 0.8803 |

### Fully verified adjacent families

Ladder counts require all four technical scores. Pair tables above require only their two endpoints, so missing intermediate scores do not erase a valid parent–severe comparison.

| Model | Operator | Complete/expected families | Missing | Non-increasing | Strictly decreasing | Adjacent ↓ / = / ↑ | Adjacent floor 1 | Mean parent–severe drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| luna | all | 73/73 | 0 | 64 | 47 | 184 / 26 / 9 | 6 | 1.4863 |
| luna | gaussian_blur | 34/34 | 0 | 32 | 22 | 86 / 14 / 2 | 5 | 1.7794 |
| luna | gaussian_noise | 33/33 | 0 | 29 | 24 | 86 / 9 / 4 | 1 | 1.3091 |
| luna | jpeg_compression | 6/6 | 0 | 3 | 1 | 12 / 3 / 3 | 0 | 0.8000 |
| deepseek | all | 73/73 | 0 | 53 | 35 | 169 / 29 / 21 | 8 | 1.5945 |
| deepseek | gaussian_blur | 34/34 | 0 | 29 | 23 | 87 / 10 / 5 | 8 | 2.4647 |
| deepseek | gaussian_noise | 33/33 | 0 | 22 | 11 | 70 / 17 / 12 | 0 | 0.7697 |
| deepseek | jpeg_compression | 6/6 | 0 | 2 | 1 | 12 / 2 / 4 | 0 | 1.2000 |
| jev | all | 73/73 | 0 | 63 | 62 | 208 / 1 / 10 | 0 | 1.7704 |
| jev | gaussian_blur | 34/34 | 0 | 34 | 34 | 102 / 0 / 0 | 0 | 2.5712 |
| jev | gaussian_noise | 33/33 | 0 | 26 | 26 | 92 / 0 / 7 | 0 | 1.1109 |
| jev | jpeg_compression | 6/6 | 0 | 3 | 2 | 14 / 1 / 3 | 0 | 0.8600 |
| liquid | all | 73/73 | 0 | 54 | 54 | 200 / 0 / 19 | 0 | 1.5877 |
| liquid | gaussian_blur | 34/34 | 0 | 32 | 32 | 100 / 0 / 2 | 0 | 2.3747 |
| liquid | gaussian_noise | 33/33 | 0 | 19 | 19 | 85 / 0 / 14 | 0 | 0.8996 |
| liquid | jpeg_compression | 6/6 | 0 | 3 | 3 | 15 / 0 / 3 | 0 | 0.9123 |
| qrealign_lite | all | 73/73 | 0 | 70 | 70 | 216 / 0 / 3 | 0 | 2.3207 |
| qrealign_lite | gaussian_blur | 34/34 | 0 | 34 | 34 | 102 / 0 / 0 | 0 | 2.7556 |
| qrealign_lite | gaussian_noise | 33/33 | 0 | 30 | 30 | 96 / 0 / 3 | 0 | 1.8527 |
| qrealign_lite | jpeg_compression | 6/6 | 0 | 6 | 6 | 18 / 0 / 0 | 0 | 2.4308 |
| luna_deepseek_mean | all | 73/73 | 0 | 59 | 51 | 192 / 12 / 15 | 5 | 1.5404 |
| luna_deepseek_mean | gaussian_blur | 34/34 | 0 | 30 | 28 | 92 / 6 / 4 | 5 | 2.1221 |
| luna_deepseek_mean | gaussian_noise | 33/33 | 0 | 27 | 22 | 87 / 5 / 7 | 0 | 1.0394 |
| luna_deepseek_mean | jpeg_compression | 6/6 | 0 | 2 | 1 | 13 / 1 / 4 | 0 | 1.0000 |

The JSON preserves all-family ladders, adjacent-pair coverage, both scope-floor flags, missing endpoint IDs and native score differences. Review verification concerns visible degradation and does not establish an independently scored quality target.
