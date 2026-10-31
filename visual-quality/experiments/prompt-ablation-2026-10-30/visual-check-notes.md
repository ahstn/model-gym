# Post-result visual checks

These checks were selected after reading DeepSeek's completed scores. They illustrate failures and improvements; they are not a representative audit, do not replace the separately preselected anonymous explanation audit, and do not change frozen references.

| Case | Visible content at original supplied scale | DeepSeek technical baseline → combined | Frozen reference | Interpretation |
|---|---|---|---|---|
| P024 | Church facade remains recognizable, but large block boundaries break up foliage, sky and grass. | 3.5 → 3.0 | 2.4 [2.1, 2.7] | A relevant improvement, although the combined score remains above the reference interval. Addition A alone scored 2.9. |
| P047 | Purple-lit drummer with coarse grain, soft facial detail, clipped bright shirt/head highlights and large foreground obstructions. | 2.6 → 3.5 | 2.5 [2.2, 2.8] | Clear regression in agreement: the combined explanation calls the face and hardware reasonably resolved, and omits the consequential clipping. B and C separately scored 3.2. |
| P042 | Outdoor white-gown photo session: garment, paving, photographers and platform retain substantial detail; distant faces are naturally small. | 4.5 → 4.0 | 4.6 [4.3, 4.9] | The combined explanation gives more weight to minor distant softness, moving away from the strong overall technical quality. A/B/C/D all lower its score. |
| P027 | Defocused product label: barcode lines merge and most lettering cannot be read. | 1.8 → 2.3 | 1.8 [1.5, 2.1] | Evidence remains similar but the numeric score becomes more generous, illustrating why better-sounding explanations do not establish better calibration. |

Original-scale inspection also confirmed severe whole-frame blur in P004/P008, dense multicolored noise in P016, and blocky stone-relief texture in P020. Recognizability survives several of these defects; the severity and visible impact differ. The checks support retaining a distinction between technical fidelity and aesthetic effect, while showing that longer clarifications do not reliably enforce that distinction.
