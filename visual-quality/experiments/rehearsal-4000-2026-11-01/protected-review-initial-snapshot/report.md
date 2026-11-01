# Protected style and scope adjudication

Visual review is complete for all 76 queued comparisons across 15 exact delivered images. This includes all 72 model/arm comparisons on the 12 prospectively selected style cases, plus newly flagged scope changes and descriptive uncertain-scope flips. The supplied score export, raw receipts, image bytes, reference freeze and reviewed input records were checked before joining the judgments. Primary references and model scores are unchanged.

These are nonblind diagnostic assistant opinions after opening the exact images. A visibly justified change has an image-based rationale; a regression has a visible quality or scope problem; ambiguous differences cannot be resolved confidently. Small score changes often remain ambiguous. The review neither provides human ground truth nor proves that a particular statistic caused a response difference. Derived-mean comparisons share their component responses and are not independent replications.

| Model | Arm | Visibly justified | Regression | Ambiguous | Not assessable |
|---|---|---:|---:|---:|---:|
| DeepSeek | Own stats | 3 | 3 | 7 | 0 |
| DeepSeek | Swapped stats | 1 | 1 | 10 | 0 |
| Luna | Own stats | 4 | 3 | 7 | 0 |
| Luna | Swapped stats | 2 | 3 | 8 | 0 |
| Luna + DeepSeek mean | Own stats | 6 | 2 | 4 | 0 |
| Luna + DeepSeek mean | Swapped stats | 1 | 1 | 10 | 0 |

The following comparisons have visually supported regressions. Both arms remain in the exports; no example or label was removed because it disagreed with the reference.

| Image | Model | Arm | Visual diagnosis |
|---|---|---|---|
| R0091 | DeepSeek | Own stats | Central petals, veins and leaf surfaces remain finely resolved. The 0.8 reduction overstates noise and low global contrast despite successful selective focus. |
| R0091 | Luna | Own stats | The -0.6 technical drop overstates frame-wide soft texture: focused central petals and leaves preserve detail, whereas most blur follows shallow depth of field. Visible fine grain does not adequately explain the larger penalty. |
| R0091 | Luna + DeepSeek mean | Own stats | The 0.7 mean reduction undervalues the crisp central flower and leaves because large intentional out-of-focus regions dominate the frame. |
| R0110 | DeepSeek | Own stats | Illuminated cloud edges and tonal layers remain distinct. Calling their natural soft forms and intentional dark silhouette a broad capture defect overstates the technical penalty. |
| R0110 | DeepSeek | Swapped stats | The cloud layers and horizon remain defined. The 0.5 deduction for haze and low contrast undervalues the deliberate low-key scene and its resolved edges. |
| R0110 | Luna | Own stats | The -1.0 technical drop treats the nearly featureless foreground silhouette and naturally soft cloud masses as substantial defects despite resolved illuminated cloud contours. That penalty is disproportionate to the visible scene. |
| R0110 | Luna + DeepSeek mean | Own stats | The mean falls 0.75 despite resolved illuminated textures; both component deductions over-penalize the deliberate dark masses and silhouette. |
| R0110 | Luna + DeepSeek mean | Swapped stats | The 0.5 technical reduction persists in the mean even though the scene has coherent edges and smooth tones rather than global capture blur. |
| R0180 | Luna | Swapped stats | The -0.4 technical reduction explicitly faults softer background branches even though selective focus leaves the main bark and lichen sharply resolved. This introduces an unsupported depth-of-field penalty. |
| R0181 | Luna | Own stats | The new declared eligibility floor is unsupported: carton edges and surrounding surface establish a physical-object photograph. A low technical score is warranted by darkness, but a categorical reproduction exclusion is a regression. Technical absolute error alone improves, so it must not conceal the scope error. The treatment forces aesthetics to 1 for an unsupported reproduction exclusion. Physical packaging and surrounding surface remain visible; poor photographic presentation should be judged within scope. |
| R0181 | Luna | Swapped stats | The new declared eligibility floor misclassifies the photographed carton despite visible boundaries and surrounding surface. Darkness can justify a low quality score, but does not justify this categorical exclusion. The 1-point aesthetic floor is caused by a mistaken reproduction exclusion rather than an assessment of the physical packaging photograph; this is a scope regression. |
| R0240 | Luna | Swapped stats | The -0.5 aesthetic drop overstates busy curtain folds and stage equipment: the curtains provide a subdued backdrop and the small devices stay at the bottom edge, while the relaxed figure remains clearly dominant. |
| R0331 | DeepSeek | Own stats | The exact view is a full-frame reproduction of printed text without surrounding scene; the declared document rule requires 1.00 despite its photographic capture. The same full-frame document eligibility rule requires 1.00; raising aesthetics to 1.50 is a scope regression rather than an artistic improvement. |

Reference-uncertain eligibility remains uncertain and its floor flips are descriptive, not confirmed scope errors. Review completion is separate from the numerical advancement tests and does not approve any labels for training. The numerical analysis artifact deliberately leaves its own protected-review status unassessed; this supplemental adjudication supplies the completed review.

The derived mean declares a scope floor only when both component models declare it. A new floor in just one component can still lower the mean without triggering a new declared mean-floor error. Component scope mistakes and their numerical effect therefore remain relevant; the absence of a new mean-floor flag does not establish scope safety.
