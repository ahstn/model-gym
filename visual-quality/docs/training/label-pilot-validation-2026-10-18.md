# Pilot score validation: 18 October 2026

**Decision: hold the requested 100-image expansion.** The 12 observations pass schema and provenance checks, but the visual review found enough rubric and evidence problems that we cannot validate the teacher for the next batch. No additional paid requests were made. The original observations remain unchanged and pending review.

## Checks and limits

`verify-labels` passed for all 12 records. Source files, prepared views, request hashes, and exported observations match the stored responses. Each score is finite and falls within 1–5 in exact 0.1 steps. All training masks remain false. The three source manifest hashes and the frozen runner code hashes are unchanged.

Two assistant reviewers inspected disjoint groups of six images at original detail. Each had the rubric and view manifest, but neither had the teacher scores or explanations. They returned a plausible score interval and visible evidence for each image. The primary assistant then checked flagged images, the teacher explanations, and source metadata. This is a model-assisted diagnostic check. It is not a human rating study and cannot establish teacher accuracy.

Five of the 12 teacher scores fell inside the reviewers' intervals. That fraction is not an accuracy estimate: the intervals are subjective model judgments, not measured targets or an acceptance threshold. The decision to hold expansion rests on concrete image-evidence problems below.

## Findings

| Example | Teacher output | Visual check and implication |
|---|---|---|
| Sharp sign with a defocused background, `megalith-e6733c0f36a006f834acfe5fb66dc1a6` | 3.6; describes heavy smearing, broken textures, and compression artifacts | The sign text and edges are sharp. The background is out of focus, without clear evidence of the claimed compression damage. The blind interval was 4.3–5.0. The explanation conflicts with the rubric's treatment of selective focus. |
| Large foreground obstruction, `VizWiz_train_00021283.jpg` | 3.2; describes faint grid lines and says no severe artifacts dominate | Most of the frame contains a broad defocused obstruction; fabric detail survives below it. The blind interval was 1.8–2.7. Source annotations record obstruction from four of five raters and blur from three of five. These votes support the defect concern but do not supply a MOS. |
| Noisy building, `distortion-4f0618145dc1b3f104a581a3` | 3.6; explains the result mainly as compression damage and smeared texture | Strong coloured noise covers the sky and building while many edges remain clear. The recorded added operation is Gaussian noise, sigma 15. The explanation omits the dominant visible degradation. |
| Carved subject with a soft background, `megalith-8254d303f6d1cd6142205c4e4f054f79` | 2.2; says heavy blur obscures the subject | Grain, softness, and low contrast are visible, but central carved detail remains inspectable. The blind interval was 2.9–3.6. This is another possible confusion between background defocus and loss of subject detail. |

The synthetic operators do not prove that other defects are absent: the parent photographs can already contain noise or compression damage. However, the stored operations and direct inspection support a concern about defect attribution. Three of the four synthetic explanations describe compression or blockiness without naming the visible added noise. The fourth does name noisy texture.

The sample also has limited coverage. All four generated images happen to use Gaussian noise, with sigma values 5, 15, 15, and 35. The cohort therefore does not test the teacher's response to added blur or JPEG compression. Some very dark or washed-out images received appropriately low scores, and the clearest architectural image received a high score, but these successes do not resolve the other findings.

## Evidence and next check

The [machine-readable audit](evidence/label-pilot-validation-2026-10-18.json) records all 12 blind assessments, teacher outputs, view hashes, source annotations, flagged cases, and the decision to hold expansion. The [original run record](label-pilot-2026-10-18.md) retains the exact request count and reported cost of $0.001243306.

The next useful test is a comparison with another approved teacher on the same frozen views, plus targeted selective-focus, obstruction, blur, noise, and JPEG examples. Resolve these failure cases and collect human anchor ratings before accepting training labels. Do not replace the stored scores with the assistant review intervals or apply a numeric correction based on this small audit.
