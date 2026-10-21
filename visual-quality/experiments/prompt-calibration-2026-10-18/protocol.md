# DeepSeek technical-quality prompt calibration

Frozen before the first new provider request on 18 October 2026. This experiment uses 100 new image views in two stages. It tests a prompt revision against assistant references; it does not establish agreement with human quality ratings.

## Design

Stage A contains 50 images scored with the existing `visual-rating-v1` prompt. Blinded assistant reviewers inspect the same fixed views without seeing provider scores. Their references are saved before model outputs are inspected. The provider explanations are then checked against visible evidence. One prompt revision is allowed, based on stage A and the earlier 12-image audit.

Stage B contains another 50 images from separate source families. Its selection is fixed before stage A. The prompt reviser must not inspect stage-B images, references, or model outputs before saving and hashing the revised prompt. Fresh reviewers then inspect stage B without either prompt's scores. Both prompts score the same stage-B views, which gives a paired comparison. The two serial runners can run concurrently to reduce time-related provider differences. This requires 150 requests for 100 unique images if every request succeeds.

Each stage contains six source parents and their 18 distorted variants, 14 VizWiz images, and 12 separate Megalith photos. The controlled families cover Gaussian blur, Gaussian noise, and JPEG compression, with two families and three stored severity levels per operator. VizWiz selection uses fixed quotas across available dominant flaw groups: three blur, three framing, and two each for brightness, darkness, obstruction, and rotation. Source votes do not supply quality scores.

Selection uses metadata and a fixed hash seed. Exclude prior-12 families and overlap between stages by family, source and decoded hashes, and a conservative dHash distance of two or less. Require source parents without embedded ICC or transparency differences. These checks reduce known overlap and preprocessing confounds; they cannot prove that all semantic duplicates are absent. A source parent is not assumed to be pristine. The exact selection and file hashes are recorded in `manifest.json`.

All images use the existing full-frame sRGB PNG policy, with a maximum edge of 1536 pixels, no enlargement, and no image correction. The original dataset and previous pilot remain unchanged. New dataset subsets, fixed views, responses, and observations are stored under `visual-quality/data/`.

## Reference and evidence review

Reviewers use the unchanged technical-quality construct and its 1–5 anchors. Each reviewer records a point judgment, a plausible interval, visible evidence, and defect categories. Intervals must contain the point judgment. They express subjective uncertainty, not statistical confidence or human MOS. References do not use source captions, transform parameters, or the provider's explanation.

Review packets use anonymous case IDs and shuffled order. Stage-B evidence review receives the two model outputs under shuffled arm names, without the prompt text or version. Reviewers identify unsupported explanations and concrete rubric conflicts. Scores remain unchanged in the raw records. Review decisions and any missing assessments are retained as separate evidence.

## Metrics and decision rule

For a model score `s` and reference interval `[L, U]`, interval distance is `max(L - s, 0, s - U)`. The primary effect is the old prompt's mean distance minus the new prompt's mean distance on the 50 stage-B images. A positive value favours the revised prompt.

Call the revision an improvement against these assistant references only if all conditions hold:

1. All 50 paired outputs are valid, scorable, and linked to the same frozen views.
2. Mean interval distance falls by at least 0.15 points on the 1–5 scale.
3. The lower bound of a paired 95% family-bootstrap interval is above zero.
4. There are no new critical contradictions: a score of at least 4 when the reference upper bound is at most 2, or a score of at most 2 when its lower bound is at least 4.
5. No source stratum's mean distance worsens by more than 0.10 points.

Use 10,000 bootstrap draws with seed 20261018. Resample whole families within three strata: six controlled families, 14 VizWiz images, and 12 separate Megalith images. Keep the planned image weights of 48%, 28%, and 24%. Bootstrap uncertainty describes this small panel and these assistant references; it does not represent human rating uncertainty.

Report interval-hit rate, mean interval width, mean absolute distance from assistant point judgments, large interval errors of at least one point, evidence issues, and provider distributions. Also report score ordering across stored severity levels. Ties and ordering violations are separate counts. Severity is an operator parameter, so ordering is a diagnostic and is not treated as measured perceptual truth. Parent-to-variant comparisons are reported separately from within-operator severity comparisons.

A missing pair, a baseline too close to the floor for a meaningful reduction, or an uncertainty interval that crosses zero is inconclusive. Do not fill failed or unscorable outputs with numeric values. A failed gate does not justify another prompt revision on stage B. Any later revision requires another unseen test cohort.

## Execution and limits

Use `deepseek/deepseek-v4.1-flash-1017` through OpenRouter, with the existing key loader. Keep image bytes, output schema, token limit of 1024, and request settings fixed between prompts. The runner fetches current model and price metadata, retains returned provider names and raw usage, and stops on failed, invalid, or unpriced attempts. Each of the three runs has a $1 admission budget; the total maximum is $3. This is a safety bound, not an expected charge.

Prompt files, rubric versions, model metadata, request hashes, source selection, reference records, analysis code, and results must remain available. Preserve all original scores. All training masks stay false, and every result remains pending human validation. No human calibration or production threshold is established by this experiment.
