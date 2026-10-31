# RunInfra comparison, 25 October 2026

RunInfra's scores are closer to the frozen assistant expectations on the available stage-A views. The largest gain is on clear photographs that the previous OpenRouter/Relace run scored too low. This supports further evaluation of the RunInfra route, but does not establish human accuracy or complete the 50-image comparison.

The requested model was `deepseek-ai/DeepSeek-V4.1-Flash-1017`. RunInfra's live catalog and all replies used API ID `deepseek-v4-1-flash-1017`. Both runs used the unchanged visual-rating-v1 prompt and schema, temperature zero, disabled reasoning, high image detail, and 1024 output tokens. Delivered image bytes were identical to the historical views. The provider route and date differ, so this comparison cannot isolate model weights or internal image processing.

RunInfra returned replies for 41 images. Nine frozen PNG requests exceeded its 3,670,016-byte body limit: A01, A10, A11, A12, A17, A21, A22, A23, and A50. These remain unscored. No smaller copies were authorized or sent. The limit is recorded in the saved live catalog and described in the [RunInfra request contract](https://runinfra.ai/docs/api-reference/chat-completions).

Only 17 of the 41 replies passed the strict label contract. Twenty-three failed solely because a decimal was written with extra precision, such as `3.8999999999999999`. A29 also exceeded the evidence limit with 31 words. All 24 rejected replies remain invalid. No response was repaired, rounded into an accepted label, or retried.

The first invalid reply stopped the original run. A separately frozen continuation then sent each remaining eligible view once. It allowed later cases to proceed after a priced rating-contract failure, while retaining the original run and reply unchanged. The policy change is recorded in [continuation-protocol.md](continuation-protocol.md).

The table below is a post-hoc numeric diagnostic on 40 matching views: the 17 valid replies plus the 23 replies with only a precision failure. It compares the actual returned numeric values with the frozen references. A29 is excluded because its failure also affects the evidence contract. This diagnostic creates no accepted labels.

| Measure on the same 40 views | Previous OpenRouter/Relace | RunInfra |
| --- | ---: | ---: |
| Mean distance outside the reference interval, lower is better | 0.5025 | 0.1925 |
| Mean absolute error from the assistant point score | 0.7450 | 0.4325 |
| Scores inside the reference interval | 8/40 | 13/40 |
| Errors at least one point outside the interval | 7 | 0 |
| Critical contradictions | 0 | 0 |

Mean interval error fell by 0.3100 points, or 61.7%. The paired 95% family-bootstrap interval for this improvement is 0.1681 to 0.4525. On the 32 views from complete original families, improvement is 0.3563 points, with interval 0.1938 to 0.5188. These intervals describe the selected views and assistant references; they do not cover the missing images or establish population accuracy.

| Source group | Matching views | Previous interval error | RunInfra interval error |
| --- | ---: | ---: | ---: |
| Controlled degradation examples | 16 | 0.3250 | 0.2125 |
| VizWiz | 13 | 0.3923 | 0.2385 |
| Standalone Megalith photos | 11 | 0.8909 | 0.1091 |

The standalone photos account for most of the gain. The controlled and VizWiz groups each improve in the point estimate, but their separate uncertainty intervals include zero. Some cases get worse: A02 moves from 3.2 to about 3.9 against a reference interval of 2.7 to 3.3; A15 moves from 3.2 to about 3.7 against an upper bound of 3.1.

The strict comparison also improves on its 17 matching valid outputs: interval error falls from 0.6235 to 0.1588, point-score error falls from 0.8706 to 0.3765, and interval hits rise from 1 to 6. This format-selected subset is too narrow to replace the full comparison. The predeclared full-50 improvement rule remains unmet because the required 50 valid, unchanged-view pairs are unavailable.

Six examples were selected for a visual explanation check before their new replies were read. Displayed scores below are approximate when the exact reply contained extra decimals; exact strings remain in the diagnostic CSV.

| Case | Previous score | RunInfra score | Frozen interval | Explanation check |
| --- | ---: | ---: | --- | --- |
| A18, clean branches and sky | 3.6 | ~4.1 | 4.4–5.0 | Better score, but the added mild-noise claim is not well supported by the smooth sky. |
| A20, strong block compression | 3.2 | 2.5 | 1.8–2.4 | Correctly identifies heavy blocks and reduced detail. |
| A28, extreme blur | 2.5 | 1.5 | 1.0–1.3 | Correctly identifies severe blur and absent fine detail. Vertical streaks are visible; the claim of blocky edge artifacts is less clear. |
| A35, foreground obstruction | 2.5 | 1.5 | 1.0–1.3 | Recognizes almost no usable detail and the small grass patch, but misses the obstruction as the main cause. |
| A40, clear wheelchair photo | 2.6 | ~4.1 | 4.2–4.8 | Correctly describes clear subject and wheelchair detail, avoiding the earlier large quality penalty. |
| A46, carved stone head | 4.2 | ~4.4 | 3.9–4.4 | Retains a high score and correctly identifies preserved texture; the mild-noise claim remains uncertain. |

This small audit supports better visual judgments in several cases. It does not establish an explanation-support rate for all replies. Unsupported or uncertain defect claims remain a concern.

Total reported cost was **$0.00776127** for 41 requests, with no unknown charges. The first request cost $0.00030555; the continuation cost $0.00745572. All observations remain pending human review, aesthetics is null, original photo eligibility is retained, and every task mask remains false. The references are frozen assistant judgments, not human ground truth. Stage A is exposed development evidence; stage B was not used or changed.

The saved results are [the 40-case diagnostic](results/raw-score-diagnostic.json), [the case CSV](results/raw-score-diagnostic.csv), and [the strict 17-case comparison](results/validated-summary.json). Raw replies, request records, model metadata, and the full diagnostic remain under the ignored local directory `visual-quality/data/runinfra-comparison-2026-10-25`. The [original protocol](protocol.md) and continuation protocol record the analysis choices and their timing. Historical inputs and records remain unchanged.

Both journals were reconstructed from their saved replies and verified. All 65 focused offline tests pass, and Ruff reports no errors. Tests cover request identity, immutable inputs, no retries, cost accounting, strict rejection, diagnostic exclusions, and comparison metrics.

To verify saved results without provider calls, run these commands from `visual-quality`:

```sh
.venv/bin/python -B experiments/runinfra-comparison-2026-10-25/runner.py verify
.venv/bin/python -B experiments/runinfra-comparison-2026-10-25/continuation.py verify
.venv/bin/python -B experiments/runinfra-comparison-2026-10-25/raw_score_diagnostic.py --write
```
