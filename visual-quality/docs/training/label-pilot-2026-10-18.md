# Technical-quality pilot: 18 October 2026

The first live run completed with **12 schema-valid technical-quality scores from 12 requests**. This verifies the data path, response contract, and saved evidence. Human agreement and teacher accuracy have not been measured. All observations remain pending review, with training masks disabled.

The [subsequent visual validation](label-pilot-validation-2026-10-18.md) found rubric and evidence problems. The requested 100-image expansion is on hold. Schema validity does not establish score accuracy.

| Item | Result |
|---|---|
| Requested and returned model | `deepseek/deepseek-v4.1-flash-1017` |
| Returned provider | Relace through OpenRouter, all 12 responses |
| Sample | Four Megalith photos, four VizWiz images, four synthetic distortions |
| Split | Training only; development and final test unchanged |
| View | Full image, sRGB, RGB PNG, long edge at most 1536 pixels, no upscale |
| Request | `visual-rating-v1` rubric, strict JSON schema, high image detail, temperature 0, reasoning disabled, 1024 output-token limit |
| Valid / scored / unscorable | 12 / 12 / 0 |
| Score range | 1.5–4.2 on the proposed 1–5 scale |
| Prompt / completion tokens | 21,173 / 702 |
| Reported cost | **$0.001243306**, from the sum of `usage.cost` |
| Budget guard | $1; full-context reserve checked before each call |
| Unknown costs | 0 |
| Training ready | No; human calibration and eligibility review remain open |

The seed was `20261018`. The sample was selected before inference, without filtering by scores. Its component quotas and small size do not support an accuracy estimate, teacher ranking, or a reliable cost forecast for 250,000 images. Cache effects, provider routes, image sizes, and price changes can alter later costs.

## Stored evidence

The [tracked evidence record](evidence/label-pilot-2026-10-18.json) contains the run settings, model/provider identity, selected image and view hashes, scores, usage totals, validation results, and artifact hashes. It also records the source manifest hashes for train, development, and test. All three match their values before this implementation.

Full local artifacts remain under the ignored data directory:

- [Prepared views](../../data/label-pilot-12/views/preparation.json) and [view manifest](../../data/label-pilot-12/views/views.jsonl).
- [Run metadata](../../data/label-pilot-12/deepseek/run.json), including the live model catalogue entry and frozen prompts.
- [Pending observations](../../data/label-pilot-12/deepseek/observations.jsonl) and [run report](../../data/label-pilot-12/deepseek/report.json).
- Raw request records and provider responses in `data/label-pilot-12/deepseek/attempts/`.

The run ID is `08d02f26720b16eb16419f4fbedb3f460f8128db13fc8f0e813f98c1ab2d2ceb`. The observations SHA-256 is `153a1e5df5317ed890e49e0435d769f2eaac35480a66d6b8aacbfc5df6a950d4`.

## Verification

The package suite passed **180 tests and four subtests**. Ruff lint and format checks passed. A built wheel included the versioned JSON schema. `verify-labels` verified all 12 observations against the prepared views and stored responses.

A second local invocation replaced both network entry points with functions that fail if called. The completed run resumed successfully without a network call, and the run, report, and observation hashes stayed identical. Tests also cover HTTP 429, missing cost, malformed output, interrupted requests, saved-response recovery, changed settings, and source/view tampering.

Use the [runner guide](label-pilot.md) to repeat the preparation or inspect the existing run. Do not promote these teacher scores to accepted training targets until the [label validation gates](label-specification-2026-10-16.md) pass.
