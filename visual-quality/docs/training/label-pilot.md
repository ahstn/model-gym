# Small technical-quality label pilot

This runner implements the [proposed label contract](label-specification-2026-10-16.md) for a small technical test. It collects one teacher judgment per image view. It does not measure agreement with humans, select a winning teacher, or approve labels for training.

The [18 October run record](label-pilot-2026-10-18.md) reports the first 12-image execution and its validation results.

The [visual validation audit](label-pilot-validation-2026-10-18.md) holds bulk labelling because some explanations conflict with the visible defects and the rubric. A later, explicitly requested [50+50 prompt experiment](../../experiments/prompt-calibration-2026-10-18/README.md) completed 150 calls and found better explanations but no reliable score improvement. The commands below describe the runner; they do not approve training.

The [reviewed score record](label-pilot-corrections-2026-10-18.md) contains seven score corrections and three evidence-only corrections. These assistant judgments remain pending human review and preserve the original teacher output.

## Label values

The machine-readable contract is [label-observation-v1.schema.json](../../src/visual_quality/schemas/label-observation-v1.schema.json). The local validator enforces the schema and exact decimal steps without floating-point rounding.

| Field | Values and meaning |
|---|---|
| `schema_version` | `label-observation-v1` |
| `label_kind` | `teacher_scalar`; never human MOS |
| `technical_quality.status` | `ok` or `unscorable` |
| `technical_quality.score` | Finite number from 1.0 to 5.0 in steps of 0.1 when status is `ok`; otherwise null |
| `technical_quality.evidence` | Visible evidence or an unscorable reason; at most 30 words |
| `technical_quality.limitations` | Unique values from `limited_detail`, `partial_view`, `ambiguous_style`, `unsupported_domain` |
| `aesthetics` | Status `not_requested`, score null |
| `photo_domain_eligible` | The source value: true, false, or null; never inferred from the score |
| `review_status` | `pending` |
| `task_masks` | All false, including the domain mask |

The quality anchors are **1 bad, 2 poor, 3 fair, 4 good, 5 excellent**. Tenths express a teacher judgment between anchors. They do not imply measured precision. The original [technical-quality prompt](../../prompts/visual-rating-v1/technical-quality.txt) is unchanged. The provider receives a JSON-schema subset; local ingestion applies the complete validation rules. Invalid output is retained in the attempt log and does not become a score. This version has no label promotion command.

Each observation links an image ID, source byte hash, view ID, delivered byte hash, split, and family group. Provenance records the requested and returned model, returned provider, prompt version and hash, request hash, timestamp, usage, run ID, and attempt ID. The original dataset records stay immutable. Normalized 0–1 or 0–100 display values are not stored as additional labels.

## Prepare fixed views

Run these commands from `visual-quality/`:

```sh
uv sync --locked
uv run visual-pilot prepare-views --dataset data/pilot-10k-v1 --views data/label-pilot-12/views --limit 12 --seed 20261018 --max-edge 1536
```

The default selection takes four Megalith photos, four VizWiz images, and four generated distortions. Selection uses a seeded hash order within each component, then cycles through the components. Only the training manifest is read. Known negative domain controls are excluded; unknown eligibility stays null. This sample is for pipeline verification, not a balanced quality evaluation. It does not require matched original/variant pairs or cover every distortion type.

Preparation checks each source byte hash before decoding. It applies EXIF orientation, converts an embedded ICC profile to sRGB, composites transparency on white, and fits the full image within 1536 pixels on the long edge. Images are never enlarged or cropped. Missing ICC profiles use an explicit sRGB assumption; invalid profiles fail. Output is RGB PNG with an sRGB chunk, without EXIF or additional lossy compression. No sharpness, exposure, colour, or contrast correction is applied.

The view record contains native, oriented, and delivered dimensions, the colour action, crop bounds, resize status, alpha handling, and software versions. Its ID binds the image ID, source hash, policy hash, and delivered hash. Scores therefore apply to this prepared view. Resizing can hide defects; these scores must not be presented as native-resolution judgments. The request fixes `image_url.detail` to `high`, but provider-side image processing remains outside our control.

Preparation publishes the complete directory atomically. Repeating an identical command verifies and reuses it. A changed source manifest, image, preparation policy, or output byte stream fails validation. Use a new directory for a different view policy. The CLI limits preparation to 100 images to keep this workflow separate from bulk labelling.

## Run and inspect

Load `OPENROUTER_API_KEY` using your existing environment mechanism. The runner reads it from the process environment and never stores it. On the current machine, the confirmed loader is `source ~/.config/zsh/.zsh-private`.

```sh
uv run visual-pilot label --dry-run
uv run visual-pilot label --model deepseek/deepseek-v4.1-flash-1017 --max-tokens 1024 --max-cost-usd 1.00
uv run visual-pilot verify-labels
```

`--dry-run` checks the prepared views and prompt without network access or credentials. It does not test provider compatibility. The live command fetches the exact model catalogue entry and requires image input and structured-output support. It sends the frozen PNG bytes as a base64 data URL, with the rubric first. There are no model substitutions, provider fallbacks, response-repair plugins, tools, or automatic retries. Reasoning is disabled; temperature is zero when supported. These settings reduce variation but do not guarantee identical model responses.

A prompt directory can include `prompt-version.json` with exactly one key, for example `{"version":"visual-rating-v2"}`. The version must match `visual-rating-v` followed by a positive integer, and the metadata content enters the bundle hash. Without this file, the original v1 version and hash remain unchanged. The experimental v2 prompt has not passed the comparison rule and is not the default.

One request is made at a time. Before each request, the runner reserves the cost of the model's entire context window plus its output limit at the highest listed token rates, including price overrides. This is a conservative admission bound, not an expected cost. Provider price caps use those rates and reject per-image or per-request fees. The first version supports token-billed models; it refuses nonzero additional fee types it cannot bound. It stops if the next reserve would exceed the run budget, which is limited to $10. The default budget is $1. Actual costs come from the provider's `usage.cost`; missing costs remain unknown and stop further calls.

The output directory defaults to `data/label-pilot-12/deepseek/`:

| File | Purpose |
|---|---|
| `run.json` | Frozen prompts, settings, model catalogue snapshot, price bound, source hashes, and implementation hashes |
| `attempts/<id>/request.json` | Attempt identity and request hash, written before the network call |
| `attempts/<id>/result.json` | Raw response body, HTTP status, Retry-After value, or transport error class |
| `observations.jsonl` | Valid, pending teacher observations |
| `report.json` | Attempt counts, failures, known cost, unknown costs, and training readiness |

The view directory contains `preparation.json`, `views.jsonl`, and content-addressed PNG files. All generated data stays under the Git-ignored `data/` directory. Preserve both directories and the original image cache to retain the full evidence chain.

## Resume and failure handling

Repeat the same live command to resume. A process lock prevents two workers from using one run directory. Saved results are parsed again without another API call, and completed views are skipped. The runner refuses changed models, prompts, view manifests, request limits, budgets, or implementation hashes in an existing run.

An HTTP error, refusal, invalid rating, unexpected returned model, truncated answer, or missing cost stops the run. A timeout or interruption after the request record was written has an unknown outcome; the runner never sends that attempt again. A partially written attempt directory also blocks the run. On HTTP 429, the raw Retry-After value is preserved and the run stops. Repeating the command does not bypass that stop or retry the request.

Inspect the raw attempt and provider usage before starting a replacement run. A new output directory can incur a second charge for the same image; it is not an automatic recovery method. There is no retry override in this version. An unscorable but valid reply can be retained with a null score. It is never replaced with zero, one, or the midpoint.

`verify-labels` is read-only and requires no API key. It checks the source and view hashes, run integrity, request hashes, observation schema, and exact agreement between exported observations and stored responses. It can verify the valid subset of a blocked run; verification does not claim that the run completed or that its scores are accurate.

## Apply and verify reviewed corrections

The review workflow requires a `label-review-plan-v1` file with the frozen observation, view-manifest, and prompt hashes. Each entry identifies the exact image and view, repeats the original score, and supplies the reviewed rating, reason, plausible interval, and independent assistant assessment. The plan must cover every valid source observation exactly once. It can cover the valid subset of a blocked source run; it cannot create a label for a failed request.

The checked-in [pilot correction plan](evidence/label-pilot-corrections-2026-10-18.json) contains the reviewed values. Apply it from `visual-quality/` without an API key:

```sh
uv run visual-pilot apply-review --review-plan docs/training/evidence/label-pilot-corrections-2026-10-18.json
uv run visual-pilot verify-review --review-plan docs/training/evidence/label-pilot-corrections-2026-10-18.json
```

Both commands accept `--views`, `--source-run`, and `--review-output`. The default source is `data/label-pilot-12/deepseek/`; the default review output is `data/label-pilot-12/review-v1/`. Output must be separate from the source run, fixed views, and review plan. These commands make no provider calls and do not modify the source files.

`apply-review` checks the original run against its saved responses before publishing two files atomically: `reviewed-observations.jsonl` and `report.json`. Repeating the same command verifies and reuses the files. A changed plan, source, or output fails verification; use a new review version and directory for a later correction. `verify-review` reconstructs the expected export and compares the stored records without writing them.

| Reviewed field | Contract |
|---|---|
| `schema_version` | `label-review-v1` |
| `label_kind` | `assistant_review` |
| `technical_quality` | Same status, 1–5 score, decimal step, evidence, and limitation rules as the original label |
| `source_observation` | Complete original teacher record, including its provenance and score |
| `review` | Review identity, timestamp, assistant method, plan hash, source hash, reason, interval, independent assessment, and decision |
| `review_status` | `pending_human_review` |
| `task_masks` | All false; a correction cannot enable training |
| `aesthetics` and `photo_domain_eligible` | Aesthetics remains unrequested; eligibility retains the source value |

Intervals are subjective review ranges, not statistical confidence intervals. Assistant corrections are not human MOS, and a corrected cohort does not establish teacher accuracy. The report therefore keeps `training_ready=false`.

## Required before training

Keep the [human calibration and teacher-selection gates](label-specification-2026-10-16.md) in place. These training images must not become a final held-out evaluation set. Gather independent human ratings under the same viewing policy, resolve domain eligibility, and test agreement before accepting any labels. Aesthetics requires its own validation. The runner deliberately writes `training_ready=false` even when all API calls succeed.

API behaviour was checked on 2026-10-18 against OpenRouter's [image-input guide](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding), [structured-output guide](https://openrouter.ai/docs/guides/features/structured-outputs), [provider price caps](https://openrouter.ai/docs/guides/routing/provider-selection#max-price), and [rate-limit documentation](https://openrouter.ai/docs/api/reference/limits). The live model snapshot is retained in each run.
