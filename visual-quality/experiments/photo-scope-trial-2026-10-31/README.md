# 72-image photography scoring grid

The expanded grid contains all 50 previous images plus 22 new images, with separate technical and aesthetic scores from DeepSeek, Luna, Gemini and Haiku. The models were re-run under the user's confirmed rule: **clearly non-photographic content receives 1.00 for both scores**. The previous experiment and its ratings remain unchanged.

Open the [complete interactive grid](collage/originals-grid.html) or [full-resolution PNG](collage/originals-grid.png). Two smaller pages provide easier inspection: [images 1–36](collage/originals-grid-page-1.png) and [images 37–72](collage/originals-grid-page-2.png). The [CSV](collage/scores.csv) retains exact two-decimal scores, evidence, validity, image paths and request provenance. Images are the original scored PNGs, fitted without cropping or generative redrawing. New cases P051–P072 are marked NEW.

## Why the poster previously scored highly

P048 is a flat exhibition poster combining a painting with promotional typography. Its visible form establishes that it is a non-photographic reproduction. The pixels do not establish whether the file was a screenshot, scan or generated image.

The previous rubric explicitly instructed the models to judge paintings and graphics within their medium and not penalize them for being non-photographic. The old scores therefore measured technical presentation and visual design, without checking suitability for a photography dataset. The user has now changed that selection policy.

All four models independently returned the requested floor under the new prompt:

| Model | Previous technical / aesthetic | New technical / aesthetic |
|---|---|---|
| DeepSeek V4.1 Flash | 3.60 / 4.20 | 1.00 / 1.00 |
| GPT-6 Luna, high reasoning | 4.30 / 4.50 | 1.00 / 1.00 |
| Gemini 3.8 Flash, medium reasoning | 3.10 / 3.50 | 1.00 / 1.00 |
| Claude Haiku 5.5, low thinking requested | 3.80 / 4.00 | 1.00 / 1.00 |

These are the models' actual responses, not manually substituted model scores. P048 was requested once per model, inspected first, and included in each route's 72 requests without resending it. Each response identifies the poster/reproduction and uses `unsupported_domain`.

## Exact rule and boundaries

The [photography policy](photo-scope.txt) precedes both existing quality rubrics. It excludes visibly non-photographic screenshots/interfaces, flat posters and page reproductions, illustrations, paintings presented as the image itself, synthetic graphics and obvious renders. Excluded cases use `status: "ok"`, `score: "1.00"`, and `limitations: ["unsupported_domain"]` in both task objects. This is a dataset eligibility floor, not an assertion that the artwork itself has no artistic merit.

Photographs can contain signs, text, packaging, screens, art or carvings. Visible depth, lighting and surrounding physical context keep those photographs eligible. This cohort includes useful boundary examples: P063 shows artwork panels along a wall and pavement; P068 shows a three-dimensional sign on a metal tower; P072 shows people holding a banner outdoors. None should be excluded merely for containing designed or printed material.

Blur, noise, compression, low resolution, vivid color or an unusual subject are insufficient evidence of a screenshot or AI origin. P054 is almost black; its type is uncertain, so the policy does not impose a domain floor. It can still receive a very low quality score based on its visible darkness. A numeric 1.00/1.00 pair alone is not proof that a model excluded the image; the exclusion flag and evidence matter. Missing or undecodable images retain the prior unscorable/null behavior.

The [prompt provenance](prompt-provenance.json) records the exact text changes. The new prompt retains the combined quality additions used in the preceding collage and changes only the scope instruction and contradictory medium-inclusive sentences. This preserves the display condition for comparison; it does not reverse the earlier recommendation to retain baseline prompts in the general scoring workflow. No production or training labels were promoted.

## Corpus, references and validation

The corpus has 72 unique delivered images from 54 families: 24 controlled views, 22 VizWiz images and 26 standalone Megalith images. The extension adds eight VizWiz and fourteen Megalith images, selected deterministically from training data with seed 20261031. Historical family, exact-image and conservative near-duplicate exclusions were checked for the additions. The selection did not use model scores or filter out non-photographic subjects. All original 50 case records and delivered bytes are preserved exactly. See [manifest](manifest.json), [selection](extension-selection.json) and [preparation validation](preparation-validation.json).

Two fresh assistant reviewers inspected 36 originals each without student scores or earlier references. They classified eligibility for all 72 and supplied independent quality scores for the 22 additions. The resulting scope review has 70 eligible images, one clear exclusion (P048), and one uncertain image (P054). For the reference row, the earlier quality ratings remain unchanged for the 49 eligible old images; P048 receives the newly confirmed policy floor. References are assistant judgments, not human gold.

The [reference freeze](references-freeze.json) describes this hybrid provenance. Reference work ran alongside inference, rather than being completed before every new model request. Reviewers did not receive model outputs. The runner's `reference_kind: no_new_blind_reference` means it required no new reference file before sending images; the later reference artifact is the authoritative description of the independently completed display references.

The strict two-task schema, two-decimal score strings in 0.10 steps, independent task validation, routes and reasoning settings are unchanged. Requests, exact image hashes, prompt hashes, returned identities, raw replies and usage are saved under `visual-quality/data/photo-scope-trial-2026-10-31/<route>/images/`. No API keys are stored. Invalid ratings are displayed explicitly and are never silently repaired. The runner's ten offline tests cover durable admission, resume without resends, invalid sibling preservation, budget bounds, identity checks and input tampering.

The [results and scope audit](summary.json) compare actual exclusion flags with the independent scope review, list all invalid ratings and report known costs. They also compare quality errors before and after the change on retained eligible images with valid ratings in both runs. Those numerical changes are descriptive: one response per condition does not isolate stochastic variation or establish a causal effect. Correctly excluding P048 does not establish that all unrelated quality ratings are unchanged.

The only confirmed non-photographic case in this sample is the user-identified development poster. Success on it does not establish broad recall for screenshots, renders or other excluded media. Neither the policy nor this experiment authenticates image origin from pixels.

## Completed results and effects on ordinary photographs

All 288 unique requests completed (72 per model), with **$0.452624775** in reported cost and no unpriced requests. There were 574 valid task ratings out of 576. Haiku's aesthetic responses for P007 and P028 exceeded the 30-word evidence limit; their aesthetic cells show `invalid`, while their valid technical scores remain visible. There were no resends or manual score repairs.

All four models flagged only P048 as `unsupported_domain` and gave it 1.00 for both scores. No reference-eligible photograph or the uncertain P054 was incorrectly flagged as outside scope. DeepSeek also gave some poor photographs 1.00/1.00 on quality grounds; these are distinguished from scope exclusions in the audit.

The scope change had a positive effect on the known poster, but aesthetic agreement on retained photographs worsened for all four models. The table compares mean absolute error against the unchanged assistant references, using the same eligible old cases with valid ratings in both conditions. Lower error is better; values are score points on the 1–5 scale.

| Model | Technical error, before → after | Aesthetic error, before → after | Aesthetic error change |
|---|---|---|---|
| DeepSeek | 0.355 → 0.369 | 0.208 → 0.294 | +0.086 |
| Luna | 0.378 → 0.367 | 0.259 → 0.282 | +0.022 |
| Gemini | 0.486 → 0.473 | 0.467 → 0.616 | +0.149 |
| Haiku | 0.394 → 0.384 | 0.238 → 0.291 | +0.053 |

Each comparison uses 49 photographs, except Haiku aesthetics (45 pairs because invalid ratings in either condition are omitted). The new 22 images have no paired prior condition and are not included in this before/after table. For example, DeepSeek's aesthetic score for the blurred photographic P008 moved from 1.60 to 1.00 against the fixed 2.50 reference, despite correctly keeping it within photography scope. Full per-image differences are saved in `summary.json`.

This supports the requested eligibility rule for the identified poster, but does **not** demonstrate improved overall scoring accuracy or preservation of all unrelated quality judgments. The experiment is a single run per condition, with a known development example and no significance test. The size and consistency of the aesthetic shifts deserve further evaluation before promoting this combined prompt to the general workflow.

## Reproduction

All provider calls are durable and resumable; the P048 check is part of the 72 rather than an extra call. Each route has a $2 admission cap, $8 overall, with conservative worst-case reserves. No automatic retries are made after failed or uncertain calls. The following commands regenerate derived artifacts without additional inference:

```sh
visual-quality/.venv/bin/python visual-quality/experiments/photo-scope-trial-2026-10-31/assemble_references.py
visual-quality/.venv/bin/python visual-quality/experiments/photo-scope-trial-2026-10-31/runner.py verify --route all
visual-quality/.venv/bin/python visual-quality/experiments/photo-scope-trial-2026-10-31/summarize.py
visual-quality/.venv/bin/python visual-quality/experiments/photo-scope-trial-2026-10-31/build_collage.py
node visual-quality/experiments/photo-scope-trial-2026-10-31/collage/render.mjs
```

The [integrity review](final-integrity-review.json) and [render checks](collage/render-validation.json) record validation status and artifact hashes.

## Follow-up: DeepSeek high reasoning

The original DeepSeek requests explicitly used `reasoning_effort: "none"`, and all 72 responses reported zero reasoning tokens. The requested repeat is now complete with `reasoning_effort: "high"` and a 16,384-token allowance instead of 2,048. Those are the only two changed request fields; the same 72 images, prompt, strict schema, photography policy and reference judgments were reused.

All 72 high-reasoning responses passed both task contracts and reported positive reasoning usage, totaling 32,329 tokens. P048 remains 1.00/1.00, and no reference-eligible photograph was excluded. On all 72 matched cases, DeepSeek technical MAE worsened from 0.3583 to 0.3931 while aesthetic MAE improved from 0.2958 to 0.2611. Their equally weighted average error is unchanged. For the DeepSeek/Luna average, technical MAE worsened 0.3139 → 0.3465 and aesthetic MAE improved 0.2160 → 0.2056, so switching both dimensions to high is not supported by this trial alone.

See the [full follow-up report](../deepseek-high-trial-2026-10-31/README.md), [none-versus-high comparison grid](../deepseek-high-trial-2026-10-31/collage/originals-grid.html) and [analysis](../deepseek-high-trial-2026-10-31/analysis.json). The original grid above remains the original four-model result. This repeat is descriptive evidence against assistant references, not a human accuracy benchmark; increasing the output allowance is also a changed setting.

## Total cost per model

Costs cover this 72-image four-model trial and the additional 72-image DeepSeek high-reasoning repeat. Earlier 50/200-image experiments are excluded. Each call produces both task scores. Costs are reconciled from actual API usage, including confirmed Luna BYOK upstream usage, with no unpriced requests. DeepSeek has twice as many calls because both reasoning settings are retained. See the [exact cost ledger](../deepseek-high-trial-2026-10-31/cost-summary.json).

| Model | Original 72-image run (USD) | High-reasoning rerun (USD) | Total calls | Total cost (USD) |
|---|---:|---:|---:|---:|
| DeepSeek V4.1 Flash 1017 | 0.016905550 | 0.036467350 | 144 | 0.053372900 |
| GPT-6 Luna, high | 0.055711975 | 0 | 72 | 0.055711975 |
| Gemini 3.8 Flash, medium | 0.321563250 | 0 | 72 | 0.321563250 |
| Claude Haiku 5.5, low | 0.058444000 | 0 | 72 | 0.058444000 |
| **Total** | **0.452624775** | **0.036467350** | **360** | **0.489092125** |
