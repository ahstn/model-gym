# Three-model trial: separate technical quality and aesthetics

Completed on 27 October 2026: the same 50 fresh images were assessed by RunInfra DeepSeek V4.1 Flash 1017, OpenRouter Gemini 3.8 Flash at medium reasoning, and OpenRouter GPT-6 Luna at high reasoning. All 150 image responses passed both task contracts, giving 300 valid scores. All 15 preceding text-only conformance probes passed too.

**No clear overall winner emerged.** Technical agreement was nearly tied. DeepSeek was closest to the frozen assistant aesthetic assessments in this sample, but every paired 95% error-difference interval included zero. Neither requested OpenRouter configuration demonstrated a clear overall advantage over DeepSeek. These are comparisons with independent assistant judgments, not validated human accuracy.

## Agreement with the frozen assessments

Scores range from 1 to 5 in 0.1 steps. MAE is the average absolute difference from the reference point score; lower is closer. Interval distance measures how far a score falls outside its reference's plausible range. Bias is student minus reference. All comparisons below use the same 50 cases, with no missing scores or exclusions.

| Model | Technical MAE | Technical interval distance | Technical bias | Aesthetic MAE | Aesthetic interval distance | Aesthetic bias |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek | 0.45 | 0.20 | +0.21 | **0.32** | **0.08** | +0.14 |
| Gemini, medium | 0.44 | **0.19** | −0.11 | 0.37 | 0.09 | −0.20 |
| Luna, high | 0.44 | 0.20 | +0.32 | 0.39 | 0.09 | +0.35 |

The unrounded technical MAEs are 0.450, 0.438 and 0.436: Luna's 0.002-point advantage over Gemini is too small to interpret as a meaningful lead. The paired family bootstrap supports that caution. DeepSeek's aesthetic MAE difference versus Gemini is −0.054, with 95% interval −0.170 to +0.052; versus Luna it is −0.066, with interval −0.150 to +0.016. All six paired MAE intervals and all six interval-distance intervals cross zero. Full estimates are in [results.json](results.json).

| Model | Technical interval hits | Technical errors ≥1 point outside interval | Aesthetic interval hits | Aesthetic errors ≥1 point outside interval |
|---|---:|---:|---:|---:|
| DeepSeek | 22/50 | 2/50 | 37/50 | 0/50 |
| Gemini | 22/50 | 0/50 | 34/50 | 0/50 |
| Luna | 25/50 | 3/50 | 34/50 | 0/50 |

Aesthetic reference intervals were wider on average: 0.82 points versus 0.61 for technical quality. Higher aesthetic interval-hit rates therefore do not establish that aesthetics is the easier or more accurate task. The predefined critical-contradiction count was zero for every model, but its narrow threshold excludes substantial misses such as D020.

The mixture matters. On the 24 controlled images, technical MAE was 0.49 for DeepSeek, 0.39 for Gemini and 0.53 for Luna. On the 12 standalone Megalith images it was 0.44, 0.60 and 0.25, respectively. Gemini was consistently lower than the reference on those standalone images, averaging −0.60 technically and −0.59 aesthetically. Luna was consistently higher on the 14 VizWiz technical judgments, averaging +0.44. These small subgroups are diagnostic rather than separate leaderboards.

## What the image judgments show

The [posthoc disagreement review](disagreement-review.json) records four directly inspected examples without revising references. They illustrate failure modes, not error rates.

- **D020, heavily compressed crowd scene:** the technical reference was 2.0, plausible interval 1.7–2.4. Gemini scored 2.3 and identified severe compression; DeepSeek scored 3.5 and Luna 4.0, emphasizing legibility and lighting while underweighting conspicuous blocks on faces and clothing.
- **D024, compressed building facade:** the reference was 2.1, interval 1.8–2.5. Gemini scored 2.7, DeepSeek 3.6 and Luna 3.5. Sky bands, facade discoloration and blocky pavement were readily visible; DeepSeek described compression as mild.
- **D031, grainy night view with dominant glare:** DeepSeek's aesthetic score of 3.2 exceeded the 1.9 reference. It credited atmosphere more strongly. This is a difference in aesthetic weighting, not proof that the scene has no atmosphere.
- **D050, low railway perspective:** all three technical scores fell below the 4.4 reference. The foreground rail and pavement retain detail while the distance is defocused. This exposes ambiguity about the principal content and acceptable depth of field; the reference interpretation is not human-validated.

A separate [preselected explanation audit](explanation-audit-summary.json) checked 72 explanations on 12 images. The auditor first inspected the images without claims, then received randomized per-image model aliases with scores, identities and references hidden. Its findings concern visible support for explanations, separately from numerical agreement.

| Model | Technical explanations supported / mixed | Aesthetic explanations supported / mixed | Total supported |
|---|---:|---:|---:|
| DeepSeek | 10 / 2 | 11 / 1 | 21/24 |
| Gemini | 10 / 2 | 11 / 1 | 21/24 |
| Luna | 12 / 0 | 11 / 1 | 23/24 |

The remaining seven explanations were mixed; none was classified as wholly unsupported or unavailable. Mixed claims included attributing softness to motion without clear directional smearing, overstating obstruction, and calling hard directional illumination flat or soft. Luna had the strongest descriptive result in this small explanation sample, despite its upward score bias. These counts do not estimate factual accuracy across all 150 image responses.

My provisional assessment is to retain DeepSeek as the inexpensive baseline, with its observed aesthetic agreement and compression failures both documented. Gemini is worth comparing further on damaged images because it identified some major artifacts the other models underweighted. Luna's stronger agreement on the small standalone subset did not translate into a clear overall lead; its upward bias on damaged and difficult images needs attention. No model is promoted to an automatically trusted label source by this trial.

## Contract, settings and cost

Both task objects contain exactly `task`, `status`, `score`, `evidence` and `limitations`. Scores use the same exact string enum, `"1.00"` through `"5.00"` in `"0.10"` steps. This guarantees the permitted representation when validated; it does not imply hundredth-point judgment precision. Valid strings are decoded through the named Decimal adapter without rounding or repair. An invalid task would remain null while a valid sibling could be retained; no such failures occurred.

The existing joint technical/aesthetic rubrics and system prompt were preserved. Only the score transport wording and evidence-length guidance changed: target 12–20 words, retain the 30-word maximum. Observed evidence ranges were 15–25 words for DeepSeek, 13–20 for Gemini and 15–22 for Luna. All 300 score strings were exact enum members. This successful sample does not guarantee future compliance.

| Route | API model | Reasoning | Other settings | Total cost including five probes |
|---|---|---|---|---:|
| RunInfra | `deepseek-v4-1-flash-1017` | none | temperature 0, max_tokens 2048 | $0.02063100 |
| OpenRouter / Google AI Studio | `google/gemini-3.8-flash` | medium | temperature 0, max_tokens 16384 | $0.207468525 |
| OpenRouter / OpenAI | `openai/gpt-6-luna` | high | temperature omitted, max_tokens 16384 | $0.034813050 upstream estimate |

RunInfra's alias identifies the requested `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Its output allowance was increased from the single-task run's 1024 to 2048 for two task objects. Both OpenRouter routes used strict `json_schema`, `require_parameters:true`, one pinned provider, `allow_fallbacks:false`, and excluded returned reasoning text. Reasoning tokens were still used and billed. Luna does not support temperature on this endpoint, so it was omitted. The 16384-token allowance includes reasoning and visible output. All routes used high image detail, non-streaming requests and at most two concurrent calls. No inference request was retried or repaired.

Total direct/gateway charges were **$0.228099525**. Luna used the account's existing BYOK configuration: its OpenRouter gateway charge was zero, while returned upstream inference cost was **$0.034813050**. Adding that estimate gives **$0.262912575**, approximately **26.29 US cents**, for images and probes; image calls alone were $0.256890025. Standard Gemini upstream details were not added twice. These are response-derived amounts, not invoice reconciliation.

The frozen admission ledger uses `usage.cost`; that field alone omits Luna's BYOK upstream cost. A separate usage-only monitor tracked the upstream estimates during execution, and the final [billing summary](billing-summary.json) records the limitation. Observed combined costs stayed below the $3-per-route bound. Before reusing this runner for larger BYOK work, include upstream costs in admission accounting. Read-only generation lookups verified all 110 OpenRouter requests: expected providers, matching gateway costs, and completed generation metadata. The original replies retain reasoning-token counts and usage details.

[Exa research and endpoint snapshots](provider-research.json) support the route choices. Relevant primary documentation: [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs), [provider selection](https://openrouter.ai/docs/guides/routing/provider-selection), [reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens), and [OpenAI's GPT-6 Luna model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).

## Method, limits and saved artifacts

The [frozen protocol](protocol.md) specifies selection and analysis. The sample contains 24 controlled images across six complete families, 14 VizWiz and 12 standalone Megalith images: 50 images in 32 families. Seed 20261027 and family/hash/near-duplicate exclusions prevent reuse of earlier exposed cohorts, including the 200-image trial. All source rows remain in the training partition.

Reviewers and students received identical full-frame RGB/sRGB PNG bytes, maximum edge 1536, with no enlargement or crop. Fourteen views across five families required smaller edges to meet the request limit; controlled siblings share one edge cap. The largest request was 3,475,417 bytes, below RunInfra's 3,670,016-byte limit. Results apply to delivered views, and resizing can hide source defects.

Three fresh assistant review contexts inspected anonymous packets of 17/17/16 images individually at original supplied detail, without provider outputs or source metadata. All 50 paired reference assessments were frozen before image inference. There is one assessment per image and task, no repeated-reviewer reliability estimate, and no human gold labels. Bootstrap intervals use 10,000 draws of whole families within source strata and condition on these fixed judgments. They do not incorporate all reviewer uncertainty. The different requested reasoning settings are part of the compared configurations, not a controlled test of reasoning itself.

This fresh cohort cannot establish improvement over earlier trial averages, and there was no randomized single-task versus joint-task arm. Both constructs remain separate; all observations are pending human review, every training task mask remains false, and `training_ready` remains false.

- [Complete report](complete-report.json): quantitative comparison, explanation audit, billing, generation metadata and posthoc examples.
- [All case/task/model comparisons](comparisons.csv): 300 rows with scores, references, intervals, evidence and errors.
- [Frozen references](references.jsonl), [reference freeze](references-freeze.json), and [manifest](manifest.json).
- [Quantitative results](results.json), [explanation audit](explanation-audit.jsonl), [usage audit](usage-audit.json), and [billing summary](billing-summary.json).
- [Runner](runner.py), [wire contract](wire.py), [analysis](report.py), and [provider settings](routes.json).

Raw replies, request journals, run snapshots, delivered PNGs and anonymous review packets remain under ignored local `visual-quality/data/dual-score-trial-2026-10-27/`. The manifest ID is `c2dbb816c6bcc8c4dea04982a65de3e11c8f24de35bc53bb8bf414396617a245`. Image run IDs are recorded in `results.json` and each route's `images/run.json`.

All 99 focused offline tests and scoped Ruff checks passed before inference. An independent reviewer recomputed score metrics from raw score strings, reproduced every paired bootstrap interval, verified all six frozen runs, and checked source identities, eligibility and disabled training masks. These commands reproduce local quantitative results without inference:

```sh
cd /home/ahstn/git/model-gym/visual-quality
.venv/bin/python -B experiments/dual-score-trial-2026-10-27/prepare.py verify
.venv/bin/python -B experiments/dual-score-trial-2026-10-27/runner.py verify --route deepseek
.venv/bin/python -B experiments/dual-score-trial-2026-10-27/runner.py verify --route gemini
.venv/bin/python -B experiments/dual-score-trial-2026-10-27/runner.py verify --route luna
.venv/bin/python -B experiments/dual-score-trial-2026-10-27/report.py
.venv/bin/python -B experiments/dual-score-trial-2026-10-27/audit.py summarize
.venv/bin/python -B experiments/dual-score-trial-2026-10-27/finalize_report.py
```
