# RunInfra: 200-image string-score trial

Completed on 26 October 2026 with `deepseek-ai/DeepSeek-V4.1-Flash-1017` through RunInfra. All 200 images returned. Every score used the required two-decimal string representation; 188 replies passed the full label contract. The other 12 contained 31–37 evidence words, exceeding the 30-word limit. They remain rejected, with null accepted scores and intact raw replies.

The larger sample supports the earlier positive impression of this route, with an average absolute difference of 0.46 points from frozen assistant judgments. Agreement is still uneven: the model tends to score damaged images too generously, and some explanations overlook obvious subject blur or compression. These results support further evaluation, not automatic acceptance as training labels.

The [earlier matched comparison](../runinfra-comparison-2026-10-25/README.md) remains the evidence for improvement over the previous provider run. This trial uses 200 fresh images and makes no new OpenRouter calls. Its differences from historical averages are unpaired and cannot establish a further improvement or a causal provider effect. Assistant references are not human ground truth.

## Results

Scores use the 1–5 scale in 0.1 steps. Formatting them as strings such as `"3.90"` does not introduce hundredth-point scoring precision.

| Measure | Result |
|---|---:|
| Image requests completed | 200/200 |
| Exact permitted score strings | 200/200 |
| Full label-contract acceptance | 188/200, 94.00% |
| Conformance probes | 5/5 |
| Mean absolute error from assistant point score | 0.46; 95% interval 0.41–0.51 |
| Mean distance outside assistant interval | 0.24; 95% interval 0.20–0.29 |
| Mean score bias, model minus assistant | +0.19; 95% interval +0.12–+0.25 |
| Scores within assistant interval | 76/188, 40.43% |
| Errors at least one point outside assistant interval | 10/188, 5.32% |
| Predefined critical contradictions | 0/188 |
| Total billed cost, including probes | 3.57 US cents |

The score measures use the 188 strictly accepted replies. Rejected replies are retained in the 200-row export but do not contribute scores to these measures. All costs were known: image calls cost $0.03555504, probes cost $0.00015675, and total spend was $0.03571179. No requests were retried or repaired.

Uncertainty intervals use 10,000 bootstrap draws of whole families within source strata. They describe sampling variability conditional on the fixed assistant judgments and accepted subset; they omit reviewer uncertainty and do not measure human accuracy. The zero-event critical-error bootstrap interval is not evidence of zero risk. The critical definition is narrow: a score at least 4 with reference upper bound at most 2, or the reverse. Large misses such as T112 still matter.

| Source stratum | Accepted/planned | Point-score MAE | Mean interval distance | Mean bias |
|---|---:|---:|---:|---:|
| Controlled parents and variants | 93/96 | 0.47 | 0.25 | +0.23 |
| VizWiz | 52/56 | 0.53 | 0.30 | +0.34 |
| Standalone Megalith | 43/48 | 0.37 | 0.17 | −0.09 |

The controlled source parents have mean bias −0.08, while degraded variants have bias +0.33. This reinforces the concern about leniency toward damage. Scores are nonincreasing on 63/67 evaluable adjacent degradation pairs, including 15 ties; four pairs reverse and five are unavailable because of rejected replies. Transformation severity is a diagnostic, not a human perceptual label. Restricting analysis to complete families leaves 179 comparisons and gives MAE 0.46 and interval distance 0.24.

## What the image checks found

The preselected explanation audit inspected 24 images without scores or assistant references: 22 explanations were supported and two were mixed. T047 understated pervasive noise; T178 overstated exposure quality despite washed-out petals. Two audited replies, T083 and T178, were among the rejected labels. These audit counts describe the selected sample and do not estimate explanation accuracy across all 200 cases.

A separate review of six large disagreements was selected after scores were available. It provides failure examples rather than a rate estimate; it did not revise any reference:

| Case and delivered view | Model | Assistant interval | Visible issue |
|---|---:|---:|---|
| [T112](../../data/runinfra-string-trial-2026-10-26/views/5cdd27922f5108fc5100cd35050cad774f89798b4a2c45f60348aca05468a8a1.png) | 3.00 | 1.00–1.10 | Almost no recognizable detail; the response treats the clipped, blurred view as an abstract gradient. |
| [T173](../../data/runinfra-string-trial-2026-10-26/views/8a0a7e5986cc5ce451f4842d0f855bd82c60162917d56f31b79624d094c97342.png) | 4.50 | 2.50–3.10 | The flower center and foreground petals are defocused, despite the response calling them sharp. |
| [T084](../../data/runinfra-string-trial-2026-10-26/views/c62d1e97a9e416bcc5ced18fbe3114eacc8ef80f757aa5c87f6422c122a56258.png) | 3.60 | 1.80–2.40 | Heavy block artifacts are omitted from an explanation focused on exposure. |
| [T026](../../data/runinfra-string-trial-2026-10-26/views/63abb24d8c15e0f829fcb4ff4394d99218b173ca8581699e201f35c34fd7cc6d.png) | 4.30 | 2.80–3.30 | The blade and its pattern are broadly soft, despite a claim of sharp subject focus. |

## Saved decisions and method

The [format decision](format-decision.md) records the Exa research, official sources, and reason for using a strict string enum instead of relying on numeric decimal spelling. The [frozen protocol](protocol.md) specifies the rubric, exclusions, request settings, budget, metrics, and audit plan.

The cohort contains 200 fresh training images across 128 families: 24 parents, 72 controlled variants, 56 VizWiz images, and 48 standalone images. Historical families and exact or near duplicates were excluded. Full-frame PNG views start at a maximum edge of 1536; 69 views required a lower edge to fit the request budget. Controlled siblings share one edge cap. The largest final request was 3,515,141 bytes, below the catalog limit of 3,670,016 bytes. Results apply to these delivered views and can miss defects hidden by resizing.

Each image received one assistant reference, across nine separate review contexts. Reviewers inspected the actual PNGs and received no provider results, case-linked source metadata, or other reviewers' judgments. All 200 references were frozen before image requests. One reviewer had previously implemented the analysis code and had seen only aggregate cohort counts and generic schemas; this is disclosed in the reference freeze. There is no inter-rater or human validation in this trial.

The provider used the original visual rubric, with only score-format instructions and the example changed. Settings were temperature zero, reasoning disabled, high image detail, and 1024 output tokens. Five text-only probes passed first. Image requests used four-way concurrency under a $1 admission bound. All valid observations remain pending human review, all task masks remain false, aesthetics is null, and `training_ready` is false.

Keep this string representation for subsequent trials. The next output-contract revision should target shorter evidence to leave room below 30 words, while retaining strict validation. Before expanding label acceptance, review low-quality and obstructed images with humans and test whether the model distinguishes subject blur from harmless background softness. No further inference or label promotion is included here.

## Artifacts and verification

- [Complete report](results/complete-report.json): quantitative results plus the separate explanation audit and rejection word counts.
- [All 200 cases](results/per-image.csv): valid scores, frozen references, errors, costs, and nulls for rejected labels.
- [Frozen quantitative report](results/report.json): output of the analysis implementation pinned before image requests. Its explanation-audit placeholder is superseded only in the complete report.
- [Post-hoc disagreement review](results/disagreement-review.json): six inspected cases, exact explanations, and reviewer notes.
- [Cohort manifest](manifest.json), [references](references.jsonl), and [reference freeze](references-freeze.json): identities, hashes, judgments, and blinding record.
- [Runner](runner.py), [wire adapter](wire.py), [analysis](report.py), and [audit assembly](finalize_report.py): reproducible implementation.

Raw replies, attempts, model snapshot, request hashes, delivered PNGs, and anonymous packets are retained under the ignored local directory `visual-quality/data/runinfra-string-trial-2026-10-26/`. The image run ID is `84ab51fb2ae6bc507b1afe5eea412807144c0bb575cb1cfdb11185197de19648`. The manifest ID is `7192073b04be5bd49258eb0a74987a6f341cfa68555ae680947d05a8fb75958c`.

All 103 focused offline tests passed before inference; the final runner pin-list change also passed its 18 focused tests. Scoped Ruff checks passed. An independent review confirmed headline arithmetic, rejection lengths, and 21 reference-related hashes. The report reconstructs observations from preserved replies and verifies frozen inputs before analysis.

These commands verify and regenerate local results without API calls:

```sh
cd /home/ahstn/git/model-gym/visual-quality
.venv/bin/python -B experiments/runinfra-string-trial-2026-10-26/prepare.py verify
.venv/bin/python -B experiments/runinfra-string-trial-2026-10-26/runner.py verify-probes
.venv/bin/python -B experiments/runinfra-string-trial-2026-10-26/runner.py verify
.venv/bin/python -B experiments/runinfra-string-trial-2026-10-26/report.py --write
.venv/bin/python -B experiments/runinfra-string-trial-2026-10-26/finalize_report.py
```
