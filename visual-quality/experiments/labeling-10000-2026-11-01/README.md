# 10,000-image photo-labeling rehearsal

Shared execution and presentation code now lives in the [scoring package](../../src/visual_quality/scoring/README.md). Older experiment folders were [archived](../../archives/experiments.md); corpus rebuilding and historical reuse read restored inputs from ignored data. Completed observations, source receipts, report and collage retain their original hashes. This cleanup did not rescore images or rewrite the frozen validation receipt.

This run expands the previous 4,000-image rehearsal to 10,000 delivered views and the assistant-reference benchmark from 500 to 1,500 images. The accepted ensemble is **37.5% Luna + 37.5% DeepSeek + 25% Liquid**, calculated separately for technical quality and aesthetics. Jev and Q-ReAlign Lite remain comparison models. All labels remain pending training review.

The baseline is `9a2f7cef8265690019d41b0158a558dcfd8a0d6d`, including the requested decision-question changes. The new profile adapts `visual-rating-v2` to the established two-outcome photography contract. All four API models receive fresh calls because the technical rubric changed; compatible Q-ReAlign Lite results for the original 4,000 views are reused.

The corpus, all 1,500 reference assessments and inference are complete. The [agreement report](report.md) records coverage, latency, scope errors and costs; its total-cost-per-model table is last. The [72-image collage](collage/README.md) is restricted to the committed example set. The [full local gallery](../../data/labeling-10000-2026-11-01/gallery/index.html) contains all 10,000 images, separate statistics and related-image links.

Luna, Liquid and Lite each have 10,000 numerical score pairs; DeepSeek has 9,999 and Jev has 9,997. The four missing assessments are one valid-unscorable DeepSeek response and three persistent Jev image-processing failures. Their nulls and every attempt remain recorded. The ensemble has 9,999 complete pairs, with no missing-component fallback. Both owned Runpod workers are stopped.

On the fresh 500-image confirmation set, ensemble technical/aesthetic MAE is **0.363 / 0.275**, against Luna's **0.358 / 0.337**. The blend improves aesthetic agreement in this sample while technical agreement remains close to Luna. Lite's corresponding errors are **0.395 / 1.064**, with substantial aesthetic underprediction. These are comparisons with assistant judgments, not human-validated accuracy or statistically established population rankings. Weights and prompts were not fitted to this confirmation set.

## Where the data lives

Full data is local and ignored by Git:

```text
/home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01/
  manifest.json                  # 10,000 exact views and source/family provenance
  benchmark-manifest.json        # 1,500 references in explicit benchmark order
  references-all.jsonl           # frozen assistant assessments
  references-freeze.json         # aggregate hash, packet receipts and freeze time
  stats.jsonl                    # exact-view statistics, separate from responses
  content-coverage.json          # 64 content clusters and source concentrations
  related-images.json            # neighbours and embedding-view provenance
  corpus-embeddings.npz           # retrieval vectors
  results/api/<model>/           # plans, every attempt and terminal observations
  results/lite/                  # compatible reuse and all new request attempts
  exports/                      # full joined scores and labelled-image records
  publication-staging/           # bounded candidate export, validated before copying
```

Only the first **1,000 benchmark images by `benchmark_order`** enter the active committed [labeling records](../../labeling/records/). Their 7,000 slots comprise five model rows, the ensemble and the reference for each image. This is an image allowlist, not the first 1,000 score rows or lexicographic case IDs. The last 500 benchmark references and every nonbenchmark score remain outside Git, including score-bearing logs, HTML and collages. Aggregate reports may cover the full run.

The corpus has 7,500 Megalith views, 1,500 VizWiz views, 750 controlled derivatives and 250 controls, with 9,250 explicit families. See [corpus-method.md](corpus-method.md) for source terms, sampling, actual EmbeddingGemma 2 inference, the bounded duplicate audit and remaining uncertainty about capture sessions. Similarity supports content organization and review; it does not transfer quality labels between neighbours.

## Scoring and reference contract

| Assessor | Interface and configuration | Reuse |
|---|---|---|
| GPT-6 Luna | OpenRouter, `openai/gpt-6-luna`, high reasoning requested | Fresh 10,000 |
| DeepSeek | RunInfra, `deepseek-ai/DeepSeek-V4.1-Flash-1017`, high reasoning requested | Fresh 10,000 |
| Liquid d1.1 | OpenRouter Decisions, revised model-specific exclusion question | Fresh 10,000 |
| Jev 1.14 | OpenRouter Decisions, revised model-specific exclusion question | Fresh 10,000 |
| Q-ReAlign Lite | Pinned `q-future/Q-ReAlign-Lite-4B`, unchanged two-head scoring server | Reuse 4,000; request 6,000 |
| Reference | Independent assistant visual review | Reuse 500; review 1,000 |

The teachers return validated schema-constrained JSON. Decisions scores preserve native expected-value precision; displays use two decimal places. Q-ReAlign Lite uses its native probability-to-score mapping, not the chat JSON schema. Clearly excluded flat nonphotographic material receives 1.00 on both outcomes in the teacher/decision contract. Lite retains its unchanged native result so the comparison does not conceal its scope limitations.

The original 500 references are a historically exposed cohort. The next 500 are fresh committed examples, and the final 500 are fresh confirmation cases. New reviewers inspect the exact delivered images without model scores, statistical measurements or source quality votes. Each packet is frozen before those images are admitted to new inference. Reports keep the three cohorts separate; reference agreement is not human-validated accuracy. No weights or prompt wording are fitted to the new confirmation results.

Reference records retain actual review windows and their timing provenance. These windows can overlap within a viewed batch and include operational interruptions, so their sum is not a measure of exclusive human attention. Reference HTTP latency is null because there is no HTTP scoring request. Every inference row records the last request's end-to-end latency; total retry time and server-only time are separate. Q-ReAlign Lite and reference prices are $0.00 by the user's reporting rule. Actual Runpod infrastructure is accounted for separately. Unknown API charges remain null.

Each labelled image carries `sharpness_peak`, `luma_entropy`, `focus_area`, `histogram_entropy`, `channel_clip` and Hasler-Suesstrunk `colourfulness` from the local `image-stats` package. Histogram entropy is an explicit alias of luma entropy. Focus area is relative to that image's own peak detail and is not an absolute in-focus fraction. Statistics are debugging metadata outside the inference response, with training-feature and training-ready flags disabled.

## Reproduce exports and check publication

For offline review, install the explicit corpus dependency group rather than borrowing packages from an embedding worker's environment:

```sh
cd visual-quality
uv sync --locked --group dev --group corpus
uv run --locked --group corpus pytest -q tests experiments/labeling-10000-2026-11-01
uv run --locked --group corpus python labeling/verify.py --images
```

The corpus group supplies scikit-learn for clustering and the corpus tests. It does not install the GPU embedding or Lite model environments. The pinned historical inference environments remain described in the runbooks; the current lockfile is for maintenance and offline checks.

Run from the repository root using the project environment. The [API runbook](api-runbook.md) and [Lite runbook](lite-runbook.md) describe frozen requests, bounded retries, cost accounting and the operational amendments. Do not restart scoring to reproduce an export: the recorded responses are its inputs.

```sh
PYTHONPATH=visual-quality/src \
  /home/ahstn/git/model-gym/visual-quality/.venv/bin/python \
  visual-quality/experiments/labeling-10000-2026-11-01/collect_references.py
```

`publish_run.py` takes `--manifest`, `--references`, `--stats`, `--api-root`, `--lite-reused` and one `--lite-run` for every executed new Lite phase. It verifies delivered-view joins and reference chronology, preserves failed attempts, derives the shared ensemble, and writes full local exports. A missing ensemble component leaves that outcome null; there is no fallback to the retired two-model mean.

The default export does not publish. `--publish` writes the bounded staging directory only after all references and statistics are present. Any final missing model outcomes require an explicit documented `--allow-incomplete-models REASON`; that flag never supplies invented scores. Review and validate staging before replacing active records. The validator rejects extra files, symlinks, unlisted nested image IDs, incorrect hashes, inconsistent score formats and views outside the ordered allowlist.

```sh
cd visual-quality
PYTHONPATH=src /home/ahstn/git/model-gym/visual-quality/.venv/bin/python \
  labeling/verify.py --images
```

The [directory rules](../../AGENTS.md) preserve the accepted ensemble, photographic scope, statistics, publication and accounting decisions for subsequent work. Historical raw responses retain their provenance; current result views use the accepted ensemble without presenting old calls as fresh results.

The aggregate [validation receipt](validation.json) records the independent data audit, request and response hash checks, all 10,000 image joins, the 1,000-image publication boundary, software tests and 188 browser assertions. All 10,000 gallery previews loaded successfully. The original browser virtual-time exit stalled; explicit completion polling now preserves the same assertions with bounded process cleanup.

`gallery_run.py` rebuilds the full ignored gallery and its frozen first-1,000-only collage selection; see the [gallery runbook](gallery-runbook.md). `publish_collage.py --gallery DATA/gallery --records DATA/publication-staging --destination EXPERIMENT/collage` publishes only the four validated PNGs and bounded provenance after checking final coverage, the publication allowlist and exact selected values. It rejects unfinished exports and never copies the full gallery into Git.
