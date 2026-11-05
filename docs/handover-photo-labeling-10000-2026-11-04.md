# Resume the 10,000-image photo-labeling work

Copy the prompt below into a new session. Paths are relative to the repository unless stated otherwise. Start by checking the current branch and PR state; this document does not authorize another paid run or a merge by itself.

```text
Continue the photo-labeling work in /home/ahstn/git/model-gym, using the completed
10,000-image experiment as the baseline. Read this entire handover and the linked
runbooks before changing prompts, scoring, publication or execution settings.

Fetch origin and inspect git status. The consolidation branch is
codex/consolidate-model-gym-2026-10-21; PR #7 targets main:
https://github.com/ahstn/model-gym/pull/7
If it has merged, continue from current main on a new codex/ branch. Otherwise
continue the PR branch. Preserve unrelated uncommitted work. At handover, that
includes visual-quality/docs/README.md, the untracked embedding corpus sampling
document and experiments/embeddinggemma2-clustering-2026-10-31/.

Read visual-quality/AGENTS.md, labeling/README.md, labeling/routes.json,
labeling/prompts/{teacher,decision}.json, labeling/ensemble.json, and
experiments/labeling-10000-2026-11-01/{README,report,api-runbook,lite-runbook,
corpus-method,gallery-runbook}.md, all under visual-quality/.

Reuse the exact two-outcome photo-labeling-v2-2026-11-01 profile. The technical
rubric derives from prompts/visual-rating-v2/, adapted to the accepted independent
aesthetic rubric and photographic eligibility policy. Do not substitute the older
technical-only visual-pilot label command or an archived experimental prompt.

The canonical ensemble is 0.375 Luna + 0.375 DeepSeek + 0.25 Liquid independently
for technical and aesthetic quality on the 1–5 scale. Use visual_quality.ensemble,
retain native precision, and round to two decimals only for display. Require all
three valid component scores per outcome; otherwise return null. No reweighting,
two-model fallback, new eligibility vote or reference-score override. Jev and Lite
are comparison models. Keep raw observations; the old 50/50 mean is superseded.

The completed run has 10,000 images and 1,500 assistant references. Reuse existing
results for review and exports. Do not automatically rerun inference, refill valid
unscorable results, regenerate frozen references or tune to confirmation labels.
First report the actual state and continue the specific next task I request.
```

## Exact models and prompts

| Model | Route | Settings to retain |
|---|---|---|
| Luna | OpenRouter chat, `openai/gpt-6-luna` | `reasoning.effort=high`; OpenAI only; fallbacks disabled; 16,384 maximum tokens |
| DeepSeek | RunInfra chat, `deepseek-ai/DeepSeek-V4.1-Flash-1017` | Payload model `deepseek-v4-1-flash-1017`; `reasoning_effort=high`; temperature 0; 16,384 maximum tokens |
| Liquid | OpenRouter Decisions, `liquid/d1.1` | Revised model-specific questions from `9a2f7ce`; preserve typed answer precision |
| Jev | OpenRouter Decisions, `typesafe/jev-1.14` | Revised model-specific questions from `9a2f7ce`; preserve typed answer precision |
| Q-ReAlign Lite | Runpod, `q-future/Q-ReAlign-Lite-4B` | Revision `06add46a6bc4a7dfcf0a59a2fb1ed15a6e3d30b7`; unchanged native quality/aesthetic scoring |

The authoritative endpoints, headers, credential variable names, accepted returned identities and provider constraints are in [routes.json](../visual-quality/labeling/routes.json). Reasoning is **requested** high; do not imply a provider independently proved its internal effort. Liquid and Jev use Decisions, without invented chat reasoning settings. Teacher scores are strict-schema two-decimal strings in 0.10 steps; Decisions expected values map from 0–4 to 1–5 by adding one. Lite uses its native probability mapping and does not receive the teacher schema or photographic gate.

Profile SHA256 values, also recorded in [profile-provenance.json](../visual-quality/labeling/profile-provenance.json):

| File under `visual-quality/labeling/` | SHA256 |
|---|---|
| `prompts/teacher.json` | `2deb592d0a0aca3279d15b289c2486e3a8aa42cf7c1bd4e76ef6a0832f8abf5f` |
| `prompts/decision.json` | `53fe00d8ae6ac874cd5f1825516071c599aa8be9408e00b1e8f16f8744d63b24` |
| `routes.json` | `6173e37e6a6942500f766547cae5228f56870ea17b03179c8be5b64b6c090d5c` |

The pinned GPU server is [lite_server.py](../visual-quality/labeling/runtime/lite_server.py), SHA256 `aaab73e53225093d30e1fbe800425c3b9754d7a5547013f735de8f358a408dbb`. Check the retained health identity before new inference. Historical environment: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Transformers 5.2.0, RTX 4090. Do not import this server during ordinary CPU checks; it initializes the model.

## Data locations and boundaries

The full dataset is local and ignored:

```text
/home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01/
  exports/labelled-images.jsonl  # canonical 10,000 image-level records
  exports/scores-long.jsonl     # 61,500 model/ensemble/reference slots
  exports/scores-long.csv
  exports/summary.json
  manifest.json
  benchmark-manifest.json
  references-all.jsonl
  references-freeze.json
  stats.jsonl
  results/api/                 # frozen plans, all attempts and observations
  results/lite/                # compatible reuse and new attempts
  gallery/index.html
```

The GitHub example dataset is [visual-quality/labeling/records/](../visual-quality/labeling/records/): `labelled-images.jsonl` has 1,000 image records; `scores.jsonl` and `scores.csv` have 7,000 assessor rows. Its manifest, references, provenance and `publication-case-ids.json` travel together. These are the **first 1,000 benchmark images by explicit benchmark/publication order**, not the first 1,000 corpus IDs or score rows. Remaining per-image labels stay out of Git, including the final 500 references, galleries, logs and collages. The committed [72-image collage](../visual-quality/experiments/labeling-10000-2026-11-01/collage/README.md) uses only this allowlist.

Image bytes and raw responses are not bundled with the example labels. Their recorded paths are host-specific; verify local files or restore/reacquire exact bytes with source provenance before reuse. Do not treat an unavailable image as a low score. The full local data directory is not a Git-backed backup.

Coverage: Luna, Liquid and Lite each have 10,000 numerical pairs; DeepSeek has 9,999; Jev has 9,997; the ensemble has 9,999. One valid-unscorable DeepSeek response and three persistent Jev image-processing failures remain null. All 1,500 reference pairs and 7,000 committed slots are numerical. Four API routes were freshly scored; Lite reused 4,000 verified observations and scored 6,000 new images.

References are assistant assessments, not human ground truth. The original 500 are historically exposed; the next 500 are new committed examples; the last 500 are fresh confirmation. Do not fit prompts or weights to the confirmation cohort and continue calling it untouched. All `training_ready` and task masks remain false pending review and explicit training approval.

## Photography, statistics and accounting

- Clearly flat posters/documents, screenshots/interfaces, illustrations, artwork reproductions and obvious renders receive both 1.00 under the teacher/Decisions policy. Photographed packaging, signs or artwork with meaningful physical context remain eligible. Blur and darkness do not establish excluded origin. Keep uncertain scope for review. Decisions uses the evaluated exclusion threshold of 0.50, which is not a calibrated probability guarantee.
- Use local `image-stats/` on exact delivered views. Store `sharpness_peak`, `luma_entropy`, `focus_area`, `histogram_entropy`, `channel_clip` and Hasler-Suesstrunk `colourfulness` separately from inference. Histogram entropy is an explicit luma-entropy alias. Focus area is relative to that image's own sharpest region, not an absolute percentage in focus.
- Statistics remain debugging/review metadata, without prompt injection, automatic label correction or training features. The own/swapped-stats trial did not justify general adoption. Proposed poor/fair wording remains unadopted.
- Every assessment keeps technical/aesthetic scores, HTTP end-to-end latency and price. Null means unavailable. References have null HTTP latency and separately measured review duration. Reference and Lite price is $0.00 by reporting convention; Runpod infrastructure is separate. Ensemble has no independent latency/charge. Preserve failed attempts and unknown charges; do not count component costs again as ensemble cost.
- The completed report records $16.961405582 known API spend and 2,067 unpriced attempts. This is not a complete invoice or a price estimate for the next run. See the report's final per-model cost table for infrastructure and uncertainty.

EmbeddingGemma 2 is for organization and sampling, not transferring quality labels. The run used `google/embeddinggemma-2`, revision `f9972ac22e696a105c482202646979e33e4b5a1e`, 768-dimensional vectors, 280 image soft tokens and FP32. There are 64 clusters; 9,130 final views have exact-view embeddings and 870 use explicitly recorded family representatives. Preserve that distinction, source licenses, view hashes and family-separated partitions. Review similarity/duplicate candidates rather than assuming equal quality.

## Safe continuation and offline checks

Current helpers live in `visual-quality/src/visual_quality/scoring/`. Historical experiment inputs were archived separately. Follow [archives/experiments.md](../visual-quality/archives/experiments.md) for checksummed restoration from local tar or pinned Git history. `VISUAL_QUALITY_HISTORY` points to the restored directory containing experiment folders. Ordinary package imports and current export verification do not require these archives; corpus rebuilding and historical reuse do.

New executions must record new code hashes. Do not force an old frozen plan to resume under extracted or modified code, or rewrite old receipts to imply the current code executed the historical run. Exact historical replay uses its pinned checkout. The November 4 review fixed multiprocessing startup and publication validation; it did not rewrite image statistics, model observations or references.

From the repository root, these checks make no model calls:

```sh
(cd visual-quality && uv sync --locked --group dev --group corpus)
(cd visual-quality && uv run --locked --group corpus pytest -q tests experiments/labeling-10000-2026-11-01)
(cd visual-quality && uv run --locked --group corpus python labeling/verify.py --images)
(cd image-stats && uv sync --locked --group dev)
(cd image-stats && uv run --locked pytest -q)
(cd image-stats && uv run --locked --with ../visual-quality pytest -q tests/test_view_parity.py)
(cd visual-triage && uv sync --locked --group dev)
(cd visual-triage && uv run --locked pytest -q)
visual-quality/.venv/bin/ruff check visual-quality/src visual-quality/tests visual-quality/experiments/labeling-10000-2026-11-01 image-stats visual-triage
visual-quality/.venv/bin/ruff format --check visual-quality/src visual-quality/tests visual-quality/experiments/labeling-10000-2026-11-01 image-stats visual-triage
git diff --check
```

Omit `--images` only on machines without the recorded image files, and report that limitation. Four visual-triage tests use local signed/research fixtures and skip in an artifact-free checkout. The separate decoding parity command supplies both packages. The `corpus` group is a maintenance/test dependency group, not the pinned GPU environment for reproducing historical embeddings.

Both owned trial Runpod workers were stopped at completion. For a newly requested run, prepare code and data first, verify current provider availability/prices and budget, then prefer the Runpod plugin for lifecycle operations. Do not restart old pods automatically. Credentials may be sourced quietly from `~/.config/zsh/.zsh-private`; use the environment names in routes.json and never print or commit secrets. Prior session spend approval covered the completed 10K work, not an unspecified future expansion.

Suggested next work is human adjudication of disagreements, false scope floors and soft/low-light cases, while preserving a random audit sample and the held-out confirmation boundary. Keep the accepted weights unless a separately designed validation study supports a change. No model training is implied by this handover.
