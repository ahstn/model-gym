# Visual triage

**The metadata trial is complete. C2PA helped where it was present, but covered only a quarter of the AI test images.** All 3,480 native and control requests succeeded. Each feature group used 20 GEPA proposals before evaluation on 160 held-out images.

| Selected prompt | Accuracy | AI detected | Brier loss | Cost / 160 images | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image-only | 76.9% | 43/80 | 0.1595 | $0.006410 | 0.738 s |
| All context | 78.1% | 45/80 | 0.1418 | $0.012371 | 0.738 s |

All context combines EXIF, C2PA and pixel measurements. It nearly doubled cost for two extra detections; both prompts had zero false positives among 80 real photos. Keep image-only as the cost baseline. The remaining 35 missed AI images and source-specific metadata coverage limit automatic rejection.

Only 20 of 80 AI images had signed AI claims that passed signature and image-binding checks, all from Firefly. The 60 DALL-E 2, DALL-E 3 and Midjourney v5 files had no C2PA manifest. The fixed trust check marked all 20 Firefly claims untrusted. A valid signature and a trusted signer are separate checks; neither proves that the scene is true.

In separate controls on those Firefly images, removing C2PA reduced detections from 15/20 to 3/20 for the provenance prompt, and from 10/20 to 4/20 for all context. This supports C2PA as useful evidence when present. Missing credentials do not prove camera origin. Camera-value swaps also increased probability error; controls remain separate from native results.

- [Native metadata trial](tracks/metadata_v3/report.md): seven feature groups, GEPA, EXIF/C2PA controls, costs and latency; [per-image results](tracks/metadata_v3/results/cases.csv) and [request audit](tracks/metadata_v3/results/request-audit.json).
- [Extended validation](EXTENDED-VALIDATION.md): earlier quality and AI results, pixel-statistic tests and next steps.
- [Dataset and method research](research/expansion-2026-10-11/README.md): new sources, the Awesome IQA catalogue, licences and research leads.
- [Separate tracks](tracks/README.md): datasets, objectives, saved evidence and run commands.
- [Earlier validation](VALIDATION.md): Decider quality and Luna AI detection with three GEPA proposals.
- [Initial benchmark report](BENCHMARKS.md): Decider, Gemini and Luna on the earlier two 60-image panels.
- [Initial per-image results](summaries/initial_benchmark.csv) and [image assessments](evidence/assessments/): earlier predictions and assistant judgements, which are not human ground truth.
- [Initial report](REPORT.md): the original 30-image screen and limited sexual-content checks.
- [Evidence archive and restore](ARCHIVE.md): checksummed raw responses, full reports and source captures; [compact results](summaries/) stay in Git.

The earlier quality trial used human KonIQ ratings. On 100 new images, GEPA reduced mean score error from 0.568 to 0.443 and raised accuracy from 76% to 84%. Its separate AI track used SynthCLIC and Synthbuster photos and still missed most generated Imagen 3 images. These panels cannot rank current Decider prompts against the earlier Luna run. Human aesthetic ratings and confirmed NSFW-positive validation remain absent.

## Reproduce the initial comparison

Run from `visual-triage` with Python 3.13 or later. Load `OPENROUTER_API_KEY` through your existing shell environment; do not put it in commands or files. Image bytes remain in ignored `data/`; Git retains manifests, prompts and compact results. Restore raw evidence before rebuilding historical reports or resuming a run:

```sh
python3 evidence_archive.py restore
python3 evidence_archive.py verify
```

Prepare the original 30 images first: both builders use them for duplicate checks. `--pilot-root` must contain `pilot-10k-v1/train.jsonl` and its images. The original cache is `/Users/ahstn/git/model-gym/visual-quality/data`; managed worktrees do not copy ignored data. Use the [pilot builder](../visual-quality/docs/datasets/pilot-10k-assembly.md) if that cache is missing.

```sh
uv sync --locked --group dev
uv run --locked python prepare.py --pilot-root ../visual-quality/data
uv run --locked python benchmark_data/ai_builder.py
uv run --locked python benchmark_data/quality_builder.py --allow-expired-source-certificate
uv run --locked python benchmark_data/ai_builder.py --verify
uv run --locked python -m pytest tests -q
```

The quality source had an expired TLS certificate. The explicit option permits that public request only; verification remains the default. Archive and image hashes detect changes. Read the source note before using this exception.

Rebuild reports from saved responses without paid calls:

```sh
uv run --locked python bench_analyze.py \
  --ledgers evidence/final/quality.jsonl evidence/final/ai.jsonl \
  --references evidence/assessments/quality-60.json evidence/assessments/ai-60.json \
  --manifest evidence/benchmarks/quality.json evidence/benchmarks/ai.json \
  --output evidence/benchmarks/summary.json
uv run --locked python bench_report.py \
  --summary evidence/benchmarks/summary.json \
  --references evidence/assessments/quality-60.json evidence/assessments/ai-60.json \
  --manifest evidence/benchmarks/quality.json evidence/benchmarks/ai.json \
  --output evidence/benchmarks/per-image.md
```

For a new paid run, preserve existing evidence and choose a fresh output path. The selected chat setting is `low`, with a 4,096-token cap. To test AI origin, replace `quality` with `ai` in the command:

```sh
uv run --locked python bench_runner.py \
  --manifest evidence/benchmarks/quality.json --prompt prompts/final/quality.json \
  --task quality --models decider gemini luna --effort low \
  --max-tokens 4096 --max-calls 180 --budget-usd 2 \
  --output evidence/new-quality-run.jsonl
```

The runner checks image hashes, excludes labels from requests, records each attempt and makes no automatic retries. Unknown costs stop spending. The budget is a local guard, not an account limit. `final_benchmark.py --task quality` or `--task ai` resumes the saved run with rotating model order; it skips completed requests.

Luna used BYOK: its OpenRouter charge was zero, but its reported upstream inference cost was nonzero. Reports retain both fields and do not claim invoice reconciliation. Decider scores map from 0–4 to 1–5; its selected rubric anchor is not a generated visual explanation.
