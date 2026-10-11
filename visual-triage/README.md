# Visual triage

**The extended Decider trial improved quality scores and AI probability estimates. Neither track is ready for automatic rejection.** On 100 new KonIQ images, the quality prompt reduced mean score error from 0.568 to 0.443 and raised accuracy from 76% to 84%. AI detection still missed most generated Imagen 3 photos.

Both tracks completed 20 GEPA proposals on new, separate datasets. Training supplied image feedback, development selected each prompt, and 260 fresh final images tested the result. A frozen diagnostic then tested the same prompts with pixel statistics. The reports retain per-image predictions, costs, latency and source limits.

- [Extended validation](EXTENDED-VALIDATION.md): current results, metadata tests and next steps.
- [Dataset and method research](research/expansion-2026-10-11/README.md): new sources, the Awesome IQA catalogue, licences and research leads.
- [Separate tracks](tracks/README.md): datasets, objectives, saved evidence and run commands.
- [Earlier validation](VALIDATION.md): Decider quality and Luna AI detection with three GEPA proposals.
- [Initial benchmark report](BENCHMARKS.md): Decider, Gemini and Luna on the earlier two 60-image panels.
- [Initial per-image results](summaries/initial_benchmark.csv) and [image assessments](evidence/assessments/): earlier predictions and assistant judgements, which are not human ground truth.
- [Initial report](REPORT.md): the original 30-image screen and limited sexual-content checks.
- [Evidence archive and restore](ARCHIVE.md): checksummed raw responses, full reports and source captures; [compact results](summaries/) stay in Git.

The current trial uses human KonIQ quality ratings and provenance-labelled SynthCLIC and Synthbuster photos. Earlier datasets and results remain separate. These panels cannot rank the current Decider prompt against the earlier Luna run. Human aesthetic ratings and confirmed NSFW-positive validation remain absent.

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
