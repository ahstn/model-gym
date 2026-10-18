# Jev 1.14 repeat benchmark, 18 October 2026

This experiment repeats the public image benchmarks used in the earlier visual-decision runs. It keeps their image bytes, prompts and score rules fixed. Read the [results](results/README.md) for comparisons with Perplexity Decider v1.1 and GPT-6 Luna.

All 378 calls returned valid results at a reported cost of **$0.03072657**. On 100 KonIQ images, Jev reduced human quality mean absolute error from Decider's 0.443 to 0.238, with similar rank correlation. It detected 42 of 50 poor images and falsely flagged 2 of 50 other images; Decider detected 34 and falsely flagged none. Jev is a useful candidate for technical-quality labels, but this run does not validate aesthetic labels or justify replacing the planned teacher selection test.

On the smaller shared set, Decider had the lowest human quality error: 0.294 versus Jev's 0.325 and Luna's 0.354. Jev had the highest AI-origin accuracy, 95.0% versus 76.7% and 83.3%, but falsely flagged three real photos. On the larger AI-origin set, Jev reached 88.1% versus Decider's 76.9%, with three real false flags versus none. Their ranking scores were nearly equal. At the existing accept/review/reject gates, Jev sent 133 of 160 images to review, versus Decider's 47. These results support further evaluation as an advisory scorer; a higher accuracy at one threshold does not establish better automatic filtering.

The shared panel contains 58 UHD-IQA quality images and 60 OpenFake AI-origin images. The larger panel contains 100 KonIQ quality images and 160 VISION/Synthbuster AI-origin images. Luna has results only on the shared panel. The two panels use different prompts and must remain separate.

## Evidence

- [Protocol](protocol.json): model revision, budget, source and execution hashes.
- [Manifest](manifest.json): all 378 cases, image hashes and prompt paths.
- [Payload audit](payload-audit.json): public dataset sources and proof that every request matches a prior OpenRouter request.
- [Results](results/README.md), [summary](results/summary.json) and [per-image data](results/cases.csv).
- [Confidence audit](confidence-analysis.md): returned uncertainty fields, the three real-photo false flags, and observed filtering tradeoffs.

The new raw responses and catalogue snapshot are stored in ignored `visual-quality/data/jev14-2026-10-18/`. The original images, prompts, baselines and Python modules remain in `/Users/ahstn/.codex/worktrees/d187/model-gym/visual-triage`. The runner reads that worktree without changing it. Its pinned files are required to verify or rebuild this experiment.

## Reproduce

Run these commands from the repository root. The existing visual-triage Python environment supplies the dependencies. A completed run sends no further requests. The runner stops on a failed request, unknown cost, changed model revision or altered input. It does not retry automatically.

```sh
source ~/.config/zsh/.zsh-private
PYTHONDONTWRITEBYTECODE=1 /Users/ahstn/.codex/worktrees/d187/model-gym/visual-triage/.venv/bin/python visual-quality/experiments/jev14-2026-10-18/run.py verify
PYTHONDONTWRITEBYTECODE=1 /Users/ahstn/.codex/worktrees/d187/model-gym/visual-triage/.venv/bin/python visual-quality/experiments/jev14-2026-10-18/run.py run
PYTHONDONTWRITEBYTECODE=1 /Users/ahstn/.codex/worktrees/d187/model-gym/visual-triage/.venv/bin/python visual-quality/experiments/jev14-2026-10-18/report.py --output /private/tmp/jev14-report-rebuild
```

Report generation is offline and requires a new output directory. The private shell configuration is needed only for inference. The key is used in the authentication header and is not stored in the evidence.

The API catalogue checked on 18 October lists `typesafe/jev-1.14-20261014`, image input, a 262,144-token context and $0.042 per million input tokens. The run uses reported costs, including upstream BYOK charges where applicable. The catalogue also lists Decider v1.2 and a separate Luna Decisions model; neither replaces the historical comparators in this experiment. See the [OpenRouter image request contract](https://openrouter.ai/docs/guides/community/multimodal-decisions).
