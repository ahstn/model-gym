# Jev 1.14 native image tests

The main comparison uses the latest **100 KonIQ quality images and 160 AI-detection images** selected by the user. Jev receives the same image bytes and selected prompts as Decider. No camera or provenance metadata is added. Read the [latest results](latest-results-2026-10-15/README.md) and per-image outcomes.

The [earlier shared panels](shared-results-2026-10-15/README.md) are separate evidence. That run had started while the set selection was pending. It compares 58 unchanged quality images and 60 AI images with Decider, Luna and Gemini. It also contains six development size probes. Two narrow panoramas needed separate padding tests: the first returned HTTP 400, and the second was not attempted. Its charge remains unknown.

## Inputs and cost

Requests use `typesafe/jev-1.14` through OpenRouter's Decisions API, with both experimental headers set to `true`. The expected build is `typesafe/jev-1.14-20261014`. Image requests use a 262,144-token context window. Reported input tokens include the prompt and questions, so they are not an isolated image-token count. [OpenRouter guide](https://openrouter.ai/docs/guides/community/multimodal-decisions)

The latest quality views are 1,024×768 JPEGs. AI views are 512×512 JPEG crops. All 260 pass the published size limits. Costs come from API usage fields. The earlier unpriced request has a $0.012 budget reserve; this is not a measured charge.

The provider returned probabilities rounded to two decimals. The [parser amendment](parser-amendment.json) preserves the first 31 shared-run records and defines a bound for rounding error. It does not change or renormalise native scores. The latest study uses that parser from the start. Its [protocol](latest/protocol.json) fixes the images, prompts, code, cost guards and request order before calls.

## Reproduce

Run from `visual-triage` with the existing Python environment. Load `OPENROUTER_API_KEY` through the private shell environment, without placing the key in commands or files. A completed run makes no further calls:

```sh
python tracks/jev14/latest.py run --max-new 260
python tracks/jev14/latest_report.py --output data/jev14-latest-v1/report-rebuild
python -m pytest tests/test_jev14_*.py -q
```

Reports use fresh output directories. Raw responses, source captures and derived images stay in ignored `data/`. Earlier raw baselines and image caches must be available to verify the frozen inputs. The reports compare historical calls made on different dates; latency is not an isolated serving test. The transferred prompts were selected for Decider, with no separate Jev optimisation.
