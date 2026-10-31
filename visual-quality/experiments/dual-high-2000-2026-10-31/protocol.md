# Frozen two-model, high-reasoning, 2,000-image trial

This expands the 72-image photography-scope experiment using DeepSeek V4.1
Flash 1017 through RunInfra and GPT-6 Luna through OpenRouter. Both requests
explicitly select **high reasoning**. The paid image count is 2,000 per model,
4,000 total. Probe images are members of those 2,000 and are never added again.

The exact photo-scope prompt, two-task schema, 1.00–5.00 string score enum in
0.10 steps, and valid-sibling parsing remain unchanged. Clearly excluded
nonphotographic content receives 1.00 for both tasks with `unsupported_domain`;
real photographs of signs, labels, art in context, and physical printed objects
remain eligible. Missing or unreadable inputs retain the unscorable/null path.
The stored prompt and scope files must be byte-identical to the previous trial.
No prompt tuning or post-hoc model score replacement takes place in this run.

## Cohort and reference subset

The preparation manifest and all delivered PNG view hashes are frozen before
inference. The selected cohort contains 1,999 fresh images plus the known P048
poster anchor, relabeled T2000. The other cases use T0001–T1999. The preparation
artifacts document strata, source identities, source families, view transforms,
historical exposure exclusions, and the fixed 72-case collage subset.

Only the selected 72 cases receive independent assistant reference judgments.
Their image selection is fixed without student model outputs. References may
be frozen alongside inference and therefore have their own freeze record;
reference files that do not exist at inference freeze are not retroactively
added to immutable run metadata. The references are assistant judgments, not
human gold labels. References, source captions, source quality labels, and
corpus metadata are never included in model messages. For all 2,000 images,
validity, score distributions, student agreement, photo-scope flags, token use,
and cost can be reported. Reference-distance claims apply only to the 72-case
subset. The known poster anchor must be distinguished from fresh evaluation.

## Provider settings and spending limits

DeepSeek uses the exact settings from `deepseek-high-trial-2026-10-31`:
RunInfra alias `deepseek-v4-1-flash-1017`, `reasoning_effort: high`, temperature
0, nonstreaming, and max_tokens 16,384. Luna uses the exact photo-scope trial
settings: `openai/gpt-6-luna`, reasoning effort high, temperature omitted,
nonstreaming, and max_tokens 16,384. OpenRouter is restricted to the OpenAI
provider, requires parameter support, disables fallbacks, and retains the
existing conservative price ceilings. The live catalog snapshots are pinned
before calls. Returned provider/model identities must match their allowlists.
An explicitly returned RunInfra reasoning effort other than high blocks that
route. Raw headers and reported reasoning-token usage are preserved.

Each route has a **$10 maximum**; the independent caps bound this trial to
**$20 total**. Before submitting each call, the runner reserves the price of
the full configured context plus the maximum output allowance. It subtracts
the reserve for every other active request and debits observed charges as
replies arrive. At most **16 requests per route** are active. Published and
configured price ceilings produce a $0.1671168 DeepSeek reserve and a $0.222288
Luna reserve per request. These are conservative admission limits, not
expected per-image costs. A response priced above its reserve blocks further
admissions. Unknown charges, transport failures, interrupted requests, missing
or unexpected provider/model identities, and incomplete envelopes also block
the affected route. Already admitted calls are allowed to finish and are
included in billing.

## Durability and continuation

Before every request, a unique attempt directory and request metadata are
written and fsynced. Each raw HTTP result is then written and fsynced. API keys
are read only from the specified environment variables and are redacted from
persisted replies/headers. There are no automatic retries or fallback models.
An existing attempt directory is never resent, even if it has no result.

One process lock prevents overlapping execution for a route. Models can run
in parallel. Rather than rereading all prior replies after every small batch,
the runner reads the durable journal once at startup, interprets each newly
finished request once, and checkpoints aggregate exports every 128 completions
and at exit. Progress is emitted every 16 completions. In-flight failures stop
new admissions; they do not cancel or obscure calls that were already sent.
On restart the complete raw journal is authoritative. Explicit verification
rebuilds all observations and metrics from raw replies and compares the saved
exports exactly. Missing or stale derived exports can be repaired by invoking
`run`; it reconstructs the journal and never repeats existing attempts.

The runner, tests, protocol, routes, prompt, scope policy, available provider
catalog evidence, corpus preparation inputs, fixed selection, review packets,
and transitive transport/parser/storage code are hash-pinned. Each delivered
image and complete request payload has its own hash. Frozen prior experiments
are imported through isolated module instances and never modified.

## Commands

Use the repository's `visual-quality/.venv/bin/python`. The shell must have
`RUN_INFRA_API_KEY` and `OPENROUTER_API_KEY` exported for paid calls. Never print
these values.

```sh
visual-quality/.venv/bin/python visual-quality/experiments/dual-high-2000-2026-10-31/runner.py freeze --route all
visual-quality/.venv/bin/python visual-quality/experiments/dual-high-2000-2026-10-31/runner.py probe --route all --case-id T2000
visual-quality/.venv/bin/python visual-quality/experiments/dual-high-2000-2026-10-31/runner.py run --route all
visual-quality/.venv/bin/python visual-quality/experiments/dual-high-2000-2026-10-31/runner.py verify --route all
```

Repeat `--case-id` for a small probe subset if needed. Each selected image still
counts once within its model's 2,000-image run. Inspect the known poster probe's
raw replies, model/provider identifiers, reasoning headers and usage, schema
validity, and pricing before beginning the bulk run. Do not regenerate any
pinned input after freezing.
