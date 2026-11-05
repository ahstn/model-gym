# GEPA prompt optimisation

The v2 pass completed 20 GEPA proposals per track. It used Decider v1.1 for both quality and AI detection, with `gepa==0.1.4`. The [extended validation](../EXTENDED-VALIDATION.md) reports final and metadata results. The completed three-proposal runs remain available in the [v1 validation report](../VALIDATION.md), [quality evidence](quality/README.md) and [AI evidence](ai_detection/README.md).

## Framework choice

Standalone GEPA retains the existing transport, validation and request ledger, so this pass needs no framework migration. The earlier assessment also reviewed [DSPy 3.4.0](https://github.com/stanfordnlp/dspy/releases/tag/3.4.0) for future pipelines with several model stages. [ReAnchor](https://dspy.ai/3.4.0/api/experimental/ReAnchor/) could provide a calibration baseline from cached predictions; it was not run here. DSPy's [TypeSafe decision API](https://dspy.ai/3.4.0/api/experimental/DecisionTypes/) was experimental at that review. Its OpenRouter Decider connection was not verified. This pass therefore uses the pinned GEPA adapter and its measured outcomes.

## Search and selection

[optimise.py](optimise.py) evaluates complete source groups with the existing request ledger. The quality objective is `1 - absolute_error / 4` against native human MOS. The AI objective is `1 - Brier_error` against provenance labels. Selection uses the mean group reward, so related images do not receive extra weight. Invalid or missing results receive zero reward.

Each v2 run attempts 20 proposals. Version 2 supports 10–40 proposals; the current protocol fixes the limit at 20. Changing it requires a new protocol and run. Quality can change prompt context and technical-quality rubric descriptions. Question IDs, output types, ascending 1–5 anchor labels and the full aesthetics question remain fixed. Aesthetics has no human target and does not affect selection. AI can change context only; its probability continues to mean generated origin.

Luna proposes edits with `low` reasoning and a 4,096-token output cap. It receives training feedback and at most four training images from complete groups with the largest errors in the current batch. Quality batches contain eight images; AI batches contain eight matched pairs. GEPA shuffles training groups by epoch. The full development panel stays fixed throughout the search. Development, final and external images cannot reach reflection. Sample IDs, source names and label-specific phrases must not become lookup rules.

An `accepted` reflection proposal passed syntax and task validation. This term does not mean that GEPA retained it, that development reward improved, or that it reached final selection. A malformed response with known cost is recorded as `rejected`; that attempt keeps the prior candidate and counts towards the 20-attempt limit. Resume reuses the saved response. HTTP failures and unknown costs stop further requests.

Keep the baseline unless the best development reward improves by at least 0.005. Freeze that choice before final calls. The original and selected prompts then use the same final images. If selection retains the original, it is evaluated once. External AI images use the same frozen choice. Neither final panel can select another prompt or adjust a threshold. Repeated development comparisons can still cause overfitting; the selection margin does not establish statistical significance. Data, rubric freedom and reflection inputs also changed from v1, so this is not a controlled comparison of three versus 20 proposals.

## Cost and final diagnostics

The combined limit is $5 per track, including task evaluation, Luna reflection, primary final calls, external calls and metadata diagnostics. The search command below allows $4.50 within that limit. Guards use reported costs, including separately reported upstream BYOK charges. They reserve local headroom and do not set provider account limits. Unknown costs require reconciliation before resume. One exact quality training failure now has a reviewed $0.10 reservation policy, described in the [recovery note](../research/expansion-2026-10-11/cost-recovery.md). It preserves the original failed row and null actual cost, permits no retry, and counts the reservation only towards the budget guard. Other unknown costs still stop the run. Reports show known charges, the unverified reservation and the guard-accounted total separately.

Primary search uses images only. Before any final call, [metadata_ablation.py](metadata_ablation.py) freezes pixel statistics and their hashes from the delivered images. The diagnostic adds those statistics to the selected prompt with a fixed context instruction. It uses the same final views and permits no retuning or new selection. There is no camera or EXIF arm. Report primary and diagnostic costs separately; both count towards the track limit.

## Reproduce or resume

From `visual-triage`, install the pinned dependencies and load `OPENROUTER_API_KEY` through the existing environment. After dataset acquisition is complete, these freeze commands apply to a fresh track. Existing protocols, context plans and evidence cannot be replaced.

```sh
uv sync --locked --group dev --group optimisation
track_name=quality_v2 # Use ai_detection_v2 for the AI track.
uv run --locked python tracks/run.py freeze --track "$track_name" --baseline "tracks/$track_name/prompts/baseline.json" --max-proposals 20
uv run --locked python tracks/metadata_ablation.py freeze --track "$track_name"
uv run --locked python tracks/optimise.py --track "$track_name" --protocol "tracks/$track_name/protocol.json" --dataset "tracks/$track_name/dataset.json" --prompt "tracks/$track_name/prompts/baseline.json" --output "tracks/$track_name/optimisation" --max-calls 4000 --budget-usd 4.5 --max-proposals 20
```

To resume search, repeat the last command with `--resume`. Keep all arguments unchanged. Candidate inputs, proposal feedback and frozen hashes must match the saved run.

After search completes, use `selected_prompt_path`, `baseline_dev_ledger_path` and `selected_dev_ledger_path` from `optimisation/summary.json` in the selection command. Replace the three quoted placeholders below with those values.

```sh
uv run --locked python tracks/run.py select --protocol "tracks/$track_name/protocol.json" --candidate "SELECTED_PROMPT_PATH" --baseline-ledger "BASELINE_DEV_LEDGER_PATH" --candidate-ledger "SELECTED_DEV_LEDGER_PATH"
uv run --locked python tracks/run.py final --track "$track_name"
uv run --locked python tracks/metadata_ablation.py run --track "$track_name"
uv run --locked python tracks/report.py --track "$track_name"
uv run --locked python tracks/metadata_ablation.py report --track "$track_name"
```

Primary final and metadata runs skip recorded attempts when resumed. Reports read saved ledgers without API calls. Metadata reports are written once; existing report evidence cannot be replaced.
