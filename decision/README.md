# decision

This project trains zero-shot decision models. The input is a state, a question, a kind and a list of options.
The output is a calibrated probability distribution over the options. The model does one forward pass and reads
the logits of the option codes only. The first model is `openbmb/MiniCPM5-2B` with a LoRA adapter.

This is a separate uv project, because MiniCPM5 needs `transformers>=5.6.2` and the root project pins 4.57.

## Layout

- `src/decision/schema.py`: the `Decision` record and JSONL input and output.
- `src/decision/prompt.py`: prompt rendering and the option-code readout (`Readout`).
- `src/decision/modeling.py`: model loading, batches, token-budget batching, `code_logits`.
- `src/decision/data.py`: the training pool and the held-out panels. The data mixture is defined here.
- `src/decision/train.py`: LoRA training from a YAML config in `configs/`.
- `src/decision/evaluate.py`, `calibration.py`, `metrics.py`: evaluation and temperature calibration.
- `src/decision/suite.py`: the decision-index suite (extra `suite`).
- `scripts/pod.sh`: RunPod helper.

## Commands

| Command | What it does |
| --- | --- |
| `mise run setup` | Install the dependencies (`uv sync`). |
| `mise run lint` / `mise run fmt` | Check or fix lint and format. |
| `mise run test` | Run the tests. |
| `mise run info` | Show versions, GPU and memory. |
| `mise run check-readout` | Check the option-code tokenization. |
| `mise run data` | Build `data/pool.jsonl`, `data/panels/*.jsonl` and `data/manifest.json`. |

Other commands: `uv run decision train --config configs/X.yaml` and
`uv run decision eval --adapter runs/X/best --panel data/panels/dev.jsonl --calib data/panels/calib.jsonl --out runs/X/eval`.
Add `--model <id>` for another base, and `--trust-remote-code` for models with custom code (for example
`IFM/K2-Horizon-7B`). `scripts/stock-frontier.sh` runs the stock (no adapter) dev evals of the candidate bases.

`uv run decision filter-pool --out data/pool-clean.jsonl --suite-rows data/suite-0.2/selected-rows.jsonl.gz
data/suite-0.2/added-rows.jsonl.gz --exclude sargedev` copies `data/pool.jsonl` without the rows that share a
13-gram with a suite row and without the excluded datasets. It keeps the row order and writes
`<out>.manifest.json`.

Training options to note (`configs/*.yaml`):

- `consistency_weight`: when > 0, every choice row gets a second option order in the same micro-batch. The loss is
  the mean soft cross-entropy of both views plus this weight times the symmetric KL between the two distributions,
  compared per original option.
- LoRA targets only the projections inside the text decoder (`model.get_decoder()`), so the vision tower of
  multimodal bases such as gemma-4-12B-it is not adapted.

## Readout design

1. The prompt lists the options as `A. text`, `B. text`, and so on. It uses the chat template without thinking.
2. The assistant turn starts with `Answer:` and has no trailing space.
3. The model gives the logits for the next token. We read only the rows of the code tokens (`" A"`, `" B"`, ...).
   The tokenizer puts the space into the code token, so `" A"` is one token.
4. `Readout` accepts a code only when it is exactly one new, unique token after `Answer:`. It keeps 255 codes:
   `A`..`Z`, then two-letter codes.
5. `code_logits` runs the backbone only and multiplies the last hidden state by the code rows of `lm_head`.
   It does not compute the full-vocabulary logits. Codes past the number of options get `-inf`. If the model config
   has `final_logit_softcapping` (Gemma), the code logits get the same `tanh` soft-cap.
6. A softmax over the valid codes gives the distribution. A temperature for each kind calibrates it.

## Run on the pod

The pod settings are in `mise.toml` (`POD_*`). The code is in `/root/model-gym/decision` on the container disk.
`/workspace` is slow; use it only for backups.

```sh
mise run pod-sync                      # copy the code
mise run pod-bootstrap                 # install uv and the dependencies
mise run pod-run NAME -- decision train --config configs/X.yaml   # pod.sh adds `uv run`
mise run pod-status                    # GPU and jobs
mise run pod-fetch NAME                # copy runs/NAME back
mise run pod-backup NAME               # copy runs/NAME to /workspace/backup on the pod
```

`pod-sync` deletes files on the pod that are not in the local copy. It keeps only `.venv/`, `data/` and `runs/`.
Put generated files (suite folders, samples) under `data/`.

## Decision Index 0.2.1 suite

You need the `suite` extra (`uv sync --extra suite`). HLE (catalog 45) is gated on Hugging Face. Accept its terms at
<https://huggingface.co/datasets/cais/hle> before you do a full rebuild.

The kit is pinned to 0.2.1 (`87d4650`). 0.2.1 uses the same suite files as 0.2 in `data/suite-0.2`. It only drops
rows at read time (unanswerable ToolRet/BRIGHT queries, duplicate Home appliances rows) and changes the weights. So a
complete 0.2 run is also a complete 0.2.1 run. `suite score --edition 0.2 --out DIR` gives the old 0.2 index.

```sh
uv run decision suite rebuild --suite-dir data/suite-0.2                 # full suite; needs HLE access
uv run decision suite rebuild --skip-hle --suite-dir data/suite-0.2-noHLE   # no HLE; writes PARTIAL.json
uv run decision suite sample --n 500 --out data/sample-500.jsonl.gz --suite-dir data/suite-0.2-noHLE
uv run decision suite run --model openbmb/MiniCPM5-2B --adapter runs/X/best \
    --temperatures runs/X/eval/temperatures.json --suite-dir data/suite-0.2-noHLE --out runs/X/suite-noHLE --shards 6
uv run decision suite score --run runs/X/suite-noHLE
uv run decision suite score --run runs/X/suite-0.2 --out runs/X/suite-0.2/score-0.2.1   # rescore an old run
```

- A suite with `PARTIAL.json` is not an official suite. Its index is not comparable to the public board.
- `--shards N` runs N engine processes on one GPU and then merges and scores their results. Each shard can resume.
  Six shards of the 2B model use about 76 GB on a 96 GB card. Do not start other GPU jobs at the same time: an
  out-of-memory error stops the shard, and you must re-run to resume.
- By default the LoRA is not merged into the base weights. On a 500-request sample a bf16 merge was about 22%
  faster, but it changed 1.5% of the argmax answers (mean TV 0.0098). `--merge` (on `suite run` and on `eval`)
  merges it; then fit the temperatures with `eval --merge` too, so that they match the merged weights.
- The engine encodes the shared prompt prefix once for each request and then runs one short pass for each
  question. Each question sees only the prefix and its own suffix.

## Results (Decision Index 0.2.1)

Our own runs of the public kit on the full suite, not board submissions. The rank is against the live board
(2026-09-27 16:59 UTC, 68 entries plus Jev, 70 with ours). The stock gemma row is an estimate from a paired
15,000-request sample (about ±1 point); the other rows are exact 0.2.1 scores.

| Run | Index 0.2.1 | Knowl. | Lang. | Retr. | Tools | Arts | Rank /70 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock MiniCPM5-2B (R0) | 19.32 | 12.3 | 12.7 | 28.0 | 32.6 | 12.8 | 45 |
| R3 MiniCPM5-2B LoRA | 31.49 | 17.6 | 30.0 | 36.7 | 55.6 | 16.8 | 34 |
| Stock gemma-4-12B-it (estimate) | ≈49.2 | 32.0 | 51.5 | 58.8 | 68.3 | 33.2 | 13 |
| **gemma-4-12B-it LoRA** (`g12-r32-40k-clean-lr5e5`) | **50.44** | 33.4 | 56.2 | 55.8 | 69.3 | 34.4 | **10** |
| Winnow-12B (board, same base) | 50.02 | 33.8 | 56.0 | 54.0 | 71.0 | 30.0 | 11 |

Details, the sample method and the per-benchmark comparison are in
[12B results §5.1](../docs/jev-decision-model-2026-09-26/results-gemma-4-12b.md#51-decision-index-021-current-board-edition).

See [results](../docs/jev-decision-model-2026-09-26/results-minicpm5-2b-lora.md) for the first trained runs.
See [12B results](../docs/jev-decision-model-2026-09-26/results-gemma-4-12b.md) for the stock frontier, the clean
recipe and the gemma-4-12B-it LoRA.
`results/<run>/` keeps the small score files of each run: config, train summary, dev metrics, temperatures and suite scores.
The 12B runs also keep their training log (`metrics.jsonl`), and `results/pool-clean.manifest.json` describes the
decontaminated pool.
The weights, predictions and suite rows stay in `runs/` (git-ignored) and on the pod.
