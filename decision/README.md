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

## Readout design

1. The prompt lists the options as `A. text`, `B. text`, and so on. It uses the chat template without thinking.
2. The assistant turn starts with `Answer:` and has no trailing space.
3. The model gives the logits for the next token. We read only the rows of the code tokens (`" A"`, `" B"`, ...).
   The tokenizer puts the space into the code token, so `" A"` is one token.
4. `Readout` accepts a code only when it is exactly one new, unique token after `Answer:`. It keeps 255 codes:
   `A`..`Z`, then two-letter codes.
5. `code_logits` runs the backbone only and multiplies the last hidden state by the code rows of `lm_head`.
   It does not compute the full-vocabulary logits. Codes past the number of options get `-inf`.
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

## Decision Index 0.2 suite

You need the `suite` extra (`uv sync --extra suite`). HLE (catalog 45) is gated on Hugging Face. Accept its terms at
<https://huggingface.co/datasets/cais/hle> before you do a full rebuild.

```sh
uv run decision suite rebuild --suite-dir data/suite-0.2                 # full suite; needs HLE access
uv run decision suite rebuild --skip-hle --suite-dir data/suite-0.2-noHLE   # no HLE; writes PARTIAL.json
uv run decision suite sample --n 500 --out data/sample-500.jsonl.gz --suite-dir data/suite-0.2-noHLE
uv run decision suite run --model openbmb/MiniCPM5-2B --adapter runs/X/best \
    --temperatures runs/X/eval/temperatures.json --suite-dir data/suite-0.2-noHLE --out runs/X/suite-noHLE --shards 6
uv run decision suite score --run runs/X/suite-noHLE
```

- A suite with `PARTIAL.json` is not an official 0.2 suite. Its index is not comparable to the public board.
- `--shards N` runs N engine processes on one GPU and then merges and scores their results. Each shard can resume.
  Six shards of the 2B model use about 76 GB on a 96 GB card. Do not start other GPU jobs at the same time: an
  out-of-memory error stops the shard, and you must re-run to resume.
- The LoRA is not merged into the base weights. We tried a bf16 merge on a 500-request sample: it was about 22%
  faster, but it changed 1.5% of the argmax answers (mean TV 0.0098), and the temperatures were fit unmerged.
- The engine encodes the shared prompt prefix once for each request and then runs one short pass for each
  question. Each question sees only the prefix and its own suffix.

See [results](../docs/jev-decision-model-2026-09-26/results-minicpm5-2b-lora.md) for the first trained runs.
`results/<run>/` keeps the small score files of each run: config, train summary, dev metrics, temperatures and suite scores.
The weights, predictions and suite rows stay in `runs/` (git-ignored) and on the pod.
