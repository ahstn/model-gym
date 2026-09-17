# model-gym

Fine-tuning pipeline for a **5-level command-risk classifier** built on ModernBERT.

A guardrail for coding agents has to answer one question fast and cheaply: *how
dangerous is this shell or SQL command?* This project trains an encoder to answer
it as a calibrated 5-way ordinal classification, then exports the result to ONNX +
INT8 so a Rust runtime (`ort` + `tokenizers`) can serve it with no Python and no
GPU.

The pipeline is designed to be built on a laptop and transported to a single
CUDA host (the target here is an RTX 3090, 24 GiB) for training.

---

## 1. The taxonomy

Levels are ordinal and monotonic: index 0 is the most severe, index 4 the least.
Classifier label index is always `level - 1`.

| index | level | slug        | decision  | meaning                                                                   |
|------:|------:|-------------|-----------|---------------------------------------------------------------------------|
|     0 |     1 | `critical`  | `block`   | Catastrophic or irreversible; broad production or data destruction.        |
|     1 |     2 | `dangerous` | `approve` | Destructive or access-impacting with significant blast radius.             |
|     2 |     3 | `high_risk` | `approve` | Can destroy local state, interrupt services, or cause operational damage.  |
|     3 |     4 | `caution`   | `flag`    | Potentially disruptive or security-weakening, not destructive by itself.   |
|     4 |     5 | `low_risk`  | `allow`   | Normally non-destructive or read-only.                                    |

Level 5 is named `low_risk`, not `safe`: arbitrary execution is never categorically
safe. The level → decision mapping lives in `src/model_gym/labels.py` and ships to
the runtime inside `model-metadata.json`, so the policy can change without retraining.

Risk is treated as `operation x scope x irreversibility x bypass flags`, not as a
property of the executable name. `rm file.tmp` and `sudo rm -rf /` are different
levels, and `--dry-run`, `--recursive`, `CASCADE`, `--auto-approve` and
`--skip-final-snapshot` move the level rather than being ignored.

## 2. Stack

| Layer | Choice |
| --- | --- |
| Training | PyTorch + Transformers `Trainer` (full fine-tune, no LoRA) |
| Model | `answerdotai/ModernBERT-base`, then `ModernBERT-large` |
| Data | `datasets` over JSONL, dynamic padding via `DataCollatorWithPadding` |
| Metrics | accuracy, macro F1, quadratic weighted kappa, MAE, ECE, asymmetric miss rates |
| Export | Optimum ONNX + dynamic INT8 quantization |
| Runtime (not in this repo) | Rust `ort` + `tokenizers` |

No TRL, Axolotl, or Unsloth: at 149M-395M parameters a discriminative head is
cheaper and better calibrated than a generative recipe, and full fine-tuning is
affordable.

## 3. Quickstart on this machine (Apple Silicon, no GPU needed)

```bash
uv sync --extra train --extra export   # or: mise run setup
uv run model-gym build-data            # corpus -> data/processed/*.jsonl
uv run model-gym train --config configs/smoke.yaml   # 6 steps, ~1 minute, proves the path
uv run model-gym info                  # versions, device, GPU memory
```

`configs/smoke.yaml` runs on CPU or MPS: it checks data loading, tokenization,
training, evaluation, reports, and metadata end to end without a GPU.

## 4. Layout

```
configs/            base.yaml (ModernBERT-base), large.yaml, smoke.yaml
data/seed/          reviewed corpus, one JSONL per tool category (14 files)
data/processed/     generated: train/validation/test.jsonl + dataset_card.json
data/synthetic/     generated: the templated bootstrap corpus, for review
src/model_gym/
  labels.py         taxonomy, index mapping, decision bands, risk score
  data/schema.py    JSONL record schema, validation, dedupe, grouping
  data/synthetic.py templated bootstrap corpus
  data/build.py     load -> dedupe -> group-aware split -> write
  config.py         dataclasses, YAML, `--set section.key=value`
  modeling.py       tokenizer, dataset encoding, model construction, inference
  metrics.py        ordinal metrics and the markdown report
  train.py          Trainer wiring, class weighting, run metadata
  evaluate.py       checkpoint scoring, reports, error analysis
  export.py         ONNX export, INT8 quantization, parity check
  env.py            environment/GPU report
  cli.py            `model-gym` command line
scripts/            server bootstrap, rsync out, rsync back
tests/              schema, splitting, generator, metrics, config
```

## 5. Data

One JSON object per line:

```json
{"command": "kubectl delete pvc --all -n production", "level": 1, "category": "kubernetes", "group": "kubectl_delete:pvc_all_prod", "shell": "bash", "reason": "Persistent volume claims behind production data.", "source": "seed:paste-2"}
```

| field | required | notes |
| --- | --- | --- |
| `command` | yes | whitespace is normalized; single line; max 8000 chars |
| `level` | yes | 1-5, or a slug (`critical`, `dangerous`, `high_risk`, `caution`, `low_risk`) |
| `category` | no | tool family, one file per category in `data/seed/`; `shell` when omitted |
| `group` | no | near-duplicate family; derived from the leading program words when omitted |
| `shell` | no | `bash` (default), `sql`, `redis`, `http`, `powershell` |
| `reason` | no | why this level was assigned; kept for review and error analysis |
| `source` | no | provenance tag, e.g. `seed:paste-2`, `synthetic:templates-v1`, `prod-traffic:2026-10` |

**`group` is the important one.** Splits are made group-wise, so `rm -r ./build`,
`rm -rf ./build/` and `sudo rm -rf ./build` always land in the same split. Random
row-level splitting would put near-identical commands in both train and test and
report a quality number that means nothing.

### Categories

The reviewed corpus is split into tool families, one file per category, so a
failure can be traced to the tool it came from. `read_jsonl` rejects a row whose
`category` contradicts its file name, and `build-data` prints per-category counts
per split into the dataset card.

`sensitive_files` follows the shape of a filesystem guard pack (`~/.config/dcg/packs/sensitive-files.yaml`):
every row is one way a credential file gets read, copied, committed, or published,
with the safe counterparts (`cat .env.example`, `cat .gitignore`, `ls -l .env`)
labelled next to them so the boundary is learnable. `onepassword` covers the same
idea for a hosted secret store: `op` commands that move, share, or delete the
source of truth for credentials.

| category | covers | seed rows |
| --- | --- | ---: |
| `shell` | POSIX shell, coreutils, block devices | 91 |
| `system` | systemd, firewalls, users, kernel knobs, cron | 79 |
| `git` | history, working tree, remotes | 74 |
| `containers` | docker, docker compose, podman | 81 |
| `kubernetes` | kubectl, helm | 94 |
| `terraform` | terraform, opentofu, terragrunt | 63 |
| `aws` | AWS CLI | 97 |
| `gcp` | Google Cloud SDK, gsutil, bq | 70 |
| `azure` | Azure CLI | 68 |
| `sql` | SQL statements and their client invocations | 87 |
| `redis` | redis-cli and Redis administration | 61 |
| `elasticsearch` | Elasticsearch and OpenSearch HTTP APIs | 62 |
| `onepassword` | 1Password CLI (`op`) and secret-store administration | 70 |
| `sensitive_files` | reading, copying, or publishing files that carry secrets | 95 |

`shell` stays a separate field: it is the syntax dialect (`bash`, `sql`, `redis`,
`http`, `powershell`), and one dialect spans several categories.

### The two corpora

* `data/seed/*.jsonl` — the reviewed corpus: hand-scored commands gathered from
  tool documentation, runbooks, and published postmortems, filed by category.
  Real labels, uneven coverage. Every row carries its provenance in `source`
  (`web:<domain>`, `knowledge:<tool>`, or the taxonomy review that started this).
* `data/synthetic/templates_v1.jsonl` — 53 command families expanded into 613
  commands in `data/synthetic.py`. Every level is an explicit table entry
  (`scope.level + variant.shift`), never a heuristic, so labels stay auditable and
  diffable. 122 rows duplicate a curated command or another generated row and are
  dropped, leaving 491.

Together they produce 1583 records in 947 groups, split 1108/237/238 across
train/validation/test, with level counts 309/334/383/252/305: every level is
present in every split, and every category is present in the corpus.

The generated corpus exists because a fine-tuning pipeline with no data cannot be
tested, and because it encodes flag sensitivity (`--dry-run` → level 5,
`--skip-final-snapshot` → level 1) that is otherwise expensive to collect. It is a
**bootstrap, not a substitute for real traffic**: templated text is repetitive and
the model will overfit its phrasing. Add real labelled commands to the category
files under `data/seed/`, then drop the templates:

```bash
uv run model-gym build-data --no-synthetic
```

`build-data` reports the level and category distributions per split, and warns when
a level or category has too few rows to learn or when a split has no examples of a
level.

## 6. Training

```bash
uv run model-gym build-data
uv run model-gym train --config configs/base.yaml
uv run model-gym train --config configs/large.yaml     # 395M params, gradient checkpointing on
uv run model-gym train --config configs/base.yaml --set train.learning_rate=2e-5 --set train.epochs=6
```

Any setting can be overridden with `--set section.key=value`; `uv run model-gym config --print`
shows every default.

### What a run writes

```
runs/modernbert-base/
  best/                    best checkpoint by quadratic weighted kappa (load_best_model_at_end)
  checkpoint-*/            last `save_total_limit` checkpoints
  logs/                    trainer logs
  reports/validation.md    metrics, severe misses, over blocks
  reports/test.md
  run_metadata.json        resolved config, corpus stats, environment, trainer history
```

### RTX 3090 guidance

| Setting | base (149M) | large (395M) | why |
| --- | --- | --- | --- |
| `batch_size` | 32 | 16 | 24 GiB of memory wall at sequence length 256 |
| `gradient_accumulation_steps` | 1 | 2 | keeps the effective batch at 32 so the LR transfers |
| `gradient_checkpointing` | off | on | large needs it for activations |
| `bf16` | on | on | Ampere has native bf16; no loss scaling, more stable than fp16 |
| `attn_implementation` | `sdpa` | `sdpa` | flash-attn 2 works on sm_86 but needs a matching build; opt in per config |
| `optim` | `auto` | `auto` | resolves to `adamw_torch_fused` on CUDA, `adamw_torch` elsewhere |

TF32 matmuls are enabled automatically on CUDA. Commands are short, so
`max_seq_length: 256` costs little; the tokenizer truncates anything longer.

`flash_attention_2` needs `flash-attn` compiled for the exact torch/CUDA pair, and
flash-attn 3 is Hopper-only, so `sdpa` is the default. If the install fails,
`build_model` raises an error that says so instead of a cryptic import trace.

`train.class_weighting` accepts `none`, `sqrt`, or `balanced`. The bootstrap corpus
is mildly skewed (1.5:1, the widest level against the narrowest), so `none` is the
default; a real corpus dominated by destructive commands will need `sqrt`.
Weighting with a missing level is refused rather than silently producing infinite
weights.

## 7. Evaluation

```bash
uv run model-gym eval --model runs/modernbert-base/best --split test
uv run model-gym eval --model runs/modernbert-base/best --split test --dump-predictions runs/preds.jsonl
```

```bash
uv run model-gym train --config configs/base.yaml --set train.max_steps=20
```

| metric | why it is here |
| --- | --- |
| `accuracy` | exact level match |
| `decision_accuracy` | match after collapsing levels to block/approve/flag/allow; what the guardrail actually does |
| `macro_f1` | unweighted mean over 5 levels; catches a model that ignores rare levels |
| `qwk` | quadratic weighted kappa; 1.0 is perfect, and near misses are penalised less |
| `mae` | mean level distance, the ordinal view of the same errors |
| `ece` | expected calibration error over 15 bins; a threshold policy needs calibrated probabilities |
| `severe_miss_rate` | share of true level 1-2 commands predicted 4-5. **This is the one that matters.** |
| `over_block_rate` | share of true level 4-5 commands predicted 1-2; the usability counterweight |

Every report also lists the worst severe misses and over-blocks with their commands,
so a regression can be read directly off the diff. `mean_expected_level` is the mean
of `E[level] = sum(p_i * level_i)`, the score the runtime can threshold on.

## 8. Export for the Rust runtime

```bash
uv run model-gym export --model runs/modernbert-base/best --precision int8
uv run model-gym export --model runs/modernbert-base/best --precision fp16 --quant-target avx512_vnni
```

Produces `artifacts/<model>-<precision>/`:

```
model.onnx              the exported graph (fp32/fp16 exports)
model_quantized.onnx    dynamic INT8 graph (--precision int8; the fp32 graph is removed after quantization)
tokenizer.json          HF tokenizer for the `tokenizers` crate
labels.json             index-ordered label slugs
model-metadata.json     runtime contract
parity.json             PyTorch vs ONNX agreement on sampled corpus commands
```

`--quant-target` selects INT8 op settings: `arm64` (Apple Silicon, default),
`avx512_vnni` or `avx512` for cloud x86, `avx2` for older x86. Quantization is
dynamic, so no calibration dataset is needed. Static INT8 with calibration is the
next step once real traffic exists.

`model-metadata.json` is the contract the runtime reads: label table, decision
bands, risk-score definition, `max_seq_length`, tokenizer file, input/output names,
precision, and a sha256 for every artifact file so a transfer can be verified.

**The export is gated on parity with the PyTorch checkpoint.** 32 commands are
sampled across the corpus and scored by both implementations:

* fp32 and fp16 must agree on every row (`min_argmax_agreement` is forced to 1.0).
* INT8 must agree on at least `export.parity_min_argmax_agreement` (0.95) of the
  *decisive* rows: rows whose top two probabilities are closer than
  `export.parity_tie_epsilon` (0.05) are excluded, because their argmax is a coin
  flip in either runtime.
* The mean probability delta must stay under `export.parity_mean_prob_delta` (0.05).

Chance agreement is 0.2, so a mis-exported or mis-labelled graph fails loudly while
genuine quantization noise does not block the build. `parity.json` reports raw and
decisive agreement, the decisive row count, and p95/max deltas — read them, do not
just read the exit code. `--no-verify` skips the check for an explicitly unverified
artifact.

## 9. Transporting to the training host

```bash
# 1. on the laptop: send code, configs, and the seed corpus
scripts/sync_to_server.sh user@gpu-host ~/model-gym

# 2. on the host: install, verify CUDA, rebuild the corpus, train
ssh user@gpu-host 'cd ~/model-gym && scripts/server_bootstrap.sh'
ssh user@gpu-host 'cd ~/model-gym && uv run model-gym build-data'
ssh user@gpu-host 'cd ~/model-gym && uv run --extra train model-gym train --config configs/base.yaml'

# 3. back on the laptop: fetch the checkpoint and reports, then export locally
scripts/fetch_artifacts.sh user@gpu-host runs/modernbert-base
uv run --extra export model-gym export --model runs/modernbert-base/best --precision int8
```

`data/processed/` is deliberately not synced: the corpus build is deterministic
(fixed seed, sorted groups), so the host produces identical splits and the local
`dataset_card.json` still describes the run.

`server_bootstrap.sh` is idempotent, prints `nvidia-smi`, installs `uv` if missing,
syncs the `train` and `export` extras, and runs `model-gym info`. If `cuda available`
is false, it tells you to reinstall torch against the PyTorch index for your driver
(`UV_TORCH_INDEX=https://download.pytorch.org/whl/cu126 ...`).

## 10. Development

```bash
mise run test     # uv run --extra train --extra export pytest
mise run lint     # ruff check + format --check
mise run fmt
```

`pytest` covers the taxonomy invariants, record validation, group-aware splitting,
the generator (uniqueness, level coverage, pinned levels, seed/generator group
alignment), the ordinal metrics, and configuration validation. It needs no GPU and
no network.

## 11. Known limits and next steps

* **The corpus is a bootstrap.** 1092 commands scored from tool documentation and
  postmortems, plus 491 templated variants. Quality numbers on it are an upper
  bound on real performance, not an estimate. Collect real labelled traffic before
  trusting the model.
* **Level 4 and 5 coverage comes mostly from templates.** Real read-only and
  caution examples are the highest-value next addition.
* **No temperature scaling yet.** `ece` is reported but not corrected; if the
  runtime thresholds on probability rather than argmax, fit a temperature on the
  validation split first.
* **Static INT8 calibration** is not implemented; dynamic quantization is. Static
  mode usually buys another speed step on x86 if latency becomes the constraint.
* **`--dry-run`-style flags are labelled from the table, not inferred.** If the
  runtime can parse the command, parsing plus the classifier will beat either alone.
* **Exact-parity export gate** is tuned for this task: argmax exactness, mean
  probability delta under 0.05. Loosen only with evidence. An undertrained
  checkpoint has near-tie margins, so INT8 noise can flip its argmax and fail the
  gate even when the graph is correct: a 6-step smoke model lands at 0.92, the same
  checkpoint after 300 steps at 1.00.
