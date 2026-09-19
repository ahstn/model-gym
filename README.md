# model-gym

Training and comparison pipeline for a **5-level command-risk classifier**, using ModernBERT, DeBERTa-v3, and lexical or frozen-encoder baselines.

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
| Training | PyTorch + Transformers `Trainer`; full or partially frozen fine-tuning |
| Model | ModernBERT, DeBERTa-v3-base, frozen ModernBERT/Nomic embeddings, case-sensitive n-grams |
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
uv run --extra train model-gym train --config configs/smoke.yaml   # 6 steps, proves the path
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
  calibration.py    validation-only temperature fitting and checkpoint sidecar
  policy.py         score bands, argmax actions, fitted probability thresholds
  baselines.py      case-sensitive n-grams and frozen-encoder linear heads
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

Seed and template variants must use the same canonical group ID. Exact duplicates with different levels, groups, categories, or shells now fail the build and name both sources. Other semantic near-duplicates still need review; matching group IDs are not a proof that every equivalent command has been found.

Labels describe command-visible risk under documented tool behavior. An opaque resource ID, localhost, a temporary path, or a development name does not prove that data is disposable, backed up, or recoverable. Score destructive operations conservatively when that context is unknown. A verified preview flag reduces risk only for the operation it suppresses; bypassing confirmation is not a dry run. Configuration-defined hooks and external programs remain outside the command-only view.

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
| `shell` | POSIX shell, coreutils, block devices | 99 |
| `system` | systemd, firewalls, users, kernel knobs, cron | 83 |
| `git` | history, working tree, remotes | 74 |
| `containers` | docker, docker compose, podman | 81 |
| `kubernetes` | kubectl, helm | 94 |
| `terraform` | terraform, opentofu, terragrunt | 71 |
| `aws` | AWS CLI | 97 |
| `gcp` | Google Cloud SDK, gsutil, bq | 70 |
| `azure` | Azure CLI | 68 |
| `sql` | SQL statements and their client invocations | 87 |
| `redis` | redis-cli and Redis administration | 61 |
| `elasticsearch` | Elasticsearch and OpenSearch HTTP APIs | 62 |
| `onepassword` | 1Password CLI (`op`) and secret-store administration | 76 |
| `sensitive_files` | reading, copying, or publishing files that carry secrets | 95 |

`shell` stays a separate field: it is the syntax dialect (`bash`, `sql`, `redis`,
`http`, `powershell`), and one dialect spans several categories.

### The two corpora

* `data/seed/*.jsonl` — the reviewed corpus: hand-scored commands gathered from
  tool documentation, runbooks, and published postmortems, filed by category.
  Real labels, uneven coverage. Every row carries its provenance in `source`
  (`web:<domain>`, `docs:<exact-url>`, `knowledge:<tool>`, or the taxonomy review that started this).
* `data/synthetic/templates_v1.jsonl` — 55 command families expanded into 618
  commands in `data/synthetic.py`. Every level is an explicit table entry
  (`scope.level + variant.shift`), never a heuristic, so labels stay auditable and
  diffable. 126 rows duplicate an agreeing curated command or another generated row and are dropped, leaving 492.

Together they produce 1610 records in 900 groups, split 1127/242/241 across train/validation/test, with level counts 308/342/383/260/317. Every level is present in every split. The 1118 seed rows include 26 new documented-command adaptations for permissions, redirection, firewall controls, Terraform/Terragrunt, and secret editing/sharing. These are reviewed examples, not captured production traffic.

The generated corpus exists because a fine-tuning pipeline with no data cannot be
tested, and because it encodes flag sensitivity (`--dry-run` → level 5,
`--skip-final-snapshot` → level 1) that is otherwise expensive to collect. It is a
**bootstrap, not a substitute for real traffic**: templated text is repetitive and
the model will overfit its phrasing. Add real labelled commands to the category
files under `data/seed/`, then drop the templates:

```bash
uv run model-gym build-data --no-synthetic
```

`build-data` records split hashes, split policy, group IDs, category group counts, and per-source row/group coverage in the dataset card. It warns on missing levels and thin held-out categories. The current test split has only four container commands; do not treat that category score as a reliable estimate.

## 6. Training

```bash
uv run model-gym build-data
uv run --extra train model-gym train --config configs/base.yaml
uv run --extra train model-gym train --config configs/large.yaml     # 395M params, gradient checkpointing on
uv run --extra train model-gym train --config configs/base.yaml --set train.learning_rate=2e-5 --set train.output_dir=runs/base-lr2e5
```

Any setting can be overridden with `--set section.key=value`; `uv run model-gym config --print`
shows every default.

Training refuses a nonempty output directory unless resuming a checkpoint. Use a distinct output directory for each experiment. By default, training loads and scores only train/validation; `train.evaluate_test=true` is an explicit opt-in after model selection.

Checkpoint selection defaults to `train.selection_metric=safety`. It uses the configured runtime policy, not necessarily the argmax label. The validation gates are `max_critical_miss_rate=0.05`, `max_unsafe_allow_rate=0.01`, and `max_unnecessary_intervention_rate=0.25`. A checkpoint that passes all gates ranks above one that fails. Passing models rank by QWK. If none passes, selection minimizes the sum of excess rates divided by `(1 - limit)`, with QWK only breaking ties. Missing population support cannot pass a gate. These are point-estimate experiment limits, not a production safety certificate. Approval bypass is reported separately and is not one of these three gates.

`train.decision_policy_path` loads a fitted policy and saves it with the best checkpoint. Without it, training uses the original expected-score bands. Set `train.calibrate_selection=true` to fit a validation temperature at each checkpoint comparison and to save calibration for the selected model. This uses validation for selection and fitting; it does not provide an independent safety estimate.

Use `train.selection_metric=qwk` for a controlled baseline. For a sweep, keep the data seed and split hashes fixed, vary the learning rate and training seed, and freeze the winner before scoring test data.

### What a run writes

```
runs/modernbert-base-v2/
  best/                    best validation checkpoint by configured selection rule
    calibration.json       explicit calibrate, or train.calibrate_selection=true
    decision-policy.json   exact rule used for runtime action metrics
  checkpoint-*/            last `save_total_limit` checkpoints
  logs/                    trainer logs
  reports/validation.md    metrics, severe misses, over blocks
  reports/test.md          only if test evaluation was explicitly enabled
  run_metadata.json        config, split hashes, corpus stats, selection status, trainer history
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

### Controlled encoder changes

ModernBERT supports `model.pooling=mean` or `cls` and `model.classifier_dropout=0.1`. The default preserves the pretrained configuration. `train.freeze_encoder_layers=N` freezes embeddings and the first N blocks; `-1` freezes the whole encoder but leaves the head trainable. This supports ModernBERT and DeBERTa-v3. `train.ordinal_loss_weight=0.5` adds a mean-squared error between predicted and target cumulative class probabilities to cross-entropy. It is an ordinal penalty, not a fitted safety-cost matrix.

DeBERTa-v3-base needs eager attention in this Transformers version:

```bash
uv run --extra train model-gym train \
  --set model.name=microsoft/deberta-v3-base \
  --set model.attn_implementation=eager \
  --set train.output_dir=runs/deberta-v3-base
```

`model_gym.baselines` provides `LexicalClassifier.fit(commands, labels, mode="char"|"word"|"union", C=...)` and `EmbeddingClassifier.fit(features, labels, encoder, C=...)`. Labels are zero-based indices, 0 through 4. Lexical features preserve case and punctuation. `structured=True` adds literal flag, operator, and path counts, not a shell parser or knowledge of external state. `FrozenEncoder.encode` supplies masked-mean or CLS features; Nomic's embedding model requires the `classification: ` prefix for this task.

These baseline APIs return raw logits and probabilities. Apply the saved temperature and `DecisionPolicy` separately. Their saved bundles are self-contained; joblib files must come from a trusted source because loading them can execute Python code. They are Python comparison artifacts, not Rust/ONNX exports.

## 7. Evaluation

```bash
uv run --extra train model-gym calibrate --model runs/modernbert-base-v2/best
uv run --extra train model-gym eval --model runs/modernbert-base-v2/best --split validation
# Only after freezing model selection:
uv run --extra train model-gym eval --model runs/modernbert-base-v2/best --split test --dump-predictions runs/preds.jsonl
```

```bash
uv run --extra train model-gym train --config configs/base.yaml --set train.max_steps=20 --set train.output_dir=runs/base-20steps
```

| metric | why it is here |
| --- | --- |
| `accuracy` | exact level match |
| `decision_accuracy` | decision match from the argmax level; distinct from the saved runtime policy |
| `macro_f1` | unweighted mean over 5 levels; catches a model that ignores rare levels |
| `qwk` | quadratic weighted kappa; 1.0 is perfect, and near misses are penalised less |
| `mae` | mean level distance, the ordinal view of the same errors |
| `ece` | expected calibration error over 15 bins; a threshold policy needs calibrated probabilities |
| `severe_miss_rate` | legacy ordinal rate: true level 1-2 predicted 4-5; this misses level 3 approval bypasses |
| `over_block_rate` | legacy ordinal rate: true level 4-5 predicted 1-2; includes approval, not just blocking |
| `critical_miss_rate` | true level 1 commands not blocked |
| `unsafe_allow_rate` | true level 1-3 commands allowed, divided by all true level 1-3 rows |
| `approval_bypass_rate` | true level 1-3 commands sent to flag/allow |
| `allow_precision`, `allow_coverage` | share of allow decisions that are truly level 5, and share of all commands allowed |
| `unnecessary_intervention_rate` | true level 4-5 commands sent to block/approve |

Top-level action metrics use the argmax level. The `score_policy` section retains the original `E[level] = sum(p_i * level_i)` diagnostic: block below 1.5, approve below 3.5, flag below 4.5, and allow from 4.5. Exact boundaries belong to the less severe band. `runtime_policy` reports the saved policy; trainer logs prefix its metrics with `runtime_`, and safety selection uses those values.

Reports include numerator counts, denominators, policy errors, and category/source breakdowns. Zero support is not evidence of safety. A single perfectly predicted class has undefined QWK, stored as JSON `null`.

`calibrate` fits a positive scalar temperature using validation logits only, then saves its value, validation hash, NLL/ECE before and after fitting, and calibrated validation metrics in `best/calibration.json`. Normal evaluation reloads it automatically. Repeating the fit starts from raw logits, so it does not compound calibration. Calibration leaves argmax unchanged but can change score-band decisions; check the validation safety gates again after fitting. Malformed calibration files fail rather than silently disabling calibration.

### Probability-based action policy

```bash
uv run --extra train model-gym fit-policy --model runs/modernbert-base-v2/best --out runs/fitted-policy.json
```

This command requires calibration on the same validation file. It fits three thresholds on a 41-point grid. Actions follow this order: block when `p(L1) >= block_threshold`; otherwise approve when `p(L1)+p(L2)+p(L3) >= approval_threshold`; otherwise allow when `p(L5) >= allow_threshold`; otherwise flag. Higher-priority actions win ties. The search minimizes gate violations, then critical misses, unsafe allows, and benign interventions, then maximizes action accuracy and allow coverage.

The output is a fitting-set estimate. Cross-fit calibration and thresholds by command group before comparing policy families. Freeze the chosen policy before comparing encoders. Changing a checkpoint's policy requires calibration to be rerun before export, so the saved validation metrics describe the exported rule.

## 8. Export for the Rust runtime

```bash
uv run --extra export model-gym export --model runs/modernbert-base-v2/best --precision int8 --quant-target avx2 --out artifacts/base-v2-int8
uv run --extra export model-gym export --model runs/modernbert-base-v2/best --precision fp32 --out artifacts/base-v2-fp32
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

Export refuses an existing output directory. Saved temperature scaling is applied inside the ONNX graph before quantization. The runtime must use ordinary `softmax(logits)` and must not divide by temperature again. Metadata records this contract and the calibrated validation metrics.

New exports use metadata schema version 2. Consumers must read `decision_policy` and the graph's actual `inputs`, including `token_type_ids` when present. `decision_bands` is null for non-score policies. An older Rust consumer that only applies expected-score bands must be updated before using these artifacts. The Rust runtime is not in this repository.

**The export is gated on parity with the PyTorch checkpoint.** By default, 32 commands are sampled from validation, with a train/source fallback. Test data is never an implicit parity sample. The PyTorch reference uses full-precision matrix multiplication, not CUDA TF32. Both implementations score the same commands:

* fp32 and fp16 must agree on every argmax label and runtime action (`min_argmax_agreement` is forced to 1.0).
* INT8 must agree on at least `export.parity_min_argmax_agreement` (0.95) of the
  *decisive* rows: rows whose top two probabilities are closer than
  `export.parity_tie_epsilon` (0.05) are excluded, because their argmax is a coin
  flip in either runtime.
* The mean probability delta must stay under `export.parity_mean_prob_delta` (0.05).

Substantial quantization drift fails the build. `parity.json` reports raw and decisive agreement, the decisive row count, score-band and runtime-action agreement and flip rows, and p95/max deltas. Read them, not just the exit code. INT8 classification parity does not certify safety or rule out action changes. `--no-verify` skips the check for an explicitly unverified artifact.

## 9. Transporting to the training host

```bash
# 1. on the laptop: send code, configs, and the seed corpus
scripts/sync_to_server.sh user@gpu-host ~/model-gym

# 2. on the host: install, verify CUDA, rebuild the corpus, train
ssh user@gpu-host 'cd ~/model-gym && scripts/server_bootstrap.sh'
ssh user@gpu-host 'cd ~/model-gym && uv run model-gym build-data'
ssh user@gpu-host 'cd ~/model-gym && uv run --extra train model-gym train --config configs/base.yaml'

# 3. back on the laptop: fetch the checkpoint and reports, then export locally
scripts/fetch_artifacts.sh user@gpu-host runs/modernbert-base-v2
uv run --extra export model-gym export --model runs/modernbert-base-v2/best --precision int8
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

* **The corpus is a bootstrap.** There are 1118 seed commands and 492 retained template variants. Scores on this corpus are not an estimate or a formal upper bound on production performance. Collect independently labelled real traffic before trusting the model.
* **Level 4 and 5 coverage is mostly seed data.** Templates supply 167/577 rows, or 28.9%. That does not establish coverage of real benign traffic.
* **Calibration is not classification repair.** Fit on validation only and verify the resulting runtime decisions. Neither low ECE nor export parity proves a safe guardrail.
* **Manual grouping has limits.** Exact annotation conflicts fail the build, but semantic equivalence still needs review. Version the dataset and compare models on the same held-out split.
* **Static INT8 calibration** is not implemented; dynamic quantization is. Static
  mode usually buys another speed step on x86 if latency becomes the constraint.
* **`--dry-run`-style flags are labelled from the table, not inferred.** Literal syntax features are now tested. A shell-aware parser remains a separate hypothesis; it cannot recover hidden account, workspace, or backup state.
* **INT8 quality is model-dependent.** Both selected encoders in iteration-v3 fail the parity gate on decisive validation rows. Smaller files and faster inference do not make those artifacts usable.

## 12. Historical validation round: `iteration-v2`

The prior data and configuration are preserved in `runs/baseline-77448bd-data/`. The corrected snapshot in `runs/iteration-v2/dataset/` has 1610 rows in 900 groups: 1127 train, 242 validation, and 241 test rows. The 26 added commands are adapted from public CLI documentation, not collected production traffic. No group crosses a split. Manual grouping still needs review; the container test category has only four rows.

The sweep compared learning rates `1e-5`, `2e-5`, and `3e-5` with seeds 13, 37, and 73, with at most four epochs and early stopping. Tuning read only train and validation. **None of the nine candidates passed the provisional validation safety limits.** The least-violating candidate used `3e-5`, seed 73, epoch 4. Selection was frozen before final test scoring.

The table compares that candidate with a separate QWK-selected baseline trained on the same corrected split (`3e-5`, seed 13). It is not a controlled comparison with the old 238-row test set.

| Final test metric, 241 rows | QWK baseline | Selected, raw | Selected, calibrated |
|---|---:|---:|---:|
| Exact-level accuracy | 38.2% | 41.9% | 41.9% |
| Macro F1 | 0.370 | 0.423 | 0.423 |
| QWK | 0.617 | 0.569 | 0.569 |
| ECE, lower is better | 0.200 | 0.195 | 0.082 |
| Runtime critical misses | 40/45 | 34/45 | 44/45 |
| Runtime unsafe allows | 2/155 | 4/155 | 1/155 |
| Runtime unnecessary interventions | 36/86 | 33/86 | 47/86 |

“Runtime” means the exported expected-score bands, not decisions derived from the most likely level. Earlier review numbers used the latter. Both rules are now reported. Validation-only temperature fitting gave `T = 1.8501`. It improved probability calibration but made blocking worse under the fixed score bands. The calibrated model blocked only one of 45 critical test commands. **Do not deploy it for autonomous safety decisions.**

FP32 export matches the full-precision PyTorch reference on all 242 validation rows and all 241 test rows. Its parity reference disables TF32; the old TF32 reference caused one validation argmax mismatch. The AVX2 INT8 export failed the unchanged 95% decisive-row gate, reaching only 57.2% on validation. Weight-format, range, and batch-size probes did not resolve that failure. The failed artifact is kept under `runs/iteration-v2/failed-exports/`, separate from verified exports.

| CPU runtime result | FP32 | INT8, failed parity |
|---|---:|---:|
| Graph size | 571.3 MiB | 144.5 MiB |
| Test exact-level accuracy | 41.9% | 34.9% |
| Test argmax agreement with PyTorch | 100% | 60.2% |
| Tokenizer + inference p50 | 17.67 ms | 8.95 ms |
| Tokenizer + inference p95 | 23.85 ms | 12.38 ms |

Latency uses an Intel Core Ultra 7 265K, ONNX Runtime CPU, four intra-op threads, one inter-op thread, and batch size one. Each artifact had 30 warm-up calls and 723 timed calls across all 241 test commands. These are Python ONNX Runtime plus Hugging Face fast-tokenizer measurements, not Rust-runtime measurements. Test quality also uses batch size one.

Full results are in `runs/iteration-v2/selection.json`, `evaluation-summary.json`, `onnx-benchmark.json`, and `evaluation/`. The verified FP32 runtime contract is in `artifacts/modernbert-base-v2-fp32/model-metadata.json`. These generated artifacts are local outputs, not source-controlled files.

## 13. Latest comparison round: `iteration-v3`

This round preserves the iteration-v2 corpus, checkpoints, and [performance snapshot](docs/performance-snapshot-iteration-v2.md). It compares four lexical baselines, two frozen encoders, 15 ModernBERT fine-tunes, and six DeBERTa-v3-base fine-tunes. The development directory contains only the unchanged 1127 training and 242 validation rows. The 241-row test set was already exposed in earlier work; it is a historical benchmark, not a fresh audit set.

The protocol and model choices were frozen before historical test scoring. Five of 27 candidates passed the three validation gates. **None passed all three on the historical test. None is ready for autonomous safety decisions.**

### Policy comparison

Three-fold group cross-fitting refit both temperature and thresholds within each fitting fold. It compared policies on the existing v2 checkpoint, so it does not remove that checkpoint's earlier validation-selection bias.

| Cross-fitted validation policy | Critical misses | Unsafe allows | Approval bypasses | Benign interventions |
|---|---:|---:|---:|---:|
| Expected-score bands | 37/40 | 0/161 | 12/161 | 33/81 |
| Argmax action | 15/40 | 6/161 | 28/161 | 19/81 |
| Risk thresholds | 3/40 | 2/161 | 60/161 | 13/81 |

The least-violating policy used probability thresholds. Full-validation refitting gave block at `p(L1) >= 0.175`, approve at `p(L1..L3) >= 0.825`, and allow at `p(L5) >= 0.7`, in that order. This one policy was fixed for every encoder; temperatures were fitted per model.

The trade-off matters: blocking improved, but approval bypass increased. On the unchanged v2 checkpoint's historical predictions, this rule reduced critical misses from 44/45 to 7/45 while increasing approval bypasses from 19/155 to 60/155. A separate approval-bypass limit is needed before treating the current three-gate objective as a release rule. This round did not change the frozen gates after seeing test results.

### Historical model comparison

All rows below use the fixed risk-threshold policy and validation-only temperature fitting. Neural rows are the validation-selected representative of each family, not the seed with the best test score.

| Model | Accuracy | QWK | ECE | Critical misses | Unsafe allows | Benign interventions |
|---|---:|---:|---:|---:|---:|---:|
| Character TF-IDF | 42.3% | 0.546 | 0.076 | 9/45 | 0/155 | 33/86 |
| Word TF-IDF | 37.8% | 0.512 | 0.046 | 12/45 | 0/155 | 35/86 |
| Character + word | 41.5% | 0.548 | 0.094 | 12/45 | 0/155 | 34/86 |
| Character + word + syntax counts | 39.8% | 0.537 | 0.059 | 8/45 | 0/155 | 34/86 |
| Frozen ModernBERT + linear head | 33.6% | 0.416 | 0.100 | 6/45 | 0/155 | 33/86 |
| Frozen Nomic + linear head | 42.7% | 0.524 | 0.144 | 8/45 | 4/155 | 34/86 |
| ModernBERT, ordinal loss 0.5 | 37.3% | 0.484 | 0.104 | 9/45 | 0/155 | 28/86 |
| DeBERTa-v3-base | 36.5% | 0.524 | 0.110 | 4/45 | 0/155 | 26/86 |

The selected ModernBERT used learning rate `3e-5`, seed 37, mean pooling, and the ordinal penalty. The selected DeBERTa used `3e-5`, seed 37, and ordinary cross-entropy. Their approval bypasses were 60/155 and 55/155. DeBERTa made no allow decisions, so its zero unsafe-allow count is not evidence of useful allow coverage.

Nomic had the highest validation accuracy, 54.5%, but reached 42.7% on the historical set. It trained only 3845 head parameters while retaining the full encoder at inference. The paired group-bootstrap interval for its accuracy gain over selected ModernBERT spans zero. These results do not establish an accuracy gain from a larger encoder or the tested enhancements.

### Controlled ModernBERT ablations

Values are validation mean ± sample standard deviation across seeds 13, 37, and 73. Every variant used learning rate `3e-5`, at most four epochs, and the same calibrated safety-selection rule.

| Variant | Accuracy | QWK | Validation gate passes |
|---|---:|---:|---:|
| Full fine-tune, mean pooling | 46.8% ± 1.0% | 0.657 ± 0.042 | 0/3 |
| CLS pooling | 43.4% ± 3.3% | 0.590 ± 0.065 | 1/3 |
| Head dropout 0.1 | 46.6% ± 1.0% | 0.664 ± 0.042 | 0/3 |
| Freeze embeddings and first 11 blocks | 42.7% ± 1.5% | 0.542 ± 0.029 | 0/3 |
| Ordinal penalty 0.5 | 47.5% ± 2.3% | 0.667 ± 0.031 | 1/3 |

The ordinal variant supplied the selected checkpoint, but its small average validation gain did not become a historical-test accuracy gain. Partial freezing reduced trainable parameters from 149.6M to 55.8M and reduced quality in this sweep. Syntax counts raised lexical validation accuracy from 42.1% to 44.6%, but historical accuracy fell from 41.5% to 39.8%.

### Runtime and export

CPU timings include tokenization or feature extraction, inference, calibration, and the action policy. Each used four CPU threads, batch size one, 30 warm-up calls, and 723 timed calls over the historical commands. Lexical models use sklearn; frozen encoders use PyTorch plus sklearn; fine-tuned exports use Python ONNX Runtime. These are not Rust measurements or backend-controlled architecture timings.

| Runtime | p50 | p95 |
|---|---:|---:|
| Character TF-IDF | 0.38 ms | 0.43 ms |
| Character + word + syntax counts | 0.76 ms | 0.83 ms |
| Frozen ModernBERT | 20.45 ms | 26.18 ms |
| Frozen Nomic | 21.40 ms | 27.40 ms |
| ModernBERT FP32 ONNX | 18.44 ms | 24.60 ms |
| DeBERTa FP32 ONNX | 17.59 ms | 24.23 ms |

Both FP32 exports match PyTorch labels and runtime actions on all 242 validation and 241 historical test rows. They are in `artifacts/modernbert-base-v3-fp32/` and `artifacts/deberta-v3-base-fp32/`. Both dynamic AVX2 INT8 exports failed the unchanged 95% decisive-row gate: 53.1% agreement for ModernBERT and 45.5% for DeBERTa in the full-validation batch. Batch-one checks also failed. Rejected graphs are kept under `runs/iteration-v3/failed-exports/`.

Transformers 4.57.6 emits a false Mistral-regex warning when reloading this DeBERTa tokenizer. An audit found identical token IDs for all 1610 corpus commands across the original SentencePiece tokenizer, saved fast tokenizer, and exported tokenizer. No Mistral patch or warning suppression was applied.

### Bespoke encoder direction

This round did **not** build a bespoke tokenizer or new transformer stack. Pooling, dropout, freezing, and loss changes preserve the pretrained architecture. A checkpoint swap is not an architecture rebuild.

The research-backed hypothesis is a case-sensitive subword encoder with an exact byte/character branch and syntax-only structure features, followed by a calibrated ordinal head. [ModernBERT](https://aclanthology.org/2025.acl-long.127/) provides the pretrained reference; [CANINE](https://aclanthology.org/2022.tacl-1.5/) and [Charformer](https://arxiv.org/abs/2106.12672) provide character/byte-front-end designs. [NeoBERT](https://arxiv.org/abs/2502.19587) provides modern block-design evidence, not a guarantee that changing tokenizers helps this task.

New vocabularies, byte front ends, or changes to normalization and feed-forward blocks need adaptation or distillation. They are not interchangeable pretrained parts. A separate unlabeled command corpus and an independent labelled audit set are missing here. The 1127 supervised training rows can test a new branch's feasibility, but this round does not establish that they can train a useful new encoder. No architecture can infer hidden environment state from command text alone.

Detailed local outputs are in `runs/iteration-v3/`: `protocol.json`, `policy-comparison.json`, `selection.json`, `features-comparison.json`, `neural-comparison.json`, `historical-test-comparison.json`, `ablation-summary.json`, `historical-paired-comparisons.json`, `cpu-benchmark.json`, `exports-comparison.json`, `bespoke-research.json`, and `verification.json`. Each candidate also has a checkpoint or baseline bundle, calibrated reports, and historical per-command predictions.
