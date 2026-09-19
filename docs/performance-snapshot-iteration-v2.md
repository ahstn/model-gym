# Performance snapshot: iteration-v2

## Status

This snapshot records the completed `iteration-v2` training and validation round. It preserves the measured results before further model or policy changes. It is not a deployment approval.

Classification accuracy improved against a baseline trained on the same corrected split, but QWK fell. No sweep candidate met the provisional validation safety limits. Temperature scaling improved calibration but worsened critical blocking under the current runtime decision bands. The INT8 export failed parity and is rejected.

## Current architecture

| Component | Configuration in this round |
|---|---|
| Pretrained checkpoint | `answerdotai/ModernBERT-base` |
| Model class | `ModernBertForSequenceClassification` |
| Trainable parameters | 149,608,709; full fine-tuning |
| Encoder | 22 layers, hidden size 768, 12 attention heads |
| Input | Raw command text, checkpoint tokenizer, maximum 256 tokens |
| Pooling | Attention-mask-aware mean pooling |
| Prediction | Five-class classification head, L1 critical through L5 low-risk |
| Training objective | Single-label cross-entropy; no class weighting |
| Calibration | One temperature fitted on validation logits |
| Runtime action | Expected-level score mapped to block, approve, flag, or allow |
| Retrieval and reranking | Neither is present |

The model learns token embeddings and contextual representations internally. There is no separate sentence-embedding service, vector index, or reranker. Category and provenance fields support evaluation and data management; they are not added to the model's command input.

The runtime computes `sum(p[level] * level)`. Its bands are block below 1.5, approve from 1.5 to below 3.5, flag from 3.5 to below 4.5, and allow from 4.5 through 5.0. Exported logits already contain temperature scaling. Consumers apply ordinary softmax and must not apply temperature a second time.

## Data and pipeline corrections

- Reconciled the six identified seed/template label conflicts. Conflicting duplicate annotations now fail the build instead of being silently discarded.
- Fixed the identified family-group drift, including build-directory removal and production RDS deletion variants.
- Replaced the invalid `gsutil` dry-run example with a valid inventory operation, not a claimed atomic deletion preview. Corrected Terraform global flag placement.
- Documented conservative labels for unknown context. Command text does not reveal the active workspace, account, current directory, backups, or resource importance.
- Added 26 documentation-backed examples covering permissions, redirection, firewall operations, Terraform/Terragrunt, and secret editing or sharing. These are adapted CLI examples, not collected production traffic.
- Added safety-aware checkpoint selection, explicit metric counts and supports, validation-only calibration, and calibrated ONNX export.
- Training does not score test data by default. Existing run and export directories are protected against accidental overwrite.

The original run remains at [`runs/modernbert-base/`](../runs/modernbert-base/). Its data and configuration are preserved in [`runs/baseline-77448bd-data/`](../runs/baseline-77448bd-data/); all 22 manifest file hashes were verified.

### Corrected corpus

There are 1,610 commands in 900 groups: 1,118 seed commands and 492 retained synthetic variants.

| Split | Rows | Groups |
|---|---:|---:|
| Train | 1,127 | 629 |
| Validation | 242 | 157 |
| Test | 241 | 114 |

No group crosses splits. This does not certify that every semantic equivalence has been found: grouping is still manual. The container test category has only four rows. Synthetic examples supply 167/577 L4/L5 rows, or 28.9%, rather than most benign examples.

The frozen dataset is in [`runs/iteration-v2/dataset/`](../runs/iteration-v2/dataset/).

| Split | SHA256 |
|---|---|
| Train | `283c5f1465a5928581542df43358d124aed55c4c6e73b748ef543ce538010a36` |
| Validation | `afa3c0da60ab0e1353c730b5125e59de22880d10fbdddb33eb27542736c79545` |
| Test | `d5816a9906a878163d7a5a4c4ce4f08dd69279c11a56832b5003ca5e305ff88f` |

## Experiment protocol

Nine candidates used learning rates `1e-5`, `2e-5`, and `3e-5`, each with seeds 13, 37, and 73. Training used an RTX 3090, BF16, batch size 32, at most four epochs, and early-stopping patience two. A separate QWK-selected baseline used `3e-5` and seed 13 on the same corrected data.

Tuning had train and validation files only. All ten run records show no test evaluation during training. Selection was frozen before final test scoring; temperature fitting used validation only.

The selected candidate was **`3e-5`, seed 73, epoch 4**, at [`runs/iteration-v2/lr-3e-05-seed-73/best/`](../runs/iteration-v2/lr-3e-05-seed-73/best/).

None of the nine safety-selected candidates met these provisional validation limits:

- Critical commands not blocked: at most 5%.
- Commands requiring block or approval that are allowed: at most 1%.
- Benign commands receiving block or approval: at most 25%.

The selected candidate had the lowest constraint-violation score under the configured rule, not a safety-qualified result. These are experimental point-estimate limits, not statistical safety guarantees or approved production requirements.

Mean validation exact-level accuracy across the three seeds was 38.3%, 44.9%, and 47.1% for `1e-5`, `2e-5`, and `3e-5`, respectively. Lower learning rates did not improve that metric in this sweep.

## Held-out performance

The following comparison uses the same corrected 241-row test set. It is not a controlled comparison with the original 238-row test set. The baseline and selected candidate also differ in seed and selection rule, so the difference does not isolate the effect of safety-aware selection.

| Test metric | QWK baseline | Selected, raw | Selected, calibrated |
|---|---:|---:|---:|
| Exact-level accuracy | 38.2% | 41.9% | 41.9% |
| Macro F1 | 0.370 | 0.423 | 0.423 |
| QWK | 0.617 | 0.569 | 0.569 |
| ECE, lower is better | 0.200 | 0.195 | 0.082 |
| Runtime critical misses | 40/45 | 34/45 | 44/45 |
| Runtime unsafe allows | 2/155 | 4/155 | 1/155 |
| Runtime unnecessary interventions | 36/86 | 33/86 | 47/86 |

The selected model reached 80.5% train accuracy, 45.9% validation accuracy, and 41.9% test accuracy. The generalization gap remains large.

### Interpretation of safety metrics

Earlier review numbers used actions derived from the most likely level, or argmax. The exported runtime instead uses expected-score bands. Reports now distinguish these two rules, and safety selection uses the runtime rule. Do not compare the earlier 25.8% argmax-derived approval-bypass rate directly with a new score-band rate.

Under the calibrated runtime policy:

- Only **one of 45 critical test commands was blocked**.
- One of 155 commands requiring block or approval was allowed.
- Nineteen of those 155 commands were downgraded to flag or allow.
- Automatic-allow precision was **8/10**, with coverage **10/241, or 4.1%**.
- Forty-seven of 86 benign commands received block or approval.

The low unsafe-allow count does not establish safety. The runtime rarely allows anything, yet fails to block most critical commands.

## Calibration

Validation-only temperature scaling fitted **T = 1.850075838377791**.

| Validation metric | Before calibration | After calibration |
|---|---:|---:|
| Negative log-likelihood | 1.399864 | 1.283796 |
| ECE | 0.158927 | 0.046058 |

Test ECE improved from about 0.195 to 0.082. Argmax classifications did not change, but expected scores did. Under the fixed decision bands, test critical misses rose from 34/45 to 44/45, and unnecessary interventions rose from 33/86 to 47/86.

Calibration is therefore not a safety improvement by itself. The calibrated validation policy also fails the safety limits: it misses 38/40 critical commands. No policy threshold was changed in response to test outcomes.

## Export validation

### FP32

FP32 matches the full-precision PyTorch reference on all 242 validation rows and all 241 test rows. Test score-band decisions also match exactly.

The initial FP32 check had one validation argmax mismatch because the PyTorch reference used CUDA TF32 arithmetic. The parity reference now uses full-precision matrix multiplication. The verified artifact is in [`artifacts/modernbert-base-v2-fp32/`](../artifacts/modernbert-base-v2-fp32/).

A correct export only reproduces the model. It does not make the model's decisions safe.

### INT8

The AVX2 INT8 export reached only **57.2% agreement on 194 decisive validation rows**, below the unchanged 95% gate. Weight-format, quantization-range, and batch-size probes did not resolve the failure.

The rejected artifact remains at [`runs/iteration-v2/failed-exports/int8-avx2-parity-failed/`](../runs/iteration-v2/failed-exports/int8-avx2-parity-failed/), outside the verified artifact directory.

On test commands, INT8 label agreement with the full-precision PyTorch reference was only 60.2%. Its test accuracy fell to 34.9%. Its smaller size and lower latency do not justify use.

## CPU runtime measurements

| CPU runtime | Test accuracy | Tokenizer + inference p50 | Tokenizer + inference p95 | Graph size |
|---|---:|---:|---:|---:|
| FP32 | 41.9% | 17.67 ms | 23.85 ms | 571.3 MiB |
| INT8, rejected | 34.9% | 8.95 ms | 12.38 ms | 144.5 MiB |

Measurement conditions:

- Intel Core Ultra 7 265K and ONNX Runtime CPU execution provider.
- Four intra-op threads, one inter-op thread, sequential execution, and batch size one.
- Thirty warm-up calls, then three passes over all 241 test commands: 723 timed calls per artifact.
- Latency includes the Hugging Face fast tokenizer and the Python ONNX Runtime call. It does not include process startup or model loading.
- These are not Rust-runtime measurements. Test quality also uses batch size one.

## Verification recorded for this round

- 175 tests passed with `uv run --frozen --extra train --extra export pytest`.
- Ruff lint and format checks passed.
- Shell-script syntax checks passed.
- Preserved-data hashes, split hashes, and all seven FP32 payload hashes were verified.
- The calibration CLI reproduced the fitted temperature exactly using an isolated checkpoint fixture and validation-only data.
- Temporary experiment runners and the obsolete FP32 attempt were removed. Trained runs, baseline evidence, the verified FP32 artifact, and the rejected INT8 artifact were retained.

These checks describe the completed round, not a claim that validation was rerun when this document was saved.

## Evidence and remaining limits

- [Sweep and checkpoint selection](../runs/iteration-v2/selection.json)
- [Evaluation and calibration summary](../runs/iteration-v2/evaluation-summary.json)
- [Per-split reports and per-command predictions](../runs/iteration-v2/evaluation/)
- [Runtime benchmark](../runs/iteration-v2/onnx-benchmark.json)
- [FP32 runtime contract](../artifacts/modernbert-base-v2-fp32/model-metadata.json)
- [Final verification evidence](../runs/iteration-v2/final-verification.json)
- [Export precision diagnosis](../runs/iteration-v2/export-diagnosis.json)
- [Quantization range and batching probes](../runs/iteration-v2/quantization-range-diagnosis.json)

The linked runs and artifacts are local generated outputs, not source-controlled files. This document preserves the key results if those outputs are unavailable later.

The model is not ready for autonomous safety decisions. Independent real traffic, broader semantic-group review, explicit context handling, and a safety action rule assessed together with calibration remain necessary. A larger encoder, more epochs, or a passing export-parity check would not by itself resolve those limits.
