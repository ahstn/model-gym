# Jev, Laya, and GLiNER versus a bespoke command-risk encoder

Date: 2026-09-19

The previous architecture response is preserved in full in [bespoke-encoder-stack-2026-09-19.md](bespoke-encoder-stack-2026-09-19.md). This document includes the authenticated Jev benchmark through OpenRouter and local GLiNER2.5 evaluation, alongside the retained Laya and selected ModernBERT measurements.

## Decision

**Do not abandon the command-risk project. Do defer a new encoder architecture until simpler alternatives have earned or failed their place.**

Jev has the highest observed exact-level accuracy of the four models: 47.3%, versus 37.3% for selected ModernBERT, 24.9% for zero-shot Laya, and 20.3% for zero-shot GLiNER2.5. Under the same frozen policy and validation-fitted calibration, Jev nevertheless misses 13 of 45 critical commands and bypasses approval on 76 of 155 hazardous commands. ModernBERT has fewer of those errors, at 9 and 60, but causes many more unnecessary interventions. GLiNER blocks every historical command, so its zero misses do not establish useful discrimination. None passes all historical safety gates.

Jev's hosted p50/p95 was 311.84/414.66 ms. Local CPU results were 291.63/313.64 ms for Laya, 132.61/143.98 ms for GLiNER, and 18.44/24.60 ms for ModernBERT ONNX. These are end-to-end implementation measurements, not hardware-controlled architecture comparisons.

[INFERENCE] The strongest near-term plan is to keep the domain-specific data, evaluation, policy, and runtime work, then compare pretrained or domain-adapted models before committing to custom pretraining. A better general classifier could replace the learned predictor without replacing the rest of this project.

The proposed 53M encoder is still a design. It has no measured latency or quality. The earlier 18.44 ms measurement belongs to our existing fine-tuned ModernBERT-base ONNX export, not to the proposed model.

## What the architectures actually do

| Area | Our proposed bespoke model | Laya | Jev |
|---|---|---|---|
| Task interface | Fixed five-level command-risk distribution | Runtime-defined Choice, Score, and Noul questions | Runtime-defined Choice, Score, and Noul questions |
| Input | Exact command text, proposed byte/syntax features, verified context where available | Text or serialized state, instructions, and candidate descriptions | Text or structured state, instructions, and candidate descriptions |
| Neural core | Proposed 8 layers, width 512, 8 heads, about 53M parameters before auxiliary branches | English: ModernBERT-large plus two Transformer head layers, about 421M parameters. Multilingual: mmBERT-base, about 322M | Layer count, width, parameter count, tokenizer, and exact neural design not publicly disclosed in the inspected sources |
| Classification | Pool command features and predict five logits | Score hidden states at candidate marker tokens, then softmax over candidates | Typed probabilities emitted in parallel; exact internal scoring mechanism undisclosed |
| Multiple decisions | A shared command encoder can feed several small fixed heads | One separate state-plus-question sequence per question, processed together as a batch | Documentation says state is ingested once and all questions are evaluated in parallel |
| Effective context | A design choice, with explicit overflow handling required | English checkpoint: 512 total tokens, including a 192-token question/option budget. Other checkpoints: 1024 total, 256-token question/option budget | Native documentation: 64k total, with state plus the longest question limited to 32k. The OpenRouter route advertises 32k |
| Adaptation | Domain fine-tuning or distillation; custom backbone would need new training | Open weights and code support fine-tuning | No customer fine-tuning or LoRA; adapt instructions, criteria, state, or a downstream model |
| Deployment | Intended local CPU/ONNX path | Local PyTorch path, Apache-2.0 code and published weights | Hosted API; no public self-hosting or downloadable-weight route found |

Sources: [Laya model code](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/laya/common.py), [Laya runtime](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/laya/agent.py), [Laya model card](https://huggingface.co/convaiinnovations/laya), and [Jev model documentation](https://docs.typesafe.ai/models.md).

Laya's actual path is:

```text
For each question:
  [CLS] type + instructions [SEP]
  [MASK] candidate 1 [MASK] candidate 2 ... [SEP]
  command/state [SEP]
      -> ModernBERT encoder
      -> question-type embedding + two Transformer layers
      -> candidate-marker scoring
      -> temperature scaling and softmax
      -> typed result
      -> application-owned action policy
```

Its "single forward pass" claim is true at the batch-call level. It does not mean the state is encoded once and reused across questions. The state is tokenized and encoded again for each question. For our fixed five labels, runtime candidate descriptions also add work that a fixed classification head does not need.

Jev and Laya are not verified architecture equivalents. The same API shape and the same RLCD name do not establish the same network or training algorithm. I could inspect and run Laya; I could not verify the author's claim of having built it before Jev. The Reddit page did not expose its post content through the reader, and the structured endpoint returned HTTP 403. The comparison therefore rests on code, model files, documentation, and measurements, not priority claims.

### GLiNER2.5: schema-based classification with an open encoder

The requested [GLiNER2.5 Multi checkpoint](https://huggingface.co/fastino/gliner2.5-multi-v1/tree/235cf92d6d4318da9bfca0d08975c8fa7250d13b) has 287,355,159 parameters and uses mDeBERTa-v3-base. `AutoExtractor` loads it as `BoundaryExtractor`. The [linked paper, arXiv:2507.18546v1](https://arxiv.org/html/2507.18546v1), describes the earlier GLiNER2 system, including a 205M-parameter model and span-grid extraction. Its model size, context, and published benchmark values must not be assigned to this newer boundary checkpoint.

For this evaluation, `ClassificationSchema.single` encodes the existing task instructions and five label descriptions. `Classifier.score` runs one encoder pass, takes contextual embeddings at the five `[L]` markers, and applies a scalar classifier to each. Native single-label probabilities are a softmax over those five logits. `Classifier.decode` reuses the scores; it does not run the encoder again. No entity extraction, span boundary search, relation extraction, or multi-task constraints are involved.

This is another dynamic-label encoder, not a generated-JSON model. Unlike hosted Jev, its weights and local runtime are available under Apache-2.0. It can also be fine-tuned, but this run is zero-shot.

The stock input pipeline is a serious fit caveat for commands. It lowercased tokens in 168 of the 483 evaluation inputs and appended a period to 482. A preprocessing probe confirmed that `git branch -D feature` and `git branch -d feature` produce identical encoded input IDs, despite different Git behavior. The same original command strings and rubric were supplied to all candidates; GLiNER's library changed the text before encoding. I did not patch that path or try another prompt.

The [input audit](../runs/classifier-comparison-2026-09-19/gliner2.5-multi-v1/input-audit.json) retains the original commands, model-facing text, word tokens, and lengths. No command was truncated. The longest encoded sequence was 349 tokens. The audit used a conservative 512-token ceiling based on the encoder configuration; the checkpoint also has a `max_len=4096` setting. This run does not establish behavior on longer inputs.

Implementation references: [classification scoring](https://github.com/fastino-ai/GLiNER2/blob/main/gliner2/classification/scoring.py), [schema compiler](https://github.com/fastino-ai/GLiNER2/blob/main/gliner2/classification/compiler.py), and [preprocessor](https://github.com/fastino-ai/GLiNER2/blob/main/gliner2/processor.py). The installed version and source-file hashes are retained in [environment.json](../runs/classifier-comparison-2026-09-19/gliner2.5-multi-v1/environment.json), rather than relying on moving `main` links alone.

## Frozen command-risk benchmark

I ran the English root Laya checkpoint, not the typed-workflow specialist:

- Model: `convaiinnovations/laya`.
- Checkpoint revision: `c5d78730f3493e4fe16d61507ef4b78eef7318cf`.
- Source revision: `6a5819129eb220570792e417e49723d697efd76f`.
- Trainable parameter count loaded: 421,293,827.
- One fixed five-way Choice prompt, using the repository's risk taxonomy. No prompt search or fine-tuning.
- Command text only. No label, reason, group, source, or category was supplied to the model.
- All 242 validation and 241 historical-test commands. No command or prompt was truncated.
- Shipped temperature reported separately from a new scalar temperature fitted only on validation.
- Existing iteration-v3 probability-threshold action policy held fixed.
- Intel Core Ultra 7 265K, CPU FP32, four intra-op threads, one inter-op thread, batch size one.
- Timing: 30 warm-up calls and 723 timed calls. Includes public `Agent.predict`, raw-logit capture, scalar calibration, and local policy evaluation. Model loading excluded.
- Repeated calls produced identical logits in this run.

### Jev extension through OpenRouter

The user supplied `OPENROUTER_API_KEY` through the local mise environment. I used OpenRouter's [native Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request.md), not a chat-completion or generated-JSON adapter:

- Requested model: `typesafe/jev-1.13`; every response resolved to `typesafe/jev-1.13-20260917`.
- Provider restricted to `typesafe`, with fallback disabled. Every response identified TypeSafe.
- Exact same Choice instructions, criteria, command-only inputs, data hashes, and frozen action policy as Laya.
- One sequential quality request for each of the 242 validation and 241 historical commands. No prompt search or fine-tuning.
- Persistent HTTPS client, concurrency one, 30 warm-ups, then three passes over all 241 historical commands: 723 timed requests.
- 1,236 successful benchmark requests, zero failures, and distinct response IDs. Access and adapter-development probes are kept separately.
- Laya and ModernBERT were not rerun. Their retained predictions and timing results were reused, with data/label alignment verified.

Jev returns rounded probabilities, not raw logits. Three quality responses summed to 0.99 rather than 1.0. All raw responses are retained; finite probabilities were normalized within the five-entry rounding-error bound. The prompt was unchanged after this interface check, and the full scoring run restarted with the final adapter.

Native outputs are reported separately from validation-only probability-power recalibration. For the latter, probabilities are floored at `1e-12`, normalized, and transformed with a scalar temperature fitted on validation only. The fitted value is 6.2971. This is not Jev raw-logit calibration, and it cannot recover information removed by rounding. Native Jev assigned zero probability to the true label on 7 validation and 5 historical-test examples.

### GLiNER2.5 extension

- Model: `fastino/gliner2.5-multi-v1`, pinned to `235cf92d6d4318da9bfca0d08975c8fa7250d13b`; all nine downloaded checkpoint files passed the Hub checksum check.
- Runtime: `gliner2==2.0.0`, PyTorch 2.14.0, Transformers 4.57.6, CPU FP32, four intra-op threads, one inter-op thread, batch one.
- The library fell back from requested SDPA to eager attention because this Transformers DeBERTa implementation does not support SDPA. This is measured as eager inference, not claimed as an optimized backend.
- Exact same original instructions, five criteria, validation/test hashes, and frozen action policy. The stock preprocessing differences are documented above.
- Raw logits and native probability maps retained for every row. Scalar temperature fitted only on the 242 validation rows, then fixed for the 241 historical-test rows.
- CPU timing covers public `Classifier.score` plus `decode`, schema and token processing, calibration, and policy. It uses 30 warm-ups and 723 timed calls, with model loading excluded.
- Other models' predictions and timings were retained, not rerun. No prompt search, fine-tuning, case-preserving adaptation, quantization, or ONNX export was performed for GLiNER.

### Historical results

The measured in-house model is the selected ModernBERT command-risk classifier, not the unbuilt 53M-parameter bespoke proposal.

"Critical misses" means a true level-1 command was not blocked. "Approval bypasses" means a true level-1/2/3 command received flag or allow, rather than block or approve. "Unsafe allows" counts level-1/2/3 commands allowed outright. "Unnecessary interventions" counts level-4/5 commands blocked or sent for approval.

| Model/configuration | Exact-level accuracy | QWK | Critical misses /45 | Approval bypasses /155 | Latency p50 | Latency p95 |
|---|---:|---:|---:|---:|---:|---:|
| Existing character TF-IDF | 42.3% | 0.546 | 9 | 55 | 0.38 ms | 0.43 ms |
| Existing selected ModernBERT | 37.3% | 0.484 | 9 | 60 | 18.44 ms | 24.60 ms |
| Existing selected DeBERTa | 36.5% | 0.524 | 4 | 55 | 17.59 ms | 24.23 ms |
| Laya, shipped temperature | 24.9% | -0.035 | 40 | 148 | Not separately timed | Not separately timed |
| Laya, validation-only calibration | 24.9% | -0.035 | 38 | 145 | 291.63 ms | 313.64 ms |
| Jev, native probabilities | 47.3% | 0.623 | 17 | 66 | Not separately timed | Not separately timed |
| Jev, validation-only recalibration | 47.3% | 0.623 | 13 | 76 | 311.84 ms, hosted | 414.66 ms, hosted |
| GLiNER2.5 Multi, native probabilities | 20.3% | -0.034 | 0 | 0 | Not separately timed | Not separately timed |
| GLiNER2.5 Multi, validation-only calibration | 20.3% | -0.034 | 0 | 0 | 132.61 ms | 143.98 ms |

**GLiNER's zero critical misses and approval bypasses come from blocking all 241 commands, not from a useful safety boundary.** This holds for both native and calibrated probabilities under the frozen policy.

The Laya runtime was about 15.8 times slower than the existing ModernBERT ONNX path on this machine. This is an implementation-level comparison, not an isolated architecture experiment: Laya uses eager PyTorch, the fine-tuned local models use ONNX Runtime, and Laya processes a question and candidate descriptions in addition to the command. I did not optimize or export Laya to ONNX.

Jev latency includes the workstation-to-OpenRouter-to-TypeSafe round trip, response handling, calibration, and the local action policy. It is not CPU inference time. The local controls exclude network transport.

### Safety and useful automation: the four models

All rows below use validation-fitted calibration and the same frozen action policy.

| Model | Critical misses /45 | Unsafe allows /155 | Approval bypasses /155 | Unnecessary interventions /86 | Correct allows /all allows | Allow coverage |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 13 | 0 | 76 | 2 | 19/19 | 7.9% |
| Laya | 38 | 0 | 145 | 0 | 0/0, undefined precision | 0.0% |
| ModernBERT | 9 | 0 | 60 | 28 | 10/10 | 4.1% |
| GLiNER2.5 Multi | 0 | 0 | 0 | 86 | 0/0, undefined precision | 0.0% |

Jev trades fewer unnecessary interventions and greater useful allow coverage for more observed critical misses and approval bypasses than ModernBERT. Jev, Laya, and ModernBERT fail the historical critical-miss limit of 5%. ModernBERT also exceeds the 25% unnecessary-intervention limit. GLiNER meets the critical-miss limit by blocking everything, but intervenes on 100% of benign/caution commands and therefore fails the intervention gate. The unsafe-allow limit is 1%, but zero observed errors does not establish that deployment rate.

Native Jev made 6 unsafe allows and 66 approval bypasses. Recalibration reduced unsafe allows to zero and critical misses from 17 to 13, while increasing approval bypasses to 76. Its historical ECE fell from 0.210 to 0.066. The calibration change is not an unqualified safety improvement.

Jev passed the existing three gates on validation: 1/40 critical misses, 1/161 unsafe allows, and 4/81 unnecessary interventions. It still bypassed approval on 68/161 validation examples. This exposes both the validation-to-historical gap and the missing approval-bypass constraint in the current gate set.

GLiNER reached 23.1% validation accuracy. Its validation-fitted raw-logit temperature was 10.7893. Historical ECE fell from 0.286 to 0.026, but accuracy stayed at 20.3% and every historical action remained block. On validation, native probabilities produced 240 blocks and two approvals; calibrated probabilities produced 242 blocks. Low ECE is not useful classification or a safety certificate.

### Uncertainty and repeated-call stability

Jev's accuracy advantage over ModernBERT was 9.96 percentage points. A paired bootstrap over the 114 historical command groups, with 2,000 resamples and seed 20260919, gives a 95% percentile interval of +0.47 to +18.02 points. Its advantage over Laya was 22.41 points, with an interval of +10.46 to +33.47 points. These intervals describe this historical corpus, not an independent production audit.

The critical-miss-rate difference between Jev and ModernBERT is uncertain: observed +8.89 points, interval -18.43 to +33.33 points. The approval-bypass difference is +10.32 points, interval -2.17 to +23.00 points. Treat the observed safety counts as measurements, not proof that one model has a lower deployment error rate.

Using the same paired group-bootstrap procedure, GLiNER's exact-level accuracy difference was -26.97 points versus Jev (95% interval -35.42 to -18.90), -17.01 points versus ModernBERT (-27.09 to -8.46), and -4.56 points versus Laya (-13.36 to +3.69). The data supports a clear accuracy deficit versus Jev and ModernBERT on this corpus; the GLiNER/Laya difference is less certain.

Identical Jev requests were not fully repeatable. Across 723 timed calls, 26 probability-derived top labels and 16 policy actions differed from the first quality pass. The largest per-class probability change was 0.13. All calls used the same resolved model version and provider. No vote or best-of-repeat selection was used for headline quality.

GLiNER produced identical raw logits across all 723 timed repeats: maximum logit difference zero, with no top-label or policy-action flips. Its typed classification path also matched native `BoundaryExtractor.extract` labels and confidence on three separate parity probes.

Laya made no allow decisions under either tested temperature with the fixed policy. Its zero unsafe-allow count therefore does not establish useful safe automation. Its calibrated ECE was 0.067 despite an 84.4% critical-miss rate and a 93.5% approval-bypass rate. This is a concrete example of why low aggregate ECE is not a safety certificate.

Limits of this result:

- This was one frozen zero-shot prompt, not the best prompt found by a search and not an in-domain fine-tune.
- Jev, Laya, and GLiNER are zero-shot on this task; ModernBERT was fine-tuned on 1,127 command-risk training rows. This is a practical candidate comparison, not a controlled training-recipe experiment.
- The same frozen policy supports comparison with iteration-v3, but is not asserted to be the best policy for every model.
- The historical corpus is already exposed, is small, and includes synthetic examples. It is not a fresh audit or production-distribution estimate.
- No current local candidate passes all historical safety gates. A weak Laya result does not make our existing models ready to deploy.
- The current three-gate objective still needs an explicit approval-bypass constraint.

The benchmark rejects these Laya and GLiNER configurations as replacements. It does not prove that a better prompt, a different checkpoint, or a command-domain fine-tune cannot work. GLiNER would also need a reviewed case/syntax-preserving input path before it could distinguish some command pairs.

### Jev access, cost, and retained evidence

The earlier direct TypeSafe request returned HTTP 403 because no direct API key was available. That historical result remains in `jev-access.json`; it is superseded for benchmark access by the successful OpenRouter run.

OpenRouter reported $0.025976538 for the 1,236 benchmark requests: 618,489 input tokens and 76,318 output tokens, with output tokens uncharged. Retained access and adapter probes add $0.001157814 in reported cost. One early adapter probe lacks a retained usage response, so total account billing cannot be stated exactly.

The current combined result is [four-model-comparison.json](../runs/classifier-comparison-2026-09-19/four-model-comparison.json); the earlier three-model result is preserved. GLiNER's protocol, checkpoint/source pins, schema, input audit, raw logits, probabilities, calibration, timing samples, comparison driver, and [verification record](../runs/classifier-comparison-2026-09-19/gliner2.5-multi-v1/verification.json) are under [gliner2.5-multi-v1](../runs/classifier-comparison-2026-09-19/gliner2.5-multi-v1). Jev's artifacts remain under [jev-openrouter](../runs/classifier-comparison-2026-09-19/jev-openrouter). The original Laya artifacts remain in the parent comparison directory. Existing ModernBERT comparator results remain in [iteration-v3](../runs/iteration-v3).

## Why the public headlines do not settle the choice

### Laya versus Jev is not a controlled head-to-head

Laya's [benchmark report](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/BENCHMARKS.md) explicitly says its Jev figures are third-party results, never measured by the author in the same run, with different prompts and sample sizes.

The reported 76.6% typed-decisions accuracy belongs to `laya-typed-decisions`, a specialist fine-tuned on the benchmark's own 1,200-case training split. It is evaluated on 400 test cases containing 2,000 decisions across four synthetic, teacher-labelled workflows. That is not direct test-set training, but it is also not a zero-shot generalization result. The English base checkpoint scores about 36.2% on that benchmark, below its 46.1% majority-class baseline. The specialist model card also reports worse soft accuracy and ECE than the listed Jev numbers. See the [specialist card](https://huggingface.co/convaiinnovations/laya-typed-decisions).

The published 32.8 ms single-question result belongs to multilingual Laya on a Tesla T4. English Laya is listed at 39.5 ms on that GPU. Those numbers cannot be substituted for CPU latency.

### Jev's speed claim is against a different baseline

The [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev) reports 70-500 ms end-to-end latency and large gains against generative LLM workflows. Our existing encoder already produces a probability vector in one forward pass. The claimed generation savings do not imply a speedup over our 18 ms local encoder.

TypeSafe's [workflow evaluation](https://evals.typesafe.ai/) measures agreement with reference probabilities from large external models, not independently reviewed command-risk labels. It includes security-incident workflows, but those do not establish critical-command recall or approval-bypass control for our taxonomy.

### Valid output is not correct judgment

Both systems constrain output shape. They can still select the wrong valid answer. Our fixed classifier already cannot invent a sixth risk label.

Laya's Choice/Score `confidence` is `1 - H(p)/log(k)`, a distribution-concentration measure, not a calibrated probability that its answer is correct. Jev's [confidence documentation](https://docs.typesafe.ai/confidence.md) also describes confidence as derived from the distribution.

Jev's own [known-limitations page](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) warns about adversarial state content, indirection, irrelevant context, numerical reasoning, and inconsistent probabilities across separately worded questions. These limits are relevant to shell syntax, quoted instructions, scripts, and missing execution context.

## Ideas worth adopting

### 1. Separate facts, uncertainty, and actions

[INFERENCE] The useful pattern is a narrow probabilistic model with code-owned decisions. Keep the full five-level distribution and apply asymmetric action thresholds outside the model. This part already exists in our pipeline.

For a fixed taxonomy, compute `P(requires approval) = P(L1) + P(L2) + P(L3)` from the same distribution. Do not replace it with a separately prompted Boolean and assume the two answers must agree. Do not treat entropy confidence or Laya's act-head output as an authorization decision.

### 2. Test factor supervision on a shared encoder

[INFERENCE] Candidate auxiliary targets include destructive operation, broad or production scope, irreversible data loss, bypassed safeguards, and missing context. These could make errors easier to inspect and test. A single command encoder could feed small heads for these targets and the five-level task.

This needs reviewed factor labels. It is not a proven gain. Do not multiply separately predicted factor probabilities and call the product calibrated command risk.

Jev's documented shared-state pattern is useful here. Laya's repeated state encoding is not the pattern to copy for a small CPU gate.

### 3. Use candidate-text scoring only if policies must change at runtime

[INFERENCE] Laya's marker scoring is attractive when the label set or descriptions change per request. It is less attractive for five stable labels, because it adds prompt tokens and large general-purpose heads. Keep it as an experimental comparator rather than the default local architecture.

### 4. Borrow the loss objective, not the RL label

Laya exposes log, spherical, and ranked-probability rewards. Its supplied fine-tuning recipe combines noisy-logit policy-gradient training with a supervised soft-target cross-entropy term. That does not show the RL component is needed for command risk, nor does it reproduce Jev's undisclosed RLCD algorithm.

[INFERENCE] Start with cross-entropy, the existing ordinal-loss experiment, and held-out calibration. When labels or teacher distributions are known, these objectives can be optimized directly. Add a policy-gradient mechanism only if a controlled comparison shows a gain.

The [published specialist notebook](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) also has details we should not copy: its calibration items come from the training items, the fine-tune loss gives the action head zero gradient, and inherited option-count temperature buckets can override the new per-type temperatures. These observations concern that supplied specialist recipe, not a claim about every stage of the English checkpoint's original training.

### 5. Preserve the full command and fail closed on unsupported input

Laya can drop the state tail to fit the context budget and rewrites literal mask-token strings. Dangerous suffixes such as a later shell command must not disappear silently.

Our existing code is not exempt: it normalizes command whitespace and enables tokenizer truncation. The lossless lexer/byte path in the earlier diagram was a proposal, not an implemented advantage. Exact input preservation and explicit overflow handling should be acceptance criteria for every candidate. The current benchmark audited all 483 inputs and found no truncation under its fixed prompt.

### 6. Do not add hierarchical beam search for five labels

The [hierarchical-classification cookbook](https://docs.typesafe.ai/cookbooks/hierarchical_classification) explores large taxonomies. Its example uses beam width 3 and gets four of four selected examples right, versus two of four with greedy search. That is a demonstration, not a broad benchmark.

Our five ordinal levels fit in one output vector. A hierarchy adds early-branch errors and more work without a demonstrated need. The cookbook's geometric-mean path score is a search-ranking heuristic, not a calibrated leaf probability. Even coherent conditional edge probabilities would require their product for joint path mass; taking a root changes the quantity.

## Does this make the project impractical?

**No for the domain-specific system; possibly for a from-scratch backbone, unless it earns its cost.**

Jev makes hosted classification cheap enough that API cost alone is a weak reason to build a new model. The measured requests averaged 500.4 input tokens and about $0.00002102 each, or $21.02 per million requests at the observed price. Latency, privacy, availability, and measured safety are the more important constraints.

The [Jev model documentation](https://docs.typesafe.ai/models.md) says customer requests are not used for training, but zero data retention is an enterprise arrangement. The [privacy policy](https://typesafe.ai/legal/privacy-policy) describes a US-hosted service. A hosted classifier adds a network and data-handling dependency that a local command gate does not have.

[INFERENCE] Revised order of work:

1. Keep the command corpus, reviewed taxonomy, policy, evaluation, and runtime contracts.
2. Treat the measured Laya and GLiNER results as failures of the tested zero-shot configurations. Collect and review independent command data before another model sweep. Resolve GLiNER's loss of case-sensitive input before considering command-domain adaptation.
3. Use the completed Jev run as a stronger zero-shot accuracy baseline. Fit any new policy with an explicit approval-bypass constraint on development data only; assess it on a fresh audit rather than retuning against this historical set.
4. Compare a pretrained small encoder with bespoke input features or auxiliary heads before creating a new backbone.
5. If a stronger model earns a clear quality gain but misses local latency or size needs, test distillation into a compact student.
6. Build the custom 53M architecture only if it solves a measured gap that these simpler routes leave open.

If Jev meets the safety, latency, privacy, and cost requirements, replacing the predictor is sensible. Keep the project as the command-specific evaluation and policy layer. If a local model is required, neither tested Laya nor tested GLiNER is currently a drop-in winner.

Finally, the audit needs more independent critical cases. Even with zero misses in 45 independent critical examples, the one-sided 95% upper bound on the miss rate is about 6.4%. Correlated command groups make that simple bound optimistic. No single headline accuracy or small test set can certify the desired safety rate.

## Would more data help?

**[INFERENCE] More independent, well-reviewed command data is a stronger next investment than another architecture sweep. The gain is not yet measured.** No retained experiment varies training-set size while holding the model and training recipe fixed, so there is no learning curve from which to promise a gain or a sufficient sample count.

The current corpus contains 1,610 rows in 900 groups. Training uses only 1,127 rows in 629 groups; 345 training rows are synthetic. Validation has 242 rows in 157 groups, and historical test has 241 rows in 114 groups. Across the corpus, 492 rows are generated templates. These counts are verified in [data-assessment.json](../runs/classifier-comparison-2026-09-19/data-assessment.json) and recorded in the [dataset card](../data/processed/dataset_card.json).

There is evidence of a generalization gap, not proof that data volume alone is the cause. The prior selected iteration-v2 model reached 80.5% training accuracy, 45.9% validation accuracy, and 41.9% historical-test accuracy. The selected iteration-v3 ModernBERT reached 50.0% validation accuracy but 37.3% on historical test. More diverse data, better labels, and better modeling are possible remedies; these runs do not isolate their effects.

Keep three data needs separate:

1. **Training data:** collect de-identified real command families with expert labels and provenance. Prioritize risk-boundary pairs: destructive versus dry-run, force versus checked operation, narrow versus recursive scope, overwrite versus append, and secret export versus safe inspection. Include matched benign controls, not just more dangerous commands. Keep related variants and workflows in one group across splits.
2. **Calibration and policy data:** reserve separate development groups for confidence calibration and action thresholds. Add an explicit approval-bypass constraint before fitting a new policy. More examples cannot compensate for a missing policy objective.
3. **Audit data:** create a fresh, untouched set with reviewed critical, approval-required, and benign cases. Do not use it for training, prompt or model selection, calibration, or threshold tuning. More rows added to the exposed historical set do not restore its independence.

Review label disagreements before expanding templates. The repository records six reconciled seed/template conflicts, and its seed labels are described as hand-scored, but there is no retained inter-rater agreement measurement. A second reviewer on difficult cases would give useful evidence about the label ceiling.

More labels do not change a frozen zero-shot model unless we use them for adaptation or development. They also cannot recover information removed by preprocessing: GLiNER's stock lowercasing loses distinctions between case-sensitive flags. Nor can command text reveal the active account, current directory, backups, or hidden resource importance. Those gaps need a clear context contract or a conservative review path, not more copies of the same command.

After collecting and reviewing new independent groups, measure a grouped training-size learning curve with fixed settings and several seeds. That will show whether further labeling is still improving accuracy and safety. Keep the fresh audit sealed until the model and policy are fixed.

Sources: [iteration-v2 measured gap and labeling corrections](performance-snapshot-iteration-v2.md), [iteration-v3 validation report](../runs/iteration-v3/modernbert-ordinal-0.5-seed-37/reports/validation.md), [iteration-v3 historical report](../runs/iteration-v3/modernbert-ordinal-0.5-seed-37/reports/test.md), and [frozen iteration-v3 protocol](../runs/iteration-v3/protocol.json).

## Verification and scope

- Prior response saved in full in a date-stamped document.
- Jev documentation, public evaluation methods, and hierarchical cookbook reviewed.
- Laya source, checkpoint configuration, model cards, demo/runtime differences, and published benchmark protocol reviewed.
- Pinned Laya checkpoint executed locally on all 242 validation and 241 historical commands, plus 723 timed CPU calls.
- Jev executed through OpenRouter on the same 483 commands, plus 30 warm-ups and 723 timed hosted calls. All 1,236 benchmark requests succeeded with the required provider and resolved model.
- Pinned GLiNER2.5 Multi executed locally on the same 483 commands, plus 30 warm-ups and 723 timed CPU calls. Logit/probability agreement, validation-only calibration, runtime metrics, timing samples, and unchanged retained controls were verified.
- GLiNER preprocessing was audited on every row. A case-sensitive flag collision and three native-API parity probes were exercised and retained separately from benchmark quality.
- Corpus row/group/source counts were checked. The data recommendation distinguishes training, calibration/policy development, and fresh audit needs; no data-size learning-curve result is claimed.
- Data hashes, prediction alignment, normalized probabilities, recalibration, action metrics, latency percentiles, usage sums, and credential absence were verified from retained artifacts.
- No new training, prompt search, production deployment, or changes to application source code.
- No credential value or authorization header was saved in the benchmark outputs.
