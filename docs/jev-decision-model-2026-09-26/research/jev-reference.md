# Jev reference analysis (TypeSafe jev-1.13.0)

Sources are linked inline. Tags: [INFERENCE] = our reasoning, not stated by a source. [3P] = third-party observation from outside the API, not confirmed by TypeSafe.

**Brief check:** both brief URLs load. `docs.typesafe.ai/models.md` matches the brief. The blog page header says "published Sep 26, 2026" (Framer metadata), but the body date is **Sep 15, 2026**. The launch was about 11 days before today. The FAQ answers on the blog are hidden in a collapsed accordion. I pulled them out of the page's embedded Framer JSON. They are quoted below and are not visible in the reader view.

Peer note (research-astra-1): live leaderboard data is **v0.2.1**, not v0.2. It has Jev 57.89, Rune v3 57.44, AutoJev 56.40, 38 index tasks, and RouterBench+SGD were removed. The brief's numbers (Jev 51.7 …) are v0.2.

---

## 1. Interface spec we must match

Endpoint: `POST https://api.typesafe.ai/v1/systemone`, Bearer auth ([api](https://docs.typesafe.ai/api)).

### Request
| field | type | notes |
|---|---|---|
| `state` | string \| object \| array | text only: strings, JSON objects, arrays of text. No images, audio or video ([models](https://docs.typesafe.ai/models.md), [system-one](https://docs.typesafe.ai/concepts/system-one.md)) |
| `model` | string | `jev-latest` / `jev-preview` → `jev-1.13.0` |
| `questions` | map<id, Question> | **The question id "is not sent to the underlying model and is not used in inference"** ([api](https://docs.typesafe.ai/api)) |

Question types. All have `type` and `instructions: string|object|array|null` ([advanced](https://docs.typesafe.ai/primitives/advanced.md)). Instructions may be structured objects that point at named fields with backtick refs, e.g. ``"Is the resume for the same person as `potential_duplicate`?"``, and may point into nested `state` paths such as `` `items[3]` `` ([jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)).

| type | criteria | limits |
|---|---|---|
| `noul` | optional `{true: desc, false: desc}`. Each desc is str/obj/array/null | — |
| `choice` | **required** map option→desc (desc may be null, str, obj or array) | **max 255 options** |
| `score` | **required** ordered array of level descriptions (low→high), level index = array position starting at 0 | ≥2 recommended, **max 10** |

### Response
```json
{"model":"jev-1.13.0",
 "answers":{
  "q_noul":  {"type":"noul","noul":0.95},
  "q_choice":{"type":"choice","choice":"billing","probabilities":{"billing":0.88,"technical":0.12,"sales":0.0},"confidence":0.81},
  "q_score": {"type":"score","score":1.05,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.0,"1":0.95,"2":0.05},"confidence":0.92}},
 "usage":{"input_tokens":304,"output_tokens":18}}
```
([api](https://docs.typesafe.ai/api))

- `choice` is the argmax. Probabilities sum to 1.
- `score` = Σ level·p(level), the expected level. It can fall between levels ([score](https://docs.typesafe.ai/primitives/score.md)). Docs warn that score levels are "weak in numerical calibration" for interpolation ([jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)).
- Noul has **no confidence** field.
- **Confidence is a deterministic function of the probabilities, not a learned output** ([confidence](https://docs.typesafe.ai/confidence.md); official adapter [confidence_metrics.py](https://raw.githubusercontent.com/typesafe-ai/system-one-adapter-python/main/src/system_one_adapter/_utils/confidence_metrics.py)):
  - Choice: `(p_max − 1/K)/(1 − 1/K)`. K=1 → 1.
  - Score: `max(0, 1 − Σ p_i·|i−mode| / MAD_uniform)`, where `MAD_uniform = mean_i |i − (K−1)/2|`.
  - Sanity check: the docs' score example gives 1 − 0.05/0.667 = 0.925 ≈ 0.92 ✓. The choice example gives 0.82 against a reported 0.81. That is a rounding gap, so the API probably computes from unrounded probabilities [INFERENCE].
- Probabilities are returned at 2-decimal precision ([archerhume methods](https://archerhume.com/posts/jevs-architecture-unmasked/)) [3P].
- `output_tokens` is a **billing/serialization count** and does not measure decoding. Archerhume fits it as 4 shared + 15 per noul answer + tokens in the question-id. A 255-option answer counted 2,714 output tokens. It depends on the id text, which the model never sees ([archerhume §1](https://archerhume.com/posts/jevs-architecture-unmasked/)) [3P].
- Errors: 401 / 422 (validation) / 429 / 529 ([api](https://docs.typesafe.ai/api)).

### Semantics we must reproduce
1. **Question isolation.** "Every question is evaluated in parallel and in isolation against the same state"; "One question's answer is not hidden context for another. You can add or remove questions without changing the others' results" ([intro](https://docs.typesafe.ai/introduction), [primitives](https://docs.typesafe.ai/primitives.md)).
   - The Parallel-questions cookbook tested 13 questions batched vs single across 5 repeats. Means match, and "batching neither shifts the answer nor adds variance" ([cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions.md)).
   - Third-party check: a secret placed in a sibling question gives p=0.00 visibility, while the same secret in the state gives 0.90–0.92 ([archerhume §2](https://archerhume.com/posts/jevs-architecture-unmasked/)) [3P].
2. **Options interact within a Choice (listwise).** Adding an irrelevant 5th option moved log-odds between two existing options from +0.38 to +0.11 in 10/10 blocks. This rules out independent per-option logits under a fixed softmax. Options are also **order-sensitive**: reversing the order moved p from 0.84–0.89 to 0.93–0.96. A reference card is read correctly 16/16 times when placed last, but only 11–12/16 when first or middle ([archerhume §3–4](https://archerhume.com/posts/jevs-architecture-unmasked/)) [3P]. Fake options injected via delimiters never displaced real ones, so option boundaries are structural and not forgeable text [3P].
3. **No structural invariants are guaranteed.** Noul("refund") = 0.22 while Choice yes = 0.01. p(q) + p(¬q) = 1.19 ([jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)). Our model does not need to be *more* coherent than Jev to match it. Coherence is still a cheap place for us to beat it [INFERENCE].
4. **Not strictly deterministic.** Most answers had std 0.0 across 5 repeats, but some Nouls show "run-to-run sampling noise" of about 0.005 ([cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions.md)). The CEO "will not commit to deterministic models" ([Latent Space](https://www.latent.space/p/jev)).
5. **>255 options** is handled client-side: "2 stage-system of scoring independently then making an explicit choice" ([blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)). The [skill-suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion.md) and [hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification.md) cookbooks show the same pattern.
6. **Noul is absolute; Choice is relative.** "Each Noul is absolute and can be low for all of them" ([jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)). "Noul" is short for **Bernoulli** (CEO on [HN](https://news.ycombinator.com/item?id=49718407)).

## 2. Operational limits and claims

| property | claim | source |
|---|---|---|
| Context | **64k tokens per request** (state + all questions). **32k for state + the longest single question** | [models](https://docs.typesafe.ai/models.md) |
| Context (probed) | about 32,768 per branch and about 65,536 per request. A 23k-token state with 5,000 questions fits | [archerhume](https://archerhume.com/posts/jevs-architecture-unmasked/) [3P] |
| Price | $0.042 / Mtok input. Output free | [models](https://docs.typesafe.ai/models.md) |
| Rate limit | 250k tok/s, 1,200 RPM (adjusting dynamically) | [models](https://docs.typesafe.ai/models.md) |
| Latency | "70ms–500ms" end-to-end. "40x–200x faster". Workflow evals: "193.6x faster, 444.6x cheaper" (self-described as the high end) | [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |
| Latency (measured) | 13 questions over a ~54k-char article in 0.27 s ($0.000497); 13 separate calls took 2.71 s | [cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions.md) |
| Latency (probed) | flat up to ~100 questions, then rising. Question tokens cost ~2× state tokens. ~30k tokens in ~160 ms. Latency does not depend on option count (2 vs 200) | [archerhume](https://archerhume.com/posts/jevs-architecture-unmasked/) [3P] |
| Our own measurement | p50 311 ms on the command-risk set | repo `docs/jev-laya-comparison-2026-09-19.md` (per brief) |
| Calibration | "calibrated probabilities", with no public number from TypeSafe | [primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md) |
| Calibration (probed) | MMLU 1,200 items: 10-bin **ECE 0.031**, with 990 of 1,200 in the 0.9–1.0 bin. MMLU-Pro acc 84.6%. Fresh 3-digit multiplication: acc 86.7% at mean p 0.83. Two-step word problems: 32% at p 0.30 | [archerhume §5](https://archerhume.com/posts/jevs-architecture-unmasked/) [3P] |
| Hallucination / type errors | "0%" by construction (schema guarantee, "not empirical") | [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |
| Language | English is primary. Other languages including CJK are weaker | [models](https://docs.typesafe.ai/models.md) |
| Public benchmarks | deliberately not published | blog FAQ (extracted) |

Known weaknesses of jev-1.13 are listed on the [jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md):
- literal reading, including negations taken at face value
- counting and math
- date comparison
- numeric representations (hex/RGB)
- multi-hop indirection
- **context rot from irrelevant state**
- adversarial or injected content in state
- contradictory instructions vs criteria (e.g. a Noul with true↔false swapped)
- generation

We should mirror these slices in our eval and could beat Jev on some of them [INFERENCE].

## 3. Every architectural and training hint (official)

| hint | quote / source |
|---|---|
| New architecture + **parallel sampler** + RLCD | "a new model architecture, parallel sampler for maximum efficiency, and training method we call Reinforcement Learning for Calibrated Decisions (RLCD)" ([blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)) |
| Non-autoregressive output | "Jev outputs all probabilities in parallel instead of autoregressively generating by token"; "Generates all outputs in a single query" (blog) |
| Not small, not an LLM | FAQ: "Jev is neither small nor an LLM, hence being off the intelligence Pareto curve." |
| Built on a pretrained LM | Primer diagram: "Pretrained language models branch into … RLHF and RLVR paths and an emphasized RLCD decision-model path" ([primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)). CEO: "if you gave me a billion dollars, I wouldn't pre-train … Anything except pre-training"; "Frankensteining … solves problems" ([Latent Space](https://www.latent.space/p/jev)) |
| RLCD objective | "optimizes for calibrated decisions: answers with epistemically honest probabilities"; "probabilities are optimized against outcomes" ([system-one](https://docs.typesafe.ai/concepts/system-one.md)). Latent Space calls it "a novel, unpublished technique" |
| Data | "We make all the data ourselves" (FAQ). CEO confirms "all your data is synthetic": "Yep", curated by a "data team" with taste; "data is probably far more interesting than architecture" ([Latent Space](https://www.latent.space/p/jev), [HN](https://news.ycombinator.com/item?id=49718824)) |
| Eval philosophy | Workflow evals use the **average of GPT-6 Astra + Fable 5.1 probabilities as the reference distribution**. They are not ground-truth labels ([blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [evals.typesafe.ai](https://evals.typesafe.ai/)). This suggests teacher-distribution targets are how they think about correctness [INFERENCE] |
| State ingested once | "Jev ingests the `state` once and evaluates every question against it in parallel" ([models](https://docs.typesafe.ai/models.md)) |
| Context rot | "Jev suffers from context rot" ([jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)) |
| Same weights for everyone, no per-customer tuning | [models](https://docs.typesafe.ai/models.md) |
| Architecture secret | CEO: "architecture is close to the chest for now, but we have talked about writing a paper" ([HN](https://news.ycombinator.com/item?id=49718824)) |
| KV-cache interest | CEO essay "Tyranny of the KV Cache", linked from [Latent Space](https://www.latent.space/p/jev) [UNVERIFIED: Google Doc, not loaded] |

## 4. Community reverse-engineering
- **Archer Hume, "Jev's Architecture Unmasked"** ([link](https://archerhume.com/posts/jevs-architecture-unmasked/)). This is the most rigorous public probe: about 10k API calls, with evidence JSON published. Conclusions:
  - a causal transformer with a **shared state prefix and isolated question suffixes** (a tree/prefix-cache like Hydragen/DeFT)
  - a **listwise option block followed by one decision position**, with a slot head (256 slots) or a pointer head. He leans pointer: a copied answer scored 1.00 at every position of a 200-option list.
  - sparse MoE of about 10B active parameters (speculative)
  - trained on proper scoring rules
  - tokenizer matches none of 192 public tokenizers. It is close to o200k, with per-digit splitting; Qwen matches 348/415 probes. That points to a custom or modified tokenizer.
- **dev.to "I Rebuilt Jev's Structure with Qwen"** ([link](https://dev.to/sennalang/i-rebuilt-jevs-structure-with-qwen-not-its-capabilities-2o1d)). One packed forward pass with a tree attention mask, **RoPE position ids reset per branch after the state**, a pointer head for Choice/Score and a linear+sigmoid Noul head. Its AG News pointer head alone reached acc 0.804 with **ECE 0.198**, which shows that calibration needs real training.
- **HN launch thread** ([49717558](https://news.ycombinator.com/item?id=49717558), about 1,981 points):
  - speculation about encoder-only with scalar/ordinal heads (quotemstr) and diffusion (theredsix)
  - cooljoseph's recipe: LLM backbone, tree position embedding + graph-sparse attention, unembedding replaced by scalar logit + confidence heads
  - porridgeraisin's description of calibrated RL: penalise confident-wrong distributions in proportion (i.e., a proper scoring reward)
- **Arcturus Labs** ([link](https://arcturus-labs.com/blog/2026/09/16/typesafes-jev-trades-text-generation-for-instant-calibrated-decisions/)) [UNVERIFIED beyond search summary]: slot or logprob readouts on a fine-tuned LLM.
- **Simon Willison** ([link](https://simonwillison.net/2026/Sep/21/jev/)):
  - bias/black-box concern
  - open recreations such as Kev (Qwen3.5 0.8B/4B/9B)
  - a JevBench at benchmarkheaven [UNVERIFIED]
  - `llm-typesafe` plugin
- **Official LLM adapter** ([system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python)). This is TypeSafe's reference way to get System One answers from LLMs:
  - `llm_answer_mode="probabilities"|"discrete"`
  - structured outputs vs prompted mode
  - probability normalization

  Useful as the **teacher/baseline harness** for distillation labels, since it matches the reference-answer methodology of TypeSafe's workflow evals.

## 5. Undisclosed
- Backbone identity and size: dense vs MoE, parameter count, base model (the tokenizer suggests it is custom).
- Encoder vs decoder, the attention mask, and how positions are encoded across branches.
- The readout head design (slot / pointer / vocab-token logprob) and how Score/Noul heads differ from Choice.
- The RLCD algorithm: reward (log/Brier?), on-policy or not, whether it uses teacher distributions or hard labels, KL regularization, and any post-hoc temperature scaling.
- Data: synthetic generation pipeline, task mix, volume, and teacher models.
- What the "parallel sampler" samples (the observed noise suggests some stochasticity or nondeterministic kernels).
- Serving hardware, precision/quantization, per-version changes (1.12→1.13), and any calibration metrics on any benchmark.

## 6. Implications for our model [INFERENCE]
- **Wire format:** implement exactly the schema above, including null descriptions, structured instructions and backtick references, and the 255/10 limits. Derive confidence with the adapter formulas and do not learn a separate confidence head. Keep question ids out of model input.
- **Architecture target:**
  - one prefill with the state as a shared prefix
  - each question as an isolated branch (block-sparse/tree mask with position ids continuing from the end of the state)
  - options listwise inside the branch with structural delimiters (special tokens)
  - readout at option-marker positions (pointer), with Noul as a sigmoid or 2-way head
  - score = expected level

  This fits a causal LLM with SDPA/flex-attention masks and needs no custom model.
- **Training:** a proper scoring loss (CE/Brier) on soft teacher distributions from frontier LLMs, then calibration (temperature scaling on held-out data, possibly per type). The RL stage is optional.
- **Eval must include:**
  - option-permutation invariance
  - irrelevant-option addition
  - isolation (adding siblings changes nothing)
  - distractor-heavy state (context rot)
  - negation/literal reading
  - p(q)+p(¬q) coherence
  - ECE per type
