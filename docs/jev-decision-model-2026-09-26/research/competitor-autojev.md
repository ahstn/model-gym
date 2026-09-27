# AutoJev-27B: reproducible mechanism, incomplete training provenance

## Verdict and corrections to the brief

**AutoJev is a strong reusable implementation, not a fully reproducible training experiment.** Its most valuable lesson is a conventional full-backbone decision SFT with a tiny vocabulary-sliced readout, separate temperature fitting, and careful data plumbing—not an exotic architecture or successful RL recipe. Exact curated training data are absent. [README](https://github.com/denis-pplx/autojev), [recipe](https://github.com/denis-pplx/autojev/blob/main/configs/training.json), [model](https://github.com/denis-pplx/autojev/blob/main/src/autojev/model.py).

**The brief's #1-open ranking is historical, not current.** Archived v0.2 records AutoJev 50.94 versus Rune 47.23; live v0.2.1 records Jev 57.89, Rune v3 57.44, AutoJev 56.40, Decider chat 51.35, Winnow-12B 50.02. The live index has changed panel/weighting and Rune checkpoint; do not interpret AutoJev's score increase as a model improvement. Loaded live `index.json` generated 2026-09-26T11:54:33Z. [Archived index](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/index-v2.json), [current index](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/index.json), [methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/methodology.json).

**Checkpoint-selection exposure is verified, and stronger than a generic overlap disclaimer.** Author explicitly calls the 4,700-row panel `monitored_validation_used_for_checkpoint_selection`, says panel membership was amended after earlier results, and disclaims an untouched test and statistical significance. Board explicitly flags WinoGrande selection and 750 BBH / 1,500 RAGTruth selection rows. This is evaluation/model-selection exposure, not evidence those rows received gradient training; exact overlap counts with each board sample are not established by these declarations. [Author results](https://github.com/denis-pplx/autojev/blob/main/assets/results.json), [board contamination notes](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/methodology.json).

No requested primary URL 404ed. Content differs from brief on current ranking/version. A secondary web-search summary incorrectly implied $3,100 was H200 training compute: the primary author says **$1,900 agents + $1,200 data**, with the swarm running 20 hours. GPU rental cost is not itemized. [Primary announcement](https://x.com/denisyarats/status/2102252088067850507), [secondary recap](https://mattheworiordan.github.io/jev-landscape/).

## Exact training and readout

| Component | Verified detail |
|---|---|
| Base | `Qwen/Qwen3.8-27B`, immutable revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Model implementation | `Qwen3_5ForConditionalGeneration` supplies original weights; retained `Qwen3_5Model` backbone plus new linear decision readout |
| Train mode | All retained weights trainable, including vision and decision readout; original full language head discarded |
| Objective | Cross-entropy over active candidate positions; hard labels or supplied soft distributions; not next-token loss over all prompt tokens |
| Learning rate | Peak/base `2e-6`; warmup first `floor(0.05 * total_steps)` (14 of 286) then cosine to 0.1 times base |
| Epoch/updates | One epoch over 73,000 unique rows, 286 updates; selected update 200, 51,200 examples / 70.137% epoch |
| Batch | Effective 256; microbatch maximum 32, additionally limited by estimated padded-token budget 8,192; therefore not fixed 32 × 8 accumulation |
| Length | Max input 8,192, no silent truncation; overlong inference raises error |
| Optimizer | Custom CPUOffloadAdamW wrapping fused PyTorch AdamW; weight decay .01; FP32 CPU master weights and moments, CPU FP32 gradient buffers |
| Precision | CUDA model/readout BF16; readout logits converted FP32; CPU fallback FP32; no 4-bit training in recipe |
| Stability | Non-reentrant gradient checkpointing; SDPA; global gradient-norm clipping 1.0; finite checks |
| Shuffle | Choice candidate order randomized per step, preserving/remapping soft targets |
| Seed/CPU | Seed 20260920; 32 CPU threads |
| Evaluation | Temperature fit and monitored-panel eval every 50 updates and final; selected by accuracy, then Brier, then earlier step among calibration-eligible checkpoints |
| Historical trajectory | Fresh base; exact optimizer pause/resume at update 50 while panel changed; one continuous trajectory, not two SFT stages |

Sources for all table rows: [training metadata](https://github.com/denis-pplx/autojev/blob/main/configs/training.json), [launcher](https://github.com/denis-pplx/autojev/blob/main/configs/train.sh), [trainer](https://github.com/denis-pplx/autojev/blob/main/src/autojev/train.py), [optimizer](https://github.com/denis-pplx/autojev/blob/main/src/autojev/optim.py), [model](https://github.com/denis-pplx/autojev/blob/main/src/autojev/model.py), [released NOTICE](https://huggingface.co/denis-pplx/autojev-27b/raw/main/NOTICE).

### Not merely constrained text generation

The loader finds 255 distinct single-token codes from `A..Z`, then uppercase pairs, validates that they remain single tokens after the chat prefix, and initializes a bias-free `Linear(hidden_size,255)` from those rows of the original LM head. The backbone's final-position hidden state produces these 255 logits; invalid positions are masked to -1e9; `softmax(logits/T)` produces normalized candidate probabilities. No decoding loop, no generated answer string, no full-vocabulary normalization. This is a vocabulary-initialized classification head with full-backbone training. The board categorizes the technique as autoregressive, but operationally this path is one forward pass per question. [model.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/model.py), [board latency methods](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/methodology.json).

Prompt: system instruction to classify state as data, then `State`, `Question`, `Options`, followed by “Return only the letter code of the best option.” Chat template uses `enable_thinking=False`. Runtime choice preserves supplied key order; noul is false/true; score maps ordinal levels to probabilities and returns their expectation. Choice confidence is chance-adjusted winning probability, not raw max probability; evaluation ECE uses raw chosen-option probability. Score confidence is a separate distance-based heuristic. Do not calibrate the user-facing `confidence` field as though it were correctness probability. Each question repeats the shared state; no shared-prefix multi-question encoder is implemented here. [model.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/model.py), [evaluate.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/evaluate.py).

**Typed API support is broader than the training/evaluation proof.** The model supports score, but the published accuracy/calibration pipeline's `options()` explicitly accepts only Choice/Noul. Score support is not proof of trained ordinal calibration. Author's tweet mentions 260k base context, while the published DecisionModel default hard limit is 8,192; distinguish backbone capability from shipped serving path. Image support is explicitly not a natural-image accuracy claim. [evaluate.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/evaluate.py), [model.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/model.py), [announcement](https://x.com/denisyarats/status/2102252088067850507), [card](https://huggingface.co/denis-pplx/autojev-27b).

## Training data: what is known and what is not

**The 73,000-row curated release corpus and images are not bundled; its exact per-source composition is UNVERIFIED.** A SHA-256 commitment is present but does not reveal the records or mixture. Do not mistake either of these published builders for the released corpus. [recipe reproduction limits](https://github.com/denis-pplx/autojev/blob/main/configs/training.json).

### Earlier public-source builder (nominal 96,000 rows)

| Source | Training quota |
|---|---:|
| VitaminC | 16,000 |
| MASSIVE en-US | 10,000 |
| MASSIVE de-DE | 10,000 |
| BoolQ | 7,000 |
| SQuAD2 | 14,000 |
| PAWS | 12,000 |
| MultiNLI | 12,000 |
| Civil Comments | 8,000 |
| Aegis2 | 6,000 |
| PubMedQA | 0 (reserved evaluation) |
| Synthetic controlled-geometry images | 1,000 default |

These are builder quotas, not audited actual curated counts. Public text quotas sum 95,000; default image retention adds 1,000. Whole-family sampling, near-duplicate removal, conflicts, and overlength filtering can lower realized counts. It uses pinned HF revisions/archive checksums and pinned Nimble converters; label-blind family selection, shared MASSIVE translation IDs, PAWS connected sentence/token-bag families, normalized passages, exact and approximate 5-word shingle overlap checks. Soft human annotations are used where supplied; not teacher-confidence targets for this builder. [data.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/data.py).

### Published synthetic and continuation machinery

- `synthetic.py` defines `gpt-5.6-sol`, medium reasoning, fixed 1,000-example transfer generation, diverse languages/conditions, separate blind-label call with opaque IDs, same-model label agreement, ambiguity rejection, length checks, up to three generation attempts. It explicitly labels this **not human gold**. API usage is written outside the public repo. [synthetic.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/synthetic.py).
- `sft_synthetic.py` plans 50,000 continuation examples: NLI 18k, intent 10k, numeric 10k, news 8k, evidence 4k, separated domains for train/calibration/confirmation and controlled phenomena such as negation, exceptions, missing premises, entity mismatches. [sft_synthetic.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/sft_synthetic.py).
- `sft_pipeline.py` has ten tranches of 5,000 new + 1,920 replay examples, total 69,200; frozen row order, SHA-bound schedule, atomic readiness receipts and audits. Trainer waits for the next audited tranche without changing frozen ordering; checks protected family/ID separation. This is an explicit data-production/consumption pipeline, not proof the released 73k run used the 69.2k schedule. [pipeline](https://github.com/denis-pplx/autojev/blob/main/src/autojev/sft_pipeline.py), [trainer](https://github.com/denis-pplx/autojev/blob/main/src/autojev/train.py), [historical recipe](https://github.com/denis-pplx/autojev/blob/main/configs/training.json).

### What “auto” actually means

Primary author: internal swarm of Astra and Fable; Sol and Luna used for synthetic data/filtering; swarm tried RL without success and fell back to SFT. User set goals/refined scope; agents performed research/data/training/eval/deployment. The public repo publishes the resulting scripts and controlled generation/checkpoint-selection loops, not the internal research swarm controller, complete research-agent prompts/transcripts, RL failure logs, or an autonomous architecture-search framework. [announcement](https://x.com/denisyarats/status/2102252088067850507), [README and file inventory](https://github.com/denis-pplx/autojev). **[INFERENCE] Copying these scripts reproduces a training workflow, not the agent search that discovered the curated recipe.**

## Calibration and selection

Fit one positive scalar T by minimizing **hard-label NLL** on 3,500 separate temperature rows. Implementation does 80 golden-section iterations in log T over [0.05,20], checking boundaries and T=1. Released T is **2.207568021892729**. Fit is repeated per checkpoint; evaluation panel never directly fits T. Selection requires ECE and Brier no more than Jev reference +0.010, then maximizes monitored accuracy, tie-breaking lower Brier then earlier step. Optional guard fold in current code adds another eligibility gate; this is not evidence the historical run used it. [evaluate.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/evaluate.py), [train.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/train.py), [recipe](https://github.com/denis-pplx/autojev/blob/main/configs/training.json).

At selected step 200, raw→fitted: ECE .058733→.042817; Brier .224004→.220267; NLL .403713→.371548; accuracy remains .845957. **Calibration is not uniformly better:** RAGTruth ECE .03587→.07254, Brier .16972→.17919; BBH ECE .05655→.07598. A scalar globally improves the mixed panel while harming individual task distributions. ECE is 15 equal-width bins; Brier sums over classes then averages rows. [all-checkpoints.csv](https://github.com/denis-pplx/autojev/blob/main/assets/all-checkpoints.csv).

## Results: author panel versus common board

### Author-monitored 4,700-row panel (accuracy %)

| Benchmark | n | Base Qwen readout | AutoJev step 200 | Jev |
|---|---:|---:|---:|---:|
| BBH |750|72.93|82.80|94.27|
| Financial PhraseBank |999|75.68|84.18|76.98|
| JevBench public hard |101|73.27|70.30|73.27|
| JudgeBench |350|69.14|78.86|78.57|
| RAGTruth |1500|62.13|88.93|77.27|
| WinoGrande |1000|73.10|83.30|90.70|
| Micro overall |4700|69.83|84.60|82.79|

Source: [results.json](https://github.com/denis-pplx/autojev/blob/main/assets/results.json). Jev here is `typesafe/jev-1.13-20260917`. AutoJev beats Jev by 85 decisions / 1.8085 percentage points **micro**, but loses **six-suite macro** 81.3953% versus 81.8415%. Public-hard includes six score-to-choice adaptations and is not official JevBench composite. One seed, no selected-comparison uncertainty interval, panel amended after early results; older step-360 results use different panels. Source: same results file.

**[INFERENCE, arithmetic from table] The headline win is mixture-sensitive:** RAGTruth contributes +175 correct versus Jev, exceeding the entire +85 overall gain. Removing RAGTruth reverses the net to -90/3,200 (-2.8125 pp). This does not invalidate useful RAG improvements; it invalidates treating “beats Jev” as a task-independent conclusion. [results.json](https://github.com/denis-pplx/autojev/blob/main/assets/results.json).

### Common board v0.2.1: every available benchmark result

Values are native metric units, not comparable across rows; ForecastBench is lower-better. RouterBench and SGD/SGD-X remain in result records but were removed from the v0.2.1 headline index. Other recorded benchmarks can also be outside its 38-task index. Source for every table row: [current index JSON](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/index.json); inclusion rules: [methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/methodology.json).

| Benchmark | Metric | AutoJev | Jev |
|---|---|---:|---:|
|BFCL|case exact accuracy|0.9764|0.9575|
|ToolRet|nDCG@10|0.6742|0.6528|
|API-Bank|accuracy|0.8406|0.8819|
|BANKING77|macro-F1|0.7903|0.7974|
|CLINC150+OOS|macro-F1|0.8780|0.8927|
|RouterBench|selected quality|0.7981|0.7992|
|Home appliance simulator|case exact accuracy|0.7386|0.5227|
|SGD/SGD-X|macro-F1|0.3976|0.4295|
|ContractNLI|macro-F1|0.7815|0.7169|
|ANLI|macro-F1|0.7065|0.7479|
|BPoMP|accuracy|0.9366|0.9060|
|Humicroedit|accuracy|0.6240|0.6187|
|POP909-CL|accuracy|0.3880|0.1810|
|cfcolor|accuracy|0.6546|0.6474|
|MMLU|accuracy|0.8401|0.9173|
|GPQA Diamond|accuracy|0.4949|0.7857|
|ARC-Easy|accuracy|0.9920|0.9933|
|ARC-Challenge|accuracy|0.9676|0.9778|
|WinoGrande|accuracy|0.8516|0.9195|
|HellaSwag|accuracy|0.9398|0.9452|
|GSM8K|accuracy|0.6736|0.7987|
|ChessBench|accuracy|0.1720|0.1722|
|MuSR|accuracy|0.6184|0.6609|
|SATA-Bench|case exact accuracy|0.2988|0.2642|
|BRIGHT|nDCG@10|0.4866|0.4752|
|Amazon ESCI|macro-F1|0.5509|0.5521|
|ACOS|per-review F1|0.2040|0.2952|
|FinEntity|macro-F1|0.9290|0.8698|
|iSarcasmEval|Sarcasm F1 English A|0.6034|0.5051|
|VAST|macro-F1|0.7079|0.6463|
|NLI4CT|macro-F1|0.8481|0.8406|
|CRUXEval|accuracy|0.7491|0.7298|
|CLadder|accuracy|0.7450|0.7264|
|HLE|accuracy|0.1198|0.2036|
|ForecastBench|Brier ↓|0.1948|0.1736|
|Habermas Machine|accuracy|0.4177|0.4594|
|PhishNChips|accuracy|0.5995|0.6255|
|MMLU-Pro|accuracy|0.6448|0.8270|
|BBH|accuracy|0.7810|0.9292|
|When2Call|accuracy|0.8168|0.8097|
|New Yorker|accuracy|0.7027|0.7008|
|RAGTruth|hallucinated-class F1|0.8034|0.7653|
|HoVer|accuracy|0.7422|0.7285|

Board records all 120,340 requests answered, no gap/lift; latency median 104.2 ms, p95 1,168.0 ms, mean 428.6 ms, in-process synchronized forward on RTX PRO 6000. Calibration summary: accuracy .7295, confidence .7375, ECE .0178, Brier .361 over 72,594 scored decisions / 32 benchmarks. Area ECE ranges from .0232 knowledge to .0879 retrieval: global ECE is not a uniform trust certificate. [index](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/index.json), [timing and calibration methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/methodology.json).

## Hardware, time, cost, licensing

- One H200 is author-confirmed. Swarm ran 20h, $3.1k total stated as $1.9k agents + $1.2k synthetic data. **This is not a measured 20h GPU training duration or a $3.1k GPU invoice.** No published standalone GPU-hours/rental rate/CPU RAM/energy cost found in loaded release assets. Author's ~120ms p95 H200 claim has unspecified request mix and differs from common-board broad-suite p95; do not compare directly. [primary announcement](https://x.com/denisyarats/status/2102252088067850507), [recipe](https://github.com/denis-pplx/autojev/blob/main/configs/training.json).
- GPU inference requires roughly 49 GiB BF16 weights plus overhead; selected artifact has 11 backbone shards and a 2.61MB readout. [card](https://huggingface.co/denis-pplx/autojev-27b), [Hub inventory](https://huggingface.co/api/models/denis-pplx/autojev-27b/tree/main).
- **[INFERENCE] “One H200” hides substantial host-memory needs.** For ~26.1B retained parameters, CPU FP32 masters + two Adam moments + gradient buffers alone approach `26.1e9 * 16 bytes = 417.6 GB` decimal, before loading/resume overhead. Local 60GB RAM cannot reproduce this offload recipe; 24GB VRAM also cannot hold the released BF16 model. Smaller full FT still requires an explicit host-memory plan rather than assuming GPU fit means training fit. [optimizer allocations](https://github.com/denis-pplx/autojev/blob/main/src/autojev/optim.py), [board retained-parameter metadata](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/methodology.json).
- Code MIT; weights Apache 2.0 with base attribution/modification notice. Reusable readout, optimizer, data-selection machinery, evaluation/calibration, server/API; preserve notices. Training-source licenses remain per-source and curated data are absent; weights/code licensing does not supply rights or reproducibility for unbundled data. [code license](https://github.com/denis-pplx/autojev/blob/main/LICENSE), [weights license](https://huggingface.co/denis-pplx/autojev-27b/blob/main/LICENSE), [NOTICE](https://huggingface.co/denis-pplx/autojev-27b/raw/main/NOTICE), [data provenance](https://github.com/denis-pplx/autojev/blob/main/src/autojev/data.py).

## Why it led, and what a smaller model can copy

**Measured:** fixed-prompt/readout base→selected accuracy improves 69.83→84.60 on the monitored panel; temperature improves probabilities but cannot change positive-temperature argmax; 255-option coverage avoids unsupported-answer penalties; training is full retained-weight SFT. **Not measured:** an ablation isolating full FT versus LoRA, curated data versus quantity, 27B versus smaller bases, or readout design versus an ordinary constrained first token. Therefore claims that any one ingredient caused its leaderboard lead are **[INFERENCE]**, not established science. [results](https://github.com/denis-pplx/autojev/blob/main/assets/results.json), [model](https://github.com/denis-pplx/autojev/blob/main/src/autojev/model.py), [board](https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/index-v2.json).

**[INFERENCE] Smaller bases can replicate the mechanism essentially completely, but “most of the accuracy” is an experiment, not a fact.** Runtime-defined labels, sliced head, CE, choice shuffling and temperature fit are size-independent; reasoning/knowledge transfer and calibration out of domain are not guaranteed. Current loader is Qwen-VL-specific and insists on 255 single-token codes, so MiniCPM/K2 ports need their own backbone loader/tokenizer validation, not merely changing a model ID. [model.py](https://github.com/denis-pplx/autojev/blob/main/src/autojev/model.py).

### Top three actionable trial recommendations

1. **Reproduce the mechanism, not the 27B spend first.** On one small-base candidate, compare frozen native candidate logits, LoRA decision CE, and full retained-weight CE using the exact same data/prompt/255-code readout and early-stop protocol. Start with a short data pilot and inspect both task transfer and step-50 saturation before increasing scale. This tests the alleged full-FT benefit directly. [Mechanistic basis](https://github.com/denis-pplx/autojev/blob/main/src/autojev/model.py), [early checkpoint gains](https://github.com/denis-pplx/autojev/blob/main/assets/all-checkpoints.csv). **[INFERENCE/recommendation]**
2. **Freeze task-family holdouts before curation and checkpoint selection.** Keep command-risk templates, candidate-description paraphrases, and whole source families disjoint; create untouched final tasks, plus explicit label-order/label-name perturbations. Report macro task accuracy, command-risk cost, Brier/NLL and per-domain ECE with/without scalar T. Do not select on the public common board. This targets AutoJev's documented weaknesses rather than copying them. [selection exposure](https://github.com/denis-pplx/autojev/blob/main/assets/results.json), [calibration heterogeneity](https://github.com/denis-pplx/autojev/blob/main/assets/all-checkpoints.csv). **[INFERENCE/recommendation]**
3. **Use AutoJev as a reference/teacher candidate, not sole gold or proof that 27B rental is necessary.** Rent an inference-capable GPU only for a bounded teacher/reference sweep; audit disagreements with independent human/programmatic labels. Defer H200 full FT until the small-base ablation identifies a genuine capacity gap. If renting full-FT hardware, specify host RAM alongside GPU and distinguish agent/data budget from GPU cost. [same-model synthetic caveat](https://github.com/denis-pplx/autojev/blob/main/src/autojev/synthetic.py), [memory mechanism](https://github.com/denis-pplx/autojev/blob/main/src/autojev/optim.py), [actual budget breakdown](https://x.com/denisyarats/status/2102252088067850507). **[INFERENCE/recommendation]**

## Research verification scope

Read primary HF/GitHub metadata and Python implementations; programmatically parsed the result JSON and both index versions to extract numeric comparisons and all 43 common-board result rows. No model weights or datasets downloaded; no inference/training run, repo modification, or tests performed. Historical curated mixture, exact GPU training hours/rental bill, and proprietary research-agent loop remain unavailable in the loaded public release; no claims of independent metric reproduction.
