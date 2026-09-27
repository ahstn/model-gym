# Jev Decision Index audit — GPT-6-Astra viewpoint

## Executive finding: the brief is already stale

The live JSON is **Decision Index 0.2.1**, generated 2026-09-26T11:54:33+00:00: **56 open entries**, **43 static benchmarks shown**, **38 counted**, **120,340 requests / 119,898 scoreable**, Jev `jev-1.13.0`. Jev is **57.89**, Rune v3 **57.44**, AutoJev **56.40**. The brief's 51.7/50.9/47.2 corresponds to the older 0.2 scoring/rune configuration, not today's live board. Rune v3 is now accepted on the common board: methodology says the author's submitted run was adopted after 3,000 rows matched exactly. No supplied leaderboard URL returned 404. **Content differs materially from the brief.** [Live results](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json) · [Methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json) · [Archived 0.2](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index-v2.json).

**Contrarian conclusion [INFERENCE]: do not optimize a single moving leaderboard scalar as the training objective.** This is a useful public stress suite, but not a clean held-out estimate of generalized decisions, calibrated probabilities, or our command-risk utility. The strongest evidence is the suite's own leakage discoveries, exposure disclosures, changing weights, and absence of interactive reproductions. [Methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json).

## What actually runs the Space

This is a **static HTML/JS app**, not a Python `app.py`: `index.html` loads the result bundle, `methodology.html` loads methodology, and `news.html` is editorial. The repo contains `data/index.json`, `index-v0.1.json`, `index-v2.json`, `index-v0.2.1.json` and matching methodology files; current aliases have the same object hashes as v0.2.1. There are no per-request `results.jsonl` files in the Space tree. The published JSON gives aggregate per-model/per-benchmark results plus internal provenance paths; I inspected those aggregates and app readout logic, **not unavailable full raw run files**. [Repository tree](https://huggingface.co/api/spaces/multimodalart/jev-decision-index/tree/main?recursive=true) · [App](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/index.html).

The Space README still describes 0.2, 40 benchmarks and equal areas. Even live `index.json` retains an obsolete equal-area `suite.formulas` field; its **`area_weights`, `gold`, `index_note`, actual scores and methodology formulas agree on 0.2.1**. I recomputed Rune 57.44, AutoJev 56.40 and Decider-chat 51.35 from stored per-benchmark skills and the new weights. Likewise methodology retains old static/reproduction benchmark counts (36/37) and `jev_benchmarks=42` despite updated result rows. Treat versioned executable/data semantics as authoritative, not every descriptive field. [README](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/README.md) · [Live results](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json) · [Methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json).

## Index definition

1. Native task metrics differ: accuracy, macro-F1, case-exact accuracy, per-review F1, nDCG@10, Brier loss, etc. Unsupported/refused/errored requests count as wrong against full denominators; linked cases require all chunks. Do not average successful-only metrics or multiply coverage twice on already track-adjusted metrics.
2. Map each task to `skill = clip((coverage-adjusted score - chance)/(1-chance), 0, 1)`. Task-specific chance comes from uniform choice, random whole-case answers, analytic F1, or expected random ranking. GSM8K's 4/10-choice tracks are corrected separately; ToolRet/BRIGHT baselines are per query, answerable queries only.
3. ForecastBench instead contributes `clip((0.25 - Brier)/0.25, 0, 1) × coverage`; p=0.5 is zero skill. RAGTruth's baseline is always predicting hallucinated (F1 0.5177), ACOS always yes (per-review F1 0.031).
4. Within an area gold tasks weight 1.2, other tasks 1.0, normalized by area weight sum. Overall `100 × Σ area_weight × weighted_area_skill`. Knowledge 0.258494; language 0.258494; retrieval 0.200229; tools 0.182783; arts 0.1. Arts is fixed; remaining 90% is proportional to square root of task count. Gold: GPQA, HLE, MMLU-Pro, BBH, ANLI, WinoGrande, HellaSwag, BANKING77, CLINC150, BRIGHT, BFCL, API-Bank, ForecastBench.
5. Raw index is secondary; the shifted geometric breadth index is computed but hidden. Calibration and latency **do not enter the headline index**.

All formula claims: [Methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json), [live stored skills](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json), [older kit scorer (0.2, not 0.2.1)](https://github.com/apolinario/decision-index/blob/main/decision_index/scoring/index02.py).

## All 43 benchmarks, category, metric and winners

Numbers below are **percent of each native metric after coverage adjustment**, except ForecastBench, which is raw Brier (lower better). Best-open comparison is exact on published rounded scores, not significance-tested. † means outside 0.2.1 index; ★ means 1.2× intra-area weight. Chess is folded into knowledge. Every row is derived from [live results](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json); definitions/explainers are in the same JSON. For task-level winners, the app likewise uses `score × answered/requests` except lower-is-better losses. [App source](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/index.html).

|ID|Benchmark|Area|Metric|Jev|Best open|Open score|Leader|
|---:|---|---|---|---:|---|---:|---|
|1|BFCL ★|tools|case exact accuracy|95.75|Solomon v1.1|97.99|Open|
|2|ToolRet|tools|nDCG@10|65.28|AutoJev-27B|67.42|Open|
|3|API-Bank ★|tools|accuracy|88.19|Jevfire|84.06|Jev|
|4|BANKING77 ★|retrieval|macro-F1|79.74|Decider 35B-A3B|89.79|Open|
|5|CLINC150 ★|retrieval|macro-F1|89.27|Decider 35B-A3B|94.68|Open|
|6|RouterBench †|tools|selected quality (quality objective)|79.92|Kev 4B|80.07|Open|
|9|Home appliances|tools|case exact accuracy|52.27|AutoJev-27B|73.86|Open|
|10|SGD †|retrieval|macro-F1|42.95|Surogate Rune 26B-A4B v3|66.40|Open|
|11|ContractNLI|language|macro-F1|71.69|AutoJev-27B|78.15|Open|
|12|ANLI ★|language|macro-F1|74.79|Decider chat · Qwen3.6-27B|74.59|Jev|
|20|BPoMP|arts|accuracy|90.60|Xor|96.62|Open|
|21|Humicroedit|arts|accuracy|61.87|Xor|63.74|Open|
|22|POP909|arts|accuracy|18.10|Surogate Rune 26B-A4B v3|66.80|Open|
|23|cfcolor|arts|accuracy|64.74|AutoJev-27B|65.46|Open|
|24|MMLU †|knowledge|accuracy|91.73|Decider chat · Qwen3.6-27B|85.21|Jev|
|25|GPQA Diamond ★|knowledge|accuracy|78.28|Decider chat · Qwen3.6-27B|54.59|Jev|
|26|ARC-Easy †|knowledge|accuracy|99.33|AutoJev-27B|99.20|Jev|
|27|ARC-Challenge †|knowledge|accuracy|97.78|Decider chat · Qwen3.6-27B|97.27|Jev|
|28|WinoGrande ★|language|accuracy|91.95|Decider 35B-A3B|88.63|Jev|
|29|HellaSwag ★|language|accuracy|94.52|Decider 35B-A3B|96.65|Open|
|30|GSM8K|knowledge|accuracy|79.87|Surogate Rune 26B-A4B v3|79.72|Jev|
|31|ChessBench|knowledge|accuracy|17.22|Surogate Rune 26B-A4B v3|18.74|Open|
|32|MuSR|knowledge|accuracy|66.01|Surogate Rune 26B-A4B v3|64.89|Jev|
|33|SATA-Bench|knowledge|case exact accuracy|26.42|Surogate Rune 26B-A4B v3|34.85|Open|
|36|BRIGHT ★|retrieval|nDCG@10|47.52|Jevfire|49.22|Open|
|37|Amazon ESCI|retrieval|macro-F1|55.21|djev|56.43|Open|
|38|ACOS|language|per-review F1|29.52|Decider 35B-A3B|27.24|Jev|
|39|FinEntity|language|macro-F1|86.98|AutoJev-27B|92.90|Open|
|40|iSarcasmEval|language|Sarcasm F1 · track A, English|50.51|JoshuaSP diffusiongemma (open-jev)|66.00|Open|
|41|VAST|language|macro-F1|64.63|Surogate Rune 26B-A4B v3|76.62|Open|
|42|NLI4CT|language|macro-F1|84.06|AutoJev-27B|84.81|Open|
|43|CRUXEval|knowledge|accuracy|72.98|AutoJev-27B|74.91|Open|
|44|CLadder|knowledge|accuracy|72.64|AutoJev-27B|74.50|Open|
|45|HLE ★|knowledge|accuracy|20.08|Qwen-2.5-1B-RLCD|17.17|Jev|
|48|ForecastBench ★|arts|Brier (lower is better)|0.1736|Kev 9B|0.1760|Jev|
|50|Habermas|arts|accuracy|45.94|razorback16 openjev diffusiongemma (NVFP4, vLLM)|47.61|Open|
|56|PhishNChips|retrieval|accuracy|62.55|djev|86.05|Open|
|57|MMLU-Pro ★|knowledge|accuracy|82.70|Surogate Rune 26B-A4B v3|67.20|Jev|
|58|BBH ★|knowledge|accuracy|92.92|Surogate Rune 26B-A4B v3|81.15|Jev|
|62|When2Call|tools|accuracy|80.97|Winnow-E4B|85.02|Open|
|64|New Yorker|arts|accuracy|70.08|Surogate Rune 26B-A4B v3|74.05|Open|
|59|RAGTruth|language|F1 on hallucinated class|76.53|AutoJev-27B|80.34|Open|
|61|HoVer|retrieval|accuracy|72.85|Surogate Rune 26B-A4B v3|80.67|Open|

**Winner count:** Jev: 14 (API-Bank, ANLI, MMLU, GPQA Diamond, ARC-Easy, ARC-Challenge, WinoGrande, GSM8K, MuSR, ACOS, HLE, ForecastBench, MMLU-Pro, BBH); Open: 29 (BFCL, ToolRet, BANKING77, CLINC150, RouterBench, Home appliances, SGD, ContractNLI, BPoMP, Humicroedit, POP909, cfcolor, HellaSwag, ChessBench, SATA-Bench, BRIGHT, Amazon ESCI, FinEntity, iSarcasmEval, VAST, NLI4CT, CRUXEval, CLadder, Habermas, PhishNChips, When2Call, New Yorker, RAGTruth, HoVer); Tie: 0 (). [Derived from live results](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json).

### Exclusions and the changing task definition

- Five displayed-but-not-counted tasks: MMLU, ARC-Easy, ARC-Challenge, RouterBench, SGD. RouterBench prompt tells the model GPT-4 is best; SGD builder includes future dialogue turns. This is evaluation leakage, not necessarily model misconduct.
- Six environments remain outside all models' indices: MiniWoB++, Boxoban, RTFM, ScienceWorld, Hanabi, Codenames. No reproduction trajectories have been run; Jev trajectories must not be reused.
- ToolRet counts only 685/1,000 sampled queries and BRIGHT 220/550 with a relevant candidate among the 32. Home appliances removes 24 duplicates and 48 rows matching published dev, leaving 88/160. ACOS changes from all-or-nothing to per-review F1.
- 442 global exclusions comprise 27 defective choice rows, 380 beyond nearly all entrants' context reach, 35 sibling chunks. The suite also contains five Jev-only calibration extras (OpenBookQA, CommonsenseQA, support tickets, phishing gradient, email spam); these are not the 43 common static tasks.
[Source for all exclusions](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json) · [Catalog](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json).

## Strongest parameter-tier entries

Uses **served total parameters**, not marketed/active size. In particular Winnow-E4B is ~8.00B served and Rune is ~25.81B total, not a <10B entry. Top three per tier, score then calibration/latency as independent tradeoffs. [Metadata/results](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json).

|Tier|Model|Served B|Index|ECE points ↓|Median ms|Method|
|---|---|---:|---:|---:|---:|---|
|<350M|GLiNER 2.5 base|0.194|6.76|36.70|14.4|full fine-tune|
|<350M|Decision 1.0 Kai|0.308|6.52|18.51|30.0|head / adapter|
|<350M|system-one-gemma|0.268|5.07|23.87|32.0|LoRA + head|
|350M–700M|Bosun v3.1 0.6B|0.596|14.32|14.19|106.7|LoRA + head|
|350M–700M|GLiNER2.5-Decide|0.486|11.21|8.84|92.8|full fine-tune|
|350M–700M|jeff|0.576|8.04|9.69|21.3|inference technique|
|700M–3B|Decider 2B|2.274|28.97|7.71|40.5|full fine-tune|
|700M–3B|this-that 1.2|1.882|28.14|19.83|38.6|full fine-tune|
|700M–3B|Decision 1.0 Sol|2.274|25.32|11.50|39.6|head / adapter|
|3B–10B|Decision 1.0 Lux|9.653|43.49|7.60|46.6|head / adapter|
|3B–10B|Decider 4B|4.660|40.70|8.37|22.5|full fine-tune|
|3B–10B|Winnow-E4B|7.996|39.89|5.80|168.7|LoRA|
|10B+|Surogate Rune 26B-A4B v3|25.806|57.44|11.96|477.6|full fine-tune|
|10B+|AutoJev-27B|27.781|56.40|1.78|104.2|full fine-tune|
|10B+|Decider chat · Qwen3.6-27B|27.781|51.35|2.09|916.9|inference technique|

### Calibration is not the index

The published calibration panel samples 1-in-6 requests, scores 32 benchmarks equally, excludes ToolRet/BRIGHT/RouterBench/ForecastBench, and uses chosen-option probability versus per-field correctness. Reported fields are accuracy, mean confidence, ten-bin ECE, Brier, confident-and-wrong share (`over95`), reliability bins and area metrics. Jev has 72,594 fields, accuracy 73.93%, mean confidence 81.15%, ECE 7.40 points, Brier 0.3558, confident-and-wrong share 2.07%. Its hosted probabilities are rounded to two decimals. The displayed diamond is **joint confident-and-wrong share**, not necessarily error conditional on confidence≥95%; don't turn 2.07% into a reliability guarantee. The artifacts do not establish a complete Brier aggregation formula; [UNVERIFIED] whether all special typed fields use identical multiclass normalization. [Results](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json) · [App definitions](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/index.html).

|Entry|Index|ECE pts ↓|Brier ↓|≥95% and wrong % ↓|
|---|---:|---:|---:|---:|
|Jev|57.89|7.40|0.3558|2.07|
|Surogate Rune 26B-A4B v3|57.44|11.96|0.3704|4.82|
|AutoJev-27B|56.40|1.78|0.3610|0.77|
|Decider chat · Qwen3.6-27B|51.35|2.09|0.3998|0.10|
|Winnow-12B|50.02|16.79|0.4862|6.24|
|JoshuaSP diffusiongemma (open-jev)|49.47|21.55|0.5241|13.66|
|Jevfire|49.37|5.21|0.4139|1.52|
|Decider 35B-A3B|47.11|2.26|0.3878|0.38|
|Xor|41.48|1.49|0.3881|0.41|

[Derived from live results](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json). **[INFERENCE]** Rune nearly matching Jev in index does not match AutoJev calibration (11.96 versus 1.78 ECE points); Xor has still lower ECE (1.49) but far lower decision index (41.48). ECE alone can reward less sharp models and cannot replace application loss/reliability analysis.

## Where top models separate from mid-size entries

These are **actual index-point contributions**, computed from published per-benchmark skill × normalized gold weight × area weight. This avoids mistaking a large raw gap on a tiny-weight task for the best optimization target. Positive values favor the first model; full-pair sum reproduces the observed index gap within rounding. [Source](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json).

### AutoJev-27B minus Decider 4B: 15.70 index points

|Benchmark|Top skill %|Mid skill %|Index-point gap|
|---|---:|---:|---:|
|Home appliances|73.86|17.05|1.923|
|RAGTruth|59.24|0.81|1.425|
|CRUXEval|60.20|10.65|1.186|
|HoVer|48.44|12.54|1.089|
|API-Bank|83.75|59.67|0.978|
|BBH|68.26|43.31|0.717|
|VAST|56.19|30.29|0.632|
|ANLI|56.04|34.63|0.627|
|iSarcasmEval|48.98|25.48|0.573|
|CLadder|49.00|26.60|0.536|
|BFCL|96.81|83.74|0.531|
|NLI4CT|70.45|48.81|0.528|

### Surogate Rune 26B-A4B v3 minus Decider 2B: 28.47 index points

|Benchmark|Top skill %|Mid skill %|Index-point gap|
|---|---:|---:|---:|
|API-Bank|82.95|30.37|2.136|
|PhishNChips|61.80|0.10|1.872|
|GSM8K|75.71|4.46|1.705|
|Home appliances|46.59|3.41|1.462|
|CRUXEval|59.36|0.00|1.421|
|HoVer|61.34|18.14|1.311|
|RAGTruth|51.90|0.00|1.266|
|BBH|72.68|28.93|1.257|
|iSarcasmEval|49.03|0.21|1.191|
|WinoGrande|71.90|35.44|1.067|
|ANLI|60.34|24.16|1.059|
|When2Call|68.01|39.61|0.961|

**[INFERENCE] Interpretation:** The 27B-versus-4B gap is not mostly already saturated MMLU/ARC (excluded anyway). Home-appliance structured exactness, hallucination detection, code execution, multi-hop verification, API selection and BBH carry major differences. The 27B-versus-2B gap adds GSM8K, phishing, sarcasm and commonsense. Training a narrow tool-risk model to maximize broad index points could spend resources on irrelevant capability while neglecting expensive false-safe errors. Sources for underlying gaps: [results](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json).

## Contamination and checkpoint-selection exposure

The board's own disclosure distinguishes direct test training from other-split training and selection exposure. These are **board-reported audits**, not my independent dataset reconstruction. [Methodology contamination section](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json).

|Entry/exposure|Published finding / board treatment|
|---|---|
|AutoJev-27B|WinoGrande used for checkpoint selection; 750 BBH and 1,500 RAGTruth rows also used for checkpoint selection. These are not described as zeroed the way directly trained rows are. Thus the headline is not wholly unseen evaluation.|
|JevK5|980 MMLU-Pro test IDs, union of v0.1/v0.2 training exposure, count wrong in full denominator.|
|open-jev (pngwn)|9,542/12,032 MMLU-Pro and 5,308 MMLU test questions in training. Trained MMLU-Pro accuracy .929 versus .614 unseen; trained rows count wrong.|
|lev|45 exact training/test overlaps found (22 BANKING77, 12 CLINC150, 1 ANLI, 9 ARC-Easy, 1 HellaSwag), count wrong. 42 RouterBench question overlaps not penalized because routing labels were not carried by training text.|
|Lumma-Fev 0.1B/0.6B|Author decision datasets contain 3,077/3,080 BANKING77 labeled test rows; cards do not declare training data. Board penalizes these rows. **[INFERENCE]** This is precautionary association evidence, not by itself proof the released weights trained on those rows.|
|reflex 4B|Removed: published adapter trained on 800 MMLU-Pro test items; 65.8% exposed versus 52.1% held out, +13.7 points (95% CI 9.8–17.5). Recommended base-only configuration differs. Reflex 27B remains.|
|Multiple families|Other-split training disclosed for MMLU auxiliary_train, ARC, HellaSwag, WinoGrande, BANKING77, CLINC150. Some public split duplicates count wrong.|
|Shuffle audit|12 entrants tested on five benchmarks; none crossed the flag threshold (≥5-point drop with CI excluding zero), but six smaller significant drops exist. **[INFERENCE]** Passing option-shuffle audit does not rule out memorized content, answer explanations, or selection overfitting.|

Additional limitations: Jev's weights/training set are not auditable from the leaderboard; pretrained public-benchmark exposure is not ruled out. Penalizing known overlap as wrong is transparency, not a statistical correction for unknown leakage, and uneven disclosure can confound ranking. Repeated public submissions/checkpoint replacements make the board a development set. **All preceding interpretation is [INFERENCE]** grounded in the disclosed controls and limits. [Methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json).

## Can we run it? Yes for public 0.2; exact live 0.2.1 parity is not yet packaged

- Public kit: [https://github.com/apolinario/decision-index](https://github.com/apolinario/decision-index); Python ≥3.10; core `huggingface_hub>=0.34,<2`, `httpx>=0.27,<1`; optional torch≥2.4, transformers≥4.51,<6, accelerate≥1.0, safetensors; rebuild pyarrow, pandas, numpy, scipy, mido, python-chess, tiktoken. [pyproject](https://github.com/apolinario/decision-index/blob/main/pyproject.toml).
- **Important blocker to claiming current-board parity:** `decision_index/editions.py` supports only 0.1/0.2 with DEFAULT 0.2, `scoring/index02.py` still averages five equal-weight areas; kit is version 0.2.0. It does not encode live 0.2.1's weights, exclusions, retrieval answerability changes, or ACOS scoring cutover. The public 0.2 kit is runnable, but cannot honestly be advertised as reproducing today's scalar unmodified. [Editions](https://github.com/apolinario/decision-index/blob/main/decision_index/editions.py) · [Scorer](https://github.com/apolinario/decision-index/blob/main/decision_index/scoring/index02.py).
- The Space says it is built from a `typesafe-diffusion-lab` checkout using `evaluation/reproductions/run.py`, `compute_indices.py`, `build_leaderboard.py`, etc.; its reproduction link is the public kit above, **not a public copy of every native adapter or raw lab result**. For exact current parity, obtain the versioned 0.2.1 builder/scorer + artifacts from maintainer, or explicitly port all published changes and prove parity against all published model aggregates. [Space README](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/README.md) · [methodology reproduce](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json).
- Rebuilding requires about **7 GB downloads / 17 GB working space**, including a **2.2 GB HoVer Wikipedia database**, and accepting HLE's gated terms. Licenses prevent suite redistribution; kit recommends your own private HF dataset. No suite/model downloads were done for this research because they exceed assignment limits. [Kit README](https://github.com/apolinario/decision-index).
- Engines: native `/v1/systemone` HTTP adapter, generic transformers reference, or custom `Engine.__call__(state,questions)->(response,raw)` with declared `Unsupported`. Preserve native prompt/readout; no truncation, no label pruning, no label leakage into adapter, no correctness-based retry. Generic transformers results are not automatically reproduction of a model author's special scorer. [Engine docs](https://github.com/apolinario/decision-index/blob/main/docs/engines.md) · [rules](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json).

Public 0.2 workflow (from kit README; **not executed**, not current-version parity proof):

```sh
pip install -e ".[transformers,rebuild]"
export HF_HUB_DISABLE_XET=1
python -m decision_index suite rebuild --work work
python -m decision_index suite import --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz
python -m decision_index suite sample --n 100 --out sample-100.jsonl.gz
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=my-model --out runs/my-model
```

### RTX PRO 6000 runtime: budget estimates, not wall-clock guarantees

Board hardware is one RTX PRO 6000 Blackwell Server Edition 96 GB per run. Local timing usually CUDA-synchronized prompt+inference wall time excluding model load/warmup/Internet HTTP; some entries use loopback batching and queueing. Lift overlay timings are excluded from headline latencies because their multi-process/extra-pass execution inflates them. Jev HTTPS latency is not directly comparable. [Latency methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json).

The table below is **[INFERENCE] serial-equivalent hours = 120,340 × published mean_ms / 3,600,000**, not measured complete sweep elapsed time. It excludes rebuild, model download, startup, failures, lifted-path cost and opportunities for continuous batching; some means themselves include queueing. Full HF job default timeout is **72h**, not a measured expected duration. [Results](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json) · [HF-job setup](https://github.com/apolinario/decision-index).

|Model|Published mean ms|Estimated serial-equivalent h|
|---|---:|---:|
|GLiNER 2.5 base|25.4|0.85|
|Bosun v3.1 0.6B|264.8|8.85|
|Decider 2B|268.5|8.98|
|Decider 4B|59.7|2.00|
|Decision 1.0 Lux|122.4|4.09|
|Winnow-12B|137.5|4.60|
|AutoJev-27B|428.6|14.33|
|Surogate Rune 26B-A4B v3|1385.0|46.30|
|Decider chat · Qwen3.6-27B|2227.2|74.45|

### Submission

Run full suite using `pipeline` or `hf-job`; upload the run directory to a HF dataset (`--upload <you>/<repo>`). Open a PR to the public kit adding a line to `submissions/README.md` (create if absent): model name, results dataset pointing to `runs/<name>/scores.json`, engine+commit, hardware and declared capacity limits. `scores.json` must say `complete: true`; reviewer re-scores untouched results. Current repo documentation asks for **0.2**, so confirm/pin the accepted edition rather than relabel 0.2 scores as 0.2.1. Raw source payload redistribution has license constraints even when the result bundle is shareable. [Submission instructions](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard).

## Is chasing this index well-posed?

**[INFERENCE] Only if “maximize this pinned, public benchmark” is literally the goal.** As evidence for a reusable zero-shot decision model it is necessary-but-insufficient:

1. **Moving objective:** 0.2→0.2.1 changes area weights, task inclusion, answerability filters, duplicates, ACOS metric and a top checkpoint. A jump of several points need not be model progress. Pin corpus, scorer, prompt adapter and edition.
2. **Tiny tasks can buy large gains:** 88 home-appliance cases receive ~3.39% of the index (one ordinary tool task). Each independent binary task with 196/220 cases is not comparably precise to 12k MMLU-Pro. Aggregate confidence intervals or resampling of linked groups are needed before spending on 0.5-point gains.
3. **Clipping hides failure severity:** all below-chance scores become zero. For safety, consistently inverted confident predictions and cautious random predictions are not equivalent.
4. **Not a probability-quality objective:** headline ignores ECE/Brier except ForecastBench. A high-index poorly calibrated engine can be a worse cost-sensitive router or safety gate. Calibration sample excludes retrieval rankings and forecasts and averages per-field, not multi-question exactness.
5. **Not pure generalization:** checkpoint-selection exposure, public test training and reused datasets remain. The zero-shot *interface* does not imply zero-shot *task knowledge*.
6. **Not task-native equivalence:** choice conversion, candidate pools, typed yes/no decomposition and constrained readouts can change what original benchmarks measure. Near-perfect finite-option code/math scoring does not prove generative solution ability or agent success. Six interactive environments are absent.
7. **Not throughput benchmarking:** serving stack, batch/queue settings, native output caps/lifts, quantization and mean-versus-median differ. Total parameter bands do not equal active compute or 3090 feasibility.
8. **The no-pruning policy is nuanced:** adapters cannot prune labels, yet Hopper's approved native large-label path is a tournament retaining 10 finalists then spreading 5% mass over all choices. Native approximation remains part of the engine. Don't treat high coverage as evidence every label got equally exact consideration.

Underlying evidence for this assessment: [methodology and lifts](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json), [app and latency display](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/index.html), [kit adaptation/scoring docs](https://github.com/apolinario/decision-index).

## Top three trial recommendations

1. **Freeze a two-track evaluation before training [RECOMMENDATION]:** use pinned public 0.2 as a reproducible regression suite until 0.2.1 scorer parity is available; create genuinely disjoint task/source-held-out command-risk/typed-choice evaluation with false-safe cost, log loss/Brier, selective risk and label-order/paraphrase perturbations. Never select checkpoints on the held-out comparison panel. Report index and operational loss separately.
2. **Run a strong small baseline and a teacher/reference, not only fashionable heads [RECOMMENDATION]:** Decider 4B (40.70, 22.5ms board median) is the first realistic quality/latency reference; Decider 2B (28.97) is the smaller floor. Compare AutoJev (56.40, ECE 1.78) versus Rune v3 (57.44, ECE 11.96) on our application before choosing a distillation teacher. Those are board metrics, not local performance promises. [Results](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json).
3. **Allocate learning to observed failure slices [RECOMMENDATION]:** prioritize multi-field tool exactness, hallucination/entailment, code effects, multi-hop evidence and many-label coverage; then calibrate on disjoint data. Budget an initial stratified sample to measure actual runtime and peak memory; rent 96GB only for teacher/full-suite jobs that cannot fit locally, not just because the board used it. The gap tables above justify these slices; hardware capacity must be measured, not inferred from marketed model size. [Gap inputs](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json) · [hardware/timing caveats](https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json).

## Verification scope

Read HF repository tree, static app source, all current aggregate JSON objects, archived 0.2 bundle, methodology and public harness README/edition/scorer/dependencies. Executed in-memory JSON ranking, per-task winner extraction, and weighted-index recomputation; reproduced top-three stored index numbers to displayed precision. No repository edits, model inference, dataset rebuild, raw-results rescore or runtime measurements performed. This is a source/data audit, not a successful harness run. Sources were live branch URLs; pin commits before using them as an experiment manifest.
