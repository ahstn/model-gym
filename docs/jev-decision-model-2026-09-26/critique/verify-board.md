# Verify: board, competitor recipes, kit (fact-check, 2026-09-26)

Sources were fetched live in this session. They are:
- IDX = https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json (`generated_utc` 2026-09-26T11:54:33+00:00, `suite.edition` = `release-v2.1`, 56 models)
- METH = .../data/methodology.json
- TREE = https://huggingface.co/api/spaces/multimodalart/jev-decision-index/tree/main?recursive=true

Headline field = `scores.balanced_skill`.

## 1. Live v0.2.1 scores: CONFIRMED (all 9)
| Entry | Claimed | IDX balanced_skill |
|---|---:|---:|
| Jev (`jev.scores`) | 57.89 | 57.89 |
| Surogate Rune 26B-A4B v3 | 57.44 | 57.44 |
| AutoJev-27B | 56.40 | 56.4 |
| Decider chat · Qwen3.6-27B | 51.35 | 51.35 |
| Winnow-12B | 50.02 | 50.02 |
| Decision 1.0 Lux | 43.49 | 43.49 |
| Decider 4B | 40.70 | 40.7 |
| Decider 2B | 28.97 | 28.97 |
| Xor | 41.48 | 41.48 |

- "Decider chat, T=1.943, zero training": CONFIRMED. IDX meta has `kind: "inference technique"`, `trained_bytes: null`, `weights_repo: null` and `served_checkpoint: "Qwen/Qwen3.6-27B"`. The Decider README says "Qwen/Qwen3.6-27B at temperature 1.943 (the Decision Index entry 'Decider chat · Qwen3.6-27B')": https://github.com/Mapika/decider/blob/main/README.md. One nuance: T was fitted, so the entry is not calibration-free. It has no weight training.
- Rune v3's placement on the board: METH says `replaced_by rune-26b-a4b-v3 … adopted after a 3,000-row reproduction matched it bit for bit`.

## 2. Xor 26-option cap → ~12.09% unanswered: CONFIRMED
- METH unanswered table: `{"engine":"xor","total":119898,"unanswered":14500,"share":0.1209,"reason":"supports at most 26 answer options per question"}`.
- IDX Xor has `counts {"ok":105398,"unsupported":14500}` and `coverage 0.6927`. Every other listed entry has 0.76.
- Its calibration panel covers only 28 of 32 benchmarks. The low ECE (1.49) is therefore measured on a narrower set.

## 3. AutoJev recipe: PARTLY
Sources:
- https://github.com/denis-pplx/autojev/blob/main/configs/training.json
- configs/train.sh
- src/autojev/optim.py
- the card at https://huggingface.co/denis-pplx/autojev-27b

| Claim | Verdict | Evidence |
|---|---|---|
| Qwen3.8-27B base | CONFIRMED | `base_model: Qwen/Qwen3.8-27B`, rev 1d4bf0f2 |
| LR 2e-6 | CONFIRMED | `"lr": 2e-06`; train.sh `--lr 2e-6`; wd 0.01; eff. batch 256 |
| ~73k rows | CONFIRMED | `training_rows: 73000` |
| 1 epoch | PARTLY | Scheduled for 1 epoch, 286 updates, but the release is update 200 (`selected_examples_seen 51200`, `selected_fraction_of_epoch 0.701`). The shipped model has seen about 0.70 epoch. |
| 8192 ctx | CONFIRMED | `max_length 8192`, `token_budget 8192` |
| CPU-offloaded FP32 AdamW | CONFIRMED | optim.py: `self.masters = … .to(device="cpu", dtype=torch.float32)`, with pinned CPU grad buffers |
| T = 2.2076 | CONFIRMED | `selected_temperature 2.207568…`, fitted on 3,500 temperature rows |
| Selection exposure on WinoGrande/BBH/RAGTruth | CONFIRMED | METH: "autojev-27b used WinoGrande rows for checkpoint selection"; "autojev-27b checkpoint selection used 750 BBH and 1,500 RAGTruth rows" |

Hardware per the card: one H200, full-weight SFT.

## 4. Decider claims: PARTLY (both numbers are right; the attribution needs a fix)
- The ~3.3-point figure is CONFIRMED, but it comes from the **4B** card, not the 2B. The source is https://huggingface.co/Mapika/decider-4b/blob/main/README.md: "controlled quarter-data comparison of the same 4B (`ref_4b` vs `ref_4b_bf16opt`) … AdamW on the bf16 parameters beat AdamW with FP32 master weights by 3.3 held-out points and 0.072 nats (0.791 against 0.758), and by 7 points on nine knowledge tasks."
  - Correct statement: a single quarter-data 4B ablation with one seed and no CI.
  - The Decider README says the 2B and 4B used no master copy. The 35B used FP32 master weights plus Muon, and a retrain is planned.
- The 2B SFT figure of 5.3 h on one GH200 is CONFIRMED. The README says `scripts/train.sh full … one epoch from Qwen3.5-2B-Base`: "One epoch is 1.47M examples and 455M tokens, 5.3 h on a GH200 plus 45 min of evaluation". It reproduces the **v9 supervised** weights.
  - Caveat: the board's Decider 2B is **v11**, which is v8 → RL v10 (unreleased RL code) → LoRA r64 for 2 epochs on 42,749 rows (https://huggingface.co/Mapika/decider-2b).
  - So 5.3 h does not reproduce the 28.97 entry. It reproduces only the SFT stage.

## 5. apolinario/decision-index kit supports only 0.1/0.2: CONFIRMED
- At main commit `19ad28ec9485493cc4f7fc07d91c178f948e6434` (2026-09-24T10:00:48Z), `decision_index/editions.py` has `DEFAULT = "0.2"` and defines only the keys "0.1" and "0.2".
- Its aliases are release-v1, release-v2, v0.1 and v0.2. `get()` raises for anything else, so "0.2.1" and "release-v2.1" are rejected.
- The 0.2 entry lists `benchmarks=44` and `scoreable=120615`. Live IDX reports 119,898 scoreable.
- Source: https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/decision_index/editions.py

## 6. Winnow-12B = LoRA r32 on gemma-4-12B-it, no calibration, ECE ~0.168: CONFIRMED (with one discrepancy)
- The card (https://huggingface.co/EldanRing/Winnow-12B) says:
  - Base `google/gemma-4-12B-it` rev 707f0a3b.
  - "LoRA rank / alpha | 32 / 64", all attention and MLP projections, merged into BF16.
  - "default decision temperature is 1.0, without a separately fitted calibration map".
- IDX calibration: `ece 0.1679`, `over95 0.0624`, `brier 0.4862`.
- Discrepancy: IDX meta lists the base as `google/gemma-4-12B`, not `-it`. The card is authoritative for the base.

## 7. Nimble 9B v2 ~39.57 with 4,926 rows: CONFIRMED
- IDX "Bespoke Nimble 9B v2": `balanced_skill 39.57`.
- training_summary.json (https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2/raw/main/training_summary.json) has:
  - `training_rows 4926`, split into 2826 original, 1000 contrastive, 100 long-form and 1000 decision-skills.
  - LoRA r16/α32, LR 5e-5, 1 epoch, 616 steps, 1,349 s, peak 24.5 GiB, max_length 8192.
  - "Fixed one epoch; no evaluation-based selection". Card T = 2.179.
- Discrepancy: IDX meta says base `Qwen/Qwen3.5-9B-Base`, while the summary and card say `Qwen/Qwen3.5-9B` (rev c2022362). Trust the summary.
- The peak of 24.5 GiB is just above a 3090's 24 GB.

## 8. Does the Space publish per-request results? NO; only aggregates
- TREE contains only index/methodology JSON (v0.1, v2, v0.2.1, current), HTML/JS and images. There are no results.jsonl or per-request files.
- Each IDX model has per-benchmark `results[bid]` with `score`, `jev`, `requests`, `answered`, `unsupported`, `errors` and `median_ms`. It also has `benchmarks[bid]` with `raw`, `skill`, `coverage`, `random` and tracks, plus calibration reliability bins.
- What this allows:
  - Model-level proxy correlations, e.g. 56 models × 43 benchmarks, to test whether a cheap subset predicts `balanced_skill`.
  - Leave-one-benchmark-out regressions.
- What it does not allow: per-item correlations or item-response-theory (IRT) item selection. Those need running models ourselves on the kit's 0.2 suite, or the maintainer's raw runs.
