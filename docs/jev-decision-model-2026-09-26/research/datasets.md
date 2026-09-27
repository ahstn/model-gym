# Candidate training datasets for an open Jev-style decision model

Sources: HF Hub API, datasets-server `/info`, `/splits`, `/statistics` (train split, full, not partial), `/rows`, dataset READMEs, and leaderboard data `data/index.json` / `data/methodology.json` (v0.2.1, generated 2026-09-26) from https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/main/data.

**Correction to the brief:** the live board is **Decision Index 0.2.1** (`balanced_skill` headline). Jev scores 57.89, Rune v3 57.44, AutoJev 56.40, Decider chat 51.35, and Winnow-12B 50.02. The brief's screenshot showed a different edition: Jev 51.7 and AutoJev 50.9. The board has 56 entrants, 43 benchmarks, and a 38-benchmark index panel. RouterBench and SGD are excluded from the index (methodology.json `benchmarks[].index_note`).

## 1. Summary table

| | SargeDev/jev-distill-corpus-v3 | ZefanCai/Open-Jev-v1.1 | tasksource/tasksource-jev-typed-decisions |
|---|---|---|---|
| License | Apache-2.0; the openjev_v2 stream is CC0 at source. [UNVERIFIED] Whether Jev API ToS allows distillation was not checked. | "other": generated content is CC0, WANLI is CC BY 4.0, code is MIT, and some TypeSafe-doc question text has an unverified license | "other", set per row. `license_use` on train rows: commercial 1,374,992; non-commercial 254,470; unspecified 870,538 |
| Rows | 740,957 total. train 655,806 / val 14,111 / calibration 13,766 / test 14,261 / ood 13,058 / test_set_30k 29,955 | 326,619 published. train 147,139 / cal 26,675 / val 25,413 / test 43,125 / ood 84,267. Omits 2,053 Wikispeedia rows | train 2,500,000 / val 15,000 / test 15,000 (~2.97 GB, 1.45 GB download) |
| Kind mix (train) | noul 341,910 (52%) / choice 163,657 (25%) / score 150,239 (23%) | choice 110,271 (75%) / noul 25,434 (17%) / score 11,434 (8%) | choice 1,945,292 (78%) / noul 372,242 (15%) / score 182,466 (7%) |
| Options per row | 2–16, mean 3.49, median 2 | 2–16, mean 3.57, median 3 | 0–235, mean 5.20, median 3. noul has `options=[]` and target `[p]` |
| Target format | Full soft distribution. noul is `[P(false),P(true)]`, score is 6 levels 0..5 | Distribution over options. Mostly one-hot rule labels (see `metadata.target_basis`). noul is `[P(no),P(yes)]` | Mostly one-hot gold. Soft targets come from ≥5-annotator votes or mean ratings (~10% "graded" share). noul is `[p]` |
| Provenance | yuri_v3 (498k): synthetic templated scenarios in 53 domains, **labels distilled from Jev 1.13 via OpenRouter**. yuri_v1 (148k): memory-relevance noul labeled by an unnamed "32B teacher" over 18 QA datasets. openjev_v2 (94.8k): reschema of ZefanCai/Open-Jev release-v2 | Rule-generated controlled scenarios (51,200 new rows in 8 authored domains, 80 policy families, 6,400 scenarios) + **101,207 WANLI NLI rows** + replayed Open-Jev controls (≤1,500 per source) | Deterministic recasts of 667 human-labeled Tasksource sources. **No teacher model** |
| Sources / domains (train) | 66 domains, 14 families. The biggest single domain is memory_relevance at 137,203. Other yuri_v3 domains have ~7.5–10k each | 73 sources. wanli-decisions-v1 has 82,045 (56% of train); each other source has ≤1,500 (community-diversity-v2 sub-sources have 640) | 667 sources. Mix is classification 47%, MC 30%, graded 10%, procedural 10%, token 3% |
| Languages | Card says `en`, but yuri_v1 contains non-English passages (a Spanish row seen at offset 300000; the stream draws on MLQA/TyDi) | en and zh per card tags | en plus multilingual (xnli, xlwic, X-CSQA, disrpt, ...) |
| state length (chars) | min 40, median 214, mean 307, max 1,264 | `state_json` median 244, mean 906, max 4,389 | median 206, mean 645, **max 128,613** (long tail) |
| question length | median 58, mean 102 | median 141, mean 168 | median 51, mean 57 |
| Extra fields | domain, family, source | group_id, metadata_json (target_basis, latents), record_json, original_line_number | variant (direct 2.15M, packed_derived 120k, label_verification 81k, criteria_permutation 81k, instruction_paraphrase 41k, paired_text_format 27k), group_id, question_id, license |

Links: https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3 · https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1 · https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions (all counts come from the datasets-server `/statistics?...&split=train` endpoint).

## 2. Per-dataset notes

### SargeDev/jev-distill-corpus-v3 (last modified 2026-09-21)
- **Only corpus with Jev soft labels.** It is the closest thing to "mimic Jev's calibrated distribution" data. Example row: `score` target `[0.01,0.01,0.05,0.41,0.51,0.01]`; `choice` target `accept_output/reassign/rollback/escalate` → `[0,0,0.14,0.86]`.
- **Label filtering.** The card says degenerate labels (max-prob ≥0.999 with confidence ≥0.95) were filtered, which removed 1.5%. noul P(true) has mean 0.45 and sd 0.28. Dedup uses sha256 of normalized state+question, and cross-stream collisions were removed.
- **Weakness:** the yuri_v3 states are short templated strings, e.g. "X; additionally Y. Context: A, B. Reported by the review queue." Diversity is low, and they don't exercise long context, large label sets (max 16 options), or knowledge.
- **Weakness:** yuri_v1 labels come from an unnamed 32B model, not Jev. Some are 0.5/0.5 (seen at the sampled row), so filter near-uniform targets.
- **Duplication with Open-Jev:** openjev_v2 is Open-Jev v2 content, and its domains (tic_tac_toe-v1, snake-v1, painting-geometry-v1, workflow-controls-v1/*, customer-control-v1, vizdoom-basic-v1) also appear as Open-Jev-v1.1 sources. Using both datasets double-counts these rows.
- **Splits:** there is a `calibration` split for temperature fitting. `test_set_30k` is the recommended held-out set.

### ZefanCai/Open-Jev-v1.1 (last modified 2026-09-23)
- **What it is:** a redistributable projection of the frozen `community-hard-mix-v2-final` mixture used to train Open-Jev-27B-v1.1.
- **Split construction:** splits are separated by parent group, and the OOD split holds out templates, rule families, and goals.
- **Metric caveat:** the card says "accuracy excludes soft-target rows".
- **Mostly WANLI:** WANLI is 56% of train. That makes it NLI-heavy, which matters because ANLI is a benchmark and WANLI is MNLI-seeded.
- **The rest is structured JSON states:** support, IR, contracts, RAG, browser tools, shell history, games, and rubric judging. These are good for multi-candidate "abstain" reasoning, and the example row has 7 options including `abstain`.
- **Leakage screen:** a lexical screen against 531 frozen JevBench/Frontier-100 requests found 0 matches (`provenance/benchmark-overlap-screen.json`). It did **not** screen against the decision-index suite.
- **Model input:** state, question, kind, and options. Metadata and latents must not be fed to the model (per the card).

### tasksource/tasksource-jev-typed-decisions (last modified 2026-09-26, i.e. brand new)
- **Largest and most diverse; real human labels.** Options are shuffled per row, and val/test rows whose text appears in train were removed.
- **Benchmarks it keeps out:** BIG-bench (so BBH), MMLU, BLiMP, and MATH test are excluded.
- **Benchmark train splits are in.** AGENTS.md says: "GLUE, SuperGLUE, HellaSwag, PIQA and many other public benchmarks are **in** the training data (their train splits), so don't report zero-shot results on them after training here."
- **Companion dataset:** `tasksource/procedural-typed-decisions` (counting, arithmetic, state tracking) is referenced but was not analyzed here.
- **Not directly a Jev-distribution target:** labels are gold, not calibrated beliefs. Temperature scaling on a held-out set is still required.

## 3. Overlap risk with the Decision Index suite

Benchmarks in the suite (methodology.json `benchmarks`) include BFCL, ToolRet, API-Bank, BANKING77 (3,080 = full test), CLINC150+OOS (5,500), ContractNLI, ANLI (3,200), BPoMP, Humicroedit, POP909, cfcolor, MMLU, GPQA-D, ARC-E/C, WinoGrande, HellaSwag (10,042 = validation), GSM8K, ChessBench, MuSR, SATA, BRIGHT, Amazon ESCI, ACOS, FinEntity, iSarcasmEval, VAST, NLI4CT, CRUXEval, CLadder, HLE, ForecastBench, Habermas, OpenBookQA, CommonsenseQA, and more. Added in 0.2: PhishNChips, MMLU-Pro, BBH, RAGTruth, HoVer, When2Call, New Yorker.

**tasksource sources that share a dataset with suite benchmarks** (train-row counts from `/statistics`):

| Source | Train rows |
|---|---|
| winogrande_xl | 13,625 |
| hellaswag | 13,542 |
| WANLI | 13,174 |
| commonsense_qa | 8,942 |
| openbookqa | 5,757 |
| esci | 4,856 |
| anli a1/a2/a3 | ~14.1k |
| hover (+3way) | ~9.1k |
| clinc_oos/plus | 4,460 |
| cladder | 4,406 |
| banking77 | 4,365 |
| sms_spam | 4,318 |
| ARC-Easy + Challenge | 3,903 |
| humicroedit subtasks | 4,549 |
| nli4ct_semeval2024 | 2,329 |
| contract-nli | 4,478 |
| Sarcasm headlines / pragmeval sarcasm | 4,691 / 340 |

- **Split separation:** these rows are the sources' train splits and the board mostly uses test/val. That separation is not enough, though. The board already counts train/test duplicates as wrong: 2 BANKING77 rows, 15 ARC rows duplicated in ARC train and MMLU auxiliary_train, and lev's 45 items (methodology.json `notes.contamination`).
- **CLadder, ContractNLI and NLI4CT are high risk.** I haven't verified which split the board uses for these; check before training.
- **Protocol-level exposure:** training on the train splits of benchmark datasets is "declared overlap / other splits of the same dataset". The board records that for the Decider family, Kev, JevK5, and others. It does not count as wrong, but it weakens zero-shot claims.
- **SargeDev:** the templated yuri_v3 content is low risk. yuri_v1 draws on HotpotQA, PubMedQA, MLQA, NarrativeQA, LSAT, and TyDi; none of these is a suite benchmark per the list above. The main contamination vector is **distillation of Jev itself**, which inflates agreement with Jev rather than true accuracy.
- **Open-Jev-v1.1:** WANLI is MNLI-seeded, and ANLI is in the suite. Content overlap is unlikely, but it is in-distribution for NLI. No suite screen was published.
- **Documented leaks elsewhere:**
  - pngwn/typed-decisions-v2-system-one was built from the MMLU-Pro test set: 9,542 items trained on, a +31-pt gap between seen and unseen items.
  - JevK5 trained on 980 MMLU-Pro test items.
  - FrontiersMind kevv datasets contain 3,077/3,080 BANKING77 test items.
  - **Do not use any of these.**

## 4. Which leaderboard entrants used which dataset

Evidence comes from HF `models?filter=dataset:` and model-card grep.

| Dataset | Models on HF declaring it | On the board? |
|---|---|---|
| SargeDev/jev-distill-corpus-v3 | autotrust/JEV-9B, autotrust/JEV-27B, alanhuangya/wev-{1.7b,4b,8b}, SargeDev/Jev_Qwen3.8-27B (+NVFP4) | None of these appear among the 56 board entrants |
| ZefanCai/Open-Jev-v1.1 | ZefanCai/Open-Jev-27B-v1.1, nicolasembleton/openjev-minicpm5-2b-lora-phase1 (MiniCPM5-2B, same as our candidate base) | Neither is on the board |
| ZefanCai/Open-Jev (v1/v2) | Gowtham25/reflex-s1, Edoigtrd/Mirave-*, Kwokou/BeeNara, roadius/peewee-mix-v1 | Not on the board. **MoJev** (score 11.69) trained on "205,084 rows from 18 Open-Jev generators" per its card |
| tasksource-jev-typed-decisions | none declared (released 2026-09-26) | none |

Other entrants' data:

| Entrant | Data |
|---|---|
| Winnow-12B/E4B | Private |
| Rune | Not stated on the card |
| Hopper | ARC, CSQA, MMLU, SNLI, MNLI, VitaminC, BoolQ, SQuAD2, CLINC, DBpedia, HelpSteer2 |
| Kev | banking77, boolq, ag_news, mnli, sst5, yelp, trec, dbpedia, amazon, imdb |
| JevK5 | avbiswas/bev-decision-150K |
| this-that | limberc/this-that-complex-decisions |
| Jev-Omni | akhilaaa3/decision-bench |
| GLiNER2.5-Decide | fastino/fast-decisions |

- **AutoJev** (`configs/training.json`): the curated run used 73,000 rows, 3,500 temperature rows, and a 4,700-row monitored panel. Its sources are identified only by hash. [UNVERIFIED] which public dataset, if any.
- **Decider:** its README mentions OpenJev only as an evaluation suite. [INFERENCE] It is not a declared training source.

## 5. Recommended mixture (for ~0.5–4B base; token budget ~1 epoch)

| Share | Source | Filter |
|---|---|---|
| 35% | tasksource, `license_use=="commercial"` (or all, if research-only) | Drop sources in the suite-overlap list above. Keep all 6 variants, but cap `packed_derived` at ≤5%. Truncate or drop states >8k tokens. Cap every source at ≤5k rows |
| 30% | SargeDev yuri_v3 (Jev soft labels) | Keep as-is. Drop rows with max-prob ≥0.99 on score/choice only if they are over-represented. Train with KL for choice/score and BCE on P(true) for noul |
| 10% | SargeDev yuri_v1 | Drop near-uniform targets (entropy >0.97 bits for noul). Downsample to ≤60k rows |
| 20% | Open-Jev-v1.1 non-WANLI sources (~65k rows) + ≤20k WANLI | Keep group_id intact. Include `abstain` rows |
| 5% | Our own command-risk set (repo) | For the domain score head |

- **Exclude SargeDev `openjev_v2`** when using Open-Jev-v1.1, to avoid duplicates. Dedup across all three datasets by sha256 of normalized (state, question, sorted options).
- **Kind balance target:** roughly choice 50 / noul 30 / score 20. tasksource and Open-Jev are choice-heavy; SargeDev supplies noul/score.

## 6. Held-out and decontamination protocol

1. **Suite blocklist.** Rebuild the suite items with https://github.com/apolinario/decision-index (reproduce inputs; the corpus sha256 is `b2b56d6f…`). Then:
   - Remove training rows with exact normalized-state matches or 13-gram overlap (Open-Jev's screen rule) against any suite item's state/question/options.
   - Also drop whole source datasets that match suite benchmarks (the tasksource list in §3), or at least drop their dev/test-derived rows.
2. **No suite-based checkpoint selection.** AutoJev was flagged for using WinoGrande, BBH, and RAGTruth rows. Select checkpoints and fit temperature on: SargeDev `calibration` + `test_set_30k`, Open-Jev `calibration`/`ood`, and tasksource `validation`.
3. **Report three held-out panels:**
   - (a) in-distribution: SargeDev test_set_30k, Open-Jev test
   - (b) OOD: Open-Jev ood, SargeDev ood (note these contain the same Open-Jev v2 OOD rows — don't double-count)
   - (c) the leaderboard suite as a final one-shot.
4. **Group-level splitting.** Split by `group_id` (Open-Jev, tasksource) so packed and variant rows of the same example never straddle splits.
5. **Shuffle test.** Run the board's shuffle test (reorder options) on our own MMLU-like panel before submitting.

## 7. Open items / unverified
- [UNVERIFIED] Which split the suite uses for CLadder, ContractNLI, NLI4CT, and VAST. I didn't load `selected-rows`.
- [UNVERIFIED] The exact item-level overlap between tasksource train rows and the suite test rows. A datasets-server `/filter` query over 2.5M rows timed out; this needs a local parquet download of the `source` subsets (~1.45 GB total, under the limit).
- [UNVERIFIED] Whether TypeSafe/OpenRouter terms allow redistributing Jev outputs. This affects SargeDev's Apache-2.0 claim.
