# Completeness critique: research notes + 3 plans vs the original request

Scope: read-only check of ../brief.md, ../research/*.md and ../plans/{plan-task,plan-astra,plan-sol}.md. I ran two new primary checks (tokenizers, MiniCPM5 config). They are marked **[NEW CHECK]**.

## 1. Coverage matrix (requested item → verdict)

| Requested item | Where covered | Concrete verdict present? | Gap |
|---|---|---|---|
| Trial / experiment program | All 3 plans (ladders E0–E17 task, E0–E9 Astra, E0–E9 Sol) | **Yes**: gates, GPU-h and $ per rung | The plans disagree on the first model to train (§2) |
| Extra hardware: what, where (Daytona / RunPod / Vast / …) | hw-rental-astra.md, hw-rental-sol.md, hw-3090.md; §6 of each plan | **Yes**. All three say Lium PRO 6000 $1.19–1.29/h, RunPod Secure $2.09 fallback, RunPod A100-80 $1.19/1.59. Daytona GPU is preemptible only, so no long runs there. Vast is for price discovery and restartable sweeps. Modal is rejected for long jobs | Nobody got a real checkout quote for Lium volume/egress ([UNVERIFIED] in hw-rental-astra.md). The Lium pod count is a single snapshot |
| Jev analysis | jev-reference.md | **Yes**: API semantics, confidence formulas, RLCD, averaged-teacher reference, jaggedness | — |
| 3 datasets | datasets.md | **Yes**: rows, kinds, licenses, contamination, languages | The Jev-ToS question for SargeDev is explicitly **not checked** (datasets.md L11). This is the root of contradiction C4 |
| AutoJev | competitor-autojev.md | **Yes**: recipe, 255-code sliced head, T, exposure, MIT/Apache | License of *outputs used as a teacher*: not analysed (§4) |
| Rune | competitor-rune.md | **Yes**: v2 self-distillation, v3 undisclosed | Rune weight license (Gemma-derived) is not stated in the note. It is moot, because task-plan rejects Rune as a teacher |
| Decider | competitor-decider.md | **Yes**: Apache harness, RL trade-off verified (OpenJev −0.8) | — |
| Winnow-12B | competitor-winnow.md | **Yes**: gated soft CE, no calibration (ECE .168) | Teacher unnamed [UNVERIFIED] |
| Xor | competitor-xor.md | **Yes**: 26-option cap → 12.09 % unanswered | — |
| 12 papers | papers-deepseek-nsa.md (V4.1-Flash, V4, FlashMemory, NSA); papers-hybrids.md (Implicit Hybrids, LoLCATs, Sparse Frontier, ODA, Limits of Speculation); papers-classification.md (GLiClass, CalRL, DHRD) | **Yes, all 12 have a mechanism, a transfer judgement and a verdict.** Verdicts are consistent across plans: CED only as layer truncation; NSA / V4 / FlashMemory / LoLCATs rejected; GLiClass as an encoder control; ODA + Speculation as a cascade router; Sparse Frontier as a slice matrix; CalRL deferred; DHRD conditional; Implicit Hybrids only as a diagnostic | Implicit-Hybrids diagnostic appears only in task-plan (inside E14). Astra and Sol drop it silently. That is acceptable, but should be recorded as "rejected" rather than omitted |
| MiniCPM5-2B | base-models.md; primary in Sol and Astra, secondary in task | **Yes**: 2.517B stored, Llama, Apache, TRL recipe | **[NEW CHECK]** `config.json` declares `transformers_version: 5.6.2` (https://huggingface.co/openbmb/MiniCPM5-2B/raw/main/config.json), but the repo lock pins 4.57.6 (hw-3090.md L13). This is a major-version jump. The plans say only "pin ≥ min". A separate `llm` env is mandatory, not optional |
| K2-Horizon-3.7B | base-models.md (5.06B stored, trust_remote_code); task E7, Astra E0 inference, Sol omits | **Yes**: inference-only first, must win by ≥1.5–2 pt to justify remote-code risk | Sol never evaluates K2. Resolution: keep the zero-shot K2 row in E0/E1 (cost ≈1 h) |
| Alternative viewpoints | top-vs-mid.md; competitor-xor.md (reflex "fine-tuning never helped"); each plan's "what not to do"; Astra's contrarian equivariant readout | **Partly.** A "don't train at all: ship stock 27B + readout" view is quantified (Decider-chat 51.35) but only as a baseline, never as a product option. The "fix the encoder product" view is only a kill fallback | See §5 A1–A3 |

## 2. Contradictions between plans, with recommended resolution

| # | Topic | task (balanced) | Astra | Sol | Resolution |
|---|---|---|---|---|---|
| C1 | **Primary tier** | 3–10B, Qwen3.5-4B (target ≥42); 2B secondary via distillation | 700M–3B MiniCPM5 (≥32) + 12B Gemma (≥52) | 700M–3B MiniCPM5 (≥30); 4B secondary | **Run the 2B and 4B tiers in parallel on the 3090, and decide the ship tier by the E4 → E8 slope.** Reasons below this table |
| C2 | **Teacher** | Mean of T1 = stock Qwen3.6-27B + Decider readout (51.35, ECE .021) and T2 = AutoJev-27B | Stock Qwen3.8-27B in reasoning mode; AutoJev only as a cheap control; Gemma-12B check on 20 % | SargeDev Jev labels (rights permitting); optional 5k Qwen teacher | **Default T1 (Qwen3.6-27B one-pass readout). Test AutoJev and Qwen3.8 reasoning in the E0 teacher audit on 1k adjudicated rows, and pick by gold NLL. Do not average by default.** Reasons below this table |
| C3 | **Budget (rental)** | Lean $35 / Medium $150–230 / Ambitious $650–900 | Lean $250 / **Medium $700** / … (includes data, annotation and contingency) | Lean $0–20 / Medium cap **$100** / Ambitious ≤$250 | **Pure GPU spend converges at ≈$60–230 for the medium tier. The $700 is Astra's all-in figure** (it adds $250 data/annotation, $190 contingency and a 12B tier). Recommend a **$250 GPU authorisation** plus a separately approved $100–250 for annotation/API. Measured throughput re-baselines this after the 500-step pilot |
| C4 | **Jev-distilled data (SargeDev yuri_v3)** | Excluded from shipped weights; internal ablation only (E5c′) | Optional 10 % substitution, only after ToS verified; excluded otherwise | **25 % of core mixture** "only if output rights…", and E3 is built around it | **Adopt task/Astra: exclude from the core recipe; allow one internal ablation arm.** Reasons below this table |
| C5 | **4B Base vs instruct** | E7 decides among Qwen3.5-4B-Base, Qwen3.5-4B and K2 | Qwen3.5-4B instruct | Stock Qwen3.5-4B (instruct), QLoRA | **Keep task-plan's E7 comparison, but run it zero-shot first.** Reasons below this table |
| C6 | **Encoder tier** | E2: keep only if within 5 proxy pts of the 2B decoder at ≥5× throughput | E2: within 2 pts, ≥3× throughput; otherwise a risk/cascade baseline | E1: ≥14 index (threshold truncated in the plan) | **One shared encoder arm (gliclass-instruct-large vs ModernBERT-large joint markers).** Reasons below this table |
| C7 | LoRA rank / lr | r64, lr 1e-4 (Decider-style) | r16, lr 5e-5 | r16 (OpenBMB recipe) | Start from r16 (the documented MiniCPM recipe; Hopper, JevK5 and Nimble were all r16) and run one r64 arm in the E4 pilot. Rank is cheap to test and plans should not pre-commit |
| C8 | Mixture size | 300K rows / 120M tok | 80K → 200K | 40–60K → 120K | Start at 60K (Nimble 4.9K rows → 39.57 shows volume isn't the bottleneck). Scale only on held-out gain, per Sol/Astra |
| C9 | Suite index edition | Own `index021.py` re-weighting port, parity-proved to ±0.01 | Pinned kit commit 19ad28e, edition 0.2, labelled honestly | Pinned 0.2, never relabel | **Both.** Report 0.2 from the kit as the official number. Task's 0.2.1 port is valid only after the ±0.01 parity proof on ≥3 published entries (leaderboard.md L186 says the maintainer's 0.2.1 builder is not public) |
| C10 | Custom readout | Letter slot (E3 compares against a pointer head) | Equivariant set-attention readout (E3, 24–48 h) | Letter slot only | Letter slot is the default everywhere. Astra's E3 is **gated** on the E1 permutation-consistency slice showing ≥3-pt order sensitivity after shuffle augmentation. Kev's pointer head < Hopper (competitor-winnow.md) is weak prior evidence against custom heads |

### C1 reasoning (primary tier)
- In task-plan's arithmetic, 4B LoRA on the 3090 costs ~27 h of wall time for 300K rows, which is affordable.
- The 3–10B tier has 4 beatable anchors: Decider 4B 40.70, Winnow-E4B 39.89, Hopper 39.67, JevK5 (task-plan §2, from leaderboard.md).
- The 2B tier has one weak anchor (Decider 2B 28.97).
- Sol and Astra prefer MiniCPM because its Llama architecture removes the DeltaNet isolation and LoRA-target risk (base-models.md L13, L38). That is a real engineering risk, not a quality argument.
- Recommended order: MiniCPM5-2B pilots first (lowest tooling risk), then the 4B main run.
- The 12B tier stays gated as in task-plan E15. Astra's ≥52 target is unanchored: Winnow is 50.02 with a private data family.

### C2 reasoning (teacher)
- T1 is Apache, has no suite-selection exposure, has the best calibration among stock models (ECE .021, competitor-xor.md L30), and costs one forward pass.
- AutoJev (56.40) is stronger, but:
  - it has board-flagged selection exposure;
  - its synthetic data came from `gpt-5.6-sol` (competitor-autojev.md L67), so using it adds a third-party-output provenance chain (§4).
- A reasoning-mode teacher costs ~10–50× the tokens. That is justified only if it beats T1 on gold NLL.
- "Mean of two teachers" mirrors TypeSafe (jev-reference.md L103), but it only helps if the teachers have errors that are not correlated. Qwen-family T1 and Qwen-based AutoJev share a base, so the mean mostly buys calibration. Astra's cross-family Gemma check is the better diversity control.

### C4 reasoning (SargeDev yuri_v3)
- Output rights are unverified (datasets.md L11).
- The states are short and templated (median 214 chars, ≤16 options).
- Imitating Jev caps us at Jev's blind spots (Astra §4).
- Sol's E3 then needs a replacement: "hard vs 0.7 gold + 0.3 soft from the **T1 teacher** on the same rows".
- The ToS check is a one-hour prerequisite that nobody did. See §4 G5.

### C5 reasoning (4B Base vs instruct)
- Kev used -Base (competitor-winnow.md L58). Hopper, JevK5 and AutoJev used instruct-lineage checkpoints.
- There is no published head-to-head, so this is a genuine empirical question.
- Running it zero-shot first is ~2 h on the 3090. Promote both variants to a 100K LoRA run only if they are within 2 pt of each other.

### C6 reasoning (encoder tier)
- Gate: within 3 pt of the 2B decoder on source-held-out macro accuracy, at ≥3× throughput.
- The product decision is the latency budget: the existing ONNX INT8 ModernBERT is the incumbent.
- task-plan's 5-pt gate is too loose to justify a third model family. Sol's absolute 14-index gate is meaningless without a parity-proven index.

## 3. Things important that NOBODY checked (or only asserted)

| # | Item | Status | Evidence / recommendation |
|---|---|---|---|
| G1 | **Serving stack for board submission** | Only asserted. The board rule is `/v1/systemone` HTTP, or a custom `Engine(state,questions)`, with no truncation or label pruning (leaderboard.md L188). task-plan says "HF/vLLM + prefix caching + GGUF". Nobody checked that **vLLM exposes last-position logits over 255 arbitrary token ids without generation** | Existing precedents: Decider `serve_vllm.py` (competitor-decider.md L77), Jevfire vLLM LM-head scoring, Winnow llama.cpp KV-fork (competitor-winnow.md L33). **Recommendation:** fork `decider.serve_vllm` and verify it on each base in E0. vLLM needs working NVML (hw-3090.md L70), so the reboot is a hard prerequisite for this path, not a nicety |
| G2 | **Question isolation on hybrid bases** | Flagged but not tested. On Qwen3.5 (3 DeltaNet : 1 attention, base-models.md L13) a tree/block mask cannot isolate branches, because the recurrent state is sequential. task-plan's risk table proposes KV/state forking. Astra says don't tree-mask DeltaNet. Sol has only a drift test | Correct design: **prefill the shared state once, snapshot both the DeltaNet recurrent state and the attention KV, and run each question branch from a copy.** This needs a per-architecture cache-copy implementation (the HF `Cache` for Qwen3.5 hybrid layers). Nobody checked that HF/vLLM supports copying the linear-attention state. The E0 gate should be the isolation probe (|Δp| < 1e-3 BF16). The fallback is one full forward per question (cost ×Q). MiniCPM (pure Llama) and Gemma local/global attention can use tree masks |
| G3 | **255 single-token codes per tokenizer** | Asserted (AutoJev loader, competitor-autojev.md L37; Astra and task "validate in context") | **[NEW CHECK] CONFIRMED: vocabulary is sufficient.** I loaded each `tokenizer.json`: MiniCPM5 has 26 A–Z + 676/676 uppercase pairs as vocab entries (512 with the Ġ-space prefix); K2 has 26 + 676 (580 Ġ); Qwen3.5-4B has 26 + 562 (544 Ġ). So ≥255 candidate codes exist in all three. **Still unverified:** that the pairs stay single tokens *after the chat-template prefix* (BPE merges across the preceding character). This must stay an E0 unit test, per AutoJev. Also note that Qwen's vocab lacks 114 bare pairs, so the code list is tokenizer-specific: AutoJev's Qwen loader is not portable (competitor-autojev.md L162) |
| G4 | **transformers version** | Plans say "pin ≥ min". [NEW CHECK] MiniCPM5 config says 5.6.2; the repo is on 4.57.6 | Use a separate `llm` uv extra or venv. Do not upgrade the ModernBERT/ONNX export path. Astra and Sol already say so; task-plan's "add to pyproject" should be an isolated extra |
| G5 | **License of teacher outputs** | Qwen3.6/3.8 and AutoJev weights are Apache (base-models.md; competitor-autojev.md L156), so outputs are usable. AutoJev, however, was trained on `gpt-5.6-sol` synthetic data (competitor-autojev.md L67): the provider's terms bind AutoJev's author, not us, but release notes should disclose the lineage. Rune is Gemma-derived; not needed. Jev outputs (SargeDev) have **unverified TypeSafe/OpenRouter ToS** in all three plans. Nobody fetched the TypeSafe terms page | Before E-T: (a) fetch TypeSafe ToS and OpenRouter's pass-through terms; (b) record teacher lineage in the model card; (c) default to T1 (Qwen, clean Apache lineage) |
| G6 | **Multilingual** | Jev is English-primary, weaker on other languages including CJK (jev-reference.md L77). tasksource includes xnli/X-CSQA; Open-Jev has zh; yuri_v1 has Spanish (datasets.md L18). The board has iSarcasmEval "track A, English" only. **None of the 3 plans sets a language target or a multilingual slice** | Recommendation: declare v1 English-only in the API contract (`Unsupported` is not needed, just documented). Keep tasksource multilingual rows at natural share (≤5 %) and add a 200-item non-English slice to the slice suite for monitoring only. mmBERT is the encoder fallback if a requirement appears (base-models.md L36) |
| G7 | Latency of the question-count dimension | Only Astra benchmarks questions = 1/8/32 | Board decisions share state across questions. Add Q ∈ {1, 8, 32} to task-plan's latency matrix |
| G8 | Score/Noul readout on MiniCPM | Isolated per-level Noul (Decider) is assumed | Fine, but check that the "true"/"false" (or yes/no) tokens are single tokens in the MiniCPM template. This belongs in the same E0 test as G3 |
| G9 | Command-risk product link | task-plan gives a product target (≥47.3 % exact-level); Sol and Astra keep the incumbent separate | Adopt task-plan's product check. It is the user's real use case (brief: Jev 47.3 vs ModernBERT 37.3) |
| G10 | Suite prerequisites | ~7 GB download / 17 GB working space, HLE gated terms, HoVer DB (leaderboard.md L187) | No plan schedules accepting the HLE terms or the build time. Add it to day 1 of week 1 |

## 4. Alternative viewpoints under-represented

- **A1: "Ship stock 27B + readout, don't train."**
  - Decider-chat reaches 51.35 with zero training.
  - Jevfire reaches 49.37 at 78 ms median with FP8 prefix batching (competitor-xor.md L31).
  - On a rented PRO 6000 this beats every plan's 4B target by ~9 points. The cost is ~$1.3/h of serving, not training.
  - Plans treat it only as a baseline or teacher. It deserves one explicit row in the decision rubric: *if the product can tolerate a 27B server, E0 already answers the question.*
- **A2: "Encoder is enough for command-risk."** This is the program kill fallback in task-plan. Astra and Sol keep the incumbent separate. That is consistent.
- **A3: "Fine-tuning hurts."** reflex found that no FT helped at 27B (competitor-xor.md L39). All plans answer this correctly with a matched stock-base control and the ≥+2 / +10 gates.

## 5. Bottom line

- **Coverage:** every requested item has a concrete verdict somewhere.
- **Main gaps:**
  - the serving and isolation implementation on DeltaNet (G1, G2)
  - Jev/TypeSafe output ToS (G5)
  - no multilingual stance (G6)
  - the transformers 5.x jump (G4)
- **Contradictions:** resolve C1–C10 as above.
- **Merged program:**
  - task-plan's harness, leakage controls and teacher T1;
  - Sol/Astra's MiniCPM-first pilots and small initial mixture;
  - Astra's cross-family teacher check;
  - SargeDev excluded from shipped weights;
  - GPU authorisation ≈ $250, with the 12B/27B tiers gated.
