# DeepSeek V4 / V4.1-Flash / FlashMemory / NSA — relevance to a short-input Jev-style decision classifier

Sources fetched 2026-09-26 (arXiv abs + full HTML): 
- V4.1-Flash https://arxiv.org/abs/2609.19969 (HTML https://arxiv.org/html/2609.19969)
- V4 https://arxiv.org/abs/2606.19348
- FlashMemory-DS-V4 https://arxiv.org/abs/2606.09079 (v3, 20 Jul 2026)
- NSA https://arxiv.org/abs/2502.11089

All four URLs loaded; no 404s.

## Brief-number verification

| Brief claim | Paper says | Verdict |
|---|---|---|
| V4.1-Flash: shallower input encoder / deeper decode; 8B active prefill vs 16B decode | 552B MoE backbone, 40 layers = 20-layer causal encoder + 20-layer decoder; decoder global KV projected from the layer-L/2 hidden state (`C_l = H_{L/2} W_l^KV`), so prefill runs only the bottom half; "8B activated per token during prefill and 16B during decode". Inspired by YOCO. SWA still runs layer-wise across all layers. (2609.19969 §2.2, §4.2.1) | **Correct, but "shallower encoder / deeper decode" is misleading**: it is a 50/50 split, and the decoder is the same depth. Prefill skips the top half for *global* KV only. |
| ~1/4 V4-Flash global KV | "global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of … DeepSeek-V4-Flash"; via CSA2 cross-layer KV reuse + FP4 KV; persistent KV ~1/8 with "SWA Bounded Replay" (abstract) | Correct. Note the 1/4 comes mostly from cross-layer reuse + FP4, not only from CED. |
| V4: local + compressed sparse + heavily compressed global attention | V4-Pro 1.6T/49B act., V4-Flash 284B/13B act., 1M ctx; hybrid CSA (compress KV by m, then DeepSeek Sparse Attention top-k via "lightning indexer") + HCA (much larger m′≫m, dense, no sparsity) + small sliding-window branch; mHC residuals; Muon; >32T tokens. At 1M ctx V4-Pro needs 27% single-token FLOPs and 10% KV of V3.2 (abstract, §2.3) | Correct. |
| FlashMemory: small trained evidence indexer on frozen backbone; avg 77.5 vs 76.9, 13.5% KV; MRCR 76.0→48.0 | "Lookahead Sparse Attention" with Neural Memory Indexer; dual-encoder trained backbone-free (only query encoder trained; indexer keys precomputed and frozen); labels from frozen DS-V4-Flash golden attention with cross-layer majority vote over 21 CSA layers. Avg 76.9 (0.93 GB) → 77.5 (0.10 GB); 13.5% KV; MRCR 76.0%→48.0%. Also: context-independent queries do NOT collapse to ~0 retrieval (LongMemEval-S no-context 96.7→95.0) and a length-generalization ceiling (§3.3). Indexer rank r=2048 — "not PEFT-style LoRA". | Numbers correct. "Small" indexer is an overstatement [INFERENCE: r=2048 query encoder is sizable]. Tested only at ≥~100K–1M ctx. |
| NSA: compressed + selected + local; 9x fwd / 6x bwd at 64K | Three branches (compressed tokens, top-n selected blocks, sliding window) gated; pretrained 27B-total/3B-active GQA+MoE, 30 layers, 260B tokens. "up to 9.0× forward and 6.0× backward speedup at 64k"; decoding up to 11.6× at 64K; expected decode speedup only 4× at 8K (memory access 2048 vs 8192 tokens) | Correct. Speedups grow with length; at ≤4K the sparse budget (~2K tokens+) ≈ full sequence → ~no gain [INFERENCE from Table 4 extrapolation]. |

## Core skepticism
Our workload: prompt = question + candidate descriptions + evidence, mostly <4K tokens, one forward pass, read label logits. At <4K:
- Attention is a small fraction of FLOPs for 2B–27B models; MLP dominates [INFERENCE, standard FLOP accounting: attention-score FLOPs ≈ MLP FLOPs only near seq ≈ 6·d_model, i.e. ~16–30K for d=2.5–5K].
- There is no decode phase; KV cache size is irrelevant to a single-pass classifier (except when batching many requests or caching shared prefixes).
- All four papers require pretraining-scale changes (new attention, new layer wiring) or an indexer trained against a frontier backbone. None has a LoRA-retrofit result on a dense 2–27B model.
So: **none of the attention mechanisms should be adopted as-is.** What transfers are two cheap *ideas*: (a) CED/YOCO-style "don't run the full depth over the whole input", (b) learned query/label-conditioned evidence selection — both only matter if we see long inputs.

## Per-paper analysis

### 1. DeepSeek-V4.1-Flash (CED)
- **Mechanism**: bottom-half layers are an encoder; top-half layers read global KV projected from layer L/2, so prefill costs ~half the depth; decode tokens run full depth.
- **Transfer**: in a classifier, "decode" ≈ the few readout tokens (answer slot / label tokens). Analogue: run the context through all layers is unavoidable in a stock decoder, BUT two retrofits are feasible:
  1. *Layer truncation / early readout*: drop top k layers, attach readout at layer L−k, LoRA-heal. Directly cuts prefill (which is 100% of our latency).
  2. *YOCO-retrofit*: for layers > L/2, replace per-layer K/V of context tokens with learned projections of H_{L/2} (new W_l^KV init from existing W_K/W_V applied to H_{L/2}), while readout tokens run full depth. Requires custom attention wiring + distillation; nobody has shown this works as a post-hoc LoRA retrofit [UNVERIFIED — no such result in this paper].
- **Experiment**: base = our best SFT/LoRA model (e.g. 4B/9B). Variable: readout depth {L, 0.75L, 0.5L} with LoRA heal + KL distillation from full-depth teacher logits over candidates. Metrics: decision index (or our held-out suite) accuracy, Brier/NLL/ECE after temperature scaling, p50/p95 latency incl. prefill at 512/2K/4K tokens, batch 1 and 16.
- **Cost**: layer truncation is trivial (config edit + LoRA on 3090 for ≤9B; hours). YOCO retrofit: 1–2 weeks eng, custom modeling code, [INFERENCE] likely 1–3 rented A100/H100-days of distillation for 9B.
- **Priority**: layer-truncation = **GO (low, cheap, after main baselines)**; YOCO retrofit = **NO-GO** for now.

### 2. DeepSeek-V4 (CSA + HCA + SWA, mHC, Muon)
- **Mechanism**: compress KV by m, lightning-indexer top-k sparse attention; heavily compressed dense global branch; SWA local branch. 1M-context efficiency.
- **Transfer**: Only conceptual: "label-conditioned evidence selection" — score evidence chunks against each candidate, keep top-k. For <4K inputs this is unnecessary; could help if some benchmarks have long evidence (check leaderboard per-benchmark length distribution first). mHC/Muon are pretraining choices, irrelevant for LoRA.
- **Using V4 itself as a base/teacher**: V4-Flash 284B/13B-active is far beyond 3090; only viable as a teacher via API for distillation labels [INFERENCE; pricing not checked here].
- **Experiment (only if long-evidence slices exist)**: baseline = full-context LoRA model; variable = retrieve top-k evidence chunks by a cheap embedding scorer conditioned on question+candidates (k∈{2,4,8}) vs truncation vs full. Metrics as above, stratified by input length and evidence dispersion.
- **Priority**: **NO-GO** for architecture; teacher use = maybe (defer to distillation workstream).

### 3. FlashMemory-DS-V4 (LSA indexer)
- **Mechanism**: dual-encoder memory indexer trained offline on golden attention labels from the frozen backbone; keeps only query-critical KV chunks at decode.
- **Transfer**: The *training recipe* (train a small scorer on the frozen model's own attention/importance signal) transfers to an evidence pruner for long inputs. But its failure modes are exactly ours: aggregation-heavy tasks (MRCR 76→48) and context-independent queries still pay overhead. Multi-label decisions that need aggregation across evidence are the MRCR-like case → risky.
- **Experiment**: only in a long-input stress slice: frozen classifier, label chunk importance by leave-one-chunk-out Δ candidate log-loss (cheaper than attention for us), train a small cross-/bi-encoder to predict it; evaluate accuracy/ECE vs kept-fraction {10,25,50,100%}, stratified by aggregation vs single-evidence tasks.
- **Cost**: moderate (3090 feasible for a ≤300M scorer), but value near-zero at <4K.
- **Priority**: **NO-GO** unless inputs >8K are ≥10% of traffic.

### 4. Native Sparse Attention
- **Mechanism**: compressed + selected-block + sliding-window branches with learned gates, Triton kernels; natively trained (27B/3B-active MoE, 260B tokens).
- **Transfer**: Speedups vanish at short length (Table 4: 4× expected decode at 8K, and our classifier has no decode). Retrofitting NSA into a pretrained dense model needs continued pretraining at 10^10+ tokens [INFERENCE]. Eval-design idea worth keeping: stratify by length/evidence count (overlaps Sparse Frontier item).
- **Priority**: **NO-GO**.

## Priority summary

| Idea | Go/No-go | Where | Cost |
|---|---|---|---|
| Layer truncation / early readout + LoRA heal + logit distillation (CED-inspired) | GO, low priority, after SFT baselines | 3090 for ≤9B | ~1–2 days |
| Shared-prefix KV caching for multi-question same-evidence batches (the practical "decode vs prefill" lesson) | GO (serving eng, no training) | any | hours |
| Label-conditioned evidence retrieval for long inputs (V4/FlashMemory-inspired) | Conditional: only if length histogram shows >8K mass | 3090 | days |
| YOCO/CED retrofit, NSA/CSA retrofit, FlashMemory indexer | NO-GO | — | pretraining-scale |

Measurement rule for any efficiency trial: report latency including prefill at fixed input-length buckets (512/2K/4K) on the same GPU, and accuracy + calibration (Brier, NLL, ECE after held-out temperature scaling) — efficiency wins that cost >0.5 decision-index points or raise ECE should be rejected.
