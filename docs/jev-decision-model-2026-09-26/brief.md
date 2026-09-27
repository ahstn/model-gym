# Shared brief: open Jev-style zero-shot decision model

Date: 2026-09-26. Repo: /home/ahstn/git/model-gym (existing project: 5-level command-risk ModernBERT/DeBERTa classifier with ONNX INT8 export; prior Jev comparison at docs/jev-laya-comparison-2026-09-19.md — Jev 47.3% exact-level vs ModernBERT 37.3% on our command-risk set; Jev hosted p50 311 ms).

## Goal
Decide how WE should train our own open Jev-style "decision model" (zero-shot classification: runtime-defined Choice / Score / Noul questions with candidate descriptions, calibrated typed probabilities). Output: which trials, experiment runs, base models, and whether/what extra hardware to rent.

## Reference (closed source)
- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://docs.typesafe.ai/introduction (also https://docs.typesafe.ai/models.md)

## Leaderboard
https://huggingface.co/spaces/multimodalart/jev-decision-index — "55 open reproductions of TypeSafe Jev's Decision Model on the same benchmark suite, 121K decisions per model across 43 benchmarks. suite v0.2, Jev jev-1.13.0, reproductions run on 1x NVIDIA RTX PRO 6000, updated 2026-09-26." Tabs: summary, head-to-head, full results, calibration, all benchmarks, methodology. Space repo data e.g. data/methodology.json.
Screenshot decision index (top): Jev 51.7 (reference); AutoJev-27B 50.9 (full FT); Surogate Rune 26B-A4B 47.2 (full FT); Decider chat Qwen3.6-27B 46.1 (inference technique); Jevfire Qwen3.8-27B 45.7 (inference); Winnow-12B 45.0 (LoRA); JoshuaSP diffgemma 26B 44.2 (inference); Decider 35B-A3B 43.5 (full FT); reflex Qwen3.8-27B-FP8 39.1 (inference); Decision 1.0 Lux 9B 39.0 (head/adapter); Xor Qwen3.6-35B-A3B 38.8 (full FT); djev diffgemma 26B 37.6 (inference); Jev-Omni 12B 37.1 (LoRA); Hopper Qwen3.5-4B LoRA 37.0; Bespoke Nimble 9B v2 36.7 (LoRA); Decider 4B 36.6 (full FT); JevK5 Qwen3.5-4B LoRA 36.4; Winnow-E4B 36.0 (LoRA); Kev 9B 35.4 (LoRA); razorback16 diffgemma 26B 35.0 (inference); Solomon v1.1 28B 32.0 (head/adapter); Kev 4B 31.3 (LoRA); mmastrac diffgemma 26B 31.1 (inference); Decision 1.0 Nox 4.7B 31.1 (head/adapter); Jobe Qwen3.5-4B ~30.9 ... Parameter tier filters: <350M, 350M-700M, 700M-3B, 3B-10B, 10B+. Categories: full fine-tune, inference technique, LoRA, head/adapter.

## Candidate datasets
- https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3
- https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1
- https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions

## Competitors to study
- https://huggingface.co/denis-pplx/autojev-27b / https://github.com/denis-pplx/autojev
- https://huggingface.co/surogate/rune-26b-a4b-GGUF / https://invergent.ai/blog/rune/
- https://github.com/Mapika/decider / https://huggingface.co/Mapika/decider-2b
- https://huggingface.co/EldanRing/Winnow-12B
- https://huggingface.co/juspay/xor

## Prior GPT-6-Astra analysis (verify, do not trust blindly)
- AutoJev: SFT + constrained answer readout + held-out temperature scaling. Board documents checkpoint-selection exposure to some benchmark rows (limits unseen-task generalization claims). configs/training.json in repo; caveats in space data/methodology.json.
- Rune: repo now hosts v3; its newer 53.39 is the authors' own run, not the common leaderboard entry.
- Decider: calibration-aware RL improves browser-task success while broad OpenJev accuracy decreases slightly (docs/RL.md). Specialization != better general decision model.

## Papers to analyze (user-supplied summaries; verify numbers)
- DeepSeek-V4.1-Flash: KV cache compression (arXiv 2609.19969) — shallower input encoder / deeper decode; 8B active prefill vs 16B decode; ~1/4 V4-Flash global KV. Idea: distil full-depth classifier into shallow input encoder + deeper label/decision processing.
- DeepSeek-V4 (2606.19348) — local + compressed sparse + heavily compressed global attention. Idea: label-conditioned evidence selection; full evidence access for decision tokens.
- FlashMemory-DeepSeek-V4 (2606.09079) — small trained evidence indexer on frozen backbone; avg 77.5 vs 76.9, 13.5% KV, but MRCR 76.0→48.0.
- The Limits of Speculation (2609.22156) — MoE speculative decoding bound ~2.34x; idea: cascade escalation trained on error-reduction per unit cost.
- On-Demand Attention (2609.20734) — learn when global recall helps; idea: cheap pass then predict whether full pass improves candidate log loss.
- Modern Transformers Are Implicit Hybrids (2609.02986) — global retrieval heads vs local heads; 1.4B/100B tok, retrieval 44.64 vs 40.17.
- The Sparse Frontier (2504.17768) — sparse attention tradeoffs; eval across length, label count, evidence count/dispersion, negation, aggregation.
- GLiClass (2508.07662) — joint text+label encoding, all labels scored in one pass; degrades with large dense label sets.
- Calibration-Aware RL for decision LLMs (ACL Findings 2026, aclanthology 2026.findings-acl.610) — Qwen3-4B OpenBookQA acc 92.79→93.70, ECE 6.74→1.91.
- Dual-Head Reasoning Distillation (2509.21487) — train-time-only reasoning.
- LoLCATs (2410.10254) — low-rank linearizing; retrieval-focused conversion data needed.
- Native Sparse Attention (2502.11089) — compressed + selected + local attention; 9x fwd / 6x bwd at 64K.

## Base model candidates (user-suggested)
- openbmb/MiniCPM5-2B — Llama arch; OpenBMB TRL/PEFT/Unsloth recipes (https://github.com/OpenBMB/MiniCPM/blob/main/docs/finetune/trl.md: BF16, r16 LoRA, grad ckpt, 2048 seq).
- IFM/K2-Horizon-3.7B — dense, Apache-2.0, intermediate checkpoints; budget as 5.06B.

## Hardware
Local: 1x RTX 3090 24 GB, 20 CPU threads, 60 GB RAM, ~440 GB free disk. NOTE: nvidia-smi currently fails "Driver/library version mismatch" (NVML 595.91) — likely needs reboot after driver update. Rentable compute possible: need to decide if needed over the 3090, what GPU(s), and where (Daytona, RunPod, Vast.ai, Lambda, Modal, etc.). Leaderboard reproductions run on 1x RTX PRO 6000 (96 GB).
