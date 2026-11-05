# 9–12B vision backbone refresh — 30 October 2026

**Recommendation:** retain Qwen3.5-9B as the general training control. Add **ZwZ2-9B** as the most promising newly released starting checkpoint, conditional on clarifying its weight licence. Add **PortraitAes** as a portrait/aesthetics specialist reference and method source. **Clef Flash** merits a separate scoring-interface experiment. None has demonstrated a matched win on our technical-quality and aesthetic targets.

The strongest additional architecture comparison is **Step3-VL-10B**, an older release omitted from the previous shortlist. Taichu5 is a secondary perception candidate. Bonsai 2 is more compelling for deployment efficiency than as the first checkpoint to fine-tune.

This is a research recommendation, not a training result. No model inference, training job, weight download, or paid teacher call was performed.

## Scope and evidence

The previous [backbone review](unified-and-backbones.md) was verified on **27 September 2026**, 33 days earlier. This review distinguishes releases since that date from older models newly added to our shortlist. An updated README or search-index date does not establish a new weight release.

The requested range is **9–12 billion total logical parameters**, including vision components and added scoring heads. Active MoE parameters and packed quantization tensor counts are insufficient. For this review the new range takes precedence over the older plan's compact-student preference; that historical plan has not been rewritten.

“Starting checkpoint” includes post-trained/instruction-tuned VLMs. Where an actual pretraining-only Base checkpoint exists, it is identified separately. Published instruction-model scores must not be attributed to its Base counterpart.

Research used **six sequential rounds each of Exa and Hugging Face**, with three parallel verification workstreams. The HF search connector failed with `Tool model_search not found`; repository lookup worked. The official Hub CLI supplied catalogue discovery. The expanded catalogue contains 1,000 records spanning 17 July–30 October, including 409 repositories created from 28 September onward. It is metadata screening, not 1,000 independently evaluated models. Exact dates and trainability were checked on shortlisted original repositories.

See the [research ledger](evidence/backbone-refresh-2026-10-30/README.md), [search audit](evidence/backbone-refresh-2026-10-30/search-audit.json), and [catalogue snapshot](evidence/backbone-refresh-2026-10-30/hf-round5-expanded-catalogue.json). Benchmarks below are publisher-reported and were not reproduced.

## General starting checkpoints

The table is a shortlist, not a common-protocol leaderboard. Parameter counts are full-precision HF totals unless a calculation is explicitly described.

| Checkpoint | Total B | New since 27 Sep? | MMMU-Pro | Training relevance and decision |
|---|---:|---|---|---|
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.653 | No; March | **70.1**, publisher post-trained model | **Default/control.** Public [Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) and post-trained weights; Apache-2.0. Same family as the existing Q-ReAlign Pro reference. |
| [ZwZ2-9B](https://huggingface.co/inclusionAI/ZwZ2-9B) | 9.410 | **Yes; weights 16 Oct** | Not reported | **First new checkpoint challenger.** Qwen3.5-9B plus perception-focused RL; BF16 weights, training code and data available. Code Apache-2.0; weight licence not explicitly stated. |
| [Step3-VL-10B](https://huggingface.co/stepfun-ai/Step3-VL-10B) | 10.172 | No; January, newly shortlisted | **64.08** single response; **67.18** with 16-rollout PaCoRe | **First additional architecture comparison.** 1.8B Perception Encoder + Qwen3-8B decoder. Public [Base](https://huggingface.co/stepfun-ai/Step3-VL-10B-Base) and Chat BF16 weights; Apache-2.0. Custom implementation; turnkey fine-tuning recipe unverified. |
| [Taichu5 / ZDTaichu5.0-9B](https://huggingface.co/TaichuAI/ZDTaichu5.0-9B) | 9.794 | No; initial release commit 15 Sep | Not reported | **Secondary perception experiment.** MMStar 76.80, CV-Bench 86.82, RealWorldQA 76.99. BF16/custom implementation, adaptive recurrent computation; NVIDIA Open Model License and upstream notices. Training integration unverified. |
| [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it) | 11.960 | No; June, already reviewed | **69.1** thinking at 1,120 visual tokens; **67.7** at 280 | **Architecture reserve.** Encoder-free; public [Base](https://huggingface.co/google/gemma-4-12B) and IT, Apache-2.0. Preserve prior fine-detail/quality reservations. Exclude the optional ~0.4B drafter under a strict 12B system limit. |
| [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) | 10.293 | No; December 2025, already reviewed | **60.6**, publisher | **Practical reserve.** MIT and [verified LLaMA-Factory integration](https://github.com/hiyouga/LlamaFactory/pull/9586). No same-release raw Base checkpoint located. |

Exact counts: Qwen 9,653,104,368; ZwZ2 9,409,813,744; Step3 10,171,750,144; Taichu 9,794,197,510 BF16 parameters plus two integer buffers; Gemma 11,959,730,224; GLM 10,292,777,472. Different Qwen derivatives omit or add components, so compare the actual released configuration rather than assuming the marketing name is an exact count.

### Why ZwZ2 changes the shortlist

The [paper's own Qwen-versus-ZwZ2 comparison](https://arxiv.org/html/2610.37037#S4.T1) is stronger evidence for its incremental value than comparing scores across unrelated model cards:

| Accuracy (%) | Qwen3.5-9B | ZwZ2-9B | Change |
|---|---:|---:|---:|
| VStar | 83.77 | 90.58 | +6.81 |
| HR-Bench 4K | 84.88 | 89.75 | +4.87 |
| HR-Bench 8K | 82.50 | 87.62 | +5.12 |
| ZoomBench | 49.57 | 59.53 | +9.96 |
| MMStar | 77.27 | 78.27 | +1.00 |
| PerceptionBench | 38.70 | 38.89 | +0.19 |
| CountQA | 37.63 | 48.43 | +10.80 |
| Mean over all 25 benchmarks | 51.63 | 56.59 | +4.96 |

These results suggest a useful fine-detail-perception starting point; they do **not** establish sensitivity to blur, compression, noise, composition or aesthetic appeal. The very small PerceptionBench gain also argues against describing every perceptual ability as substantially improved.

The work uses about 20.4K verified multi-region questions and GRPO, without additional SFT. Training sees full images; zoomed views help construct supervision, and inference requires no zooming tool. The [v2 code](https://github.com/inclusionAI/Zooming-without-Zooming-v2) and [data](https://huggingface.co/datasets/inclusionAI/ZwZ2-RL-VQA) are public. Its 16 October [weight upload](https://huggingface.co/inclusionAI/ZwZ2-9B/commit/381a132f4ddfb0a5aa8638113a2bf6c457a29d4d) predates the late-October paper/card updates.

The paper evaluates in low-thinking/non-thinking modes, but exact per-benchmark equality of processor budgets was not independently established. Its [9B launcher](https://github.com/inclusionAI/Zooming-without-Zooming-v2/blob/main/verl/train_zwz2_qwen3_5_9b.sh) defaults also differ from the paper's training settings: 10 versus 2 epochs, train batch 512 versus 128, maximum response 1,900 versus 32K. Public code is useful, but its current defaults should not be called a verified reproduction recipe. The weight repo lacks an explicit LICENSE and licence metadata, despite the code's Apache-2.0 licence.

### How to read the visual benchmark numbers

- **MMMU-Pro tests multimodal reasoning, not photographic quality.** Its overall score averages standard ten-option questions and vision-input versions. Record the subset and protocol; ordinary MMMU, MMLU-Pro and MMMU-Pro are different benchmarks. [Benchmark paper](https://arxiv.org/html/2409.02813v2).
- **Qwen's 70.1 belongs to the post-trained checkpoint.** Its card says thinking is enabled by default but does not establish the exact MMMU-Pro generation budget in the inspected table. Do not label this a verified non-thinking result. Other card scores include MMStar 79.7, RealWorldQA 80.3 and OCRBench 89.2. [Publisher card](https://huggingface.co/Qwen/Qwen3.5-9B).
- **Step3's compute matters.** Its single-response maximum is 65,536 tokens. PaCoRe uses 16 rollouts plus synthesis, with a 131,072 maximum. The 67.18 result is unsuitable as a cheap single-pass scoring claim. The same report gives BLINK 66.79, MMStar 77.48 and OCRBench 86.75. Its multi-crop input and 16× spatial downsampling require checking on subtle image defects. [Report](https://arxiv.org/html/2601.09668v2), [architecture](https://github.com/stepfun-ai/Step3-VL-10B).
- **Gemma's pixel budget matters.** The [report](https://arxiv.org/html/2607.02770v2) explicitly distinguishes thinking-mode results at 1,120 and 280 tokens. The previous review's single-split BrightRate-LM result was weaker for this 12B model; it remains a task-specific caution, not proof that the architecture is generally unsuitable. See [the prior comparison](unified-and-backbones.md#22-brightrate-lm-study-one-task-one-split).
- **GLM's 60.6 is the publisher result.** The [original chart](https://raw.githubusercontent.com/zai-org/GLM-V/refs/heads/main/resources/bench_46v.jpeg) and card use a 16K output budget. Step3's re-evaluation reports 59.93 under its setup; retain separate attribution.
- **Missing scores stay missing.** Neither Taichu nor ZwZ2 reports MMMU-Pro in the inspected sources. Taichu's 77.20 is **MMLU-Pro**. Its [release page](https://taichu-ai.github.io/ZDTaichu5.0-9B/) reports spatial/perception results with adaptive computation, without a sufficiently specified common inference budget.

## New specialist and decision checkpoints

### PortraitAes: strongest new direct aesthetics connection

[PortraitAes](https://huggingface.co/Junzou/PortraitAes) is a **9.410B Qwen3.5 derivative**, with an October 4 paper and an October 11 HF repository creation date. The code and model card declare Apache-2.0. It routes portraits across nine photographic intents and six assessment dimensions, including image quality, composition and tone. Its public [training code](https://github.com/junzhouxie/PortraitAes) includes LoRA SFT, score calibration and cross-image ranking/GRPO.

The [paper](https://arxiv.org/html/2610.05010) reports overall **PLCC / SRCC 0.924 / 0.934** on the standard set and **0.829 / 0.795** on the hard set. These measure agreement with rubric-driven multimodal-model labels checked by experts, rather than broad independent human MOS. The standard set is **1,600 images expanded into 11,200 linked tasks**, and the hard set has 1,000 images. Benchmark data are not distributed under the code licence. No MMMU-Pro result was located.

Use it as a portrait-specific frozen reference and possibly a continue-tuning control. Its intent-aware supervision is a useful training idea. Its fused overall score should not replace our separate technical and aesthetic targets, and its portrait-only evidence does not establish performance on landscapes, documents, products or graphics.

### Clef Flash: useful output-head experiment

[Clef Flash](https://huggingface.co/Cloudflare/clef-flash) launched **October 1**, Apache-2.0. It adds a typed decision head to Qwen3.5-9B and returns probability distributions over allowed outcomes, including ordinal scores. The calculated total is **9,531,576,564**: 9,409,813,744 backbone parameters plus a separate 121,762,820-parameter head. The HF backbone badge alone omits this head. [Head configuration](https://huggingface.co/Cloudflare/clef-flash/blob/main/joint_head_config.json), [implementation](https://huggingface.co/Cloudflare/clef-flash/blob/main/joint_schema_model.py).

The [training description](https://blog.cloudflare.com/clef-decision-models/) uses frozen Qwen weights, adapters and the head, with calibration and ordinal partial credit. No standalone public training recipe/dataset was verified. No MMMU-Pro or IQA/IAA result was found; its 65.3 is **MMLU-Pro**. Its value here is a possible scoring/readout design, not evidence of a stronger visual representation. Compare it with a simple ordinal head on stock Qwen before taking on its additional machinery.

### Bonsai 2: deployment watchlist

[Ternary Bonsai 2 9B](https://docs.prismml.com/models/bonsai-2-9b), released **October 15**, is an approximately **9.41B** Qwen3.5-derived model, Apache-2.0. The publisher reports **MMMU-Pro 68.73**, versus **72.43 for its own Qwen3.5-9B run**; that paired comparison is more useful than mixing it with Qwen's 70.1 card result. [Release and evaluation](https://prismml.com/news/bonsai-2-family).

Publisher releases are [GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf) and [MLX packed weights](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit). An [unvalidated community BF16 conversion](https://huggingface.co/jonathan-kellerai/Ternary-Bonsai-2-9B-hf-bf16) makes ordinary fine-tuning plausible, but dequantization does not recover the original dense weights or retain the ternary memory advantage. Benchmark parity and a working training path remain unverified. Prefer a full-precision training control, then assess deployment compression separately.

## Other candidates screened

| Model | Outcome |
|---|---|
| [JEV-9B vision path](https://huggingface.co/autotrust/JEV-9B) | Image support added 3 Oct. About **9.741B** including its stored vision adapter; main 8.954B HF badge is text-only. Unchanged Qwen vision plus a text-trained decision adapter used zero-shot on images. Author VL-RewardBench 74.34% is not IQA/IAA MOS. Optional decision control; no MMMU-Pro found. |
| [Mapika Decider 12B Vision](https://huggingface.co/Mapika/decider-12b-vision) | 20 Oct, **11.960B**, Gemma derivative with merged visual LoRA and token-logit scoring. Author mean 87.0% versus 78.1% on eleven small evaluations; no MMMU-Pro/IQA/IAA. Documented 280-token input limit and some non-commercial training data. Lower-priority readout experiment. |
| [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) | **9.410B**, September 21 weights/22 announcement, not an October release. MIT metadata. Shipped SFT visual-coding mini score 64.0; 72.4 belongs to a separate RL experiment. No direct quality or MMMU-Pro evidence. |
| [Apple LensVLM-9B](https://huggingface.co/apple/LensVLM-9B) | **9.410B**, September 22 weights. Document compression/tool-use specialization. [AMLR licence](https://huggingface.co/apple/LensVLM-9B/blob/main/LICENSE) restricts weights and derivatives to non-commercial research; exclude from a commercial-training shortlist. |
| [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) | February release; **9.372B** with audio, approximately **8.696B** in its documented image-only configuration (configuration-derived). Its October commits update the card. Ordinary MMMU 70.2 thinking is not MMMU-Pro. Audio capacity adds little to this still-image project. |
| [VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B) | **10.017B**, July, Apache-2.0. Specialized region grounding/object detection; lower priority for whole-image quality scoring. |
| [FLUX 3 Klein prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) | October Qwen derivative, **9.410B**, Apache-2.0. Trained to rewrite generation/editing prompts, not to score existing image quality. |
| [BMT-GRAFT-12B](https://huggingface.co/JWang580/BMT-GRAFT-12B) | October Gemma derivative, **11.960B**; clinical-note extraction, with no relevant image evaluation. |
| [Granite Switch 4.1/4.1.1](https://huggingface.co/ibm-granite/granite-switch-4.1-8b) | New October releases but **8.776B**, text plus speech transcription, no vision encoder. Out of scope. |
| [Clef Omni](https://huggingface.co/Cloudflare/clef-omni), larger MoE releases | Clef Omni has a ~30B comprehension backbone; “3B active” does not qualify. Similarly, Nemotron 120B-A12B and Qwen 122B-A10B exceed the total-parameter limit. |

## Proposed comparison once the corpus is ready

1. **Keep Qwen3.5-9B post-trained as the training control**, with frozen Q-ReAlign Pro as the task-specialist reference. Add raw Qwen Base as a separate ablation only if the corpus and budget justify testing whether prior instruction tuning helps. General benchmark scores do not answer that question.
2. **Compare Qwen with ZwZ2-9B**, once checkpoint terms are clear. Use the same corpus, held-out source groups, output contract and adapter/vision-training budget. Starting from released ZwZ2 avoids reproducing its RL stage.
3. **Add Step3-VL-10B as the architecture comparison.** Taichu is next if integration and budget permit. Keep Gemma and GLM as reserves rather than running every candidate immediately.
4. **Evaluate PortraitAes on the portrait subset**, retaining technical and aesthetic results separately. Test Clef's decision head in a separate output-method comparison so architecture, starting weights and scoring design do not all change at once.
5. **Select on independent task evidence.** Report SRCC, raw PLCC, MAE/calibration and pairwise ordering separately for technical quality and aesthetics. Include human-reviewed score extremes, subtle defect pairs, intentional blur/grain, different subjects and cross-source holdouts. Group original images and all their degraded variants together when splitting and bootstrapping.
6. **Control what the model sees and what it costs.** Use identical delivered images, logged resize/crop settings, measured visual tokens and a high-resolution check. Equal token counts across architectures do not guarantee equal visual detail. Match data exposure and measure GPU-hours, peak VRAM and images/second. Disable long generated reasoning for the intended scorer, then test any reasoning variant separately.

The Luna/DeepSeek average can remain a versioned teacher target under the current preference. It should not be the sole test truth for selecting a student trained on those same teachers. Assistant assessments also remain reference judgments rather than independent human ground truth. A better MMMU-Pro score, cleaner JSON, or closer teacher imitation is insufficient on its own to establish better quality scoring.

This proposal supplements the existing [evaluation protocol](../evaluation/protocol.md) and [label specification](../training/label-specification-2026-10-16.md); it does not populate missing corpus labels or change split manifests.
