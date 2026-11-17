# Qwen implementation evidence

Checked 2026-11-17 through official model cards, framework documentation and GitHub source. These are compatibility leads and inspected configurations, not a tested dependency lock or measured training result for our corpus.

## Checkpoint choice

`Qwen/Qwen3.5-9B` is a dense vision-language checkpoint with a hybrid language architecture: GatedDeltaNet linear-attention layers interleaved with standard attention. The model name does not precisely describe all stored parameters, including vision components. Use the actual loaded parameter count when calculating memory. Its model license is Apache-2.0. A separate `Qwen3.5-9B-Base` checkpoint exists for pretraining-only research/adaptation.

Start with the post-trained checkpoint to retain rubric and output-format behavior. This is a practical choice for the current data, not evidence that it outperforms `-Base` after matched training. Treat 4B as a later deployment-cost challenger. A newer Qwen family name is not itself evidence of better technical/aesthetic scoring.

- [9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [9B Base card](https://huggingface.co/Qwen/Qwen3.5-9B-Base)
- [Model license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE)
- [Transformers architecture documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)

## Frameworks inspected

| Framework | Verified support | Qualification for our implementation |
|---|---|---|
| Transformers + PEFT + TRL | Multimodal processor/model classes, VLM SFT data, adapters and configurable answer loss | Best transparent control for short-answer SFT; custom score heads/losses need explicit implementation. Generic VLM support is not a completed Qwen9B smoke test |
| LLaMA-Factory | Explicit Qwen3.5 visual/projector/language registration and model-specific tutorial | Useful YAML/CLI alternative; freeze flags affect adapter placement and must be recorded |
| ms-swift | Explicit Qwen3.5-9B, 9B-Base and 4B registration with vision handling | Broad SFT/RL support; the inspected family launcher is 35B MoE on four GPUs, not our 9B single-GPU recipe |
| Unsloth | Official Qwen3.5 fine-tuning guide links a vision notebook; separate visual/language tuning switches | Optimization candidate after parity control. Defaults may adapt more components than intended; vendor memory/speed figures do not predict our image workload |
| Axolotl | Exact Qwen3.5-9B vision LoRA and full-tuning YAML examples | Concrete alternative. Hybrid-attention LoRA targets need inspection; vision examples disable sample packing |

Source inspections:

- [TRL SFT documentation, inspected revision](https://github.com/huggingface/trl/blob/b2686bf5c300756cdf878b84f83a48f794e2fb94/docs/source/sft_trainer.md): current VLM assistant-only loss has version/template requirements, including Transformers >=5.18.0 in this revision. Truncation can remove image tokens. Bound image processing explicitly; use `max_length=None` unless truncation safety is demonstrated for every sample.
- [LLaMA-Factory visual registration](https://github.com/hiyouga/LlamaFactory/blob/885a4ca65cebf17a92e869af13c2ce42caca314a/src/llamafactory/model/model_utils/visual.py), [Qwen3.5 tutorial](https://blog.llamafactory.net/en/posts/qwen3_5_finetuning/). The tutorial's robot examples are implementation evidence, not an IQA benchmark.
- [ms-swift Qwen registration](https://github.com/modelscope/ms-swift/blob/cf627bdd5d7e739a6cefc5829ab5696575615e8b/swift/model/models/qwen.py), [family launcher](https://github.com/modelscope/ms-swift/blob/cf627bdd5d7e739a6cefc5829ab5696575615e8b/examples/models/qwen3_5/transformers.sh).
- [Unsloth guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune): its current warning discourages 4-bit Qwen3.5 adaptation, although an older code comment is less restrictive. Treat this as vendor caution. The advertised approximately 22 GB BF16 LoRA figure for 9B is not a measured requirement for our views.
- [Axolotl documentation](https://docs.axolotl.ai/docs/models/qwen3.5.html), [LoRA YAML](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/qwen3.5/9b-lora-vision.yaml), [full-tuning YAML](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/qwen3.5/9b-fft-vision.yaml).
- [PEFT LoRA guidance](https://huggingface.co/docs/peft/en/developer_guides/lora), [DoRA guidance](https://huggingface.co/docs/peft/en/package_reference/lora_variant_dora).

## Memory evidence

For a conventional mixed-precision Adam implementation, weights, gradients, two moments and possibly master weights can require roughly 12–16 bytes/trainable parameter. For 9B language parameters this means 108–144 decimal GB, excluding additional vision parameters, activations and workspace. It is an accounting example, not a universal lower bound.

Axolotl documents approximately 61 GiB for its 9B vision full-tuning configuration. The inspected YAML uses `adamw_bnb_8bit`, microbatch 1, sequence length 4096 and activation checkpointing. This makes a full-tuning trial on 96 GB plausible, subject to actual visual tokens and optimizer/kernel behavior. The example also has `val_set_size: 0.0` and a learning rate intended for its demonstration; neither is an endorsed setting for this project.

LoRA saves trainable gradient/optimizer memory but still needs the backbone and activations. QLoRA reduces frozen base-weight memory; an 8-bit optimizer reduces optimizer-state memory. Gradient accumulation changes effective batch size without reducing the memory required by the largest individual example. Measure peak allocated and reserved memory on those large examples.

## Required smoke tests before renting for a long run

1. Load the conditional-generation vision model and matching `AutoProcessor`; ensure the image grid/position metadata is correct. A text-only causal model class is not equivalent.
2. Enumerate the actual adapters and trainable parameters. Qwen3.5's `linear_attn.in_proj_qkv`, `in_proj_z` and `out_proj` need deliberate treatment; a generic `q_proj/v_proj` list covers only part of the language architecture. Choose MLP/projector/vision scope explicitly.
3. Verify that inputs contain pixels and rubric only. Mask prompt/image/padding positions from answer loss; do not include reference labels, teacher text or image statistics in the input. Preserve null score masks.
4. Run a backward pass and inspect nonzero gradients in intended modules. Check output scores and task masks rather than relying only on decreasing loss.
5. Save optimizer, scheduler, adapter/head, random state and processor configuration; resume and compare the next steps. A weights-only file is not full training resume.
6. Compare the loaded adapter and merged/served checkpoint on the same frozen image examples. If deployment quantization is later introduced, recheck score calibration, tails and eligibility; do not assume format or export parity.
7. Freeze exact Torch/CUDA, Transformers, PEFT/TRL, attention-kernel, processor and checkpoint revisions after the successful Blackwell test. The code revisions above are evidence pins, not an environment validated end to end.

Do not add expensive embedding/language-output matrices to trainable `modules_to_save` merely because an example does so. We do not currently need new vocabulary tokens. If score-token probabilities are used instead of explicit heads, verify tokenization at the actual answer position; normalize the correct candidate tokens and test the expectation calculation.
