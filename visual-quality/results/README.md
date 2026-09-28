# Results

All numbers on this page are **ours**, measured with the `vq` package on the locked manifests, unless a row says "author-reported". Metric pairs are **PLCC / SRCC**. DMOS sets (LIVE, CSIQ) are negated, so higher is better everywhere.

Machine: one RTX PRO 6000 Blackwell (96 GB), RunPod container limited to about 13.6 CPUs.

- Q-ReAlign runs: torch 2.14.0+cu130, transformers 5.17.0, BF16, SDPA, batch 16, last-position logits, no pixel cap.
- OneAlign runs: its own venv, torch 2.8.0+cu128, transformers 4.36.1 (the model card pins it), FP16, eager attention, batch 32 images or 4 videos.

Raw files: `anchors/<model>/metrics.json` (config, environment, per-set metrics, 95% bootstrap intervals). Per-item predictions stay on the pod under `runs/anchors/<model>/preds/`.

## Stage 0: accuracy anchors

Sets are the ones in [protocol.md](../docs/evaluation/protocol.md). T1 = in-domain test splits of the ONE-ALIGN training sources, plus AGIQA-3K and LIVE. T2 = cross-dataset sets that no anchor trained on. Full sets, no subsampling. Videos: 8 frames per video, long side 448.

### T1

| Model | KonIQ | SPAQ | KADID | AGIQA-3K | LIVE | AVA | LSVQ | Mean |
|---|---|---|---|---|---|---|---|---|
| Q-Align OneAlign (8.2B) | 0.950 / 0.941 | 0.935 / 0.932 | 0.966 / 0.963 | 0.838 / 0.801 | 0.856 / 0.887 | 0.819 / 0.822 | 0.872 / 0.874 | 0.891 / 0.889 |
| Q-ReAlign Mini (0.8B) | 0.913 / 0.909 | 0.914 / 0.914 | 0.846 / 0.854 | 0.821 / 0.785 | 0.848 / 0.892 | 0.749 / 0.746 | 0.837 / 0.840 | 0.847 / 0.849 |
| Q-ReAlign Lite (4B) | 0.941 / 0.943 | 0.934 / 0.932 | 0.930 / 0.927 | 0.870 / 0.828 | 0.862 / 0.899 | 0.804 / 0.814 | 0.878 / 0.881 | 0.889 / 0.889 |
| Q-ReAlign Pro (9B) | 0.952 / 0.950 | 0.937 / 0.935 | 0.939 / 0.934 | 0.884 / 0.843 | 0.876 / 0.902 | 0.828 / 0.832 | 0.885 / 0.884 | 0.900 / 0.897 |

SRCC minus OneAlign, paired bootstrap over items (1,000 resamples, 95% interval). Bold = the interval does not contain 0.

| Model | KonIQ | SPAQ | KADID | AGIQA-3K | LIVE | AVA | LSVQ |
|---|---|---|---|---|---|---|---|
| Mini | **-0.031** [-0.039, -0.024] | **-0.018** [-0.023, -0.014] | **-0.109** [-0.122, -0.098] | **-0.016** [-0.029, -0.005] | +0.005 [-0.006, +0.016] | **-0.077** [-0.082, -0.071] | **-0.035** [-0.040, -0.030] |
| Lite | +0.002 [-0.003, +0.008] | -0.001 [-0.003, +0.002] | **-0.036** [-0.043, -0.030] | **+0.027** [+0.016, +0.037] | **+0.012** [+0.004, +0.021] | **-0.009** [-0.012, -0.005] | **+0.006** [+0.003, +0.010] |
| Pro | **+0.009** [+0.005, +0.014] | +0.003 [-0.000, +0.005] | **-0.029** [-0.035, -0.024] | **+0.042** [+0.030, +0.052] | **+0.015** [+0.007, +0.024] | **+0.010** [+0.007, +0.013] | **+0.010** [+0.007, +0.013] |

### T2

| Model | LSVQ-1080p | LIVE-C | CSIQ | KoNViD-1k | MaxWell | Mean |
|---|---|---|---|---|---|---|
| Q-Align OneAlign | 0.820 / 0.782 | 0.894 / 0.881 | 0.906 / 0.881 | 0.883 / 0.874 | 0.764 / 0.759 | 0.853 / 0.835 |
| Q-ReAlign Mini | 0.779 / 0.744 | 0.834 / 0.793 | 0.839 / 0.828 | 0.848 / 0.841 | 0.738 / 0.728 | 0.808 / 0.787 |
| Q-ReAlign Lite | 0.822 / 0.795 | 0.884 / 0.868 | 0.887 / 0.862 | 0.877 / 0.873 | 0.779 / 0.773 | 0.850 / 0.834 |
| Q-ReAlign Pro | 0.830 / 0.800 | 0.889 / 0.880 | 0.890 / 0.883 | 0.882 / 0.876 | 0.776 / 0.770 | 0.853 / 0.842 |

| Model (SRCC minus OneAlign) | LSVQ-1080p | LIVE-C | CSIQ | KoNViD-1k | MaxWell |
|---|---|---|---|---|---|
| Mini | **-0.038** [-0.050, -0.027] | **-0.088** [-0.107, -0.069] | **-0.053** [-0.075, -0.033] | **-0.034** [-0.048, -0.018] | **-0.031** [-0.054, -0.009] |
| Lite | **+0.013** [+0.003, +0.023] | **-0.013** [-0.025, -0.001] | **-0.019** [-0.034, -0.005] | -0.001 [-0.012, +0.009] | +0.014 [-0.001, +0.029] |
| Pro | **+0.018** [+0.010, +0.027] | -0.001 [-0.013, +0.011] | +0.003 [-0.011, +0.016] | +0.002 [-0.007, +0.011] | +0.011 [-0.004, +0.026] |

### What the anchors show

1. **The accuracy bar is OneAlign, not the README "Q-Align" row.** On our harness OneAlign reaches a T1 mean of 0.891 / 0.889. The Q-ReAlign README lists Q-Align at 0.873 / 0.869 (author-reported, converted to PLCC / SRCC). The biggest differences are KADID (ours 0.963 SRCC, README 0.912) and AGIQA-3K (ours 0.801, README 0.738). [INFERENCE] The README row is probably an IQA-only Q-Align or paper numbers, not the OneAlign checkpoint.
2. **Our harness reproduces Lite and Pro.** Lite T1 mean: ours 0.889 / 0.889, README 0.889 / 0.889. Pro: ours 0.900 / 0.897, README 0.900 / 0.896. Per-set values agree within about 0.01.
3. **The released Mini checkpoint does not reproduce its README numbers.** Ours: 0.847 / 0.849. README: 0.880 / 0.879. The gap is largest on KADID (0.854 vs 0.903 SRCC) and AVA (0.746 vs 0.797). The same harness reproduces Lite and Pro, and Mini gives the same SRCC in FP32 and with eager attention (KonIQ and KADID, 600 items each: every setting within 0.001). The Q-ReAlign example config evaluates on the first 200 items per set and keeps the best checkpoint by that score. [INFERENCE] The README Mini row may come from such a selection, or from a checkpoint that is not the one on the Hub.
4. **Mini is below OneAlign on 11 of 12 sets**, with intervals that exclude 0. It ties only on LIVE. For our goal (lower cost than Q-Align at matched accuracy), a 0.8B model trained with the released recipe is not enough by itself. R0 checks whether our own run of that recipe lands at the Mini level or higher.
5. Lite matches OneAlign on mean SRCC (T1 equal, T2 within 0.001) but is below it on KADID, AVA, LIVE-C, and CSIQ. Pro is above OneAlign on 6 of 12 sets, ties on 5, and is below it on KADID.

## Harness checks

- **Batched scorer = reference path.** Mini, 256 items: batch 1 with full-sequence logits (the Q-ReAlign scorer path) vs batch 16 with last-position logits. Prediction correlation 0.99995 (KonIQ) and 0.99987 (AVA). Largest absolute difference 0.008 on a 0–1 scale.
- **Prompt wording does not matter.** Mini, KonIQ, 600 items: our fixed prompt vs the per-item prompt in the test JSON, with and without a space before the answer stem. SRCC 0.9062–0.9066 in all four cases.
- **Image resizing is the same as ms-swift.** ms-swift resizes Qwen3.5 images with `qwen_vl_utils.fetch_image` (at most 16,384 visual tokens; minimum 4). The Hugging Face processor default gives the same size limits. KonIQ images need no resize under either path.

## Training (R0)

Qwen3.5-0.8B, full fine-tune on the ONE-ALIGN mix (287,702 rows; one LSVQ training video has no frames), released Q-ReAlign recipe: AdamW lr 2e-5, betas (0.9, 0.95), weight decay 0.1, cosine with 3% warm-up, 2 epochs, global batch 16. Config: [`configs/r0-mini-full.yaml`](../configs/r0-mini-full.yaml).

Differences from the released run, and why:

- One GPU (micro-batch 8 × accumulation 2) instead of 2 GPUs × 4 × 2. Same global batch.
- FP32 master weights with BF16 autocast. ms-swift trains with `--bf16 true`.
- Loss on the answer tokens plus `<|im_end|>\n`. The empty thinking block is part of the prompt, not the target.
- **No gradient checkpointing.** It does not change the maths. On this model it made each step about 5 times slower: 6.6 vs 33.3 samples/s (micro-batch 8, SDPA, measured while another job shared the GPU). Peak VRAM without it is about 65 GB with video rows.

Smoke run (384 rows, 24 steps): loss fell from 0.77 to 0.15. The saved model scored KonIQ 0.729 / 0.661 on 256 items, so saving, loading, and scoring work end to end.

## Pipeline notes

- Some AVA JPEGs are truncated. Q-Align sets `ImageFile.LOAD_TRUNCATED_IMAGES = True` in its AVA eval and training scripts. We do the same in the scorer, the trainer, and the OneAlign script.
- transformers 5.17 needs `kernels>=0.16,<0.17` for Hub kernels. With `use_kernels=True`, Qwen3.5 has no kernel mapping for `RMSNormZeroCentered`. The `causal_conv1d` warning also appears with Hub kernels on. The speed effect is measured in the speed harness, not here.
- OneAlign runs out of memory at 32 videos per batch (eager attention, 8 frames each). The script now divides the batch size by the frames per item.
