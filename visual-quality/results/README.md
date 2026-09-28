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

## Stage 0: speed harness

`vq bench` (Q-ReAlign, R0) and `scripts/onealign_eval.py bench` (OneAlign), run one after the other with the GPU free. Fixed inputs: 1,000 images (500 KonIQ + 500 full-size SPAQ test images) and 200 videos (100 LSVQ + 100 LSVQ-1080p test videos, 8 frames each). Times include image decode and preprocessing.

- Batch-1 latency: 100 items, one at a time, in the main process (decode, processor, forward).
- Peak throughput: DataLoader with 8 workers, batch sizes 1, 4, 8, 16, 32, 64. The table shows the best batch size.
- Cost: GPU time only, at the pod price of $2.09/h.

Raw files: `bench/<name>/bench.json`.

| Model | Mode | Visual tokens / item | Batch-1 latency p50 / p95 (ms) | Peak items/s (batch) | Peak VRAM (GB) | USD per 1M items |
|---|---|---:|---|---|---:|---:|
| Q-Align OneAlign | image | 64 | 56 / 64 | 66.4 (64) | 21.9 | 8.75 |
| Q-Align OneAlign | video | 512 | 164 / 170 | 9.5 (8) | 22.0 | 61.24 |
| Q-ReAlign Mini | image | 789 | 36 / 57 | 56.2 (4) | 2.6 | 10.33 |
| Q-ReAlign Mini | video | 924 | 51 / 69 | 47.4 (4) | 2.5 | 12.25 |
| Q-ReAlign Lite | image | 789 | 63 / 79 | 21.9 (4) | 11.5 | 26.49 |
| Q-ReAlign Lite | video | 924 | 76 / 90 | 17.9 (4) | 11.3 | 32.37 |
| Q-ReAlign Pro | image | 789 | 84 / 140 | 14.8 (4) | 20.1 | 39.10 |
| Q-ReAlign Pro | video | 924 | 97 / 115 | 12.3 (4) | 19.9 | 47.11 |
| R0 (ours, 0.8B) | image | 789 | 38 / 49 | 58.2 (4) | 2.6 | 9.97 |
| R0 (ours, 0.8B) | video | 924 | 41 / 65 | 48.1 (4) | 2.5 | 12.07 |

### Inference-config sweep (Mini, one factor at a time)

Accuracy for the pixel caps: full KonIQ, SPAQ, KADID, AGIQA-3K, LIVE, AVA, LIVE-C, and CSIQ test sets. SRCC change vs no cap is a paired bootstrap; only sets that change are listed.

| Setting | Image tokens | Image items/s (batch) | Image p50 (ms) | Video items/s | SRCC change vs default |
|---|---:|---|---:|---:|---|
| Default (SDPA, last-position logits, no cap) | 789 | 56.2 (4) | 36 | 47.4 | – |
| Full-sequence logits (Q-ReAlign scorer) | 789 | 56.7 (4) | 35 | 47.1 | none (VRAM 4.7 vs 2.6 GB) |
| Hub kernels (`use_kernels=True`) | 789 | 59.1 (8) | 34 | 48.6 | not measured |
| FlashAttention 2 (Hub kernel) | 789 | 63.6 (4) | 33 | 50.1 | not measured |
| Cap 1,024 tokens | 768 | 59.8 (4) | 35 | 47.1 | none (all within 0.001) |
| Cap 576 tokens | 538 | 84.1 (4) | 33 | 48.2 | KonIQ **-0.011** [-0.014, -0.008]; others within ±0.001 |
| Cap 256 tokens | 234 | 168.1 (8) | 27 | 49.7 | KonIQ **-0.090**, LIVE **-0.032**, AVA **-0.007**, SPAQ **-0.006** |

Frames are already at most 448 px on the long side, so the caps do not change video.

### What the speed harness shows

1. **Video: the 0.8B models are 5 times faster and 5 times cheaper than OneAlign** (47–48 vs 9.5 videos/s; about $12 vs $61 per 1M videos). Batch-1 latency is 41–51 ms vs 164 ms.
2. **Images: at the default setting, Mini and R0 are not cheaper than OneAlign.** Peak throughput is 56–58 vs 66 images/s. Batch-1 latency is lower (36–38 vs 56 ms), and VRAM is 8 times lower. OneAlign sends 64 visual tokens per image; Qwen3.5 sends 789 on this mix.
3. **The visual-token cap is the main lever.** A 576-token cap gives 84 images/s (1.5 times the default, 1.27 times OneAlign) for a KonIQ SRCC loss of 0.011 and no change elsewhere. A 256-token cap gives 168 images/s (2.5 times OneAlign) but costs 0.09 SRCC on KonIQ.
4. Kernels help a little: FlashAttention 2 +13%, Hub kernels +5%. Full-sequence logits cost only memory.
5. For Qwen3.5, throughput falls when the batch grows past 4–8 (Mini images: 56 at batch 4, 24 at batch 64). [INFERENCE] Padding between images of different sizes and the longer attention span cost more than batching saves. Sort-by-size batching is already on.
6. Caveat: OneAlign runs in FP16 with eager attention, as its model card requires (transformers 4.36). We did not try to make it faster. Q-ReAlign Lite and Pro run 3–4.5 times slower than OneAlign on images, as the published chart suggests ([protocol.md section 9](../docs/evaluation/protocol.md#9-efficiency-measurement)).

## Harness checks

- **Batched scorer = reference path.** Mini, 256 items: batch 1 with full-sequence logits (the Q-ReAlign scorer path) vs batch 16 with last-position logits. Prediction correlation 0.99995 (KonIQ) and 0.99987 (AVA). Largest absolute difference 0.008 on a 0–1 scale.
- **Prompt wording does not matter.** Mini, KonIQ, 600 items: our fixed prompt vs the per-item prompt in the test JSON, with and without a space before the answer stem. SRCC 0.9062–0.9066 in all four cases.
- **Image resizing is the same as ms-swift.** ms-swift resizes Qwen3.5 images with `qwen_vl_utils.fetch_image` (at most 16,384 visual tokens; minimum 4). The Hugging Face processor default gives the same size limits. KonIQ images need no resize under either path.

## Training (R0)

Qwen3.5-0.8B, full fine-tune on the ONE-ALIGN mix (287,702 rows; one LSVQ training video has no frames), released Q-ReAlign recipe: AdamW lr 2e-5, betas (0.9, 0.95), weight decay 0.1, cosine with 3% warm-up, 2 epochs, global batch 16. Config: [`configs/r0-mini-full.yaml`](../configs/r0-mini-full.yaml). Metrics: `r0/eval-final/metrics.json` and `r0/eval-epoch1/metrics.json`. Training log: `r0/train_log.jsonl`.

Run: 35,962 steps, 575,392 samples, 249M tokens, 9.1 h on one RTX PRO 6000 (about $19 at $2.09/h). Peak VRAM 66 GB. Anchor evals shared the GPU for most of the first 5 hours; alone, the run reached about 17 samples/s.

Differences from the released run, and why:

- One GPU (micro-batch 8 × accumulation 2) instead of 2 GPUs × 4 × 2. Same global batch.
- FP32 master weights with BF16 autocast. ms-swift trains with `--bf16 true`.
- Loss on the answer tokens plus `<|im_end|>\n`. The empty thinking block is part of the prompt, not the target.
- **No gradient checkpointing.** It does not change the maths. On this model it made each step about 5 times slower: 6.6 vs 33.3 samples/s (micro-batch 8, SDPA, measured while another job shared the GPU). Peak VRAM without it is about 65 GB with video rows.

Smoke run (384 rows, 24 steps): loss fell from 0.77 to 0.15. The saved model scored KonIQ 0.729 / 0.661 on 256 items, so saving, loading, and scoring work end to end.

### R0 accuracy

| Model | KonIQ | SPAQ | KADID | AGIQA-3K | LIVE | AVA | LSVQ | T1 mean | T2 mean |
|---|---|---|---|---|---|---|---|---|---|
| R0 final | 0.907 / 0.896 | 0.917 / 0.914 | 0.858 / 0.863 | 0.660 / 0.628 | 0.865 / 0.888 | 0.698 / 0.691 | 0.816 / 0.818 | 0.817 / 0.814 | 0.771 / 0.751 |
| R0 after epoch 1 | 0.880 / 0.866 | 0.900 / 0.897 | 0.808 / 0.797 | 0.714 / 0.660 | 0.835 / 0.856 | 0.658 / 0.651 | 0.778 / 0.780 | 0.796 / 0.787 | 0.740 / 0.717 |
| Q-ReAlign Mini (released) | 0.913 / 0.909 | 0.914 / 0.914 | 0.846 / 0.854 | 0.821 / 0.785 | 0.848 / 0.892 | 0.749 / 0.746 | 0.837 / 0.840 | 0.847 / 0.849 | 0.808 / 0.787 |
| Q-Align OneAlign | 0.950 / 0.941 | 0.935 / 0.932 | 0.966 / 0.963 | 0.838 / 0.801 | 0.856 / 0.887 | 0.819 / 0.822 | 0.872 / 0.874 | 0.891 / 0.889 | 0.853 / 0.835 |

R0 final on T2: LSVQ-1080p 0.735 / 0.688, LIVE-C 0.757 / 0.731, CSIQ 0.826 / 0.820, KoNViD-1k 0.823 / 0.815, MaxWell 0.711 / 0.699.

SRCC minus released Mini (paired bootstrap, 95% interval):

| Model | KonIQ | SPAQ | KADID | AGIQA-3K | LIVE | AVA | LSVQ | LSVQ-1080p | LIVE-C | CSIQ | KoNViD-1k | MaxWell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 final | **-0.013** | +0.000 | +0.009 | **-0.157** | -0.004 | **-0.055** | **-0.022** | **-0.056** | **-0.062** | -0.008 | **-0.026** | **-0.028** |

SRCC minus OneAlign: R0 final is below on 11 of 12 sets (intervals exclude 0) and ties on LIVE (+0.001 [-0.010, +0.012]). The largest gaps are AGIQA-3K (-0.173), LIVE-C (-0.151), AVA (-0.131), and KADID (-0.101).

### What R0 shows

1. **Our run of the public recipe does not reach the released Mini.** It ties on SPAQ, KADID, LIVE, and CSIQ. It loses most on AGIQA-3K (-0.157) and AVA (-0.055). It also loses 0.013 on KonIQ, 0.062 on LIVE-C, and 0.022–0.056 on the four video sets.
2. [INFERENCE] **The AGIQA-3K gap points to a missing data source.** The Q-ReAlign README lists AGIQA-20K in the training mix, but the public config (`mix: [koniq, spaq, kadid, ava, lsvq]`) leaves it out, and so did we. The released Mini scores 0.785 on AGIQA-3K; R0 scores 0.628. AI-generated images are the one domain that no source in our mix covers.
3. The AVA gap (-0.055) has no clear cause yet. Candidates: BF16 vs FP32 master weights, the loss on the empty thinking block, or a different AVA sampling in the released run.
4. The second epoch helps on every set except AGIQA-3K and CSIQ (T1 mean SRCC 0.787 to 0.814). The final checkpoint is the result; we did no checkpoint selection.
5. **R0 is not yet a replacement for OneAlign.** It is 5 times cheaper on video and about as fast on images, but it is 0.075 lower in T1 mean SRCC and 0.084 lower in T2 mean SRCC.

### Next steps

- Add AGIQA-20K to the mix (the README mix), then re-run R0 at the same budget. This tests point 2.
- Train and evaluate with a 576-token cap. It costs 0.011 KonIQ SRCC at inference with the released weights and makes images 1.5 times faster. Training at the cap may recover some of that.
- Try FlashAttention 2 at inference (+13% images/s).
- Larger backbone at a cap (for example Qwen3.5-2B or 4B with 256–576 tokens): Lite already matches OneAlign on accuracy, but it is 3 times slower at no cap.

## Pipeline notes

- Some AVA JPEGs are truncated. Q-Align sets `ImageFile.LOAD_TRUNCATED_IMAGES = True` in its AVA eval and training scripts. We do the same in the scorer, the trainer, and the OneAlign script.
- transformers 5.17 needs `kernels>=0.16,<0.17` for Hub kernels. With `use_kernels=True`, Qwen3.5 has no kernel mapping for `RMSNormZeroCentered`, and the `causal_conv1d` fallback warning still appears. Speed effect: +5% images/s (see the sweep).
- OneAlign runs out of memory at 32 videos per batch (eager attention, 8 frames each). The script now divides the batch size by the frames per item.
