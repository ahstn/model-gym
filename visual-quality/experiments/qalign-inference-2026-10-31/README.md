# Q-Align and Q-ReAlign inference availability

Checked 31 October 2026 using Exa, Hugging Face repository metadata and live Space/provider status. After the user approved uploading the seven selected images, all 14 Q-ReAlign Mini scoring calls completed. See the [comparison report](comparison.md), [per-image scores](comparison.csv), and native responses under `space-results/`. No Q-Align OneAlign scores were generated.

The [Q-Align OneAlign checkpoint](https://huggingface.co/q-future/one-align) provides separate `quality` and `aesthetics` scores on a 1–5 scale. Its author recommends OneAlign as the unified checkpoint. The [author's OneAlign Space](https://huggingface.co/spaces/q-future/OneAlign) was in `RUNTIME_ERROR`; two checked OneAlign forks were also in runtime error. No inference provider was listed in the live Hugging Face model mapping for OneAlign, Q-ReAlign Mini or Pro. A different Space named OneAlign was running but its current app source routes to an unrelated model; its name is not proof of OneAlign inference.

The third-party [Q-ReAlign Mini Space](https://huggingface.co/spaces/VOIDER/Q-ReAlign-0.8B) was `RUNNING` and its public Gradio API describes `score_image` with separate quality and aesthetic tasks. The [Space source](https://huggingface.co/spaces/VOIDER/Q-ReAlign-0.8B/blob/main/app.py), revision `9fb0e64200c1615a962551f002e11c1fc981ec04`, loads `q-future/Q-ReAlign-Mini-0.8B`, resizes the input to at most 768 pixels on its longest side and returns a score in [0,1]. A linear conversion `1 + 4 * native` maps those level weights to [1,5]. The Space revision was unchanged when checked after the calls; the Space code itself does not pin the underlying model revision.

The Space assesses image quality/aesthetics and has **no photography-eligibility gate**. It will not necessarily assign our required 1.00/1.00 floor to a poster. Record raw scores and the eligibility decision separately; apply any existing scope policy transparently, never present a post-processed floor as the model's native answer. Its downscaled input also differs from the 1,536-pixel delivered view used by the prior teachers.

## Completed bounded comparison

The following seven already-scored images were sent to the public Space after upload approval. Six are eligible held-out images sampled across the prior reference technical range; T2000 is the known non-photographic poster development control. Both dimensions were scored sequentially in 14 successful inference calls. The files and original scores are in [photo-scoring](../photo-scoring-2026-10-31/scores-72.csv).

| Case | Role | Reference technical | Reference aesthetic |
|---|---|---:|---:|
| T0028 | Held out | 1.30 | 1.70 |
| T0192 | Held out | 2.10 | 2.90 |
| T0714 | Held out | 2.60 | 2.50 |
| T0965 | Held out | 3.50 | 3.80 |
| T0914 | Held out | 4.10 | 3.10 |
| T0724 | Held out | 4.50 | 3.90 |
| T2000 | Known poster | 1.00 | 1.00 |

The subset is a feasibility comparison, not an accuracy leaderboard. The poster is separate from the six photos. Converted scores are compared with the existing reference, Luna, DeepSeek, mean, Haiku, Jev and Liquid scores; responses, image hashes and endpoint revision are preserved.

## RunPod option for preferred Q-Align

A dedicated GPU pod runs OneAlign without depending on the public Space. A practical starting specification is **one RTX A6000 with 48 GB VRAM**, 32–64 GB host RAM and at least 50 GB free disk; one image per batch, FP16, eager attention. This is a planning configuration, not a tested measurement. The pinned checkpoint comprises roughly 16.41 GB of model weight files, so 24 GB VRAM may work with batch size one but leaves less headroom for the model, activations and allocator. A 48 GB card is the more forgiving starting point. Current listed [RunPod pricing](https://www.runpod.io/pricing) is approximately $0.33/hour community or $0.59/hour secure for an RTX A6000; availability, storage and other charges vary.

The author uses `transformers==4.36.1` and `torch_dtype=torch.float16` with `trust_remote_code=True`. Start with an isolated Python 3.10 environment and the [author's pinned dependencies](https://github.com/Q-Future/Q-Align/blob/main/pyproject.toml); matching CUDA-enabled PyTorch and drivers must be installed for the selected pod image. Verify the author's documented sample score before the comparison. The model uses checkpoint-specific custom remote code, and its `.bin` checkpoint is pickle-based; use an isolated trusted environment. The model card's MIT tag does not by itself settle all upstream model and training-data terms; the [author's model zoo](https://q-future.github.io/Q-Align/model_zoo/) calls out LLaMA-2 licensing for commercial use.

The prepared [score_local.py](score_local.py) pins OneAlign revision `dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb`, verifies source image hashes, scores the same seven images with both tasks, and stores native results beside prior references and selected teacher scores. It is **not GPU-tested**. Transfer the existing `photo-scoring-2026-10-31/images/` and `scores-72.csv` into the same relative repository layout, then run:

```sh
python visual-quality/experiments/qalign-inference-2026-10-31/score_local.py --output onealign-results.jsonl
```

The script preserves native model scores; it does not force the poster to 1.00. Keep failed or partial runs and record the model revision, software versions, GPU memory, input resize and pod cost before comparing outcomes.
