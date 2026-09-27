"""Score locked manifests with the original Q-Align OneAlign (q-future/one-align, mPLUG-Owl2).

Runs in its own venv (the remote code needs transformers 4.36; see scripts/onealign_env.sh) and writes
the same preds/ + metrics.json layout as `vq eval`. Scoring is the model's own `score()`: images padded to
square, CLIP 448x448, fp16, prompt "USER: How would you rate the {task} of this {input}?\\n<|image|>\\n
ASSISTANT: The {task} of the {input} is", softmax over the 5 level logits, weights 5..1. Video = the same
8 frames per video as `vq eval`.

    python scripts/onealign_eval.py --root data --out runs/eval-onealign --batch-size 32 test_koniq ...
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

from pathlib import Path

import numpy as np
import torch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.data import read_manifest
from vq.evaluate import subset
from vq.metrics import bootstrap, correlations

TASK = {"iqa": ("quality", "image"), "iaa": ("aesthetics", "image"), "vqa": ("quality", "video")}


class _Items(torch.utils.data.Dataset):
    def __init__(self, root: Path, rows: list[dict], processor, mean) -> None:
        self.root, self.rows, self.processor, self.mean = root, rows, processor, mean

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        imgs = [_expand(Image.open(self.root / m).convert("RGB"), self.mean) for m in self.rows[i]["media"]]
        return self.processor.preprocess(imgs, return_tensors="pt")["pixel_values"]


def _expand(img: Image.Image, background) -> Image.Image:
    """mPLUG-Owl2 expand2square: pad the short side with the mean colour, image centred."""
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    out = Image.new(img.mode, (s, s), background)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="q-future/one-align")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("sets", nargs="+")
    a = p.parse_args()

    import transformers

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, attn_implementation="eager", torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    proc = model.image_processor
    mean = tuple(int(x * 255) for x in proc.image_mean)
    weights = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device="cuda")
    tok = model.tokenizer
    level_ids = model.preferential_ids_
    from importlib import import_module

    mod = import_module(model.__class__.__module__)
    image_token_index = mod.IMAGE_TOKEN_INDEX

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "preds").mkdir(exist_ok=True)
    metrics = {
        "config": {"model": a.model, "batch_size": a.batch_size, "dtype": "float16", "attn": "eager"},
        "limit": a.limit,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "sets": {},
    }
    warm = True
    for name in a.sets:
        rows = subset(read_manifest(a.root / "manifests" / f"{name}.jsonl"), a.limit)
        task, inp = TASK[rows[0]["task"]]
        prompt = f"USER: How would you rate the {task} of this {inp}?\n<|image|>\nASSISTANT: The {task} of the {inp} is"
        ids = mod.tokenizer_image_token(prompt, tok, image_token_index, return_tensors="pt").unsqueeze(0).cuda()
        loader = torch.utils.data.DataLoader(
            _Items(a.root, rows, proc, mean),
            batch_size=a.batch_size,
            num_workers=a.workers,
            collate_fn=list,
            prefetch_factor=4,
        )
        preds = np.zeros(len(rows))
        probs_all = np.zeros((len(rows), 5), dtype=np.float32)

        torch.cuda.reset_peak_memory_stats()
        k = 0
        t0 = None
        for batch in loader:
            if warm:  # kernel warm-up on the first batch of the run, not timed
                _forward(model, ids, batch, level_ids, weights, video=inp == "video")
                warm = False
            if t0 is None:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            s, pr = _forward(model, ids, batch, level_ids, weights, video=inp == "video")
            n = len(batch)
            preds[k : k + n] = s.cpu().numpy()
            probs_all[k : k + n] = pr.cpu().numpy()
            k += n
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        # OneAlign scores are 1..5; map to 0..1 like the 5-level scorer (monotone, correlations unchanged)
        preds01 = (preds - 1.0) / 4.0
        mos = np.array([r["mos"] for r in rows], dtype=np.float64)
        hb = rows[0]["higher_better"]
        m = correlations(preds01, mos, higher_better=hb)
        m["ci95"] = bootstrap(preds01, mos, higher_better=hb, n=a.boot) if a.boot else None
        m.update(
            n=len(rows),
            items_per_s=len(rows) / wall,
            wall_s=wall,
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
        )
        metrics["sets"][name] = m
        with (a.out / "preds" / f"{name}.jsonl").open("w") as f:
            for r, sc, pr in zip(rows, preds01, probs_all, strict=True):
                f.write(json.dumps({"id": r["id"], "pred": float(sc), "probs": pr.round(5).tolist(), "mos": r["mos"]}))
                f.write("\n")
        print(json.dumps({name: {k2: v for k2, v in m.items() if k2 != "ci95"}}), flush=True)
        (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2))


def _forward(model, ids, batch: list[torch.Tensor], level_ids, weights, *, video: bool):
    with torch.inference_mode():
        if video:
            x = [v.half().cuda(non_blocking=True) for v in batch]
        else:
            x = torch.cat(batch).half().cuda(non_blocking=True)
        logits = model(ids.repeat(len(batch), 1), images=x)["logits"][:, -1, level_ids]
        pr = torch.softmax(logits.float(), -1)
        return pr @ weights, pr


if __name__ == "__main__":
    main()
