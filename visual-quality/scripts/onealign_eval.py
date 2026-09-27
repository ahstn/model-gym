"""Score locked manifests with the original Q-Align OneAlign (q-future/one-align, mPLUG-Owl2).

Runs in its own venv (the remote code needs transformers 4.36; see scripts/onealign_env.sh) and writes
the same layout as `vq eval` (preds/ + metrics.json) or `vq bench` (bench.json). Scoring follows the
model's own `score()`: images padded to square, CLIP 448x448, fp16, prompt "USER: How would you rate the
{task} of this {input}?\\n<|image|>\\nASSISTANT: The {task} of the {input} is", softmax over the five level
logits, weights 5..1 (mapped to 0..1 here). Video = the same 8 frames per video as `vq eval`.

    python scripts/onealign_eval.py eval --out runs/anchors/onealign test_koniq ...
    python scripts/onealign_eval.py bench --out runs/bench/onealign
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vq.bench import BATCH_SIZES, IMAGE_MIX, LATENCY_ITEMS, VIDEO_MIX, bench_rows
from vq.data import read_manifest
from vq.evaluate import subset
from vq.metrics import bootstrap, correlations

TASK = {"iqa": ("quality", "image"), "iaa": ("aesthetics", "image"), "vqa": ("quality", "video")}
VISUAL_TOKENS_PER_IMAGE = 64  # visual abstractor output


def _expand(img: Image.Image, background) -> Image.Image:
    """mPLUG-Owl2 expand2square: pad the short side with the mean colour, image centred."""
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    out = Image.new(img.mode, (s, s), background)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


class _Items(torch.utils.data.Dataset):
    def __init__(self, root: Path, rows: list[dict], processor, mean) -> None:
        self.root, self.rows, self.processor, self.mean = root, rows, processor, mean

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        imgs = [_expand(Image.open(self.root / m).convert("RGB"), self.mean) for m in self.rows[i]["media"]]
        return self.processor.preprocess(imgs, return_tensors="pt")["pixel_values"]


class OneAlign:
    def __init__(self, model_id: str, root: Path, workers: int) -> None:
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="eager", torch_dtype=torch.float16
        ).to("cuda")
        self.model.eval()
        self.root, self.workers = root, workers
        self.proc = self.model.image_processor
        self.mean = tuple(int(x * 255) for x in self.proc.image_mean)
        self.weights = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device="cuda")
        self.level_ids = self.model.preferential_ids_
        self.mod = import_module(self.model.__class__.__module__)

    def _ids(self, task: str) -> torch.Tensor:
        t, inp = TASK[task]
        prompt = f"USER: How would you rate the {t} of this {inp}?\n<|image|>\nASSISTANT: The {t} of the {inp} is"
        ids = self.mod.tokenizer_image_token(
            prompt, self.model.tokenizer, self.mod.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        return ids.unsqueeze(0).cuda()

    def forward(self, ids: torch.Tensor, batch: list[torch.Tensor], *, video: bool):
        with torch.inference_mode():
            if video:
                x = [v.half().cuda(non_blocking=True) for v in batch]
            else:
                x = torch.cat(batch).half().cuda(non_blocking=True)
            logits = self.model(ids.repeat(len(batch), 1), images=x)["logits"][:, -1, self.level_ids]
            pr = torch.softmax(logits.float(), -1)
            return (pr @ self.weights - 1.0) / 4.0, pr  # 1..5 -> 0..1 (monotone)

    def score_rows(self, rows: list[dict], batch_size: int) -> dict:
        video = rows[0]["task"] == "vqa"
        ids = self._ids(rows[0]["task"])
        loader = torch.utils.data.DataLoader(
            _Items(self.root, rows, self.proc, self.mean),
            batch_size=batch_size,
            num_workers=self.workers,
            collate_fn=list,
            prefetch_factor=4 if self.workers else None,
        )
        preds = np.zeros(len(rows))
        probs = np.zeros((len(rows), 5), dtype=np.float32)
        torch.cuda.reset_peak_memory_stats()
        k, t0 = 0, None
        for batch in loader:
            if t0 is None:  # timing starts when the first batch arrives, as in vq.scorer.score_rows
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            s, pr = self.forward(ids, batch, video=video)
            preds[k : k + len(batch)] = s.cpu().numpy()
            probs[k : k + len(batch)] = pr.cpu().numpy()
            k += len(batch)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        return {
            "preds": preds,
            "probs": probs,
            "wall_s": wall,
            "items_per_s": len(rows) / wall,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        }

    def latency_ms(self, rows: list[dict]) -> list[float]:
        video = rows[0]["task"] == "vqa"
        ids = self._ids(rows[0]["task"])
        ds = _Items(self.root, rows, self.proc, self.mean)
        for i in range(5):  # warm-up
            self.forward(ids, [ds[i]], video=video)
        out = []
        for i in range(len(rows)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.forward(ids, [ds[i]], video=video)
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1000)
        return out


def _env() -> dict:
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cpus": len(os.sched_getaffinity(0)),
    }


def cmd_eval(a: argparse.Namespace, oa: OneAlign) -> None:
    (a.out / "preds").mkdir(parents=True, exist_ok=True)
    metrics = {
        "config": {"model": a.model, "batch_size": a.batch_size, "dtype": "float16", "attn": "eager"},
        "limit": a.limit,
        "env": _env(),
        "sets": {},
    }
    oa.score_rows(subset(read_manifest(a.root / "manifests" / f"{a.sets[0]}.jsonl"), 2 * a.batch_size), a.batch_size)
    for name in a.sets:
        rows = subset(read_manifest(a.root / "manifests" / f"{name}.jsonl"), a.limit)
        res = oa.score_rows(rows, a.batch_size)
        mos = np.array([r["mos"] for r in rows], dtype=np.float64)
        hb = rows[0]["higher_better"]
        m = correlations(res["preds"], mos, higher_better=hb)
        m["ci95"] = bootstrap(res["preds"], mos, higher_better=hb, n=a.boot) if a.boot else None
        m.update(n=len(rows), items_per_s=res["items_per_s"], wall_s=res["wall_s"], peak_vram_gb=res["peak_vram_gb"])
        metrics["sets"][name] = m
        with (a.out / "preds" / f"{name}.jsonl").open("w") as f:
            for r, sc, pr in zip(rows, res["preds"], res["probs"], strict=True):
                rec = {"id": r["id"], "pred": float(sc), "probs": pr.round(5).tolist(), "mos": r["mos"]}
                f.write(json.dumps(rec) + "\n")
        print(json.dumps({name: {k: v for k, v in m.items() if k != "ci95"}}), flush=True)
        (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2))


def cmd_bench(a: argparse.Namespace, oa: OneAlign) -> None:
    result = {"config": {"model": a.model, "dtype": "float16", "attn": "eager"}, "env": _env(), "modes": {}}
    for mode, mix in (("image", IMAGE_MIX), ("video", VIDEO_MIX)):
        rows = bench_rows(a.root, mix)
        lat = oa.latency_ms(subset(rows, LATENCY_ITEMS))
        n_media = len(rows[0]["media"])
        entry = {
            "items": len(rows),
            "visual_tokens_mean": float(VISUAL_TOKENS_PER_IMAGE * n_media),
            "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p95": float(np.percentile(lat, 95)),
            "throughput": {},
        }
        for bs in a.batch_sizes or BATCH_SIZES:
            try:
                res = oa.score_rows(rows, bs)
            except torch.OutOfMemoryError:
                entry["throughput"][str(bs)] = "oom"
                torch.cuda.empty_cache()
                break
            entry["throughput"][str(bs)] = {"items_per_s": res["items_per_s"], "peak_vram_gb": res["peak_vram_gb"]}
            print(json.dumps({mode: {"bs": bs, "items_per_s": round(res["items_per_s"], 2)}}), flush=True)
        ok = {k: v for k, v in entry["throughput"].items() if isinstance(v, dict)}
        best = max(ok, key=lambda k: ok[k]["items_per_s"])
        entry["peak"] = {"batch_size": int(best), **ok[best]}
        result["modes"][mode] = entry
        print(json.dumps({mode: {k: v for k, v in entry.items() if k != "throughput"}}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "bench.json").write_text(json.dumps(result, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--model", default="q-future/one-align")
    p.add_argument("--workers", type=int, default=8)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--boot", type=int, default=1000)
    e.add_argument("sets", nargs="+")
    b = sub.add_parser("bench")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--batch-sizes", type=int, nargs="*")
    a = p.parse_args()
    oa = OneAlign(a.model, a.root, a.workers)
    {"eval": cmd_eval, "bench": cmd_bench}[a.cmd](a, oa)


if __name__ == "__main__":
    main()
