"""Batched Q-Align level-token scorer for Hugging Face image-text-to-text models (Qwen3.5 family).

Prompt: chat template with add_generation_prompt (thinking off) + the task stem; the next token is
a level word. Score = sum(softmax(logits[level ids]) * WEIGHTS), in [0, 1].

`ScorerConfig` exposes the inference settings audited in docs/evaluation/protocol.md section 9:
pixel cap, attention kernel, batch size, and last-position-only logits.
"""

from __future__ import annotations

import time

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from PIL import Image

from vq.levels import LEVELS, STEMS, WEIGHTS

# Qwen2-VL-style processors: pixels per merged visual token.
QWEN_PATCH = 16
QWEN_MERGE = 2


@dataclass(frozen=True, slots=True)
class ScorerConfig:
    model: str
    max_pixels: int | None = None  # per image/frame; None = processor default (Q-ReAlign: no cap)
    attn: str = "sdpa"  # sdpa | eager | a hub kernel repo such as kernels-community/flash-attn2
    use_kernels: bool = False  # HF hub kernels (causal-conv1d, fla) instead of the torch fallbacks
    dtype: str = "bfloat16"
    batch_size: int = 16
    last_logits_only: bool = True  # logits_to_keep=1; False = full-sequence logits (Q-ReAlign scorer)
    workers: int = 8

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def level_token_ids(tokenizer: Any) -> list[int]:
    """First token of ' <level>'; the scorer requires each level to be one distinct token."""
    ids = []
    for w in LEVELS:
        toks = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
        if len(toks) != 1:
            raise ValueError(f"level {w!r} is {len(toks)} tokens: {toks}")
        ids.append(toks[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"level tokens are not distinct: {ids}")
    return ids


def build_prompt(processor: Any, n_images: int, prompt: str, stem: str) -> str:
    content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    return text + stem


def load_images(root: Path, row: dict[str, Any]) -> list[Image.Image]:
    return [Image.open(root / m).convert("RGB") for m in row["media"]]


def encode(processor: Any, rows: list[dict[str, Any]], images: list[list[Image.Image]]) -> dict[str, Any]:
    """CPU side: prompts + pixel tensors for one batch (runs in DataLoader workers)."""
    texts = [
        build_prompt(processor, len(im), r["prompt"], STEMS[r["task"]]) for r, im in zip(rows, images, strict=True)
    ]
    flat = [img for im in images for img in im]
    return dict(processor(text=texts, images=flat, padding=True, return_tensors="pt"))


def set_pixel_cap(processor: Any, max_pixels: int | None) -> None:
    if max_pixels is None:
        return
    ip = processor.image_processor
    size = dict(getattr(ip, "size", {}) or {})
    size["longest_edge"] = int(max_pixels)
    size["shortest_edge"] = min(int(size.get("shortest_edge", 65536)), int(max_pixels))
    ip.size = size
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = int(max_pixels)


class Scorer:
    def __init__(self, cfg: ScorerConfig, device: str = "cuda") -> None:
        import torch

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = cfg
        self.device = device
        self.processor = AutoProcessor.from_pretrained(cfg.model)
        self.processor.tokenizer.padding_side = "left"
        set_pixel_cap(self.processor, cfg.max_pixels)
        dtype = getattr(torch, cfg.dtype)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                cfg.model, dtype=dtype, attn_implementation=cfg.attn, use_kernels=cfg.use_kernels
            )
            .to(device)
            .eval()
        )
        self.level_ids = level_token_ids(self.processor.tokenizer)
        self.weights = torch.tensor(WEIGHTS, device=device, dtype=torch.float32)

    def forward(self, batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        batch = {k: v.to(self.device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
        with torch.inference_mode():
            if self.cfg.last_logits_only:
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1]
            else:
                logits = self.model(**batch).logits[:, -1]
            probs = logits[:, self.level_ids].float().softmax(-1)
            score = probs @ self.weights
        return score.cpu().numpy(), probs.cpu().numpy()


class _Batches:
    """Map-style dataset over pre-grouped batches so encoding runs in DataLoader workers."""

    def __init__(self, processor: Any, root: Path, rows: list[dict[str, Any]], groups: list[list[int]]) -> None:
        self.processor, self.root, self.rows, self.groups = processor, root, rows, groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> tuple[list[int], dict[str, Any]]:
        idx = self.groups[i]
        rows = [self.rows[j] for j in idx]
        images = [load_images(self.root, r) for r in rows]
        return idx, encode(self.processor, rows, images)


def pixel_count(root: Path, row: dict[str, Any]) -> int:
    total = 0
    for m in row["media"]:
        with Image.open(root / m) as im:
            w, h = im.size
        total += w * h
    return total


def group_by_size(root: Path, rows: list[dict[str, Any]], batch_size: int) -> list[list[int]]:
    """Batches of similar pixel count (less padding); order is restored by index afterwards."""
    if batch_size == 1:
        return [[i] for i in range(len(rows))]
    order = sorted(range(len(rows)), key=lambda i: pixel_count(root, rows[i]))
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def score_rows(scorer: Scorer, root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score every row; returns preds, probs, and timing (wall seconds excluding model load)."""
    import torch

    from torch.utils.data import DataLoader

    groups = group_by_size(root, rows, scorer.cfg.batch_size)
    ds = _Batches(scorer.processor, root, rows, groups)
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=scorer.cfg.workers,
        prefetch_factor=4 if scorer.cfg.workers else None,
        persistent_workers=False,
        pin_memory=True,
    )
    preds = np.zeros(len(rows), dtype=np.float64)
    probs = np.zeros((len(rows), len(LEVELS)), dtype=np.float32)
    torch.cuda.reset_peak_memory_stats()
    t0 = None  # timing starts when the first batch arrives (excludes worker start-up)
    for idx, batch in loader:
        if t0 is None:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        s, p = scorer.forward(batch)
        preds[idx] = s
        probs[idx] = p
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {
        "preds": preds,
        "probs": probs,
        "wall_s": wall,
        "items_per_s": len(rows) / wall if wall else float("nan"),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
