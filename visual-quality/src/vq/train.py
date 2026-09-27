"""Full-parameter SFT of a Qwen3.5-style VLM on Q-Align conversations (the Q-ReAlign recipe on one GPU).

Recipe (Q-ReAlign configs/onealign.yaml on ms-swift 4.0.2 defaults):
    AdamW lr 2e-5, betas (0.9, 0.95), weight decay 0.1 (not on biases / norms), grad clip 1.0,
    cosine schedule with 3% linear warmup, 2 epochs, global batch 16, bf16 autocast with fp32 master weights,
    vision tower and merger trainable, gradient checkpointing, loss on the assistant answer + end-of-turn.

The prompt is rendered exactly as the scorer renders it (chat template with thinking off), so the level word
is predicted at the same position in training and scoring. No checkpoint selection: the final weights are
the result (no test set is looked at during training).
"""

from __future__ import annotations

import json
import math
import random
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vq.data import read_manifest
from vq.scorer import build_prompt, load_images, set_pixel_cap

END_OF_TURN = "<|im_end|>\n"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    name: str
    model: str
    mix: list[str]
    data_root: str = "data"
    out_root: str = "runs"
    max_pixels: int | None = None
    epochs: int = 2
    lr: float = 2.0e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    micro_batch: int = 8
    grad_accum: int = 2
    freeze_vision: bool = False
    gradient_checkpointing: bool = True
    attn: str = "sdpa"
    workers: int = 10
    seed: int = 0
    limit_per_set: dict[str, int] = field(default_factory=dict)  # optional subsample for pilots
    log_every: int = 10

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        raw = yaml.safe_load(path.read_text())
        if "betas" in raw:
            raw["betas"] = tuple(raw["betas"])
        return cls(**raw)


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class _Rows:
    """Map-style dataset: row -> (input_ids, labels, pixel tensors) via the processor, in workers."""

    def __init__(self, processor: Any, root: Path, rows: list[dict[str, Any]]) -> None:
        self.processor, self.root, self.rows = processor, root, rows
        tok = processor.tokenizer
        self._answer_ids = {}
        for r in rows:
            a = r["answer"]
            if a not in self._answer_ids:
                self._answer_ids[a] = tok(a + END_OF_TURN, add_special_tokens=False)["input_ids"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self.rows[i]
        images = load_images(self.root, r)
        text = build_prompt(self.processor, len(images), r["prompt"], "") + r["answer"] + END_OF_TURN
        enc = self.processor(text=[text], images=images, return_tensors="pt")
        ids = enc["input_ids"][0]
        ans = self._answer_ids[r["answer"]]
        if ids[-len(ans) :].tolist() != ans:
            raise ValueError(f"answer tokens not at the end of the sequence for {r['id']}")
        labels = ids.clone()
        labels[: len(ids) - len(ans)] = -100
        out = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")}
        # Per-token outputs (such as mm_token_type_ids) are padded in the collate; drop their batch dim here.
        for k, v in out.items():
            if v.ndim == 2 and v.shape == enc["input_ids"].shape:
                out[k] = v[0]
        out["input_ids"], out["labels"] = ids, labels
        return out


def _collate(pad_id: int):
    import torch

    def fn(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = max(len(x["input_ids"]) for x in items)
        ids = torch.full((len(items), n), pad_id, dtype=torch.long)
        labels = torch.full((len(items), n), -100, dtype=torch.long)
        mask = torch.zeros((len(items), n), dtype=torch.long)
        for i, x in enumerate(items):
            k = len(x["input_ids"])
            ids[i, :k], labels[i, :k], mask[i, :k] = x["input_ids"], x["labels"], 1
        batch = {"input_ids": ids, "labels": labels, "attention_mask": mask}
        for key in items[0]:
            if key in batch:
                continue
            if items[0][key].ndim == 1 and len(items[0][key]) == len(items[0]["input_ids"]):  # per-token
                t = torch.zeros((len(items), n), dtype=items[0][key].dtype)
                for i, x in enumerate(items):
                    t[i, : len(x[key])] = x[key]
                batch[key] = t
            else:
                batch[key] = torch.cat([x[key] for x in items], dim=0)
        return batch

    return fn


def _load_rows(cfg: TrainConfig, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in cfg.mix:
        part = read_manifest(root / "manifests" / f"{name}.jsonl")
        lim = cfg.limit_per_set.get(name)
        if lim:
            part = random.Random(cfg.seed).sample(part, min(lim, len(part)))
        rows.extend(part)
    return rows


def _param_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n.lower() or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def run(config_path: Path) -> Path:
    import torch

    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = TrainConfig.load(config_path)
    root = Path(cfg.data_root)
    run_dir = Path(cfg.out_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    log_path = run_dir / "train_log.jsonl"

    torch.manual_seed(cfg.seed)
    processor = AutoProcessor.from_pretrained(cfg.model)
    set_pixel_cap(processor, cfg.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(cfg.model, dtype=torch.float32, attn_implementation=cfg.attn)
    model.to("cuda")
    model.config.use_cache = False
    if cfg.freeze_vision:
        for p in model.model.visual.parameters():
            p.requires_grad = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    rows = _load_rows(cfg, root)
    ds = _Rows(processor, root, rows)
    pad_id = processor.tokenizer.pad_token_id
    steps_per_epoch = len(rows) // (cfg.micro_batch * cfg.grad_accum)
    total = steps_per_epoch * cfg.epochs
    warmup = max(1, round(total * cfg.warmup_ratio))
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas, eps=1e-8, fused=True)
    summary: dict[str, Any] = {"rows": len(rows), "steps": total, "warmup": warmup, "config": asdict(cfg)}
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}), flush=True)

    step = 0
    t0 = time.perf_counter()
    seen = tokens = 0
    for epoch in range(cfg.epochs):
        gen = torch.Generator().manual_seed(cfg.seed * 1000 + epoch)
        loader = DataLoader(
            ds,
            batch_size=cfg.micro_batch,
            shuffle=True,
            generator=gen,
            drop_last=True,
            num_workers=cfg.workers,
            prefetch_factor=4,
            persistent_workers=False,
            pin_memory=True,
            collate_fn=_collate(pad_id),
        )
        it = iter(loader)
        for _ in range(steps_per_epoch):
            micro = [next(it) for _ in range(cfg.grad_accum)]
            n_label = sum(int((m["labels"][:, 1:] != -100).sum()) for m in micro)
            loss_sum = torch.zeros((), device="cuda")
            for m in micro:
                m = {k: v.to("cuda", non_blocking=True) for k, v in m.items()}
                labels = m.pop("labels")
                target = labels[:, 1:]
                sel = target != -100
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model.model(**m).last_hidden_state
                    # lm_head only on answer positions: full-vocab logits for every token would cost ~GBs
                    logits = model.lm_head(hidden[:, :-1][sel])
                # token-mean over the whole global batch (as HF Trainer with num_items_in_batch)
                loss = torch.nn.functional.cross_entropy(logits.float(), target[sel], reduction="sum") / n_label
                loss.backward()
                loss_sum += loss.detach()
                seen += labels.size(0)
                tokens += int(m["attention_mask"].sum())
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm))
            for g in opt.param_groups:
                g["lr"] = cfg.lr * lr_factor(step, total, warmup)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == total:
                el = time.perf_counter() - t0
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": round(float(loss_sum), 5),
                    "grad_norm": round(gnorm, 4),
                    "lr": opt.param_groups[0]["lr"],
                    "samples": seen,
                    "samples_per_s": round(seen / el, 2),
                    "tokens_per_s": round(tokens / el, 1),
                    "elapsed_h": round(el / 3600, 3),
                    "eta_h": round(el / step * (total - step) / 3600, 2),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                }
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)
        _save(model, processor, run_dir / f"epoch{epoch + 1}")

    final = run_dir / "final"
    _save(model, processor, final)
    summary.update(
        samples=seen,
        tokens=tokens,
        wall_h=(time.perf_counter() - t0) / 3600,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return final


def _save(model: Any, processor: Any, path: Path) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
    processor.save_pretrained(path)
