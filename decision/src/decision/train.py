"""LoRA fine-tuning of a causal LM as a decision model with a candidate-slot soft cross-entropy readout.

Resume is not supported: a run directory is written once. Nothing is cached to disk except run outputs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import platform
import random
import re
import shutil
import time

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import yaml

from decision.modeling import DEFAULT_MODEL, Batch, code_logits, collate, token_budget_batches
from decision.prompt import Readout
from decision.schema import Decision, read_jsonl

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LOG_EVERY = 10
EMA_DECAY = 0.98


@dataclass(frozen=True, slots=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    def __post_init__(self) -> None:
        if self.r <= 0 or self.alpha <= 0:
            raise ValueError("lora.r and lora.alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("lora.dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("lora.target_modules must be non-empty")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    name: str
    lr: float
    model: str = DEFAULT_MODEL
    pool: str = "data/pool.jsonl"
    n_rows: int | None = None
    exclude_datasets: tuple[str, ...] = ()
    dev_panel: str = "data/panels/dev.jsonl"
    dev_subset: int = 1500
    max_seq_tokens: int = 2048
    batch_tokens: int = 65536
    max_rows_per_micro: int = 64
    rows_per_step: int = 64
    epochs: int = 1
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    lora: LoraSettings = field(default_factory=LoraSettings)
    shuffle_options: bool = True
    consistency_weight: float = 0.0
    eval_every: int = 100
    save_every: int = 0
    seed: int = 0
    gradient_checkpointing: bool = True
    out_root: str = "runs"

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("name must be a non-empty path component")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.n_rows is not None and self.n_rows <= 0:
            raise ValueError("n_rows must be positive")
        positive = ("max_seq_tokens", "batch_tokens", "max_rows_per_micro", "rows_per_step", "epochs")
        for key in positive:
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ("dev_subset", "eval_every", "save_every", "weight_decay", "consistency_weight"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be >= 0")
        if self.batch_tokens < self.max_seq_tokens:
            raise ValueError("batch_tokens must be >= max_seq_tokens")
        if not 0.0 <= self.warmup_ratio < 1.0 or not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("warmup_ratio must be in [0, 1) and min_lr_ratio in [0, 1]")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        data = dict(raw)
        lora_raw = data.pop("lora", None) or {}
        if not isinstance(lora_raw, dict):
            raise TypeError("lora must be a mapping")
        lora_known = {f.name for f in dataclasses.fields(LoraSettings)}
        lora_unknown = set(lora_raw) - lora_known
        if lora_unknown:
            raise ValueError(f"unknown lora keys: {sorted(lora_unknown)}")
        if "target_modules" in lora_raw:
            lora_raw = {**lora_raw, "target_modules": tuple(lora_raw["target_modules"])}
        if "exclude_datasets" in data:
            data["exclude_datasets"] = tuple(data["exclude_datasets"] or ())
        return cls(**data, lora=LoraSettings(**lora_raw))

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: config must be a mapping")
        return cls.from_dict(raw)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-row soft CE over candidate slots; -inf logits (invalid slots) contribute zero. Returns [B]."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    terms = torch.where(targets > 0, targets * logp, torch.zeros_like(logp))
    return -terms.sum(dim=-1)


def symmetric_kl(
    logits_a: torch.Tensor, logits_b: torch.Tensor, inv_a: torch.Tensor, inv_b: torch.Tensor, num_options: torch.Tensor
) -> torch.Tensor:
    """Half of KL(p||q) + KL(q||p) between two option orders of the same rows, compared per original option.

    ``inv_x[r, o]`` is the slot of original option ``o`` in view x (padded past ``num_options[r]``). Returns [B].
    """
    valid = torch.arange(logits_a.shape[1], device=logits_a.device).unsqueeze(0) < num_options.unsqueeze(1)
    lp_a = torch.log_softmax(logits_a.float(), dim=-1).gather(1, inv_a)
    lp_b = torch.log_softmax(logits_b.float(), dim=-1).gather(1, inv_b)
    lp_a = torch.where(valid, lp_a, torch.zeros_like(lp_a))
    lp_b = torch.where(valid, lp_b, torch.zeros_like(lp_b))
    terms = (lp_a.exp() - lp_b.exp()) * (lp_a - lp_b)
    return 0.5 * torch.where(valid, terms, torch.zeros_like(terms)).sum(dim=-1)


def lr_factor(step: int, *, total: int, warmup: int, min_ratio: float) -> float:
    """Linear warmup then cosine decay from 1 to `min_ratio` at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def stratified_subset(rows: list[Decision], n: int, *, seed: int) -> list[Decision]:
    """Proportional per-(dataset, kind) quotas; within a stratum a seeded shuffle then first-N."""
    if n >= len(rows):
        return rows
    strata: dict[tuple[str, str], list[Decision]] = defaultdict(list)
    for d in rows:
        strata[(d.dataset, d.kind)].append(d)
    rng = random.Random(seed)
    out: list[Decision] = []
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        out.extend(group[: max(1, round(n * len(group) / len(rows)))])
    return out


def _encode_all(tokenizer: Any, readout: Readout, rows: list[Decision], max_tokens: int) -> list[list[int] | None]:
    """Batched `Readout.encode(max_tokens=...)`: ids per row, None where the prompt is longer than max_tokens."""
    texts = [readout.render(d) for d in rows]
    ids_all: list[list[int]] = []
    for start in range(0, len(texts), 1024):
        ids_all.extend(tokenizer(texts[start : start + 1024], add_special_tokens=False)["input_ids"])
    return [ids if len(ids) <= max_tokens else None for ids in ids_all]


def _encode(
    tokenizer: Any, readout: Readout, rows: list[Decision], max_tokens: int
) -> tuple[list[list[int]], list[Decision], int]:
    """Kept ids, kept rows and the dropped count."""
    encoded: list[list[int]] = []
    kept: list[Decision] = []
    dropped = 0
    for ids, d in zip(_encode_all(tokenizer, readout, rows, max_tokens), rows, strict=True):
        if ids is None:
            dropped += 1
            continue
        encoded.append(ids)
        kept.append(d)
    return encoded, kept, dropped


def _permute(rows: list[Decision], rng: random.Random) -> tuple[list[Decision], list[list[int] | None]]:
    """Shuffle option order of choice rows only: score levels are ordered and noul is always ("No", "Yes").

    Returns the rows and, per row, the order used (``new[p] = old[order[p]]``) or None when unchanged.
    """
    out: list[Decision] = []
    orders: list[list[int] | None] = []
    for d in rows:
        if d.kind != "choice":
            out.append(d)
            orders.append(None)
            continue
        order = list(range(len(d.options)))
        rng.shuffle(order)
        out.append(d.permuted(order))
        orders.append(order)
    return out, orders


def _inverse(order: list[int] | None, n: int) -> list[int]:
    """Slot of each original option in a view built with ``order`` (identity when None)."""
    if order is None:
        return list(range(n))
    inv = [0] * n
    for pos, orig in enumerate(order):
        inv[orig] = pos
    return inv


def _load_pool(cfg: TrainConfig) -> list[Decision]:
    excluded = set(cfg.exclude_datasets)
    rows: list[Decision] = []
    for d in read_jsonl(Path(cfg.pool)):
        if d.dataset in excluded:
            continue
        rows.append(d)
        if cfg.n_rows is not None and len(rows) >= cfg.n_rows:
            break
    if cfg.n_rows is not None and len(rows) < cfg.n_rows:
        logger.warning("pool has only %d rows after filtering (requested %d)", len(rows), cfg.n_rows)
    return rows


def lora_targets(model: Any, names: tuple[str, ...]) -> str:
    """PEFT regex for the named projections inside the text decoder only (skips vision/audio towers)."""
    decoder = model.get_decoder()
    prefix = next(n for n, m in model.named_modules() if m is decoder)
    alternatives = "|".join(re.escape(n) for n in names)
    return rf"{re.escape(prefix)}\..*\.({alternatives})" if prefix else rf".*\.({alternatives})"


@dataclass(slots=True)
class _Epoch:
    encoded: list[list[int]]
    rows: list[Decision]
    steps: list[list[list[int]]]  # optimizer steps -> micro-batches -> row indices
    # Consistency pairs: partner[i] is the second option order of row i (-1 if none); inv[i] maps original option
    # -> slot in row i. Partners sit after the first views and never appear in `steps` themselves.
    partner: list[int] = field(default_factory=list)
    inv: list[list[int]] = field(default_factory=list)

    def with_partners(self, idx: list[int]) -> list[int]:
        return idx + [self.partner[i] for i in idx if self.partner and self.partner[i] >= 0]

    @property
    def first_rows(self) -> list[Decision]:
        return self.rows[: len(self.partner)] if self.partner else self.rows


def _plan_epoch(
    cfg: TrainConfig, tokenizer: Any, readout: Readout, base: list[Decision], epoch: int
) -> tuple[_Epoch, int]:
    rng = random.Random(cfg.seed * 1000 + epoch)
    rows, orders = _permute(base, rng) if cfg.shuffle_options else (base, [None] * len(base))
    ids_all = _encode_all(tokenizer, readout, rows, cfg.max_seq_tokens)
    keep = [i for i, ids in enumerate(ids_all) if ids is not None]
    dropped = len(rows) - len(keep)
    encoded = [ids_all[i] for i in keep]
    kept = [rows[i] for i in keep]
    partner: list[int] = []
    inv: list[list[int]] = []
    lengths = [len(x) for x in encoded]
    max_tokens, max_rows = cfg.batch_tokens, cfg.max_rows_per_micro
    if cfg.consistency_weight > 0:
        inv = [_inverse(orders[i], len(rows[i].options)) for i in keep]
        partner = [-1] * len(keep)
        second_rows: list[Decision] = []
        second_orders: list[list[int]] = []
        firsts: list[int] = []
        for j, i in enumerate(keep):
            d = base[i]
            if d.kind != "choice" or len(d.options) < 2:
                continue
            first = orders[i] or list(range(len(d.options)))
            order = list(range(len(d.options)))
            while True:
                rng.shuffle(order)
                if order != first:
                    break
            second_rows.append(d.permuted(order))
            second_orders.append(list(order))
            firsts.append(j)
        for j, order, d, ids in zip(
            firsts,
            second_orders,
            second_rows,
            _encode_all(tokenizer, readout, second_rows, cfg.max_seq_tokens),
            strict=True,
        ):
            if ids is None:
                continue
            partner[j] = len(encoded)
            lengths[j] = max(lengths[j], len(ids))
            encoded.append(ids)
            kept.append(d)
            inv.append(_inverse(order, len(d.options)))
        # A micro-batch holds up to two views per row: halve the budgets so its padded size stays the same.
        max_tokens, max_rows = max(cfg.max_seq_tokens, cfg.batch_tokens // 2), max(1, cfg.max_rows_per_micro // 2)
    micros = token_budget_batches(
        lengths,
        max_tokens=max_tokens,
        max_rows=max_rows,
        shuffle=True,
        seed=cfg.seed + epoch,
    )
    steps: list[list[list[int]]] = []
    current: list[list[int]] = []
    count = 0
    for micro in micros:
        current.append(micro)
        count += len(micro)
        if count >= cfg.rows_per_step:
            steps.append(current)
            current, count = [], 0
    if current:
        steps.append(current)
    return _Epoch(encoded, kept, steps, partner, inv), dropped


class _Trainer:
    def __init__(self, cfg: TrainConfig, model: Any, code_ids: torch.Tensor, pad_id: int) -> None:
        self.cfg = cfg
        self.model = model
        self.code_ids = code_ids
        self.pad_id = pad_id
        self.oom_retries = 0

    def _batch(self, encoded: list[list[int]], rows: list[Decision], idx: list[int]) -> Batch:
        return collate([encoded[i] for i in idx], [rows[i] for i in idx], pad_id=self.pad_id).to("cuda")

    def _backward(self, ep: _Epoch, idx: list[int], denom: int) -> float:
        """Loss per first-view row: soft CE, or with a partner view the mean CE of both views plus
        ``consistency_weight`` * symmetric KL between them (compared per original option)."""
        full = ep.with_partners(idx)
        batch = self._batch(ep.encoded, ep.rows, full)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = code_logits(self.model, batch, self.code_ids)
        ce = soft_cross_entropy(logits, batch.targets)
        n_first = len(idx)
        if len(full) == n_first:
            loss = ce.sum()
        else:
            paired = [k for k, i in enumerate(idx) if ep.partner[i] >= 0]
            pos_a = torch.tensor(paired, dtype=torch.long, device=logits.device)
            pos_b = torch.arange(n_first, len(full), device=logits.device)

            def inv(indices: list[int]) -> torch.Tensor:
                out = torch.zeros((len(indices), logits.shape[1]), dtype=torch.long)
                for r, i in enumerate(indices):
                    out[r, : len(ep.inv[i])] = torch.tensor(ep.inv[i], dtype=torch.long)
                return out.to(logits.device)

            kl = symmetric_kl(
                logits[pos_a],
                logits[pos_b],
                inv([idx[k] for k in paired]),
                inv(full[n_first:]),
                batch.num_options[pos_a],
            )
            single = ce[:n_first].sum() - ce[pos_a].sum()
            loss = single + (0.5 * (ce[pos_a] + ce[pos_b]) + self.cfg.consistency_weight * kl).sum()
        (loss / denom).backward()
        return float(loss.detach())

    def accumulate(self, ep: _Epoch, micros: list[list[int]], denom: int) -> float:
        """Accumulate grads for one optimizer step, loss scaled by 1/denom. Returns loss sum.

        On OOM, grads are cleared and the whole step is redone with every piece halved, so each row
        contributes exactly once. Re-raises once all pieces are single rows.
        """
        pieces = micros
        while True:
            try:
                return sum(self._backward(ep, p, denom) for p in pieces)
            except torch.cuda.OutOfMemoryError:
                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if all(len(p) == 1 for p in pieces):
                    raise
                self.oom_retries += 1
                pieces = [h for p in pieces for h in (p[: (len(p) + 1) // 2], p[(len(p) + 1) // 2 :]) if h]
                logger.warning(
                    "OOM during step; retrying with %d pieces (max %d rows)", len(pieces), max(map(len, pieces))
                )

    @torch.no_grad()
    def evaluate(self, encoded: list[list[int]], rows: list[Decision]) -> dict[str, Any]:
        self.model.eval()
        lengths = [len(x) for x in encoded]
        nll: list[float] = []
        agree: list[float] = []
        order: list[int] = []
        for idx in token_budget_batches(
            lengths, max_tokens=self.cfg.batch_tokens, max_rows=self.cfg.max_rows_per_micro, shuffle=False, seed=0
        ):
            batch = self._batch(encoded, rows, idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = code_logits(self.model, batch, self.code_ids)
            nll.extend(soft_cross_entropy(logits, batch.targets).tolist())
            agree.extend((logits.argmax(-1) == batch.targets.argmax(-1)).float().tolist())
            order.extend(idx)
        self.model.train()
        groups: dict[str, list[int]] = defaultdict(list)
        for pos, i in enumerate(order):
            groups[f"{rows[i].dataset}/{rows[i].kind}"].append(pos)
        per_group = {
            key: {
                "n": len(pos),
                "nll": sum(nll[p] for p in pos) / len(pos),
                "top1": sum(agree[p] for p in pos) / len(pos),
            }
            for key, pos in sorted(groups.items())
        }
        return {"n": len(nll), "nll": sum(nll) / len(nll), "top1": sum(agree) / len(agree), "groups": per_group}


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _prepare_run_dir(cfg: TrainConfig, config_path: Path) -> Path:
    run_dir = Path(cfg.out_root) / cfg.name
    if (run_dir / "train_summary.json").exists():
        raise FileExistsError(f"{run_dir} already holds a finished run; resume is not supported, pick a new name")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    return run_dir


def run(config_path: Path) -> Path:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = TrainConfig.load(config_path)
    run_dir = _prepare_run_dir(cfg, config_path)
    metrics_path = run_dir / "metrics.jsonl"
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    readout = Readout(tokenizer)
    code_ids = torch.tensor(readout.code_token_ids, dtype=torch.long, device="cuda")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    base_rows = _load_pool(cfg)
    epochs: list[_Epoch] = []
    dropped = 0
    for epoch in range(cfg.epochs):
        planned, n_dropped = _plan_epoch(cfg, tokenizer, readout, base_rows, epoch)
        epochs.append(planned)
        dropped += n_dropped
    dev_rows = stratified_subset(list(read_jsonl(Path(cfg.dev_panel))), cfg.dev_subset, seed=cfg.seed)
    dev_encoded, dev_kept, dev_dropped = _encode(tokenizer, readout, dev_rows, cfg.max_seq_tokens)
    total_steps = sum(len(e.steps) for e in epochs)
    logger.info("rows=%d dropped=%d steps=%d dev=%d", len(epochs[0].first_rows), dropped, total_steps, len(dev_kept))

    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda")
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=lora_targets(model, cfg.lora.target_modules),
    )
    model = get_peft_model(model, lora)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    warmup = max(1, round(cfg.warmup_ratio * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total=total_steps, warmup=warmup, min_ratio=cfg.min_lr_ratio)
    )
    trainer = _Trainer(cfg, model, code_ids, pad_id)

    step = rows_seen = tokens_seen = 0
    ema: float | None = None
    best: dict[str, Any] | None = None
    last_eval: dict[str, Any] | None = None
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def do_eval() -> None:
        nonlocal best, last_eval
        result = trainer.evaluate(dev_encoded, dev_kept)
        last_eval = {"type": "eval", "step": step, **result}
        _append(metrics_path, last_eval)
        logger.info("eval step=%d nll=%.4f top1=%.4f", step, result["nll"], result["top1"])
        if best is None or result["nll"] < best["nll"]:
            best = {"step": step, **result}
            model.save_pretrained(run_dir / "best")

    for ep in epochs:
        for micros in ep.steps:
            denom = sum(len(m) for m in micros)
            loss_sum = trainer.accumulate(ep, micros, denom)
            tokens_seen += sum(len(ep.encoded[i]) for micro in micros for i in ep.with_partners(micro))
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip))
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            rows_seen += denom
            loss = loss_sum / denom
            ema = loss if ema is None else EMA_DECAY * ema + (1 - EMA_DECAY) * loss
            if step % LOG_EVERY == 0 or step == total_steps:
                elapsed = time.perf_counter() - start
                _append(
                    metrics_path,
                    {
                        "type": "train",
                        "step": step,
                        "rows_seen": rows_seen,
                        "tokens_seen": tokens_seen,
                        "loss": loss,
                        "loss_ema": ema,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "tokens_per_s": tokens_seen / elapsed,
                        "rows_per_s": rows_seen / elapsed,
                        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "elapsed_s": elapsed,
                    },
                )
                logger.info("step %d/%d loss=%.4f ema=%.4f", step, total_steps, loss, ema)
            if cfg.eval_every and step % cfg.eval_every == 0:
                do_eval()
            if cfg.save_every and step % cfg.save_every == 0:
                model.save_pretrained(run_dir / f"step-{step}")
    if last_eval is None or last_eval["step"] != step:
        do_eval()
    model.save_pretrained(run_dir / "adapter")
    wall = time.perf_counter() - start

    counts = Counter(f"{d.dataset}/{d.kind}" for d in epochs[0].first_rows)
    summary = {
        "config": dataclasses.asdict(cfg),
        "base_model": cfg.model,
        "counts": {
            "rows_used": len(epochs[0].first_rows),
            "consistency_pairs": sum(p >= 0 for p in epochs[0].partner),
            "by_dataset_kind": dict(sorted(counts.items())),
            "dropped_too_long": dropped,
            "dev_rows": len(dev_kept),
            "dev_dropped_too_long": dev_dropped,
        },
        "trainable_params": n_trainable,
        "steps": step,
        "rows_seen": rows_seen,
        "total_tokens": tokens_seen,
        "wall_time_s": wall,
        "tokens_per_s": tokens_seen / wall,
        "rows_per_s": rows_seen / wall,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
        "oom_retries": trainer.oom_retries,
        "best": best,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "gpu": torch.cuda.get_device_name(),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir
