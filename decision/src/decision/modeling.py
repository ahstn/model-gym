"""Model loading, batching and the constrained option-code readout."""

from __future__ import annotations

import logging

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from decision.prompt import MAX_OPTIONS

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from decision.schema import Decision

log = logging.getLogger(__name__)

DEFAULT_MODEL = "openbmb/MiniCPM5-2B"


def load_model(
    model_id: str = DEFAULT_MODEL,
    *,
    adapter: str | Path | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
    merge: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base (+ adapter). ``merge`` folds the adapter into the base weights: inference only, faster, and the
    logits differ from the unmerged model by bf16 rounding."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa", trust_remote_code=trust_remote_code
    )
    model.to(device)
    if adapter is not None:
        from peft import PeftModel

        log.info("loading adapter %s", adapter)
        model = PeftModel.from_pretrained(model, str(adapter)).to(device)
        if merge:
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


@dataclass(slots=True)
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_index: torch.Tensor
    num_options: torch.Tensor
    targets: torch.Tensor

    def to(self, device: torch.device | str) -> Batch:
        return Batch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            last_index=self.last_index.to(device, non_blocking=True),
            num_options=self.num_options.to(device, non_blocking=True),
            targets=self.targets.to(device, non_blocking=True),
        )


def collate(encoded: Sequence[list[int]], decisions: Sequence[Decision], *, pad_id: int) -> Batch:
    if len(encoded) != len(decisions):
        raise ValueError(f"{len(encoded)} encodings for {len(decisions)} decisions")
    rows = len(encoded)
    width = max(len(ids) for ids in encoded)
    input_ids = torch.full((rows, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((rows, width), dtype=torch.long)
    targets = torch.zeros((rows, MAX_OPTIONS), dtype=torch.float32)
    for i, (ids, d) in enumerate(zip(encoded, decisions, strict=True)):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
        targets[i, : len(d.target)] = torch.tensor(d.target, dtype=torch.float32)
    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        last_index=torch.tensor([len(ids) - 1 for ids in encoded], dtype=torch.long),
        num_options=torch.tensor([len(d.options) for d in decisions], dtype=torch.long),
        targets=targets,
    )


def token_budget_batches(
    lengths: Sequence[int], *, max_tokens: int, max_rows: int, shuffle: bool, seed: int
) -> list[list[int]]:
    """Group indices by length so padded tokens (rows * longest row) stay within max_tokens.

    A single row longer than max_tokens still gets its own batch. With shuffle=True, the order of indices inside
    equal lengths and the order of batches are shuffled with a seeded numpy Generator.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(lengths))
    if shuffle:
        idx = rng.permutation(idx)
    lens = np.asarray(lengths, dtype=np.int64)
    order = idx[np.argsort(lens[idx], kind="stable")]
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for i in order.tolist():
        new_longest = max(longest, int(lens[i]))
        if current and (len(current) + 1 > max_rows or (len(current) + 1) * new_longest > max_tokens):
            batches.append(current)
            current, new_longest = [], int(lens[i])
        current.append(i)
        longest = new_longest
    if current:
        batches.append(current)
    if shuffle:
        batches = [batches[j] for j in rng.permutation(len(batches))]
    return batches


def _backbone_and_head(model: PreTrainedModel) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Text decoder and lm_head. Multimodal checkpoints (Gemma 4) return their language model, not the wrapper."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.get_decoder(), base.lm_head


def _softcap(model: PreTrainedModel) -> float | None:
    """Gemma-style final logit soft-capping (logits = cap * tanh(logits / cap)), or None."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(base.config.get_text_config(), "final_logit_softcapping", None) or None


def _code_logits_from_hidden(
    h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, softcap: float | None
) -> torch.Tensor:
    logits = (h @ weight.T).float()
    if bias is not None:
        logits = logits + bias
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def _code_rows(head: torch.nn.Module, code_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = head.weight.index_select(0, code_token_ids.to(head.weight.device))
    bias = getattr(head, "bias", None)
    code_bias = bias.index_select(0, code_token_ids.to(bias.device)).float() if bias is not None else None
    return weight, code_bias


def code_logits(model: PreTrainedModel, batch: Batch, code_token_ids: torch.Tensor) -> torch.Tensor:
    """Logits of the option codes at the answer position, [B, MAX_OPTIONS] float32; -inf past each row's options.

    Runs only the decoder backbone and multiplies the answer hidden state by the gathered code rows of lm_head,
    so the full-vocab logits are never computed. Final logit soft-capping is applied when the model defines it.
    """
    backbone, head = _backbone_and_head(model)
    hidden = backbone(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    h = hidden[rows, batch.last_index]
    weight, bias = _code_rows(head, code_token_ids)
    logits = _code_logits_from_hidden(h, weight, bias, _softcap(model))
    positions = torch.arange(logits.shape[1], device=logits.device)
    mask = positions.unsqueeze(0) >= batch.num_options.to(logits.device).unsqueeze(1)
    return logits.masked_fill(mask, float("-inf"))


MIN_SHARED_PREFIX = 64


def code_logits_shared_prefix(
    model: PreTrainedModel,
    encoded: Sequence[list[int]],
    num_options: Sequence[int],
    code_token_ids: torch.Tensor,
    *,
    max_batch_tokens: int = 65536,
    device: str = "cuda",
) -> torch.Tensor:
    """``code_logits`` for rows sharing a token prefix, encoding that prefix once; [N, MAX_OPTIONS] float32.

    The longest common token prefix (kept at least one token short of every row) is run through the backbone once;
    its cache is expanded (as views) to each suffix chunk. Each row attends only to the prefix and its own suffix,
    exactly as in an independent forward, so rows stay isolated. The cache layers are shallow-copied, not rebuilt,
    so sliding-window layers keep their cropped keys and cumulative length. Falls back to ``collate``-style
    ``code_logits`` when there is a single row or the shared prefix is shorter than ``MIN_SHARED_PREFIX`` tokens.
    """
    if len(encoded) != len(num_options):
        raise ValueError(f"{len(encoded)} encodings for {len(num_options)} option counts")
    n = len(encoded)
    shortest = min(len(ids) for ids in encoded)
    prefix = 0
    first = encoded[0]
    while prefix < shortest - 1 and all(ids[prefix] == first[prefix] for ids in encoded):
        prefix += 1
    backbone, head = _backbone_and_head(model)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    pad_id = getattr(base.config.get_text_config(), "pad_token_id", None) or 0
    if n == 1 or prefix < MIN_SHARED_PREFIX:
        width = max(len(ids) for ids in encoded)
        input_ids = torch.full((n, width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, width), dtype=torch.long)
        for i, ids in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        batch = Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            last_index=torch.tensor([len(ids) - 1 for ids in encoded], dtype=torch.long),
            num_options=torch.tensor(list(num_options), dtype=torch.long),
            targets=torch.zeros((n, MAX_OPTIONS), dtype=torch.float32),
        ).to(device)
        return code_logits(model, batch, code_token_ids)

    prefix_ids = torch.tensor([first[:prefix]], dtype=torch.long, device=device)
    cached = backbone(input_ids=prefix_ids, use_cache=True).past_key_values
    weight, code_bias = _code_rows(head, code_token_ids)
    softcap = _softcap(model)

    suffixes = [ids[prefix:] for ids in encoded]
    max_suffix = max(len(s) for s in suffixes)
    chunk = max(1, max_batch_tokens // (prefix + max_suffix))
    out = torch.empty((n, MAX_OPTIONS), dtype=torch.float32, device=device)
    for start in range(0, n, chunk):
        rows = suffixes[start : start + chunk]
        b = len(rows)
        width = max(len(s) for s in rows)
        input_ids = torch.full((b, width), pad_id, dtype=torch.long)
        suffix_mask = torch.zeros((b, width), dtype=torch.long)
        for i, s in enumerate(rows):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            suffix_mask[i, : len(s)] = 1
        input_ids, suffix_mask = input_ids.to(device), suffix_mask.to(device)
        attention_mask = torch.cat([torch.ones((b, prefix), dtype=torch.long, device=device), suffix_mask], dim=1)
        position_ids = (prefix + torch.arange(width, device=device)).unsqueeze(0).expand(b, -1)
        # expand() is a view; cache layers concatenate into new tensors on update, so `cached` is never mutated.
        cache = copy(cached)
        cache.layers = []
        for layer in cached.layers:
            view = copy(layer)
            view.keys = layer.keys.expand(b, -1, -1, -1)
            view.values = layer.values.expand(b, -1, -1, -1)
            cache.layers.append(view)
        hidden = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state
        last = torch.tensor([len(s) - 1 for s in rows], dtype=torch.long, device=device)
        h = hidden[torch.arange(b, device=device), last]
        out[start : start + b] = _code_logits_from_hidden(h, weight, code_bias, softcap)
    k = torch.tensor(list(num_options), dtype=torch.long, device=device)
    mask = torch.arange(MAX_OPTIONS, device=device).unsqueeze(0) >= k.unsqueeze(1)
    return out.masked_fill(mask, float("-inf"))
