"""Build the training pool and held-out panels from the three Jev-style decision datasets.

Every Decision id is ``f"{dataset}/{split}/{orig_id}"`` so evaluation can recover the originating split via
:func:`split_of`. The pool uses only train splits; panels use only non-train splits and exclude any group/content that
appears in the pool.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import re

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decision.schema import KINDS, NOUL_OPTIONS, Decision, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

REPOS: dict[str, tuple[str, str]] = {
    "sargedev": ("SargeDev/jev-distill-corpus-v3", "default"),
    "openjev": ("ZefanCai/Open-Jev-v1.1", "community-hard-mix-v2-redistributable"),
    "tasksource": ("tasksource/tasksource-jev-typed-decisions", "default"),
}
DATASET_PRIORITY: tuple[str, ...] = ("tasksource", "openjev", "sargedev")  # first wins on cross-dataset duplicates
COLUMNS: dict[str, tuple[str, ...]] = {
    "sargedev": ("id", "kind", "options", "target", "state", "question", "source"),
    # metadata_json / record_json are deliberately never loaded: they must not reach the model.
    "openjev": ("id", "group_id", "kind", "options", "target", "state_json", "question", "source"),
    "tasksource": (
        "id",
        "group_id",
        "kind",
        "options",
        "target",
        "state",
        "question",
        "source",
        "variant",
        "license_use",
    ),
}
SARGEDEV_SOURCES = ("yuri_v3", "yuri_v1")  # openjev_v2 is a reschema of Open-Jev v2 and is dropped entirely
MAX_OPTIONS = 255
MAX_STATE_CHARS = 24_000
NEGATIVE_TOL = 1e-4
MAX_NOUL_ENTROPY_BITS = 0.97
WANLI_MAX_SHARE = 0.05
PACKED_DERIVED_MAX_SHARE = 0.05
NGRAM = 13
MIXTURE: dict[str, float] = {
    "tasksource": 0.37,
    "sargedev/yuri_v3": 0.32,
    "sargedev/yuri_v1": 0.10,
    "openjev/other": 0.16,
    "openjev/wanli": 0.05,
}
DEV_PANEL: tuple[tuple[str, str, int], ...] = (
    ("sargedev", "test_set_30k", 1500),
    ("openjev", "test", 1500),
    ("openjev", "ood", 1000),
    ("tasksource", "test", 1500),
)
CALIB_PANEL: tuple[tuple[str, str, int], ...] = (
    ("sargedev", "calibration", 1000),
    ("openjev", "calibration", 1000),
    ("tasksource", "validation", 1000),
)

# tasksource sources overlapping the decision-index evaluation suite (suite overlap; rationale in
# docs/jev-decision-model-2026-09-26/research/datasets.md §3). Long names match as case-insensitive substrings.
SUITE_BLOCKLIST_SUBSTRINGS: tuple[str, ...] = (
    "winogrande", "hellaswag", "wanli", "commonsense_qa", "csqa", "openbookqa", "cladder", "banking77", "sms_spam",
    "humicroedit", "nli4ct", "contract-nli", "contract_nli", "sarcasm", "mmlu", "gpqa", "big_bench", "bigbench",
    "gsm8k", "finentity", "cruxeval", "ragtruth", "api-bank", "api_bank", "toolret", "when2call", "forecastbench",
    "habermas",
)  # fmt: skip
# Short names match only as whole tokens so e.g. "search" does not hit "arc" and "brightness" does not hit "bright".
SUITE_BLOCKLIST_TOKENS: tuple[str, ...] = (
    "esci", "anli", "hover", "clinc", "bbh", "musr", "vast", "acos", "bfcl", "bright",
    "ai2_arc", "arc_easy", "arc_challenge", "arc-easy", "arc-challenge",
)  # fmt: skip
_BLOCK_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(re.escape(t) for t in SUITE_BLOCKLIST_TOKENS) + r")(?![a-z0-9])"
)
_WORD_RE = re.compile(r"\w+")
_NGRAM_BASE = np.uint64(1099511628211)
SUITE_TEXT_KEYS = ("state", "questions", "context", "text", "question", "prompt", "options")


@dataclass(frozen=True, slots=True)
class Candidate:
    decision: Decision
    key: str  # cross-dataset dedup hash
    variant: str = ""
    license_use: str = ""


def split_of(decision_id: str) -> str:
    """Originating split of a Decision id of the form ``dataset/split/orig_id``."""
    parts = decision_id.split("/", 2)
    if len(parts) < 3:
        raise ValueError(f"not a dataset/split/id decision id: {decision_id!r}")
    return parts[1]


def is_blocklisted(source: str) -> bool:
    s = source.casefold()
    return any(t in s for t in SUITE_BLOCKLIST_SUBSTRINGS) or _BLOCK_TOKEN_RE.search(s) is not None


def is_wanli(source: str) -> bool:
    return "wanli" in source.casefold()


def binary_entropy_bits(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _norm_text(s: str) -> str:
    return " ".join(s.casefold().split())


def dedup_key(d: Decision) -> str:
    opts = "\x1e".join(sorted(_norm_text(o) for o in d.options))
    return hashlib.sha256(f"{_norm_text(d.state)}\x1f{_norm_text(d.question)}\x1f{opts}".encode()).hexdigest()


def _noul(options: list[str], target: list[float]) -> tuple[list[str], list[float]] | str:
    if not options and len(target) == 1:  # tasksource: target [P(yes)]
        return list(NOUL_OPTIONS), [1.0 - target[0], target[0]]
    folded = [o.strip().casefold() for o in options]
    if folded in (["no", "yes"], ["false", "true"]):
        return list(NOUL_OPTIONS), target
    if folded in (["yes", "no"], ["true", "false"]):
        return list(NOUL_OPTIONS), target[::-1]
    return "noul_options"


def normalize(dataset: str, split: str, raw: Mapping[str, Any]) -> Decision | str:
    """Map one raw dataset row to a Decision, or return the drop reason."""
    kind = raw["kind"]
    if kind not in KINDS:
        return "bad_kind"
    state = raw["state_json"] if dataset == "openjev" else raw["state"]
    if state is not None and not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    question = raw["question"]
    if not state or not state.strip() or not question or not str(question).strip():
        return "empty_text"
    if len(state) > MAX_STATE_CHARS:
        return "state_too_long"
    options = [str(o) for o in raw["options"]] if raw["options"] is not None else []
    target = [float(t) for t in raw["target"]] if raw["target"] is not None else []
    if kind == "noul":
        mapped = _noul(options, target)
        if isinstance(mapped, str):
            return mapped
        options, target = mapped
    if not 2 <= len(options) <= MAX_OPTIONS:
        return "option_count"
    if len(target) != len(options):
        return "target_length"
    if len({o.strip().casefold() for o in options}) != len(options):
        return "duplicate_options"
    t = np.asarray(target, dtype=np.float64)
    if not np.isfinite(t).all():
        return "nonfinite_target"
    if (t < -NEGATIVE_TOL).any():
        return "negative_target"
    t = np.clip(t, 0.0, None)
    total = t.sum()
    if total <= 0:
        return "zero_target"
    raw_id = str(raw["id"])
    return Decision(
        id=f"{dataset}/{split}/{raw_id}",
        dataset=dataset,
        source=str(raw.get("source") or ""),
        kind=kind,
        state=state,
        question=str(question),
        options=tuple(options),
        target=tuple((t / total).tolist()),
        group_id=str(raw.get("group_id") or raw_id),
    )


def allocate(total: int, shares: Mapping[str, float], available: Mapping[str, int]) -> dict[str, int]:
    """Split ``total`` by ``shares``; streams short of their share give all they have and the deficit is
    redistributed proportionally over the remaining streams (largest-remainder rounding)."""
    alloc = dict.fromkeys(shares, 0)
    open_ = {k for k in shares if available[k] > 0 and shares[k] > 0}
    remaining = total
    while open_ and remaining > 0:
        weight = sum(shares[k] for k in open_)
        exact = {k: alloc[k] + remaining * shares[k] / weight for k in open_}
        full = [k for k in open_ if exact[k] >= available[k]]
        if full:
            for k in full:
                remaining -= available[k] - alloc[k]
                alloc[k] = available[k]
                open_.remove(k)
            continue
        floors = {k: math.floor(exact[k]) for k in open_}
        left = total - sum(floors.values()) - sum(v for k, v in alloc.items() if k not in open_)
        for k in sorted(open_, key=lambda k: (floors[k] - exact[k], k))[:left]:
            floors[k] += 1
        alloc.update(floors)
        remaining = 0
    return alloc


def ngram_hashes(text: str, n: int = NGRAM) -> np.ndarray:
    """Hashes of all word n-grams of the normalized text (process-local hashing; compare within one process)."""
    words = _WORD_RE.findall(text.casefold())
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter((hash(w) for w in words), dtype=np.int64, count=len(words)).view(np.uint64)
    m = len(words) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    for k in range(n):
        h = h * _NGRAM_BASE + ids[k : k + m]
    return h


class SuiteFilter:
    """Detects train states sharing any word 13-gram with any suite item text."""

    def __init__(self, texts: Sequence[str]) -> None:
        parts = [ngram_hashes(t) for t in texts]
        self.hashes = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)

    def hits(self, text: str) -> bool:
        if not len(self.hashes):
            return False
        h = ngram_hashes(text)
        if not len(h):
            return False
        idx = np.minimum(np.searchsorted(self.hashes, h), len(self.hashes) - 1)
        return bool((self.hashes[idx] == h).any())


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _strings(v)


def read_suite_texts(path: Path) -> list[str]:
    with path.open("rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    opener = gzip.open if gz else open
    texts: list[str] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, Mapping):
                texts.extend(s for k in SUITE_TEXT_KEYS if k in row for s in _strings(row[k]))
    return texts


def _load_frame(dataset: str, split: str, revision: str) -> Any:
    from datasets import load_dataset

    repo, config = REPOS[dataset]
    ds = load_dataset(repo, config, split=split, revision=revision)
    cols = [c for c in COLUMNS[dataset] if c in ds.column_names]
    return ds.select_columns(cols).to_pandas()


def load_candidates(
    dataset: str, split: str, revision: str, drops: Counter[str], blocked: Counter[str]
) -> list[Candidate]:
    df = _load_frame(dataset, split, revision)
    log.info("loaded %s/%s: %d rows", dataset, split, len(df))
    prefix = f"{dataset}/{split}"
    if dataset == "tasksource":
        counts = df["source"].astype(str).value_counts()
        hit = {s: int(c) for s, c in counts.items() if is_blocklisted(s)}
        blocked.update(hit)
        mask = df["source"].astype(str).isin(hit)
        drops[f"{prefix}/suite_blocklist"] += int(mask.sum())
        df = df[~mask]
    if dataset == "sargedev":
        mask = ~df["source"].isin(SARGEDEV_SOURCES)
        drops[f"{prefix}/sargedev_source"] += int(mask.sum())
        df = df[~mask]
    state_col = "state_json" if dataset == "openjev" else "state"
    mask = df[state_col].str.len() > MAX_STATE_CHARS
    drops[f"{prefix}/state_too_long"] += int(mask.sum())
    df = df[~mask]
    cols = {c: df[c].tolist() for c in df.columns}
    out: list[Candidate] = []
    for i in range(len(df)):
        raw = {c: v[i] for c, v in cols.items()}
        d = normalize(dataset, split, raw)
        if isinstance(d, str):
            drops[f"{prefix}/{d}"] += 1
            continue
        if d.source == "yuri_v1" and d.kind == "noul" and binary_entropy_bits(d.target[1]) > MAX_NOUL_ENTROPY_BITS:
            drops[f"{prefix}/uniform_noul"] += 1
            continue
        out.append(Candidate(d, dedup_key(d), str(raw.get("variant") or ""), str(raw.get("license_use") or "")))
    log.info("normalized %s: kept %d", prefix, len(out))
    return out


def dedup(by_dataset: Mapping[str, list[Candidate]], drops: Counter[str], split: str) -> dict[str, list[Candidate]]:
    """Exact dedup across datasets; the first dataset in DATASET_PRIORITY keeps the row."""
    seen: set[str] = set()
    out: dict[str, list[Candidate]] = {}
    for ds in DATASET_PRIORITY:
        kept = []
        for c in by_dataset.get(ds, []):
            if c.key in seen:
                drops[f"{ds}/{split}/duplicate"] += 1
                continue
            seen.add(c.key)
            kept.append(c)
        out[ds] = kept
    return out


def select_uniform(
    cands: Sequence[Candidate], n: int, rng: np.random.Generator, keep: Callable[[Candidate], bool]
) -> list[Candidate]:
    out: list[Candidate] = []
    for i in rng.permutation(len(cands)):
        if len(out) >= n:
            break
        if keep(cands[i]):
            out.append(cands[i])
    return out


def tasksource_cap(quota: int, n_sources: int) -> int:
    return max(20, math.ceil(3 * quota / max(n_sources, 1)))


def select_round_robin(
    cands: Sequence[Candidate], quota: int, rng: np.random.Generator, keep: Callable[[Candidate], bool]
) -> list[Candidate]:
    """Round-robin over sources (per-source cap, packed_derived <= 5% of quota) so small sources are not starved."""
    by_source: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        by_source[c.decision.source].append(c)
    cap = tasksource_cap(quota, len(by_source))
    packed_max = math.floor(PACKED_DERIVED_MAX_SHARE * quota)
    queues = {s: iter([v[i] for i in rng.permutation(len(v))]) for s, v in sorted(by_source.items())}
    taken: Counter[str] = Counter()
    out: list[Candidate] = []
    packed = 0
    active = list(queues)
    while active and len(out) < quota:
        nxt = []
        for s in active:
            if len(out) >= quota:
                break
            for c in queues[s]:
                is_packed = c.variant == "packed_derived"
                if (is_packed and packed >= packed_max) or not keep(c):
                    continue
                out.append(c)
                taken[s] += 1
                packed += is_packed
                break
            else:
                continue
            if taken[s] < cap:
                nxt.append(s)
        active = nxt
    return out


def tasksource_available(cands: Sequence[Candidate], quota: int) -> int:
    sizes = Counter(c.decision.source for c in cands)
    cap = tasksource_cap(quota, len(sizes))
    return sum(min(v, cap) for v in sizes.values())


def select_stratified(cands: Sequence[Candidate], n: int, rng: np.random.Generator) -> list[Candidate]:
    by_kind: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        by_kind[c.decision.kind].append(c)
    total = len(cands)
    alloc = allocate(n, {k: len(v) / total for k, v in by_kind.items()}, {k: len(v) for k, v in by_kind.items()})
    out: list[Candidate] = []
    for k in sorted(by_kind):
        v = by_kind[k]
        out.extend(v[i] for i in sorted(rng.choice(len(v), size=alloc[k], replace=False)))
    return out


def _streams(train: Mapping[str, list[Candidate]]) -> dict[str, list[Candidate]]:
    sd, oj = train["sargedev"], train["openjev"]
    return {
        "tasksource": train["tasksource"],
        "sargedev/yuri_v3": [c for c in sd if c.decision.source == "yuri_v3"],
        "sargedev/yuri_v1": [c for c in sd if c.decision.source == "yuri_v1"],
        "openjev/other": [c for c in oj if not is_wanli(c.decision.source)],
        "openjev/wanli": [c for c in oj if is_wanli(c.decision.source)],
    }


def _plan(pool_size: int, streams: Mapping[str, list[Candidate]]) -> dict[str, int]:
    available = {k: len(v) for k, v in streams.items()}
    available["openjev/wanli"] = min(available["openjev/wanli"], math.floor(WANLI_MAX_SHARE * pool_size))
    for _ in range(20):
        plan = allocate(pool_size, MIXTURE, available)
        ts_avail = tasksource_available(streams["tasksource"], plan["tasksource"])
        if ts_avail >= plan["tasksource"] or ts_avail == available["tasksource"]:
            return plan
        available["tasksource"] = ts_avail
    return allocate(pool_size, MIXTURE, available)


def _counts(rows: Sequence[Decision]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "kind": dict(Counter(d.kind for d in rows)),
        "dataset_kind": dict(Counter(f"{d.dataset}/{d.kind}" for d in rows)),
        "dataset_split": dict(Counter(f"{d.dataset}/{split_of(d.id)}" for d in rows)),
    }


def _build_panel(
    spec: Sequence[tuple[str, str, int]],
    revisions: Mapping[str, str],
    pool_groups: set[tuple[str, str]],
    panel_groups: set[tuple[str, str]],
    seen_keys: set[str],
    rng: np.random.Generator,
    drops: Counter[str],
    blocked: Counter[str],
) -> list[Decision]:
    """`panel_groups` holds groups of earlier panels; it is extended with this panel's picks."""
    rows: list[Decision] = []
    for dataset, split, n in spec:
        cands = load_candidates(dataset, split, revisions[dataset], drops, blocked)
        clean = []
        for c in cands:
            group = (dataset, c.decision.group_id)
            if group in pool_groups or group in panel_groups or c.key in seen_keys:
                drops[f"{dataset}/{split}/panel_leak"] += 1
                continue
            seen_keys.add(c.key)
            clean.append(c)
        picked = select_stratified(clean, n, rng)
        log.info("panel %s/%s: %d of %d", dataset, split, len(picked), len(clean))
        rows.extend(c.decision for c in picked)
    panel_groups.update((d.dataset, d.group_id) for d in rows)
    return rows


def build(out_dir: Path, *, pool_size: int, seed: int, suite_rows: Path | None) -> dict:
    from huggingface_hub import HfApi

    rng = np.random.default_rng(seed)
    api = HfApi()
    revisions = {ds: str(api.dataset_info(repo).sha) for ds, (repo, _) in REPOS.items()}
    log.info("revisions: %s", revisions)
    drops: Counter[str] = Counter()
    blocked: Counter[str] = Counter()

    train = dedup(
        {ds: load_candidates(ds, "train", revisions[ds], drops, blocked) for ds in DATASET_PRIORITY}, drops, "train"
    )
    streams = _streams(train)
    plan = _plan(pool_size, streams)
    log.info("mixture plan: %s", plan)

    suite = SuiteFilter(read_suite_texts(suite_rows)) if suite_rows is not None else None
    suite_dropped: Counter[str] = Counter()

    def keep(c: Candidate) -> bool:
        d = c.decision
        if suite is not None and suite.hits("\n".join((d.state, d.question, *d.options))):
            suite_dropped[c.decision.dataset] += 1
            return False
        return True

    selected: dict[str, list[Candidate]] = {}
    for name, cands in streams.items():
        pick = select_round_robin if name == "tasksource" else select_uniform
        selected[name] = pick(cands, plan[name], rng, keep)
        if len(selected[name]) < plan[name]:
            log.warning("stream %s short: %d of %d planned", name, len(selected[name]), plan[name])
    n_total = sum(len(v) for v in selected.values())
    wanli_max = math.floor(WANLI_MAX_SHARE * n_total)
    selected["openjev/wanli"] = selected["openjev/wanli"][:wanli_max]
    for ds, n in suite_dropped.items():
        drops[f"{ds}/train/suite_ngram"] += n

    pool_cands = [c for v in selected.values() for c in v]
    pool = [pool_cands[i].decision for i in rng.permutation(len(pool_cands))]
    pool_groups = {(d.dataset, d.group_id) for d in pool}
    seen_keys = {c.key for c in pool_cands}

    panel_groups: set[tuple[str, str]] = set()
    dev = _build_panel(DEV_PANEL, revisions, pool_groups, panel_groups, seen_keys, rng, drops, blocked)
    calib = _build_panel(CALIB_PANEL, revisions, pool_groups, panel_groups, seen_keys, rng, drops, blocked)

    write_jsonl(out_dir / "pool.jsonl", pool)
    write_jsonl(out_dir / "panels" / "dev.jsonl", dev)
    write_jsonl(out_dir / "panels" / "calib.jsonl", calib)

    ts_pool = selected["tasksource"]
    manifest: dict[str, Any] = {
        "seed": seed,
        "pool_size": pool_size,
        "revisions": {ds: {"repo": REPOS[ds][0], "config": REPOS[ds][1], "sha": sha} for ds, sha in revisions.items()},
        "mixture_target": MIXTURE,
        "streams": {
            k: {"available": len(streams[k]), "planned": plan[k], "selected": len(selected[k])} for k in streams
        },
        "realized_shares": {k: len(v) / max(len(pool), 1) for k, v in selected.items()},
        "pool": _counts(pool),
        "pool_tasksource_sources": len({c.decision.source for c in ts_pool}),
        "pool_tasksource_variants": dict(Counter(c.variant for c in ts_pool)),
        "license_use": {
            "available": dict(Counter(c.license_use for c in streams["tasksource"])),
            "selected": dict(Counter(c.license_use for c in ts_pool)),
        },
        "drops": dict(sorted(drops.items())),
        "blocklisted_tasksource_sources": dict(sorted(blocked.items())),
        "suite_rows": str(suite_rows) if suite_rows is not None else None,
        "panels": {"dev": _counts(dev), "calib": _counts(calib)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log.info("wrote pool=%d dev=%d calib=%d to %s", len(pool), len(dev), len(calib), out_dir)
    return manifest


def filter_pool(pool: Path, out: Path, *, suite_rows: Sequence[Path], exclude: Sequence[str]) -> dict:
    """Copy `pool` to `out` without excluded datasets and without rows sharing a 13-gram with any suite row.

    The row order is kept, so the first N rows of the output are the first N clean rows of the input, and the
    panels built with the pool stay valid. Writes ``<out>.manifest.json`` and returns it.
    """
    suite = SuiteFilter([t for path in suite_rows for t in read_suite_texts(path)])
    excluded = set(exclude)
    kept: list[Decision] = []
    dropped: Counter[str] = Counter()
    total = 0
    for d in read_jsonl(pool):
        total += 1
        if d.dataset in excluded:
            dropped[f"{d.dataset}/excluded"] += 1
        elif suite.hits("\n".join((d.state, d.question, *d.options))):
            dropped[f"{d.dataset}/suite_ngram"] += 1
        else:
            kept.append(d)
    write_jsonl(out, kept)
    manifest = {
        "pool": str(pool),
        "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
        "suite_rows": [str(p) for p in suite_rows],
        "suite_ngrams": len(suite.hashes),
        "exclude_datasets": sorted(excluded),
        "rows_in": total,
        "rows_out": len(kept),
        "dropped": dict(sorted(dropped.items())),
        "kept_by_dataset": dict(sorted(Counter(d.dataset for d in kept).items())),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("filter-pool: %d of %d rows kept -> %s", len(kept), total, out)
    return manifest
