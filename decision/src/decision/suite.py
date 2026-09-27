"""Decision Index 0.2.1 adapter: score our model with the public kit (apolinario/decision-index @87d4650).

0.2.1 rescores the 0.2 suite files (same hashes, same ``data/suite-0.2/``) with three more read-time subsets
(ToolRet/BRIGHT answerable queries, deduplicated Home appliances), so a complete 0.2 run is a complete 0.2.1 run.
``suite score --edition 0.2`` still reproduces the 0.2 index.

The kit calls an engine once per suite request with ``(state, questions)``. Questions are ``choice`` (``criteria`` maps
option key -> description, 2..255 options) or ``noul`` (yes/no, answered with ``p_yes``); the kit has no ordered-level
kind, so our ``score`` kind is never produced here. Every question becomes its own ``Decision`` (siblings never share
context), all questions of a request go through ``modeling.code_logits_shared_prefix`` (the shared prompt prefix is
encoded once and its KV cache reused; each question attends only to that prefix and itself), and per-kind
temperatures from an ``evaluate.run`` ``temperatures.json`` are applied. A question longer than ``--max-tokens`` or
with an unsupported shape makes the whole request ``Unsupported`` (no truncation, no option pruning), as the kit's
rules require.

Commands (from ``decision/``, with the ``suite`` extra installed). HLE is gated: accept its terms and ``hf auth login``
first. Otherwise the kit's rebuild aborts on the gated download; ``rebuild`` checks access up front:

    uv run decision suite rebuild                          # rebuild into data/suite-work, import into data/suite-0.2/
    uv run decision suite sample --n 500 --out data/sample-500.jsonl.gz
    uv run decision suite run --model openbmb/MiniCPM5-2B --adapter runs/X/best \\
        --temperatures runs/X/eval/temperatures.json --out runs/X/suite-sample --sample data/sample-500.jsonl.gz
    uv run decision suite run --model openbmb/MiniCPM5-2B --adapter runs/X/best --out runs/X/suite    # full suite
    uv run decision suite score --run runs/X/suite

The kit runner sends one request at a time, leaving most of a large GPU idle. ``--shards N`` splits the selected rows
(whole case groups, stable hash) into ``<out>/shards/rows-<k>.jsonl``, runs N child processes on the same GPU (logs in
``<out>/shards/<k>.log``, each resumable), then merges into ``<out>/results.jsonl`` and scores once:

    uv run decision suite run --model openbmb/MiniCPM5-2B --adapter runs/X/best --out runs/X/suite --shards 8

The pip-installed kit ships without its ``hub/`` files, so ``rebuild`` downloads the pinned exclusions and manifests
into ``<work>/hub/`` and passes them explicitly.

Without HLE access, ``rebuild --skip-hle`` builds every other catalog id, cuts v2 by hand and imports it unverified
with a ``PARTIAL.json`` marker; ``run`` then skips strict verification and every score carries ``"partial"``. This is
NOT an official 0.2.1 suite (the index excludes HLE):

    uv run decision suite rebuild --skip-hle --suite-dir data/suite-0.2-noHLE
    uv run decision suite run --model ... --suite-dir data/suite-0.2-noHLE --out runs/X/suite

Once HLE access is granted, rebuild the full suite into a new directory and re-run into the same ``--out``. The kit
runner resumes by ``run_id`` (rows already in ``results.jsonl`` with a non-error status are skipped), so only the
missing HLE rows are computed:

    uv run decision suite rebuild --suite-dir data/suite-0.2
    uv run decision suite run --model ... --suite-dir data/suite-0.2 --out runs/X/suite

The engine is also usable from the kit CLI directly:
``python -m decision_index run --engine decision.suite:DecisionEngine --model ID --option adapter=PATH ...``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from decision_index.engines import Engine, Unsupported
from decision_index.engines.base import text

from decision.prompt import MAX_OPTIONS
from decision.schema import NOUL_OPTIONS, Decision, Kind

logger = logging.getLogger(__name__)

ENGINE = "decision.suite:DecisionEngine"
EDITION = "0.2.1"
EDITIONS = ("0.2.1", "0.2")
DEFAULT_SUITE_DIR = "data/suite-0.2"  # under data/: gitignored and never touched by pod.sh sync
DEFAULT_MAX_TOKENS = 8192
DEFAULT_BATCH_TOKENS = 65536
HLE_REPO = "cais/hle"
HLE_CATALOG_ID = 45
KIT_COMMIT = "87d4650b42b377c0291a89c1f1a879f9b31082bf"
HUB_URL = f"https://raw.githubusercontent.com/apolinario/decision-index/{KIT_COMMIT}/hub/"
HUB_FILES = {"exclusions": "excluded-questions.json", "manifest": "0.2.1/manifest.json", "manifest_v1": "manifest.json"}
PARTIAL_FILE = "PARTIAL.json"


def _option_text(key: str, description: object) -> str:
    if description is None:
        return key
    rendered = text(description)
    return rendered if rendered.strip() else key


def question_to_decision(key: str, state: object, question: Mapping[str, Any]) -> Decision:
    """One kit question -> one isolated Decision (uniform dummy target).

    Raises Unsupported for shapes we cannot score.
    """
    kind = question.get("type")
    options: tuple[str, ...]
    if kind == "choice":
        criteria = question["criteria"]
        if not 2 <= len(criteria) <= MAX_OPTIONS:
            raise Unsupported(f"question {key!r} has {len(criteria)} options; declared limit is 2..{MAX_OPTIONS}")
        options = tuple(_option_text(k, v) for k, v in criteria.items())
        our_kind: Kind = "choice"
    elif kind == "noul":
        options = NOUL_OPTIONS
        our_kind = "noul"
    else:
        raise Unsupported(f"question {key!r} has unsupported type {kind!r}")
    return Decision(
        id=key,
        dataset="decision-index",
        source=EDITION,
        kind=our_kind,
        state="" if state in ("", None, {}, []) else text(state),
        question=text(question.get("instructions", "")),
        options=options,
        target=tuple(1.0 / len(options) for _ in options),
        group_id="",
    )


def build_answer(question: Mapping[str, Any], probs: Sequence[float]) -> dict[str, Any]:
    """Kit answer for one question from probabilities over the Decision's options (same order)."""
    total = float(sum(probs))
    p = [float(x) / total for x in probs]
    if question["type"] == "noul":
        return {"type": "noul", "noul": p[NOUL_OPTIONS.index("Yes")]}
    keys = list(question["criteria"])
    if len(keys) != len(p):
        raise ValueError(f"{len(p)} probabilities for {len(keys)} options")
    dist = dict(zip(keys, p, strict=True))
    return {"type": "choice", "choice": max(dist, key=dist.__getitem__), "probabilities": dist}


def load_temperatures(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    temps = json.loads(Path(path).read_text())
    return {k: float(v) for k, v in temps.items()}


class DecisionEngine(Engine):
    """Kit engine over our LoRA decision model.

    Options: model, adapter, merge, temperatures, max_tokens, batch_tokens, device.
    """

    name = "decision"
    latency = "In-process request wall time: prompt rendering, tokenization and one batched backbone pass per request."

    def __init__(
        self,
        *,
        model: str | None = None,
        adapter: str | None = None,
        merge: bool = False,
        temperatures: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        batch_tokens: int = DEFAULT_BATCH_TOKENS,
        device: str = "cuda",
        **options: Any,
    ) -> None:
        from decision.modeling import DEFAULT_MODEL, load_model
        from decision.prompt import Readout

        super().__init__(
            model=model, adapter=adapter, merge=merge, temperatures=temperatures, max_tokens=max_tokens,
            batch_tokens=batch_tokens, device=device, **options,
        )  # fmt: skip
        self.model_id = model or DEFAULT_MODEL
        self.max_tokens = int(max_tokens)
        self.batch_tokens = int(batch_tokens)
        self.device = device
        self.temperatures = load_temperatures(temperatures)
        self.model, self.tokenizer = load_model(self.model_id, adapter=adapter, device=device, merge=merge)
        self.readout = Readout(self.tokenizer)
        self.code_ids = torch.tensor(self.readout.code_token_ids, dtype=torch.long, device=device)
        self.provenance = {
            "model": self.model_id,
            "adapter": str(adapter) if adapter else None,
            "adapter_merged": bool(adapter) and merge,
            "temperatures": self.temperatures,
            "max_tokens": self.max_tokens,
            "readout": (
                "shared prompt prefix KV cache per request, one suffix pass per question (questions isolated); "
                "softmax over option-code logits at the answer position"
            ),
        }

    def _temperature(self, kind: Kind) -> float:
        if not self.temperatures:
            return 1.0
        if kind not in self.temperatures:
            raise KeyError(f"temperatures file has no entry for kind {kind!r}")
        return self.temperatures[kind]

    @torch.inference_mode()
    def probabilities(self, decisions: Sequence[Decision], encoded: Sequence[list[int]]) -> list[list[float]]:
        from decision.modeling import code_logits_shared_prefix

        logits = code_logits_shared_prefix(
            self.model,
            encoded,
            [len(d.options) for d in decisions],
            self.code_ids,
            max_batch_tokens=self.batch_tokens,
            device=self.device,
        ).double()
        return [
            torch.softmax(logits[r, : len(d.options)] / self._temperature(d.kind), dim=-1).tolist()
            for r, d in enumerate(decisions)
        ]

    def __call__(self, state: object, questions: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], None]:
        decisions = [question_to_decision(key, state, q) for key, q in questions.items()]
        encoded: list[list[int]] = []
        for d in decisions:
            ids = self.readout.encode(d, max_tokens=self.max_tokens)
            if ids is None:
                raise Unsupported(f"question {d.id!r} exceeds the declared context limit of {self.max_tokens} tokens")
            encoded.append(ids)
        probs = self.probabilities(decisions, encoded)
        answers = {d.id: build_answer(questions[d.id], p) for d, p in zip(decisions, probs, strict=True)}
        usage = {"input_tokens": sum(len(e) for e in encoded)}
        return {"model": self.model_id, "answers": answers, "usage": usage}, None

    def runtime(self) -> dict[str, Any]:
        info: dict[str, Any] = {"torch": torch.__version__}
        if torch.cuda.is_available():
            info |= {"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
        return info

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()


def _check_hle_access() -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        HfApi().auth_check(HLE_REPO, repo_type="dataset")
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        msg = f"no access to gated {HLE_REPO}: accept its terms on the Hub and `hf auth login` before rebuilding"
        raise SystemExit(msg) from exc


def _brief(scores: Mapping[str, Any], partial: Any = None) -> dict[str, Any]:
    return {
        "edition": scores.get("edition"),
        "partial": partial,
        "decision_index": scores["decision_index"],
        "raw_index": scores.get("raw_index"),
        "scores": scores["scores"],
        "complete": scores["complete"],
        "completed": scores["completed"],
        "counts": scores.get("counts"),
        "areas": {a["id"]: {"skill": a["skill"], "raw": a["raw"], "coverage": a["coverage"]} for a in scores["areas"]},
        "benchmarks": {
            b.get("dataset", n): {
                "skill": b.get("index_skill"),
                "raw": b.get("index_raw"),
                "coverage": b.get("coverage"),
                "in_index": b.get("in_index"),
            }
            for n, b in scores["benchmarks"].items()
        },
    }


def _score(suite_dir: Path, run_dir: Path, edition: str = EDITION, out: Path | None = None) -> dict[str, Any]:
    from decision_index.pipeline import score_run
    from decision_index.suite.io import Suite

    scores = score_run(Suite(suite_dir, edition), run_dir / "results.jsonl", "decision", out or run_dir)
    return _brief(scores, _partial(suite_dir))


def _partial(suite_dir: Path) -> Any:
    path = Path(suite_dir) / PARTIAL_FILE
    return json.loads(path.read_text()) if path.exists() else None


def _hub_files(work: Path) -> dict[str, Path]:
    """Pinned kit hub files (absent from the pip install) downloaded into ``work/hub/``."""
    paths = {}
    for key, name in HUB_FILES.items():
        path = work / "hub" / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HUB_URL + name, path)
        paths[key] = path
    return paths


def _cmd_rebuild(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_hle:
        return _rebuild_without_hle(args)
    from decision_index.suite.build import rebuild
    from decision_index.suite.download import from_local

    _check_hle_access()
    hub = _hub_files(args.work)
    result = rebuild.main(args.work, edition=EDITION, exclusions=hub["exclusions"])
    out = Path(result["out"])
    imported = from_local(
        args.suite_dir,
        out / "selected-rows.jsonl.gz",
        exclusions=hub["exclusions"],
        manifest=hub["manifest"],
        added=out / "added-rows.jsonl.gz",
        edition=EDITION,
    )
    return {"rebuild": result, "import": imported}


def _rebuild_without_hle(args: argparse.Namespace) -> dict[str, Any]:
    """Every catalog id except HLE; v2 cut by hand (the kit only cuts v2 from a byte-identical v1)."""
    from decision_index.suite.build import adapters_added, rebuild, release_v2
    from decision_index.suite.download import from_local

    hub = _hub_files(args.work)
    only = sorted((set(rebuild.BUILDERS) | set(adapters_added.SPECS)) - {HLE_CATALOG_ID})
    result = rebuild.main(args.work, only=only, edition=EDITION, exclusions=hub["exclusions"])
    suite = args.work / "artifacts" / "benchmark-suite"
    out = suite / "release-v2-noHLE"
    out.mkdir(parents=True, exist_ok=True)
    excluded = set(json.loads(hub["exclusions"].read_text())["rows"])
    rows = out / "selected-rows.jsonl"
    release_v2.cut(suite / "release-v1-rebuilt" / "selected-rows.jsonl", rows, excluded)
    release_v2.gz(rows)
    imported = from_local(
        args.suite_dir,
        out / "selected-rows.jsonl.gz",
        exclusions=hub["exclusions"],
        manifest=hub["manifest"],
        verify=False,
        edition=EDITION,
        added=suite / "release-v2-rebuilt" / "added-rows.jsonl.gz",
    )
    partial = {
        "missing": [f"HLE (catalog {HLE_CATALOG_ID})"],
        "kit_commit": KIT_COMMIT,
        "note": "not an official Decision Index 0.2.1 suite; index excludes HLE",
    }
    (Path(args.suite_dir) / PARTIAL_FILE).write_text(json.dumps(partial, indent=2) + "\n")
    return {"rebuild": result, "import": imported, "partial": partial}


def _cmd_sample(args: argparse.Namespace) -> dict[str, Any]:
    from decision_index.suite.io import Suite
    from decision_index.suite.sample import stratified_sample

    return stratified_sample(Suite(args.suite_dir, EDITION), args.n, args.out, seed=args.seed)


def _cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    from decision_index.runner import run
    from decision_index.suite.io import Suite

    suite = Suite(args.suite_dir, EDITION)
    options: dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "batch_tokens": args.batch_tokens,
    }
    if args.adapter:
        options["adapter"] = str(args.adapter)
        if args.merge:
            options["merge"] = True
    if args.temperatures:
        options["temperatures"] = str(args.temperatures)
    if args.sample:
        rows: Any = args.sample
        keep, corpus = None, None
    elif _partial(args.suite_dir) is not None:
        logger.warning("%s has %s: unverified partial suite, scores are not official", args.suite_dir, PARTIAL_FILE)
        rows, keep, corpus = suite.row_paths, suite.in_edition, None
    else:
        rows, keep, corpus = suite.row_paths, suite.in_edition, suite.verify(strict=True)["sha256"]
    if args.shards > 1:
        return _run_sharded(args, rows, keep, corpus)
    final = run(ENGINE, options, rows, args.out, corpus_sha256=corpus, keep=keep, resume=not args.fresh)
    if args.no_score:
        return {"run": final}
    return {"run": final, "scores": _score(args.suite_dir, args.out)}


def _shard_of(evaluation: Mapping[str, Any], n: int) -> int:
    key = str(evaluation.get("group_id") or evaluation["run_id"])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % n


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _run_sharded(args: argparse.Namespace, rows: Any, keep: Any, corpus: Any) -> dict[str, Any]:
    """Split the selected rows into ``args.shards`` group-whole shards, run one child each, merge and score."""
    from decision_index.runner import iter_rows

    n = args.shards
    out: Path = args.out
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    row_files = [shard_dir / f"rows-{k}.jsonl" for k in range(n)]
    sizes = [0] * n
    handles = [p.open("w", encoding="utf-8") for p in row_files]
    try:
        for row in iter_rows(rows, keep):
            k = _shard_of(row["_evaluation"], n)
            handles[k].write(json.dumps(row, ensure_ascii=False) + "\n")
            sizes[k] += 1
    finally:
        for h in handles:
            h.close()
    logger.info("split %d rows into %d shards: %s", sum(sizes), n, sizes)

    procs = []
    for k in range(n):
        cmd = [sys.executable, "-m", "decision.cli", "suite", "run", "--model", args.model,
               "--sample", str(row_files[k]), "--out", str(shard_dir / str(k)), "--shards", "1", "--no-score",
               "--max-tokens", str(args.max_tokens), "--batch-tokens", str(args.batch_tokens),
               "--suite-dir", str(args.suite_dir)]  # fmt: skip
        if args.adapter:
            cmd += ["--adapter", str(args.adapter)]
        if args.merge:
            cmd.append("--merge")
        if args.temperatures:
            cmd += ["--temperatures", str(args.temperatures)]
        if args.fresh:
            cmd.append("--fresh")
        log = (shard_dir / f"{k}.log").open("a", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))

    results = [shard_dir / str(k) / "results.jsonl" for k in range(n)]
    last = time.monotonic()
    while any(p.poll() is None for p, _ in procs):
        time.sleep(5)
        if time.monotonic() - last >= 60:
            last = time.monotonic()
            done = [_count_lines(r) for r in results]
            logger.info("shard progress %d/%d: %s", sum(done), sum(sizes), [f"{d}/{s}" for d, s in zip(done, sizes)])
    for _, log in procs:
        log.close()
    failed = {k: p.returncode for k, (p, _) in enumerate(procs) if p.returncode != 0}
    if failed:
        raise SystemExit(f"shards failed (shard: exit code) {failed}; see {shard_dir}/<k>.log; re-run to resume")

    merged: dict[str, str] = {}
    for r in results:
        if r.exists():
            # JSONL records end in "\n" only; str.splitlines() also splits on U+2028, U+0085... inside strings.
            for line in r.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    merged[json.loads(line)["run_id"]] = line
    (out / "results.jsonl").write_text("".join(line + "\n" for line in merged.values()), encoding="utf-8")
    env = json.loads((shard_dir / "0" / "environment.json").read_text())
    env.update(shards=n, frozen_corpus_sha256=corpus)
    (out / "environment.json").write_text(json.dumps(env, indent=2, default=str))
    return {"run": {"shards": n, "rows": sizes, "results": len(merged)}, "scores": _score(args.suite_dir, out)}


def main(argv: list[str]) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser(prog="decision suite", description="Decision Index 0.2.1 with our model")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rebuild", help="rebuild the 0.2 suite files (also 0.2.1) from pinned sources and import them")
    p.add_argument("--work", type=Path, default=Path("data/suite-work"))
    p.add_argument("--suite-dir", type=Path, default=Path(DEFAULT_SUITE_DIR))
    p.add_argument("--skip-hle", action="store_true", help="rebuild without HLE (catalog 45); marks the suite PARTIAL")
    p.set_defaults(func=_cmd_rebuild)

    p = sub.add_parser("sample", help="stratified sample of whole case groups")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--suite-dir", type=Path, default=Path(DEFAULT_SUITE_DIR))
    p.set_defaults(func=_cmd_sample)

    p = sub.add_parser("run", help="run our engine over the suite (or a sample) and score it")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument(
        "--merge", action="store_true", help="merge the adapter into the bf16 weights (fit temperatures merged too)"
    )
    p.add_argument("--temperatures", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=Path)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--batch-tokens", type=int, default=DEFAULT_BATCH_TOKENS)
    p.add_argument("--fresh", action="store_true", help="ignore an existing results.jsonl")
    p.add_argument("--suite-dir", type=Path, default=Path(DEFAULT_SUITE_DIR))
    p.add_argument("--shards", type=int, default=1, help="parallel engine processes on the same GPU")
    p.add_argument("--no-score", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("score", help="print the index, areas and per-benchmark skills of a run")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--suite-dir", type=Path, default=Path(DEFAULT_SUITE_DIR))
    p.add_argument(
        "--edition", choices=EDITIONS, default=EDITION, help="scoring edition (0.2 and 0.2.1 share the files)"
    )
    p.add_argument("--out", type=Path, help="write the score files here instead of into --run")
    p.set_defaults(func=lambda a: _score(a.suite_dir, a.run, a.edition, a.out))

    args = ap.parse_args(argv)
    print(json.dumps(args.func(args), indent=2, default=str))
    return 0
