"""vq command line: labels, build, frames, eval, train, info."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

from pathlib import Path

from vq.data import LABEL_BASE, SOURCES, T1, T2, TRAIN_MIX, build, read_manifest

GROUPS = {"t1": T1, "t2": T2, "train": TRAIN_MIX, "all": tuple(SOURCES)}


def _sets(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        for s in GROUPS.get(n.lower(), (n,)):
            if s not in SOURCES:
                raise SystemExit(f"unknown set {s!r}; known: {sorted(SOURCES)} or groups {sorted(GROUPS)}")
            if s not in out:
                out.append(s)
    return out


def cmd_labels(a: argparse.Namespace) -> None:
    dest = a.root / "labels"
    dest.mkdir(parents=True, exist_ok=True)
    for src in SOURCES.values():
        path = dest / Path(src.label_file).name
        if not path.exists():
            urllib.request.urlretrieve(f"{LABEL_BASE}/{src.label_file}", path)
        print(src.name, path.stat().st_size)


def cmd_build(a: argparse.Namespace) -> None:
    report = build(a.root, _sets(a.sets))
    (a.root / "manifests").mkdir(exist_ok=True)
    (a.root / "manifests" / "report.json").write_text(json.dumps(report, indent=2))
    for k, v in report.items():
        print(f"{k:18s} labels={v['labels']:7d} kept={v['kept']:7d} missing={v['missing']:6d}")


def cmd_frames(a: argparse.Namespace) -> None:
    from vq.data import frame_paths
    from vq.frames import extract_many

    rels: list[str] = []
    for name in _sets(a.sets):
        src = SOURCES[name]
        raw = json.loads((a.root / "labels" / Path(src.label_file).name).read_text())
        for r in raw:
            rel = r.get("image") or r.get("img_path")
            rels.append(f"{src.media_dir}/{rel}" if src.media_dir else rel)
    failures = extract_many(a.root, rels, a.workers, delete=a.delete)
    done = sum((a.root / frame_paths(r)[-1]).exists() for r in rels)
    print(json.dumps({"videos": len(rels), "with_frames": done, "failures": len(failures)}))
    for k, v in list(failures.items())[:20]:
        print("FAIL", k, v, file=sys.stderr)


def _scorer_cfg(a: argparse.Namespace):
    from vq.scorer import ScorerConfig

    return ScorerConfig(
        model=a.model,
        max_pixels=a.max_pixels,
        attn=a.attn,
        use_kernels=a.use_kernels,
        batch_size=a.batch_size,
        last_logits_only=not a.full_logits,
        workers=a.workers,
    )


def cmd_eval(a: argparse.Namespace) -> None:
    from vq.evaluate import run

    run(_scorer_cfg(a), a.root, _sets(a.sets), a.out, limit=a.limit, boot=a.boot)


def cmd_bench(a: argparse.Namespace) -> None:
    from vq.bench import BATCH_SIZES, run

    run(_scorer_cfg(a), a.root, a.out, batch_sizes=tuple(a.batch_sizes or BATCH_SIZES))


def _add_scorer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-pixels", type=int, default=None)
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--use-kernels", action="store_true", help="load HF hub kernels (causal-conv1d, fla)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--full-logits", action="store_true", help="compute logits for every position (Q-ReAlign scorer)")
    p.add_argument("--workers", type=int, default=8)


def cmd_train(a: argparse.Namespace) -> None:
    from vq.train import run

    run(a.config)


def cmd_report(a: argparse.Namespace) -> None:
    from vq.report import bench_table, table

    if a.bench:
        print(bench_table(a.runs, usd_per_hour=a.usd_per_hour))
    else:
        print(table(a.runs, _sets(a.sets), ref=a.ref, boot=a.boot))


def cmd_info(a: argparse.Namespace) -> None:
    from vq.evaluate import environment

    print(json.dumps(environment(), indent=2))
    for name in _sets(a.sets):
        p = a.root / "manifests" / f"{name}.jsonl"
        print(name, len(read_manifest(p)) if p.exists() else "no manifest")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vq")
    p.add_argument("--root", type=Path, default=Path("data"))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("labels", help="download the Q-Align label JSONs").set_defaults(fn=cmd_labels)

    b = sub.add_parser("build", help="write locked manifests")
    b.add_argument("sets", nargs="*", default=["all"])
    b.set_defaults(fn=cmd_build)

    f = sub.add_parser("frames", help="extract 8 frames per video")
    f.add_argument("sets", nargs="+")
    f.add_argument("--workers", type=int, default=12)
    f.add_argument("--delete", action="store_true", help="remove each video after its frames are written")
    f.set_defaults(fn=cmd_frames)

    e = sub.add_parser("eval", help="score sets and write metrics")
    _add_scorer_args(e)
    e.add_argument("sets", nargs="+")
    e.add_argument("--limit", type=int, default=0, help="evenly spaced subset per set; 0 = all")
    e.add_argument("--boot", type=int, default=1000)
    e.set_defaults(fn=cmd_eval)

    bn = sub.add_parser("bench", help="speed harness: batch-1 latency and throughput per batch size")
    _add_scorer_args(bn)
    bn.add_argument("--batch-sizes", type=int, nargs="*")
    bn.set_defaults(fn=cmd_bench)

    t = sub.add_parser("train", help="supervised fine-tuning from a YAML config")
    t.add_argument("config", type=Path)
    t.set_defaults(fn=cmd_train)

    rp = sub.add_parser("report", help="markdown table of eval runs; optional paired SRCC gaps to --ref")
    rp.add_argument("runs", type=Path, nargs="+")
    rp.add_argument("--sets", nargs="+", default=["t1", "t2"])
    rp.add_argument("--ref", type=Path)
    rp.add_argument("--boot", type=int, default=1000)
    rp.add_argument("--bench", action="store_true", help="speed-harness table from bench.json files")
    rp.add_argument("--usd-per-hour", type=float, default=2.09, help="GPU price for the cost column")
    rp.set_defaults(fn=cmd_report)

    i = sub.add_parser("info", help="environment and manifest counts")
    i.add_argument("sets", nargs="*", default=["all"])
    i.set_defaults(fn=cmd_info)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
