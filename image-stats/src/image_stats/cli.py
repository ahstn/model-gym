"""Analyse images, build a corpus reference, render teacher prompt blocks, and write evidence maps."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import sys

from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import PIL

from image_stats.analysis import Analysis, analyse_file
from image_stats.describe import observations, render
from image_stats.evidence import evidence_maps
from image_stats.loading import DEFAULT_MAX_EDGE
from image_stats.metrics import ALGORITHM_VERSION, METRICS
from image_stats.reference import Reference

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("analyse", help="Write one JSONL record per image")
    run.add_argument("inputs", nargs="*", type=Path, help="Image files or directories (searched recursively)")
    run.add_argument("--manifest", type=Path, help="JSONL manifest, e.g. visual-quality views.jsonl")
    run.add_argument("--root", type=Path, help="Root for manifest image paths (default: manifest directory)")
    run.add_argument("--path-key", default="image_path")
    run.add_argument("--id-key", default="view_id")
    run.add_argument("--source-key", help="Manifest key with the original file, for metadata and blockiness")
    run.add_argument("--source-root", type=Path, help="Root for --source-key paths (default: --root)")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE)
    run.add_argument("--workers", type=int, default=1)

    calibrate = commands.add_parser("calibrate", help="Build percentile references from analysed records")
    calibrate.add_argument("stats", type=Path)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--name", required=True, help="Short corpus name shown in prompts")

    describe = commands.add_parser("describe", help="Render teacher prompt blocks")
    describe.add_argument("stats", type=Path)
    describe.add_argument("--reference", type=Path)
    describe.add_argument("--output", type=Path, help="JSONL output; prints text when omitted")

    evidence = commands.add_parser("evidence", help="Write evidence PNGs for one image")
    evidence.add_argument("image", type=Path)
    evidence.add_argument("--output-dir", type=Path, required=True)
    evidence.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE)

    commands.add_parser("metrics", help="List metric definitions")

    args = parser.parse_args(argv)
    if hasattr(args, "max_edge") and not 64 <= args.max_edge <= 8192:
        parser.error("--max-edge must be from 64 to 8192")
    try:
        return _run(args, parser)
    except (ValueError, OSError) as error:  # OSError covers missing and undecodable files.
        parser.exit(2, f"{parser.prog}: error: {error}\n")


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.command == "analyse":
        return _analyse(args, parser)
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "describe":
        return _describe(args)
    if args.command == "evidence":
        try:
            maps = evidence_maps(args.image, max_edge=args.max_edge)
        except PIL.UnidentifiedImageError as error:
            raise ValueError(f"{args.image} is not a readable image") from error
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, image in maps.items():
            image.save(args.output_dir / f"{args.image.stem}.{name}.png")
        print(json.dumps({"written": sorted(maps), "output_dir": str(args.output_dir)}))
        return 0
    for metric in METRICS:
        print(f"{metric.name:22} {metric.group:12} {metric.unit:12} {metric.description}")
    return 0


def _analyse(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if bool(args.manifest) == bool(args.inputs):
        parser.error("analyse takes either image inputs or --manifest")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    jobs = _manifest_jobs(args) if args.manifest else _input_jobs(args.inputs)
    if not jobs:
        parser.error("No images found")
    duplicates = sorted(identity for identity, count in Counter(job[0] for job in jobs).items() if count > 1)
    if duplicates:
        parser.error(f"Duplicate image IDs: {', '.join(duplicates[:5])}")
    tasks = [(*job, args.max_edge) for job in jobs]
    counts, flags = Counter(), Counter()
    with ExitStack() as stack:
        if args.workers > 1:
            # Forking after OpenCV has started native threads can inherit locked
            # thread-pool state. Fresh interpreters also make library use safe.
            pool = stack.enter_context(
                ProcessPoolExecutor(
                    args.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=cv2.setNumThreads,
                    initargs=(1,),
                )
            )
            records = pool.map(_record, tasks, chunksize=8)
        else:
            records = map(_record, tasks)
        # Stream records so memory stays flat; the rename publishes only a complete file.
        with _staged(args.output) as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                analysis = record.get("analysis")
                counts.update(images=1, errors=analysis is None, gated=bool(analysis and analysis["gate"]))
                flags.update(analysis["flags"] if analysis else ())
    summary = {
        "version": ALGORITHM_VERSION,
        "max_edge": args.max_edge,
        "images": counts["images"],
        "errors": counts["errors"],
        "gated": counts["gated"],
        "flags": dict(sorted(flags.items())),
        "output": str(args.output),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
        },
    }
    args.output.with_suffix(".run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(summary["errors"] > 0)


def _input_jobs(inputs: list[Path]) -> list[tuple[str, Path, None]]:
    jobs = []
    for item in inputs:
        if item.is_dir():
            # With several directories, prefix each ID with its directory name to keep IDs distinct.
            prefix = f"{item.resolve().name}/" if len(inputs) > 1 else ""
            files = sorted(path for path in item.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
            jobs += [(prefix + path.relative_to(item).as_posix(), path, None) for path in files]
        else:
            jobs.append((item.as_posix(), item, None))
    return jobs


def _manifest_jobs(args: argparse.Namespace) -> list[tuple[str, Path, Path | None]]:
    root = args.root or args.manifest.parent
    source_root = args.source_root or root
    jobs = []
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source = source_root / row[args.source_key] if args.source_key and row.get(args.source_key) else None
        jobs.append((str(row[args.id_key]), root / row[args.path_key], source))
    return jobs


def _record(task: tuple[str, Path, Path | None, int]) -> dict:
    identity, path, source, max_edge = task
    try:
        return {"id": identity, "path": str(path), **analyse_file(path, max_edge=max_edge, source_path=source)}
    except Exception as error:  # noqa: BLE001 - one bad file (e.g. corrupt EXIF) must not end a corpus run
        return {"id": identity, "path": str(path), "error": f"{type(error).__name__}: {error}"}


def _calibrate(args: argparse.Namespace) -> int:
    analysed = [record["analysis"] for record in _read_jsonl(args.stats) if "analysis" in record]
    settings = {(item["version"], item["max_edge"]) for item in analysed}
    if len(settings) != 1:
        raise ValueError(f"Expected one version and max_edge in {args.stats}; found {sorted(settings)}")
    ((version, max_edge),) = settings
    if version != ALGORITHM_VERSION:
        raise ValueError(f"Records use {version}; this package computes {ALGORITHM_VERSION}. Re-run analyse.")
    reference = Reference.build((item["metrics"] for item in analysed), name=args.name, max_edge=max_edge)
    reference.save(args.output)
    print(json.dumps({"reference": str(args.output), "count": reference.count, "version": version}))
    return 0


def _describe(args: argparse.Namespace) -> int:
    reference = Reference.load(args.reference) if args.reference else None
    rows = []
    for record in _read_jsonl(args.stats):
        if "analysis" not in record:
            continue
        analysis = Analysis.from_dict(record["analysis"])
        rows.append(
            {
                "id": record["id"],
                "gate": analysis.gate,
                "prompt_block": render(analysis, reference),
                "observations": [asdict(item) for item in observations(analysis, reference)],
            }
        )
    if args.output:
        _write_jsonl(args.output, rows)
    else:
        print("\n\n".join(f"## {row['id']}\n{row['prompt_block']}" for row in rows))
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with _staged(path) as handle:
        handle.writelines(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)


@contextmanager
def _staged(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with staging.open("w") as handle:
            yield handle
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
