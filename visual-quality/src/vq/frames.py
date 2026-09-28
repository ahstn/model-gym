"""Video -> 8 uniform frames, as in Q-ReAlign `sample_frames`: indices round(linspace(0, n-1, 8)),
long side resized to 448 (PIL default resample), saved as JPEG quality 90."""

from __future__ import annotations

import os

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vq.data import N_FRAMES, frame_key

RESIZE_LONG = 448


def _frame_count(container, stream) -> int:
    if stream.frames:
        return int(stream.frames)
    n = sum(1 for p in container.demux(stream) if p.size)
    container.seek(0)
    return n


def extract(video: Path, out_dir: Path, n: int = N_FRAMES, resize_long: int = RESIZE_LONG) -> int:
    """Write f0..f{n-1}.jpg; returns the number of frames in the video."""
    import av

    out_dir.mkdir(parents=True, exist_ok=True)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = _frame_count(container, stream)
        want = [round(i) for i in np.linspace(0, max(total - 1, 0), n).tolist()]
        slots: dict[int, list[int]] = {}
        for slot, idx in enumerate(want):
            slots.setdefault(idx, []).append(slot)
        last = None
        written = 0
        for i, frame in enumerate(container.decode(stream)):
            last = frame
            if i in slots:
                img = _resize(frame.to_image(), resize_long)
                for slot in slots.pop(i):
                    img.save(out_dir / f"f{slot}.jpg", quality=90)
                    written += 1
            if not slots:
                break
        # Metadata can over-count frames; fill the tail with the last decoded frame.
        if slots and last is not None:
            img = _resize(last.to_image(), resize_long)
            for rest in slots.values():
                for slot in rest:
                    img.save(out_dir / f"f{slot}.jpg", quality=90)
                    written += 1
    if written != n:
        raise RuntimeError(f"{video}: wrote {written}/{n} frames")
    return total


def _resize(img, resize_long: int):
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > resize_long:
        s = resize_long / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    return img


def _job(args: tuple[str, str, str]) -> tuple[str, str | None]:
    video, out_dir, rel = args
    try:
        extract(Path(video), Path(out_dir))
        return rel, None
    except Exception as e:  # noqa: BLE001 - report and continue; caller counts failures
        return rel, f"{type(e).__name__}: {e}"


def extract_many(root: Path, rels: list[str], workers: int, *, delete: bool) -> dict[str, str]:
    """Extract frames for video paths under root/video; skip ones already done. Returns failures."""
    jobs = []
    for rel in rels:
        out = root / "frames" / frame_key(rel)
        if (out / f"f{N_FRAMES - 1}.jpg").exists():
            continue
        src = root / "video" / rel
        if src.exists():
            jobs.append((str(src), str(out), rel))
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rel, err in pool.map(_job, jobs, chunksize=4):
            if err:
                failures[rel] = err
            elif delete:
                os.remove(root / "video" / rel)
    return failures
