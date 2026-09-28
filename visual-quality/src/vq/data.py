"""Locked manifests built from the released Q-Align label JSONs.

Layout under the data root (on the pod a symlink to /dev/shm/vq):

    labels/<file>.json      Q-Align label JSONs (playground/data/{training_sft,test_jsons})
    img/<rel>               images, extracted from q-future/q-align-datasets tars
    video/<rel>             videos (transient; removed after frame extraction)
    frames/<key>/f{i}.jpg   8 frames per video (Q-ReAlign recipe: uniform, long side 448, JPEG q90)
    manifests/<set>.jsonl   one row per item (see `Row`)

Row fields: id, set, task, media (paths relative to the data root), mos (float or null),
higher_better, prompt, answer (train only).
"""

from __future__ import annotations

import hashlib
import json
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vq.levels import EVAL_PROMPTS

LABEL_BASE = "https://raw.githubusercontent.com/Q-Future/Q-Align/main/playground/data"
N_FRAMES = 8


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    task: str
    label_file: str  # path under LABEL_BASE
    split: str  # "train" or "test"
    media_dir: str  # prefix under img/ or video/ joined with the JSON path
    higher_better: bool = True

    @property
    def is_video(self) -> bool:
        return self.task == "vqa"


SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        # ONE-ALIGN training mix (Q-Align release; Q-ReAlign public config `mix`).
        Source("koniq", "iqa", "training_sft/train_koniq.json", "train", ""),
        Source("spaq", "iqa", "training_sft/train_spaq.json", "train", ""),
        Source("kadid", "iqa", "training_sft/train_kadid.json", "train", ""),
        Source("ava", "iaa", "training_sft/train_ava.json", "train", ""),
        Source("lsvq", "vqa", "training_sft/train_lsvq.json", "train", ""),
        # T1: same sets as the Q-ReAlign README table.
        Source("test_koniq", "iqa", "test_jsons/test_koniq.json", "test", ""),
        Source("test_spaq", "iqa", "test_jsons/test_spaq.json", "test", ""),
        Source("test_kadid", "iqa", "test_jsons/test_kadid.json", "test", ""),
        Source("agiqa3k", "iqa", "test_jsons/agi.json", "test", ""),
        Source("live", "iqa", "test_jsons/live.json", "test", "", higher_better=False),
        Source("test_ava", "iaa", "test_jsons/test_ava.json", "test", ""),
        Source("test_lsvq", "vqa", "test_jsons/test_lsvq.json", "test", ""),
        # T2: cross-dataset.
        Source("test_lsvq_1080p", "vqa", "test_jsons/test_lsvq_1080p.json", "test", ""),
        Source("livec", "iqa", "test_jsons/livec.json", "test", ""),
        Source("csiq", "iqa", "test_jsons/csiq.json", "test", "", higher_better=False),
        Source("konvid", "vqa", "test_jsons/konvid.json", "test", "konvid"),
        Source("maxwell_test", "vqa", "test_jsons/maxwell_test.json", "test", "maxwell"),
    )
}

TRAIN_MIX = ("koniq", "spaq", "kadid", "ava", "lsvq")
T1 = ("test_koniq", "test_spaq", "test_kadid", "agiqa3k", "live", "test_ava", "test_lsvq")
T2 = ("test_lsvq_1080p", "livec", "csiq", "konvid", "maxwell_test")

_ARROW = re.compile(r"->\s*([-+0-9.eE]+)\s*$")


def frame_key(video_rel: str) -> str:
    """Collision-free frame directory name for a video path (LSVQ batches reuse basenames)."""
    stem = video_rel.rsplit(".", 1)[0]
    return stem.replace("/", "__")


def frame_paths(video_rel: str, n: int = N_FRAMES) -> list[str]:
    key = frame_key(video_rel)
    return [f"frames/{key}/f{i}.jpg" for i in range(n)]


def _media_rel(src: Source, raw: dict[str, Any]) -> str:
    rel = raw.get("image") or raw.get("img_path")
    if not rel:
        raise KeyError(f"{src.name}: row has no image/img_path: {raw}")
    return f"{src.media_dir}/{rel}" if src.media_dir else rel


def _mos(raw: dict[str, Any]) -> float | None:
    if raw.get("gt_score") is not None:
        return float(raw["gt_score"])
    m = _ARROW.search(str(raw.get("id", "")))
    return float(m.group(1)) if m else None


def _conversation(raw: dict[str, Any]) -> tuple[str, str]:
    conv = raw["conversations"]
    human = conv[0]["value"].replace("<|image|>", "").strip()
    return human, conv[1]["value"].strip()


def rows_for(src: Source, raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for raw in raw_rows:
        rel = _media_rel(src, raw)
        media = frame_paths(rel) if src.is_video else [f"img/{rel}"]
        row: dict[str, Any] = {
            "id": rel,
            "set": src.name,
            "task": src.task,
            "media": media,
            "mos": _mos(raw),
            "higher_better": src.higher_better,
        }
        if src.split == "train":
            # Verbatim, as Q-ReAlign passes it through. 21 KADID answers use non-level words
            # ("average", "pristine", ...): kept, to match the released recipe.
            prompt, answer = _conversation(raw)
            row["prompt"], row["answer"] = prompt, answer
        else:
            row["prompt"] = EVAL_PROMPTS[src.task]
            if row["mos"] is None:
                raise ValueError(f"{src.name}: test row without a score: {raw}")
        out.append(row)
    return out


def write_manifest(rows: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build(root: Path, names: list[str] | None = None) -> dict[str, Any]:
    """Build manifests from labels/ and report counts, missing media, and hashes."""
    report: dict[str, Any] = {}
    for name in names or list(SOURCES):
        src = SOURCES[name]
        raw = json.loads((root / "labels" / Path(src.label_file).name).read_text())
        rows = rows_for(src, raw)
        present = [all((root / m).exists() for m in r["media"]) for r in rows]
        kept = [r for r, ok in zip(rows, present, strict=True) if ok]
        missing = [r["id"] for r, ok in zip(rows, present, strict=True) if not ok]
        digest = write_manifest(kept, root / "manifests" / f"{name}.jsonl")
        report[name] = {
            "labels": len(raw),
            "kept": len(kept),
            "missing": len(missing),
            "missing_examples": missing[:5],
            "sha256": digest,
        }
    return report
