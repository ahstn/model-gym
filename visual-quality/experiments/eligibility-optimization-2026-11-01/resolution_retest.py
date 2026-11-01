"""Separately test an aspect-preserving enlarged copy of one rejected poster."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path

import runner
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = runner.DATA


def main():
    runner.freeze_check()
    manifest = runner.read(HERE / "reference-manifest.json")
    case = next(c for c in manifest["cases"] if c["case_id"] == "W0002")
    assert runner.file_sha(case["image_path"]) == case["view_sha256"]
    original = {}
    for arm in ("heldout-baseline", "heldout-selected"):
        p = DATA / "attempts" / arm / "jev" / "W0002" / "01" / "result.json"
        r = runner.read(p)
        assert r["http_status"] == 400 and "Invalid image dimensions." in str(r.get("body"))
        original[arm] = {"path": str(p), "sha256": runner.file_sha(p), "http_status": 400}
    with Image.open(case["image_path"]) as im:
        assert min(im.size) == 236
        scale = 280 / min(im.size)
        newsize = tuple(math.ceil(v * scale) for v in im.size)
        im = im.convert("RGB").resize(newsize, Image.Resampling.LANCZOS)
        b = io.BytesIO()
        im.save(b, format="PNG", optimize=True)
        raw = b.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    target = DATA / "views" / f"{sha}.png"
    target.write_bytes(raw)
    new = {
        **case,
        "case_id": "U0002",
        "partition": "resolution_retest",
        "image_path": str(target),
        "view_sha256": sha,
        "width": newsize[0],
        "height": newsize[1],
        "delivered_bytes": len(raw),
        "group": "poster_rescaled",
        "evidence": "Same flat poster as W0002, uniformly enlarged after Jev rejected the original dimensions.",
    }
    freeze = {
        "created_at": runner.now(),
        "kind": "post-hoc resolution diagnostic; not a replacement for the original held-out result",
        "source_case_id": "W0002",
        "source_view_sha256": case["view_sha256"],
        "original_size": [case["width"], case["height"]],
        "new_size": list(newsize),
        "interpolation": "Lanczos; no crop or padding",
        "original_failed_receipts": original,
        "selected_prompts_sha256": runner.file_sha(HERE / "winners.json"),
        "case": new,
    }
    path = HERE / "resolution-retest-freeze.json"
    if path.exists():
        existing = runner.read(path)
        if existing["case"]["view_sha256"] != sha or existing["original_failed_receipts"] != original:
            raise ValueError("Resolution diagnostic changed")
    else:
        path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n")
    if len(raw) > 2650000:
        raise ValueError("Rescaled view exceeds delivery limit")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY must be set")
    winner = runner.read(HERE / "winners.json")
    wfreeze = runner.read(HERE / "winners-freeze.json")
    assert runner.file_sha(HERE / "winners.json") == wfreeze["sha256"]
    variants = runner.read(HERE / "rounds-first.json") | runner.read(HERE / "rounds-second.json")
    routes = runner.read(runner.OLD / "api-routes.json")
    prompts = runner.read(runner.OLD / "api-decision-prompts.json")
    runner.init_ledger()
    for arm in ("resolution-baseline", "resolution-selected"):
        questions = {
            m: None if arm == "resolution-baseline" else variants[winner["models"][m]["candidate_round"]]
            for m in runner.ROUTES
        }
        runner.run_round(arm, [(m, new) for m in runner.ROUTES], questions, routes, prompts, key)


if __name__ == "__main__":
    main()
