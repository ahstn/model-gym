"""Inspect image provenance, pixel statistics and Smaug screening results."""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from image_stats.authenticity import CANDIDATE_ID, inspect_image
from image_stats.authenticity.smaug import DEFAULT_THRESHOLD, Smaug


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    model = Smaug(device=args.device, cache_dir=args.cache_dir, offline=args.offline)
    records = [inspect_image(path, model, threshold=args.threshold) for path in args.images]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"candidate": CANDIDATE_ID, "images": records}, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"images": len(records), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
