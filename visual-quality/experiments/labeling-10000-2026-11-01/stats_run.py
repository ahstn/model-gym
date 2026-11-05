"""Compute separate debugging metadata with the local image-stats package."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2

from visual_quality.image_statistics import analyse_case, configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = configuration()
    raw = args.manifest.read_bytes()
    cases = json.loads(raw)["cases"]
    assert len(cases) == len({r["case_id"] for r in cases})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    temporary = args.output.with_suffix(".partial.jsonl")
    with temporary.open("w") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, result in enumerate(
            pool.map(partial(analyse_case, cache_dir=args.cache, config=config), cases), 1
        ):
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            if count % 100 == 0:
                stream.flush()
                print(f"Statistics {count}/{len(cases)}", flush=True)
    temporary.replace(args.output)
    provenance = {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": len(cases),
        "configuration": config,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Completed {len(cases)} records: {args.output}", flush=True)


if __name__ == "__main__":
    main()
