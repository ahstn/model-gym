"""Local pilot acquisition, assembly, and verification commands."""

import argparse
import json
from pathlib import Path

from visual_quality.assemble import SOURCES, acquire, assemble, load_config
from visual_quality.validation import verify_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["acquire", "assemble", "build", "verify"])
    parser.add_argument("--config", type=Path, default=Path("configs/pilot-10k.json"))
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--source", action="append", choices=SOURCES)
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command in {"acquire", "build"}:
        print(json.dumps(acquire(args.root, config, args.source), indent=2))
    if args.command in {"assemble", "build"}:
        print(json.dumps(assemble(args.root, config, args.exclude), indent=2))
    if args.command == "verify":
        report = verify_dataset(args.root.resolve(), config)
        print(json.dumps(report, indent=2))
        return bool(report["errors"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
