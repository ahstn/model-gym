"""Acquire images, prepare fixed views, and run a small quality-label pilot."""

import argparse
import json
from pathlib import Path

from visual_quality.assemble import SOURCES, acquire, assemble, load_config
from visual_quality.validation import verify_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["acquire", "assemble", "build", "verify", "prepare-views", "label", "verify-labels"],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pilot-10k.json"))
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--source", action="append", choices=SOURCES)
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    parser.add_argument("--dataset", type=Path, default=Path("data/pilot-10k-v1"))
    parser.add_argument("--views", type=Path, default=Path("data/label-pilot-12/views"))
    parser.add_argument("--output", type=Path, default=Path("data/label-pilot-12/deepseek"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--max-edge", type=int, default=1536)
    parser.add_argument("--prompt-dir", type=Path, default=Path("prompts/visual-rating-v1"))
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-cost-usd", default="1.00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command in {"prepare-views", "label", "verify-labels"}:
        return _labels(args, parser)
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


def _labels(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from visual_quality.labelling import label_pilot, verify_labels
    from visual_quality.views import prepare_views

    try:
        if args.command == "prepare-views":
            report = prepare_views(
                args.dataset, args.views, limit=args.limit, seed=args.seed, max_edge=args.max_edge
            )
        elif args.command == "verify-labels":
            report = verify_labels(args.views, args.output)
        else:
            report = label_pilot(
                args.views,
                args.output,
                args.prompt_dir,
                model=args.model,
                max_tokens=args.max_tokens,
                max_cost_usd=args.max_cost_usd,
                dry_run=args.dry_run,
            )
    except (ValueError, TypeError, OSError) as error:
        parser.exit(2, f"Pilot error: {error}\n")
    print(json.dumps(report, indent=2))
    return int(report.get("status") in {"blocked", "budget_stopped"})


if __name__ == "__main__":
    raise SystemExit(main())
