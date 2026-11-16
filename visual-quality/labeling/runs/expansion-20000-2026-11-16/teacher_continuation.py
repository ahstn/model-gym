"""Resume teachers at the historical concurrency after transport/rate failures."""

# ruff: noqa: I001 -- configure the frozen runner only after corpus bootstrap.
import argparse
from types import SimpleNamespace
from corpus import DATA, Path, digest
import api_runner as runner

ORIGINAL_PINS = runner.code_pins


def pins():
    return {**ORIGINAL_PINS(), str(Path(__file__).resolve()): digest(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--model", choices=["luna", "deepseek"], required=True)
    args = parser.parse_args()
    options = SimpleNamespace(
        data=DATA,
        model=args.model,
        phase="workers16-recovery",
        selection="rehearsal",
        manifest=DATA / "manifest.json",
        references=None,
        case_ids=None,
        limit=None,
        worker_limit=16,
    )
    if args.command == "freeze":
        runner.code_pins = pins
        runner.POLICY[args.model].update(
            workers=16,
            budget_usd="120.00",
            max_attempts=5 if args.model == "luna" else 7,
            recovery_policy="Reuse all valid/invalid/content-denied results; at most two additional attempts for previously exhausted transient failures after lowering concurrency; preserve all unknown-charge reserves",
        )
        print(runner.freeze(options), flush=True)
    else:
        print(runner.run(options), flush=True)


if __name__ == "__main__":
    main()
