# ruff: noqa: I001
# Imports follow the explicit legacy run-context bootstrap order.
"""Run the unchanged v2 graders with an explicit expansion output directory."""

import argparse
from types import SimpleNamespace
import corpus
from corpus import DATA
import api_runner as runner

# Request content is unchanged. Higher bounded concurrency amortizes the long
# high-reasoning HTTP latency; bounded retries, accounting and circuit breakers
# remain owned by the existing runner. Limits are maximum reservations, not fees.
POLICY = {
    "luna": {"workers": 64, "budget_usd": "40.00"},
    "deepseek": {"workers": 64, "budget_usd": "35.00"},
    "liquid": {"workers": 16, "budget_usd": "5.00"},
    "jev": {"workers": 16, "budget_usd": "12.00"},
}
ORIGINAL_PINS = runner.code_pins


def code_pins():
    return {
        **ORIGINAL_PINS(),
        str(corpus.Path(__file__).resolve()): corpus.digest(__file__),
        str(corpus.Path(corpus.__file__).resolve()): corpus.digest(corpus.__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--model", choices=list(POLICY), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    runner.code_pins = code_pins
    runner.POLICY[args.model].update(POLICY[args.model])
    options = SimpleNamespace(
        data=DATA,
        model=args.model,
        phase="expansion",
        selection="rehearsal",
        manifest=DATA / "manifest.json",
        references=None,
        case_ids=None,
        limit=args.limit,
    )
    print(runner.freeze(options) if args.command == "freeze" else runner.run(options), flush=True)


if __name__ == "__main__":
    main()
