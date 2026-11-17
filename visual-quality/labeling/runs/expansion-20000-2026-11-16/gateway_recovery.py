"""One unchanged retry for HTTP520 failures after the approved guardrail recovery."""

# ruff: noqa: I001 -- establish the original adapter context before imports.
import argparse
from corpus import DATA, Path, digest
import api_runner as runner
import api_missing_postpass as postpass

ORIGINAL_ELIGIBLE = postpass.eligible
ORIGINAL_PINS = runner.code_pins


def eligible(row):
    if not ORIGINAL_ELIGIBLE(row) or not row.get("raw_response_path"):
        return False
    return runner.read(row["raw_response_path"]).get("http_status") == 520


def pins():
    return {**ORIGINAL_PINS(), str(Path(__file__).resolve()): digest(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--model", choices=["liquid", "jev"], required=True)
    args = parser.parse_args()
    root = DATA / "results/api" / args.model
    state = runner.read(root / "secret-guardrail-disabled-status.json")
    if state["status"] != "complete" or sum(state["outcomes"].values()) != 2855:
        raise ValueError("Full guardrail recovery must finish before selecting HTTP520 failures")
    postpass.eligible = eligible
    runner.code_pins = pins
    print(postpass.freeze(args.model, DATA) if args.command == "freeze" else postpass.run(args.model, DATA), flush=True)


if __name__ == "__main__":
    main()
