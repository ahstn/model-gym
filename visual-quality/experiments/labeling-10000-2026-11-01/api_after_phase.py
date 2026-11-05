"""Wait for one completed route phase, then run the next without doubling concurrency."""

import argparse
import json
import time
from pathlib import Path

import api_runner as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=runner.MODELS)
    parser.add_argument("--after-phase", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--data", type=Path, default=runner.DATA)
    parser.add_argument("--maximum-wait-seconds", type=float, default=21600)
    args = parser.parse_args()
    root = args.data / "results/api" / args.model
    state = root / (args.after_phase + "-status.json")
    start = time.monotonic()
    while True:
        if time.monotonic() - start > args.maximum_wait_seconds:
            raise RuntimeError("Previous phase did not complete within six hours")
        previous = runner.read(state) if state.exists() else {}
        if previous.get("status") == "stopped":
            raise RuntimeError("Previous phase stopped; review its status before continuing")
        if previous.get("status") == "complete":
            args.limit = None
            try:
                print(json.dumps(runner.run(args)), flush=True)
                return
            except BlockingIOError:
                pass
        time.sleep(10)


if __name__ == "__main__":
    main()
