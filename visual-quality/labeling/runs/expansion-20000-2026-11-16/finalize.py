"""Wait for scoring to release its locks, then export and audit without API calls."""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from corpus import DATA, write_json

PHASES = {
    "luna": "workers16-recovery",
    "deepseek": "workers16-recovery",
    "liquid": "per-image-denials",
    "jev": "per-image-denials",
}


def finalize():
    with ExitStack() as stack:
        for model in PHASES:
            lock = stack.enter_context((DATA / "results/api" / model / ".lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        for model, phase in PHASES.items():
            root = DATA / "results/api" / model
            if (root / "plans/secret-guardrail-disabled.json").exists():
                phase = "secret-guardrail-disabled"
            if (root / "plans/missing-output-postpass.json").exists():
                phase = "missing-output-postpass"
            plan = json.loads((root / "plans" / (phase + ".json")).read_text())
            state = json.loads((root / (phase + "-status.json")).read_text())
            count = len(list((root / "observations").glob("*.json")))
            if (state["status"] != "complete" or count != 10000
                    or state.get("planned_cases", len(plan["cases"])) != len(plan["cases"])
                    or sum(state["outcomes"].values()) != len(plan["cases"])):
                raise RuntimeError(f"{model}: stopped or incomplete ({state['status']}, {count} observations)")
        write_json(DATA / "operations/finalization-status.json", {"status": "exporting_and_auditing", "updated_unix": time.time()})
        for script in ("export.py", "audit.py"):
            subprocess.run([sys.executable, str(Path(__file__).with_name(script))], check=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    path = DATA / "operations/finalization-status.json"
    try:
        while not finalize():
            write_json(path, {"status": "waiting_for_inference_locks", "updated_unix": time.time()})
            if not args.wait:
                return
            time.sleep(30)
        write_json(path, {"status": "audited", "updated_unix": time.time()})
    except Exception as error:
        write_json(path, {"status": "failed", "error": str(error), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
