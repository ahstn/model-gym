"""Lock all graders, validate their current plans, then export and audit locally."""

from __future__ import annotations

import argparse
import fcntl
import re
import time
from contextlib import ExitStack


def selected_phases(overrides, models, default):
    phases = dict.fromkeys(models, default)
    seen = set()
    for value in overrides:
        model, separator, phase = value.partition("=")
        if model not in phases or not separator or not re.fullmatch(r"[a-z0-9_-]+", phase) or model in seen:
            raise ValueError("Expected a unique --phase MODEL=PHASE for a supported grader")
        phases[model] = phase
        seen.add(model)
    return phases


def finalize(data, old, phases):
    import audit
    import export
    from corpus import write_json

    with ExitStack() as stack:
        for model in phases:
            root = data / "results/api" / model
            if not root.is_dir():
                raise ValueError(f"Missing grader output directory: {model}")
            lock = stack.enter_context((root / ".lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        write_json(
            data / "operations/finalization-status.json",
            {
                "status": "exporting_and_auditing",
                "phases": phases,
                "updated_unix": time.time(),
            },
        )
        export.export_run(data, old, phases)
        audit.audit_run(data, old, phases)
    return True


def main():
    from corpus import DATA, OLD, write_json
    from score import MODELS, PHASE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--phase", action="append", default=[], help="Explicit latest phase as MODEL=NAME")
    args = parser.parse_args()
    phases = selected_phases(args.phase, MODELS, PHASE)
    path = DATA / "operations/finalization-status.json"
    try:
        while not finalize(DATA, OLD, phases):
            write_json(
                path, {"status": "waiting_for_inference_locks", "phases": phases, "updated_unix": time.time()}
            )
            if not args.wait:
                return
            time.sleep(30)
        write_json(path, {"status": "audited", "phases": phases, "updated_unix": time.time()})
    except Exception as error:
        write_json(
            path, {"status": "failed", "error": str(error), "phases": phases, "updated_unix": time.time()}
        )
        raise


if __name__ == "__main__":
    main()
