"""Resume frozen phases in order; stop on any incomplete or blocked phase."""

from __future__ import annotations

import argparse
import json

import api_runner as runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, choices=runner.ROUTES)
    args = parser.parse_args()
    state_path = runner.directory(args.route) / "driver-status.json"
    state = {"route": args.route, "started_at": runner.now(), "status": "running", "phases": {}}
    runner.write(state_path, state)
    try:
        for phase in runner.PHASES:
            state.update(active_phase=phase, updated_at=runner.now())
            runner.write(state_path, state)
            plan = runner.read(runner.directory(args.route, phase) / "plan.json")
            summary = runner.execute(plan)
            state["phases"][phase] = summary
            runner.write(state_path, state)
            if summary["status"] != "complete":
                raise RuntimeError(f"{phase} is {summary['status']}; inspect journals before continuing")
            runner.validate_plan(plan, check_payloads=True)
            verified = runner.save(plan, runner.reconstruct(plan))
            if verified != summary:
                raise RuntimeError("Post-phase verification differs from execution summary")
            state["phases"][phase]["verified_at"] = runner.now()
            runner.write(state_path, state)
        state.update(status="complete", finished_at=runner.now())
        runner.write(state_path, state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status="stopped_requires_audit", finished_at=runner.now(), error_type=type(error).__name__,
                     error=str(error))
        runner.write(state_path, state)
        raise


if __name__ == "__main__":
    main()
