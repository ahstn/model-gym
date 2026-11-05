"""Explicit operational recovery driver; original protocol and runtime stay frozen."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tracks.metadata_v3 import study
from tracks.metadata_v3.cost_recovery import (
    AMENDMENT_NAME,
    RecoveryBudget,
    evaluate_recovered,
    freeze_amendment,
)
from tracks.metadata_v3.optimise import optimise_arm


def run(mode):
    original_load = study.load_study

    def load_recovered():
        protocol, dataset, baseline, _budget = original_load()
        budget = RecoveryBudget(
            study.RUNS,
            protocol["global_budget_usd"],
            protocol["arm_budget_usd"],
            protocol["max_task_calls_per_arm"],
            protocol["reserve_per_inflight_usd"],
            amendment_path=study.HERE / AMENDMENT_NAME,
        )
        budget.approve_failures()
        return protocol, dataset, baseline, budget

    def search(*args, **kwargs):
        return optimise_arm(*args, **kwargs, evaluate_fn=evaluate_recovered)

    # Same-process overrides are explicit, scoped and restored after this run.
    with (
        patch.object(study, "load_study", load_recovered),
        patch.object(study, "evaluate", evaluate_recovered),
        patch.object(study, "optimise_arm", search),
    ):
        study.run(mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "optimise", "final", "controls"))
    parser.add_argument("--pricing-capture", type=Path)
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.mode == "freeze":
        if args.pricing_capture is None or args.reason is None:
            parser.error("Freeze requires --pricing-capture and an explicit --reason")
        study.load_study()  # Verify every original input and execution-code pin first.
        path = freeze_amendment(study.HERE / "protocol.json", study.RUNS, args.pricing_capture, args.reason)
        print(str(path))
        from tracks.metadata_v3.cost_recovery import allowance_summary

        protocol = study.read(study.HERE / "protocol.json")
        budget = RecoveryBudget(
            study.RUNS,
            max_calls=protocol["max_task_calls_per_arm"],
            reserve_usd=protocol["reserve_per_inflight_usd"],
            amendment_path=path,
        )
        budget.approve_failures()
        print(json.dumps(allowance_summary(study.RUNS, path)["costs"]))
    else:
        run(args.mode)


if __name__ == "__main__":
    main()
