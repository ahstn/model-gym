"""Explicit driver for the third, before-test timeout recovery amendment."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tracks.metadata_v3 import budget_reallocation, cost_recovery, study
from tracks.metadata_v3.optimise import optimise_arm
from tracks.metadata_v3.timeout_recovery import (
    AMENDMENT_NAME,
    TimeoutBudget,
    evaluate_timeouts,
    freeze_extension,
    timeout_summary,
)


def run(mode):
    original_load = study.load_study

    def load_timeouts():
        protocol, dataset, baseline, _budget = original_load()
        budget = TimeoutBudget(
            study.RUNS,
            protocol["global_budget_usd"],
            protocol["arm_budget_usd"],
            protocol["max_task_calls_per_arm"],
            protocol["reserve_per_inflight_usd"],
            amendment_path=study.HERE / cost_recovery.AMENDMENT_NAME,
            reallocation_path=study.HERE / budget_reallocation.AMENDMENT_NAME,
            extension_path=study.HERE / AMENDMENT_NAME,
        )
        budget.approve_failures()
        return protocol, dataset, baseline, budget

    def search(*args, **kwargs):
        return optimise_arm(*args, **kwargs, evaluate_fn=evaluate_timeouts)

    with (
        patch.object(study, "load_study", load_timeouts),
        patch.object(study, "evaluate", evaluate_timeouts),
        patch.object(study, "optimise_arm", search),
    ):
        study.run(mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "optimise", "final", "controls"))
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.mode == "freeze":
        if args.reason is None:
            parser.error("Freeze requires an explicit --reason")
        protocol, _dataset, _baseline, _budget = study.load_study()
        path = freeze_extension(study.HERE / budget_reallocation.AMENDMENT_NAME, study.RUNS, args.reason)
        budget = TimeoutBudget(
            study.RUNS,
            max_calls=protocol["max_task_calls_per_arm"],
            reserve_usd=protocol["reserve_per_inflight_usd"],
            amendment_path=study.HERE / cost_recovery.AMENDMENT_NAME,
            reallocation_path=study.HERE / budget_reallocation.AMENDMENT_NAME,
            extension_path=path,
        )
        budget.approve_failures()
        print(json.dumps(timeout_summary(study.RUNS, path)["costs"]))
    else:
        run(args.mode)


if __name__ == "__main__":
    main()
