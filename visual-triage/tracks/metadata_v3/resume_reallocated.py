"""Resume the frozen study under two explicit cost amendments, without request retries."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tracks.metadata_v3 import cost_recovery, study
from tracks.metadata_v3.budget_reallocation import (
    AMENDMENT_NAME,
    ReallocatedBudget,
    evaluate_reallocated,
    freeze_reallocation,
    reallocation_summary,
)
from tracks.metadata_v3.optimise import optimise_arm


def run(mode):
    original_load = study.load_study

    def load_reallocated():
        protocol, dataset, baseline, _budget = original_load()
        budget = ReallocatedBudget(
            study.RUNS,
            protocol["global_budget_usd"],
            protocol["arm_budget_usd"],
            protocol["max_task_calls_per_arm"],
            protocol["reserve_per_inflight_usd"],
            amendment_path=study.HERE / cost_recovery.AMENDMENT_NAME,
            reallocation_path=study.HERE / AMENDMENT_NAME,
        )
        budget.approve_failures()
        return protocol, dataset, baseline, budget

    def search(*args, **kwargs):
        return optimise_arm(*args, **kwargs, evaluate_fn=evaluate_reallocated)

    with (
        patch.object(study, "load_study", load_reallocated),
        patch.object(study, "evaluate", evaluate_reallocated),
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
        study.load_study()
        path = freeze_reallocation(study.HERE / cost_recovery.AMENDMENT_NAME, study.RUNS, args.reason)
        print(json.dumps(reallocation_summary(study.RUNS, path)["costs"]))
    else:
        run(args.mode)


if __name__ == "__main__":
    main()
