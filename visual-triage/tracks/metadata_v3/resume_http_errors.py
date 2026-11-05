"""Scoped fifth-amendment driver reuses the frozen shared-pool execution path."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tracks.metadata_v3 import (
    budget_reallocation,
    cost_recovery,
    resume_shared_pool,
    shared_allowance_pool,
    study,
    timeout_recovery,
)
from tracks.metadata_v3.http_error_recovery import (
    AMENDMENT_NAME,
    HTTPBudget,
    evaluate_http,
    freeze_http_extension,
    http_summary,
)


def make_budget(*args, **kwargs):
    return HTTPBudget(*args, **kwargs, http_extension_path=study.HERE / AMENDMENT_NAME)


def run(mode, diagnostics=None):
    with (
        patch.object(resume_shared_pool, "SharedPoolBudget", make_budget),
        patch.object(resume_shared_pool, "evaluate_shared_pool", evaluate_http),
    ):
        resume_shared_pool.run(mode, diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "optimise", "final", "controls"))
    parser.add_argument("--reason")
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        if args.reason is None:
            parser.error("Freeze requires an explicit --reason")
        protocol, _dataset, _baseline, _budget = study.load_study()
        path = freeze_http_extension(
            study.HERE / shared_allowance_pool.AMENDMENT_NAME, study.RUNS, args.reason
        )
        budget = HTTPBudget(
            study.RUNS,
            max_calls=protocol["max_task_calls_per_arm"],
            reserve_usd=protocol["reserve_per_inflight_usd"],
            amendment_path=study.HERE / cost_recovery.AMENDMENT_NAME,
            reallocation_path=study.HERE / budget_reallocation.AMENDMENT_NAME,
            extension_path=study.HERE / timeout_recovery.AMENDMENT_NAME,
            shared_pool_path=study.HERE / shared_allowance_pool.AMENDMENT_NAME,
            http_extension_path=path,
        )
        budget.approve_failures()
        print(json.dumps(http_summary(study.RUNS, path)["costs"]))
    else:
        run(args.mode, args.diagnostics)


if __name__ == "__main__":
    main()
