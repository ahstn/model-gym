"""Explicit shared-pool driver with optional secret-free transport diagnostics."""

import argparse
import json
import sys
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_protocol import digest
from tracks.metadata_v3 import budget_reallocation, cost_recovery, runtime, study, timeout_recovery
from tracks.metadata_v3.optimise import optimise_arm
from tracks.metadata_v3.shared_allowance_pool import (
    AMENDMENT_NAME,
    SharedPoolBudget,
    evaluate_shared_pool,
    freeze_shared_pool,
    shared_pool_summary,
)


def exception_classes(exc):
    queue, seen, result = [exc], set(), []
    while queue and len(result) < 20:
        current = queue.pop(0)
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        errno = getattr(current, "errno", None)
        result.append({"class": type(current).__name__, "errno": errno if type(errno) is int else None})
        queue.extend(
            [
                current.__cause__,
                current.__context__,
                getattr(current, "reason", None),
                getattr(current, "_reason", None),
            ]
        )
        queue.extend(value for value in current.args if isinstance(value, BaseException))
    return result


@contextmanager
def diagnostic_transport(path, runs_root):
    """Forward the original HTTP call unchanged; retain no exception strings or secrets."""
    if path is None:
        yield
        return
    path = Path(path).resolve()
    if path.is_relative_to(Path(runs_root).resolve()) or path.suffix != ".jsonl":
        raise ValueError("Transport diagnostics must use a JSONL path outside raw runs")
    original_post, lock = requests.post, threading.Lock()

    def observed(endpoint, *args, **kwargs):
        try:
            return original_post(endpoint, *args, **kwargs)
        except requests.RequestException as exc:
            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "endpoint_path": urlsplit(endpoint).path,
                "request_sha256": digest({"endpoint": endpoint, "payload": kwargs.get("json")}),
                "exception_chain": exception_classes(exc),
            }
            try:
                with lock:
                    runtime.append_record(path, record)
            except OSError:
                pass  # Diagnostics cannot replace the original transport exception.
            raise

    with patch.object(requests, "post", observed):
        yield


def run(mode, diagnostics=None):
    original_load = study.load_study

    def load_shared():
        protocol, dataset, baseline, _budget = original_load()
        budget = SharedPoolBudget(
            study.RUNS,
            protocol["global_budget_usd"],
            protocol["arm_budget_usd"],
            protocol["max_task_calls_per_arm"],
            protocol["reserve_per_inflight_usd"],
            amendment_path=study.HERE / cost_recovery.AMENDMENT_NAME,
            reallocation_path=study.HERE / budget_reallocation.AMENDMENT_NAME,
            extension_path=study.HERE / timeout_recovery.AMENDMENT_NAME,
            shared_pool_path=study.HERE / AMENDMENT_NAME,
        )
        budget.approve_failures()
        return protocol, dataset, baseline, budget

    def search(*args, **kwargs):
        return optimise_arm(*args, **kwargs, evaluate_fn=evaluate_shared_pool)

    with (
        diagnostic_transport(diagnostics, study.RUNS),
        patch.object(study, "load_study", load_shared),
        patch.object(study, "evaluate", evaluate_shared_pool),
        patch.object(study, "optimise_arm", search),
    ):
        study.run(mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "optimise", "final", "controls"))
    parser.add_argument("--reason")
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        if args.reason is None:
            parser.error("Freeze requires an explicit --reason")
        study.load_study()
        path = freeze_shared_pool(study.HERE / timeout_recovery.AMENDMENT_NAME, study.RUNS, args.reason)
        print(json.dumps(shared_pool_summary(study.RUNS, path)["costs"]))
    else:
        run(args.mode, args.diagnostics)


if __name__ == "__main__":
    main()
