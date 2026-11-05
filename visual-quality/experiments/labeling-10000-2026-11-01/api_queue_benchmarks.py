"""Admit completed reference packets and queue serial model phases without viewing scores."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import api_runner as runner

WORKTREE = runner.PACKAGE.parent


def validate_packet(path, state):
    freeze = runner.read(path)
    if runner.sha(Path(freeze["packet_path"])) != freeze["packet_sha256"]:
        raise ValueError("Reference packet changed")
    if runner.sha(Path(freeze["references_path"])) != freeze["references_sha256"]:
        raise ValueError("Reference labels changed")
    ids = freeze["case_ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Unique reference case IDs required")
    if any(not re.fullmatch(r"R4\d{3}|R5000", case_id) for case_id in ids):
        raise ValueError("Unexpected new benchmark case ID")
    if set(ids) & set(state["admitted_case_ids"]):
        raise ValueError("Overlapping reference packet admission")
    group = "examples" if "/examples/" in str(path) else "confirmation"
    phase = "ref-" + group + "-" + path.name.removesuffix("-freeze.json")
    if not re.fullmatch(r"[a-z0-9-]+", phase):
        raise ValueError("Unsafe phase name")
    return freeze, ids, phase


def enqueue(path, state, data):
    freeze, ids, phase = validate_packet(path, state)
    selection = data / "api-admission" / (phase + "-ids.json")
    runner.atomic(selection, ids)
    runner.POLICY["jev"]["budget_usd"] = "40.00"
    runner.POLICY["deepseek"]["budget_usd"] = "40.00"
    runner.POLICY["liquid"]["workers"] = 8
    plans = {}
    for model in runner.MODELS:
        plan_path = data / "results/api" / model / "plans" / (phase + ".json")
        if plan_path.exists():
            plan = runner.read(plan_path)
            runner.validate_plan(plan)
            if {c["case_id"] for c in plan["cases"]} != set(ids):
                raise ValueError("Existing packet plan differs")
            plans[model] = {"plan_path": str(plan_path), "plan_id": plan["plan_id"], "cases": len(ids)}
        else:
            plans[model] = runner.freeze(
                SimpleNamespace(
                    data=data,
                    model=model,
                    phase=phase,
                    manifest=data / "manifest.json",
                    selection="benchmark",
                    references=path,
                    case_ids=selection,
                )
            )
    record = {
        "phase": phase,
        "freeze_path": str(path),
        "freeze_sha256": runner.sha(path),
        "frozen_at": freeze["frozen_at"],
        "admitted_at": runner.now(),
        "case_count": len(ids),
        "case_ids": ids,
        "plans": plans,
        "after_phase": dict(state["last_phase"]),
        "units": {},
    }
    # Persist the admission before launching any billable work; resumption is idempotent.
    runner.atomic(data / "api-admission" / (phase + ".json"), record)
    for model in runner.MODELS:
        unit = f"modelgym-10000-{model}-{phase}"
        check = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=LoadState", "--value"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.stdout.strip() == "not-found":
            command = (
                "source ~/.config/zsh/.zsh-private >/dev/null 2>&1; "
                "exec /home/ahstn/git/model-gym/visual-quality/.venv/bin/python "
                + str(runner.HERE / "api_after_phase.py")
                + " --model "
                + model
                + " --after-phase "
                + state["last_phase"][model]
                + " --phase "
                + phase
            )
            subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    "--unit=" + unit,
                    "--property=WorkingDirectory=" + str(WORKTREE),
                    "/usr/bin/zsh",
                    "-c",
                    command,
                ],
                check=True,
            )
        record["units"][model] = unit + ".service"
        runner.atomic(data / "api-admission" / (phase + ".json"), record)
    state["packets"][str(path)] = record
    state["admitted_case_ids"].extend(ids)
    state["last_phase"] = dict.fromkeys(runner.MODELS, phase)
    state["updated_at"] = runner.now()
    runner.atomic(data / "api-benchmark-queue.json", state)
    print(
        json.dumps(
            {
                "phase": phase,
                "admitted_cases": len(ids),
                "total_new_benchmark_cases": len(state["admitted_case_ids"]),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=runner.DATA)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    state_path = args.data / "api-benchmark-queue.json"
    state = (
        runner.read(state_path)
        if state_path.exists()
        else {
            "started_at": runner.now(),
            "last_phase": dict.fromkeys(runner.MODELS, "expanded-rehearsal"),
            "packets": {},
            "admitted_case_ids": [],
            "operational_amendment": str(args.data / "api-operational-amendment-01.json"),
        }
    )
    while True:
        paths = sorted((args.data / "references").glob("**/packet-*-freeze.json"))
        for path in paths:
            if str(path) not in state["packets"]:
                enqueue(path, state, args.data)
            elif runner.sha(path) != state["packets"][str(path)]["freeze_sha256"]:
                raise ValueError("An admitted reference freeze changed")
        if not args.watch or len(state["admitted_case_ids"]) == 1000:
            state["status"] = (
                "all_1000_admitted"
                if len(state["admitted_case_ids"]) == 1000
                else "waiting_for_reference_packets"
            )
            runner.atomic(state_path, state)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
