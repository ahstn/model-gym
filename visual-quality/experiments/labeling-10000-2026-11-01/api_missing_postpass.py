"""One additional identical request for terminal missing/non-answer observations only."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import api_runner as runner


def eligible(row):
    """Never replace a numeric outcome or a valid unscorable assessment."""
    return row.get("status") in {"failed", "retryable_failure", "unknown", "invalid"} and all(
        row.get(key) is None
        for key in ("technical_score", "aesthetic_score", "technical_score_native", "aesthetic_score_native")
    )


def freeze(model, data):
    root = data / "results/api" / model
    target = root / "plans/missing-output-postpass.json"
    if target.exists():
        raise ValueError("Postpass is already frozen")
    manifest = runner.read(data / "manifest.json")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    completed, source_plans = set(), {}
    for path in (root / "plans").glob("*.json"):
        plan = runner.read(path)
        status = root / (plan["phase"] + "-status.json")
        if status.exists() and runner.read(status)["status"] == "complete":
            runner.validate_plan(plan)
            completed.update(case["case_id"] for case in plan["cases"])
            source_plans[plan["plan_id"]] = (path, plan)
    if set(cases) != completed:
        raise ValueError("All 10,000 main-phase cases must be terminal before postpass selection")
    if len(cases) != 10000:
        raise ValueError("Expected the complete 10,000-image corpus")
    rows = {path.stem: runner.read(path) for path in (root / "observations").glob("*.json")}
    if set(rows) != set(cases):
        raise ValueError("Every main-phase observation must exist, including missing results")
    selected = [row for row in rows.values() if eligible(row)]
    template = next(iter(source_plans.values()))[1]
    plan = {key: value for key, value in template.items() if key not in {"plan_id", "cases"}}
    plan.update(
        phase="missing-output-postpass",
        selection="mixed_previously_frozen_missing_outputs",
        created_at=runner.now(),
        cases=[],
        reference_pins={},
        manifest_path=str(data / "manifest.json"),
        manifest_sha256=runner.sha(data / "manifest.json"),
    )
    plan["policy"] = dict(plan["policy"])
    if model in {"jev", "deepseek"}:
        plan["policy"]["budget_usd"] = "40.00"
    plan["policy"]["workers"] = 1
    plan["recovery_policy"] = {
        "authorization": "Root authorized one same-request recovery for truly missing/non-answer outputs after main phases; user approved necessary session spend",
        "additional_attempts_per_case": 1,
        "change_prompt_or_settings": False,
        "retry_valid_numeric_or_unscorable": False,
        "selection_uses_score_values": False,
        "source_observations": {},
    }
    for row in selected:
        source_path, original = source_plans[row["plan_id"]]
        if original["profile"] != template["profile"]:
            raise ValueError("Cannot combine different request profiles")
        plan["reference_pins"].update(original["reference_pins"])
        plan["reference_pins"][str(source_path)] = runner.sha(source_path)
        case_id = row["case_id"]
        if not any(
            case["case_id"] == case_id and case["view_sha256"] == cases[case_id]["view_sha256"]
            for case in original["cases"]
        ):
            raise ValueError("Source plan did not admit this exact image")
        if cases[case_id]["benchmark_partition"] != "rehearsal" and not original["reference_pins"]:
            raise ValueError("Benchmark source plan lacks reference admission")
        plan["cases"].append({key: cases[case_id].get(key) for key in runner.CASE_FIELDS})
        path = root / "observations" / (case_id + ".json")
        snapshot = root / "postpass-source-observations" / (case_id + ".json")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and snapshot.read_bytes() != path.read_bytes():
            raise ValueError("Existing source observation snapshot differs")
        snapshot.write_bytes(path.read_bytes())
        plan["reference_pins"][str(snapshot)] = runner.sha(snapshot)
        plan["recovery_policy"]["source_observations"][case_id] = {
            "path": str(snapshot),
            "current_path": str(path),
            "sha256": runner.sha(path),
            "source_plan_id": row["plan_id"],
            "request_sha256": row["request_sha256"],
            "prior_attempt_count": row["attempt_count"],
        }
    plan["cases"].sort(key=lambda case: case["case_id"])
    plan["code_pins"] = {**runner.code_pins(), str(Path(__file__)): runner.sha(Path(__file__))}
    plan["plan_id"] = runner.contract.digest(plan)
    runner.atomic(target, plan)
    return {"plan_path": str(target), "cases": len(selected), "plan_id": plan["plan_id"]}


def run(model, data):
    root = data / "results/api" / model
    plan = runner.read(root / "plans/missing-output-postpass.json")
    runner.validate_plan(plan)
    route = plan["profile"]["routes"]["routes"][model]
    secret = os.environ[route["key_env"]]
    state = {
        "model": model,
        "phase": plan["phase"],
        "plan_id": plan["plan_id"],
        "started_at": runner.now(),
        "outcomes": {},
        "status": "running",
    }
    state_path = root / (plan["phase"] + "-status.json")
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = runner.Ledger(root, plan["policy"])
        runner.atomic(state_path, state)
        for case in plan["cases"]:
            source = plan["recovery_policy"]["source_observations"][case["case_id"]]
            number = source["prior_attempt_count"] + 1
            attempt = root / "attempts" / case["case_id"] / f"{number:02d}" / "request.json"
            body, request_sha = runner.prepare(plan, case)
            if request_sha != source["request_sha256"]:
                raise ValueError("Postpass request differs from original")
            if not attempt.exists():
                if runner.sha(source["current_path"]) != source["sha256"] or not eligible(
                    runner.read(source["current_path"])
                ):
                    raise ValueError("Missing-output selection changed before request")
                runner.request_once(plan, case, root, number, body, request_sha, secret, ledger)
            elif runner.read(attempt)["plan_id"] != plan["plan_id"]:
                raise ValueError("Unexpected prior postpass request")
            row = runner.observation(plan, case, root, request_sha)
            state["outcomes"][row["status"]] = state["outcomes"].get(row["status"], 0) + 1
            state.update(updated_at=runner.now(), **ledger.summary())
            runner.atomic(state_path, state)
        state.update(status="complete", updated_at=runner.now(), **ledger.summary())
        runner.atomic(state_path, state)
    return state


def main_phases_complete(model, data, expected_ids):
    """Require exact observation coverage and completed admissions before recovery."""
    root = data / "results/api" / model
    observed = {path.stem for path in (root / "observations").glob("*.json")}
    if observed - expected_ids:
        raise ValueError("Unexpected API observations outside the frozen corpus")
    if observed != expected_ids:
        return False
    completed = set()
    for path in (root / "plans").glob("*.json"):
        plan = runner.read(path)
        if plan["phase"] == "missing-output-postpass":
            continue
        status = root / (plan["phase"] + "-status.json")
        if status.exists() and runner.read(status)["status"] == "complete":
            completed.update(case["case_id"] for case in plan["cases"])
    return completed == expected_ids


def watch(data):
    """Run the authorized recovery once per model after all its main phases finish."""
    expected_ids = {case["case_id"] for case in runner.read(data / "manifest.json")["cases"]}
    if len(expected_ids) != 10000:
        raise ValueError("Expected the complete 10,000-image corpus")
    state_path = data / "api-completion-state.json"
    state = runner.read(state_path) if state_path.exists() else {"started_at": runner.now(), "models": {}}
    with (data / ".api-completion-watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            for model in runner.MODELS:
                if state["models"].get(model, {}).get("status") == "complete":
                    continue
                if not main_phases_complete(model, data, expected_ids):
                    continue
                target = data / "results/api" / model / "plans/missing-output-postpass.json"
                if not target.exists():
                    freeze(model, data)
                try:
                    result = run(model, data)
                except BlockingIOError:
                    # A final main-phase process may still be releasing its route lock.
                    continue
                state["models"][model] = result
                state["updated_at"] = runner.now()
                runner.atomic(state_path, state)
                print(json.dumps({"model": model, "postpass": result}), flush=True)
            if len(state["models"]) == len(runner.MODELS):
                state.update(status="complete", updated_at=runner.now())
                runner.atomic(state_path, state)
                return state
            state.update(status="waiting_for_main_phases", updated_at=runner.now())
            runner.atomic(state_path, state)
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "watch"))
    parser.add_argument("--model", choices=runner.MODELS)
    parser.add_argument("--data", type=Path, default=runner.DATA)
    args = parser.parse_args()
    if args.command == "watch":
        print(json.dumps(watch(args.data), indent=2))
        return
    if args.model is None:
        parser.error("--model is required for freeze and run")
    print(
        json.dumps(
            freeze(args.model, args.data) if args.command == "freeze" else run(args.model, args.data),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
