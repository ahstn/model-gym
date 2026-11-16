"""Continue allowed Decisions images while retaining terminal content denials."""

# ruff: noqa: I001 -- bootstrap the existing immutable request implementation first.
import argparse
import concurrent.futures
import fcntl
import os
import threading
from collections import Counter, deque
from types import SimpleNamespace
from corpus import DATA, Path, digest
import api_runner as runner

ORIGINAL_PINS = runner.code_pins


def pins():
    return {**ORIGINAL_PINS(), str(Path(__file__).resolve()): digest(__file__)}


def run(args):
    root = DATA / "results/api" / args.model
    plan = runner.read(root / "plans" / f"{args.phase}.json")
    runner.validate_plan(plan)
    route = plan["profile"]["routes"]["routes"][args.model]
    secret = os.environ[route["key_env"]]
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = runner.Ledger(root, plan["policy"])
        stop = threading.Event()
        counts = Counter()
        recent = deque(maxlen=100)
        denials = 0
        state = {
            "model": args.model,
            "phase": args.phase,
            "plan_id": plan["plan_id"],
            "started_at": runner.now(),
            "planned_cases": len(plan["cases"]),
            "active_worker_limit": plan["policy"]["workers"],
            "content_denial_policy": "terminal missing assessment; no retry; excluded from shared service-outage circuit",
        }

        def save(status):
            state.update(
                status=status,
                updated_at=runner.now(),
                outcomes=dict(counts),
                content_denials=denials,
                **ledger.summary(),
            )
            runner.atomic(root / (args.phase + "-status.json"), state)

        save("running")
        pending = iter(plan["cases"])
        active = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=plan["policy"]["workers"]) as pool:
            try:
                while True:
                    while not stop.is_set() and len(active) < plan["policy"]["workers"]:
                        case = next(pending, None)
                        if case is None:
                            break
                        active[pool.submit(runner.job, plan, case, root, secret, ledger, stop)] = case
                    if not active:
                        break
                    done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        active.pop(future)
                        row = future.result()
                        if row is None:
                            continue
                        counts[row["status"]] += 1
                        raw = runner.read(row["raw_response_path"])
                        denied = runner.image_content_rejection(raw)
                        denials += int(denied)
                        # Respect the rejected image. It is not evidence that the
                        # service is failing for unrelated images.
                        if not denied:
                            recent.append(row["status"] not in {"valid", "invalid"})
                        if len(recent) >= 20 and sum(recent) / len(recent) > 0.25:
                            stop.set()
                            state["stop_reason"] = (
                                "More than 25% exhausted service failures, excluding terminal image denials"
                            )
                        if sum(counts.values()) % 50 == 0:
                            save("stopping" if stop.is_set() else "running")
                            print(state, flush=True)
            except BaseException as error:
                stop.set()
                state["stop_reason"] = type(error).__name__ + ": " + str(error).replace(secret, "[REDACTED]")
                save("stopped")
                raise
        save("stopped" if stop.is_set() else "complete")
        print(state, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--model", choices=["liquid", "jev"], required=True)
    args = parser.parse_args()
    args.phase = "per-image-denials"
    if args.command == "freeze":
        runner.code_pins = pins
        runner.POLICY[args.model].update(
            workers=16,
            budget_usd="10.00" if args.model == "liquid" else "60.00",
            content_denials="terminal; no retry; continue unrelated allowed images",
        )
        options = SimpleNamespace(
            data=DATA,
            model=args.model,
            phase=args.phase,
            selection="rehearsal",
            manifest=DATA / "manifest.json",
            references=None,
            case_ids=None,
        )
        print(runner.freeze(options), flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
