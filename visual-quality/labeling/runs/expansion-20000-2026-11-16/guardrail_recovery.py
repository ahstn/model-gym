"""Replay secret-scanner denials after the user changed their gateway guardrail."""

# ruff: noqa: I001 -- establish the frozen adapter context first.
import argparse
import json
from types import SimpleNamespace
from corpus import DATA, Path, digest
import api_runner as runner

PHASE = "secret-guardrail-disabled"
AUTHORIZATION = DATA / "operations/guardrail-change-authorization.json"
ORIGINAL_PINS = runner.code_pins


def secret_scanner_denial(result):
    if not runner.image_content_rejection(result):
        return False
    message = json.loads(result["body"])["error"]["message"]
    return message == "Request blocked by content filter: [SECRET:bitcoin-wif-compressed]"


def pins():
    return {
        **ORIGINAL_PINS(),
        str(Path(__file__).resolve()): digest(__file__),
        str(AUTHORIZATION): digest(AUTHORIZATION),
    }


def recovery_job(plan, case, root, secret, ledger, stop):
    body, request_sha = runner.prepare(plan, case)
    previous = runner.attempts_for(root, case["case_id"])
    baseline = plan["policy"]["authorized_denials"][case["case_id"]]
    prior_count = baseline["attempt_count"]
    if len(previous) < prior_count:
        raise ValueError("Original attempts missing")
    last_old = previous[prior_count - 1].with_name("result.json")
    if digest(last_old) != baseline["result_sha256"] or not secret_scanner_denial(runner.read(last_old)):
        raise ValueError("Original secret-scanner denial changed")
    for i, path in enumerate(previous):
        request = runner.read(path)
        if request["request_sha256"] != request_sha or request["view_sha256"] != case["view_sha256"]:
            raise ValueError("Recovery request differs from original view/profile")
        result_path = path.with_name("result.json")
        if i >= prior_count and result_path.exists():
            result = runner.read(result_path)
            if not result["parsed"]["retryable"]:
                return runner.observation(plan, case, root, request_sha)
        elif i < prior_count and result_path.exists():
            result = runner.read(result_path)
            if not result["parsed"]["retryable"] and not secret_scanner_denial(result):
                raise ValueError("Recovery must not replace a prior assessment or other rejection")
    maximum = prior_count + plan["policy"]["additional_attempts"]
    for number in range(len(previous) + 1, maximum + 1):
        if stop.is_set():
            return None
        result = runner.request_once(plan, case, root, number, body, request_sha, secret, ledger)
        if result.get("http_status") in (401, 402, 403):
            stop.set()
        if not result["parsed"]["retryable"]:
            break
        if number < maximum:
            try:
                delay = float(result.get("headers", {}).get("retry-after", 0))
            except ValueError:
                delay = 0
            stop.wait(min(60, max(delay, 2 ** (number - prior_count))))
    return runner.observation(plan, case, root, request_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--model", choices=["liquid", "jev"], required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = DATA / "results/api" / args.model
    options = SimpleNamespace(
        data=DATA, model=args.model, phase=PHASE, selection="rehearsal",
        manifest=DATA / "manifest.json", references=None, case_ids=None,
        limit=args.limit, worker_limit=16,
    )
    if args.command == "freeze":
        denied = {}
        for path in sorted((root / "observations").glob("*.json")):
            row = runner.read(path)
            raw = runner.read(row["raw_response_path"])
            if secret_scanner_denial(raw):
                denied[row["case_id"]] = {
                    "attempt_count": row["attempt_count"],
                    "result_sha256": digest(row["raw_response_path"]),
                }
        if len(denied) != 2855:
            raise ValueError(f"Expected the documented 2855 denials; found {len(denied)}")
        options.case_ids = DATA / "operations" / (args.model + "-guardrail-recovery-ids.json")
        runner.atomic(options.case_ids, sorted(denied))
        runner.code_pins = pins
        runner.POLICY[args.model].update(
            workers=16, budget_usd="10.00" if args.model == "liquid" else "60.00",
            additional_attempts=3, authorized_denials=denied,
            max_attempts=max(r["attempt_count"] for r in denied.values()) + 3,
            recovery_policy="User disabled their secret-redaction guardrail; identical requests; preserve prior attempts; never repeat a new terminal result; stop on any new 403",
        )
        print(runner.freeze(options), flush=True)
    else:
        runner.job = recovery_job
        print(runner.run(options), flush=True)


if __name__ == "__main__":
    main()
