"""Fourth amendment funds unresolved allowances from the fixed shared study cap."""

import json
from datetime import UTC, datetime
from pathlib import Path

from tracks.metadata_v3 import cost_recovery, runtime, timeout_recovery
from tracks.metrics import summarize

AMENDMENT_NAME = "cost-recovery-shared-pool.json"
CODE = ("shared_allowance_pool.py", "resume_shared_pool.py")
SCOPE = "Known reported costs per arm; unresolved allowances and all inflight reservations use the shared global cap"


def freeze_shared_pool(extension_path, runs_root, reason):
    extension_path, runs_root = Path(extension_path).resolve(), Path(runs_root).resolve()
    previous = timeout_recovery.timeout_summary(runs_root, extension_path)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Shared allowance pool requires an explicit review reason")
    if previous["costs"]["inflight"]:
        raise runtime.BudgetStop("Resolve in-flight requests before the shared-pool amendment")
    prefixes = {}
    for path in sorted(runs_root.rglob("*.jsonl")):
        rows = runtime.read_ledger(path)
        if any(row.get("split") in ("test", "external") for row in rows):
            raise ValueError("Shared-pool amendment must precede held-out attempts")
        if rows:
            prefixes[str(path.relative_to(runs_root))] = {
                "prefix_bytes": path.stat().st_size,
                "prefix_sha256": cost_recovery.sha(path),
            }
    here = Path(__file__).resolve().parent
    amendment = {
        "version": 4,
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_reason": reason,
        "timeout_extension_path": str(extension_path),
        "timeout_extension_sha256": cost_recovery.sha(extension_path),
        "original_protocol_sha256": previous["amendment"]["original_protocol_sha256"],
        "runs_root": str(runs_root),
        "global_budget_usd": 5.0,
        "known_spend_budget_per_arm_usd": 0.65,
        "unknown_allowance_usd": cost_recovery.ALLOWANCE_USD,
        "unknown_allowance_scope": "shared_global_only",
        "global_guard_scope": "Known costs plus unresolved allowances plus all inflight reservations",
        "arm_guard_scope": "Known costs plus original prospective known-cost headroom; excludes unresolved allowances",
        "known_inflight_headroom_per_request_usd": 0.01,
        "original_run_receipt_unchanged": True,
        "actual_cost_remains_unknown": True,
        "no_retry": True,
        "transport_policy_unchanged": True,
        "circuit_stop_unchanged": True,
        "diagnostics": "Optional exception class and errno chains, endpoint path, request hash and UTC timestamp; outside raw runs",
        "initial_ledger_prefixes": prefixes,
        "initial_failure_receipts": {
            str(path.relative_to(runs_root)): cost_recovery.sha(path)
            for path in sorted((runs_root / "recovery-receipts").glob("*.json"))
        },
        "shared_pool_code_sha256": {name: cost_recovery.sha(here / name) for name in CODE},
    }
    path = extension_path.parent / AMENDMENT_NAME
    runtime.write_frozen(path, amendment)
    return path


def validate_shared_pool(path, runs_root):
    path, runs_root = Path(path).resolve(), Path(runs_root).resolve()
    amendment = json.loads(path.read_text())
    if (
        amendment.get("version") != 4
        or amendment.get("runs_root") != str(runs_root)
        or amendment.get("global_budget_usd") != 5.0
        or amendment.get("known_spend_budget_per_arm_usd") != 0.65
        or amendment.get("unknown_allowance_usd") != cost_recovery.ALLOWANCE_USD
        or amendment.get("unknown_allowance_scope") != "shared_global_only"
        or amendment.get("known_inflight_headroom_per_request_usd") != 0.01
        or amendment.get("original_run_receipt_unchanged") is not True
        or amendment.get("actual_cost_remains_unknown") is not True
        or amendment.get("no_retry") is not True
        or amendment.get("transport_policy_unchanged") is not True
        or amendment.get("circuit_stop_unchanged") is not True
        or not amendment.get("review_reason")
    ):
        raise ValueError("Invalid shared allowance pool amendment")
    if cost_recovery.sha(amendment["timeout_extension_path"]) != amendment["timeout_extension_sha256"]:
        raise ValueError("Previous timeout amendment changed")
    third = timeout_recovery.validate_extension(amendment["timeout_extension_path"], runs_root)
    if amendment["original_protocol_sha256"] != third["original_protocol_sha256"]:
        raise ValueError("Shared-pool original protocol changed")
    here = Path(__file__).resolve().parent
    if set(amendment["shared_pool_code_sha256"]) != set(CODE):
        raise ValueError("Incomplete shared-pool code pins")
    for name, expected in amendment["shared_pool_code_sha256"].items():
        if cost_recovery.sha(here / name) != expected:
            raise ValueError("Approved shared-pool code changed")
    for relative, entry in amendment["initial_ledger_prefixes"].items():
        ledger = (runs_root / relative).resolve()
        if (
            not ledger.is_relative_to(runs_root)
            or cost_recovery.prefix(ledger, entry["prefix_bytes"]) != entry["prefix_sha256"]
        ):
            raise ValueError("Shared-pool raw ledger prefix changed")
    for relative, expected in amendment["initial_failure_receipts"].items():
        receipt = (runs_root / relative).resolve()
        if not receipt.is_relative_to(runs_root) or cost_recovery.sha(receipt) != expected:
            raise ValueError("Previous failure receipt changed")
    return amendment


class SharedPoolBudget(timeout_recovery.TimeoutBudget):
    """Unknown actual costs remain unknown; their allowances consume only global funds."""

    def __init__(self, *args, shared_pool_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_pool_path = Path(shared_pool_path).resolve()
        self.shared_pool = validate_shared_pool(self.shared_pool_path, self.root)
        self.shared_pool_sha256 = cost_recovery.sha(self.shared_pool_path)

    def totals(self):
        if cost_recovery.sha(self.shared_pool_path) != self.shared_pool_sha256:
            raise ValueError("Shared-pool operational amendment changed")
        result = super().totals()
        result.pop("guard_budget_by_arm", None)
        result.update(
            shared_pool_sha256=self.shared_pool_sha256,
            budget_guard_scope=SCOPE,
            shared_unknown_allowance_pool_usd=result["reserved_unverified_usd"],
            known_spend_budget_per_arm_usd=self.arm_usd,
            known_spend_budget_by_arm={arm: self.arm_usd for arm in runtime.ARMS},
        )
        for value in result["arms"].values():
            value.pop("guard_budget_usd", None)
            value.update(known_spend_budget_usd=self.arm_usd, guard_scope="known_reported_usd only")
        return result

    def check(self, arm, task_calls=0):
        if arm not in runtime.ARMS:
            raise ValueError("Unknown metadata arm")
        totals = self.totals()
        active = [entry for entry in totals["inflight"].values() if entry["metadata_arm"] == arm]
        if (
            totals["arms"][arm]["task_calls"] + sum(e["record_type"] == "task" for e in active) + task_calls
            > self.max_calls
        ):
            raise runtime.BudgetStop("Arm task call cap reached")
        global_headroom = cost_recovery.ALLOWANCE_USD if task_calls else self.reserve_usd
        if (
            totals["budget_accounted_usd"] + totals["reserved_inflight_usd"] + global_headroom
            > self.global_usd
        ):
            raise runtime.BudgetStop("Shared global spend guard reached")
        if totals["arms"][arm]["known_reported_usd"] + (len(active) + 1) * self.reserve_usd > self.arm_usd:
            raise runtime.BudgetStop("Arm known-spend guard reached")
        return totals


def evaluate_shared_pool(samples, prompt, arm, output, budget, protocol_sha256, split, **kwargs):
    if not isinstance(budget, SharedPoolBudget):
        raise TypeError("Shared-pool evaluation requires the approved fourth amendment")
    if protocol_sha256 != budget.amendment["original_protocol_sha256"]:
        raise ValueError("Shared-pool original protocol identity changed")
    before = budget.approve_failures()["unknown_cost_count"]
    try:
        return runtime.evaluate(samples, prompt, arm, output, budget, protocol_sha256, split, **kwargs)
    except runtime.BudgetStop as exc:
        if not str(exc).startswith("Unknown request cost"):
            raise
        totals = budget.approve_failures()
        if totals["unknown_cost_count"] > before:
            raise runtime.BudgetStop(
                "New unknown transport batch retained; verify network before manual resume"
            ) from exc
        records = runtime.read_ledger(output)
        by_id = {row["sample"]: row for row in records}
        if not all(sample["id"] in by_id for sample in samples):
            raise runtime.BudgetStop("Incomplete cached evaluation; explicit resume required") from exc
        if (
            totals["budget_accounted_usd"] > budget.global_usd
            or totals["arms"][arm]["known_reported_usd"] > budget.arm_usd
        ):
            raise runtime.BudgetStop("Shared global or arm known-spend guard reached") from exc
        metric_samples = [{**sample, "labels": {"ai_generated": bool(sample["gold"])}} for sample in samples]
        result = summarize("ai_detection", metric_samples, [by_id[s["id"]] for s in samples])
        result["summary"].update(track="metadata_v3", metadata_arm=arm)
        return {**result, "ledger_path": str(output), "ledger_records": records}


def shared_pool_summary(runs_root, shared_pool_path):
    """Read-only report validator preserves all four linked operational receipts."""
    fourth = validate_shared_pool(shared_pool_path, runs_root)
    previous = timeout_recovery.timeout_summary(runs_root, fourth["timeout_extension_path"])
    protocol = json.loads(Path(previous["amendment"]["original_protocol_path"]).read_text())
    budget = SharedPoolBudget(
        runs_root,
        max_calls=protocol["max_task_calls_per_arm"],
        reserve_usd=protocol["reserve_per_inflight_usd"],
        amendment_path=previous["reallocation"]["first_amendment_path"],
        reallocation_path=previous["timeout_extension"]["reallocation_path"],
        extension_path=fourth["timeout_extension_path"],
        shared_pool_path=shared_pool_path,
    )
    return {**previous, "shared_pool_amendment": fourth, "costs": budget.totals()}
