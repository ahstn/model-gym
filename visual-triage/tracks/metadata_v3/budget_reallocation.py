"""Second operational amendment reallocates guard headroom without changing run receipts."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from tracks.metadata_v3 import cost_recovery, runtime
from tracks.metrics import summarize

AMENDMENT_NAME = "cost-recovery-reallocation.json"
CODE = ("budget_reallocation.py", "resume_reallocated.py")
CAPS = {arm: 1.10 if arm == "image_only" else 0.65 for arm in runtime.ARMS}


def freeze_reallocation(first_amendment_path, runs_root, reason):
    first_amendment_path, runs_root = Path(first_amendment_path).resolve(), Path(runs_root).resolve()
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Reallocation requires an explicit operational review reason")
    prefixes = {}
    for ledger in sorted(runs_root.rglob("*.jsonl")):
        rows = runtime.read_ledger(ledger)
        if any(row.get("split") in ("test", "external") for row in rows):
            raise ValueError("Reallocation must be frozen before held-out attempts")
        if rows:
            prefixes[str(ledger.relative_to(runs_root))] = {
                "prefix_bytes": ledger.stat().st_size,
                "prefix_sha256": cost_recovery.sha(ledger),
            }
    original = cost_recovery.allowance_summary(runs_root, first_amendment_path)
    if original["costs"]["inflight"]:
        raise runtime.BudgetStop("Resolve in-flight attempts before the second amendment")
    here = Path(__file__).resolve().parent
    amendment = {
        "version": 2,
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_reason": reason,
        "first_amendment_path": str(first_amendment_path),
        "first_amendment_sha256": cost_recovery.sha(first_amendment_path),
        "original_protocol_sha256": original["amendment"]["original_protocol_sha256"],
        "runs_root": str(runs_root),
        "global_budget_usd": 5.0,
        "original_run_receipt_arm_usd": 0.65,
        "guard_budget_by_arm": CAPS,
        "reallocated_unused_global_headroom_usd": 0.45,
        "allowance_usd": cost_recovery.ALLOWANCE_USD,
        "no_retry": True,
        "new_unknown_batch_action": "Record approved allowances and stop; check network before manual resume",
        "initial_ledger_prefixes": prefixes,
        "initial_failure_receipts": {
            str(path.relative_to(runs_root)): cost_recovery.sha(path)
            for path in sorted((runs_root / "recovery-receipts").glob("*.json"))
        },
        "reallocation_code_sha256": {name: cost_recovery.sha(here / name) for name in CODE},
    }
    path = first_amendment_path.parent / AMENDMENT_NAME
    runtime.write_frozen(path, amendment)
    return path


def validate_reallocation(path, runs_root):
    path, runs_root = Path(path).resolve(), Path(runs_root).resolve()
    amendment = json.loads(path.read_text())
    if (
        amendment.get("version") != 2
        or amendment.get("runs_root") != str(runs_root)
        or amendment.get("global_budget_usd") != 5.0
        or amendment.get("guard_budget_by_arm") != CAPS
        or amendment.get("original_run_receipt_arm_usd") != 0.65
        or amendment.get("reallocated_unused_global_headroom_usd") != 0.45
        or amendment.get("allowance_usd") != cost_recovery.ALLOWANCE_USD
        or amendment.get("no_retry") is not True
        or not amendment.get("review_reason")
        or amendment.get("new_unknown_batch_action")
        != "Record approved allowances and stop; check network before manual resume"
        or math.fsum(amendment["guard_budget_by_arm"].values()) > 5.0 + 1e-9
    ):
        raise ValueError("Invalid second operational amendment")
    if cost_recovery.sha(amendment["first_amendment_path"]) != amendment["first_amendment_sha256"]:
        raise ValueError("Original recovery amendment changed")
    first = cost_recovery.validate_amendment(amendment["first_amendment_path"], runs_root)
    if first["original_protocol_sha256"] != amendment["original_protocol_sha256"]:
        raise ValueError("Original protocol identity changed")
    here = Path(__file__).resolve().parent
    if set(amendment["reallocation_code_sha256"]) != set(CODE):
        raise ValueError("Incomplete reallocation code pins")
    for name, expected in amendment["reallocation_code_sha256"].items():
        if cost_recovery.sha(here / name) != expected:
            raise ValueError("Approved reallocation code changed")
    for relative, entry in amendment["initial_ledger_prefixes"].items():
        ledger = (runs_root / relative).resolve()
        if (
            not ledger.is_relative_to(runs_root)
            or cost_recovery.prefix(ledger, entry["prefix_bytes"]) != entry["prefix_sha256"]
        ):
            raise ValueError("Second amendment raw ledger prefix changed")
    for relative, expected in amendment["initial_failure_receipts"].items():
        receipt = (runs_root / relative).resolve()
        if not receipt.is_relative_to(runs_root) or cost_recovery.sha(receipt) != expected:
            raise ValueError("Original failure receipt changed")
    return amendment


class ReallocatedBudget(cost_recovery.RecoveryBudget):
    """Keep arm_usd=.65 for exact resume; apply the separately frozen guard cap map."""

    def __init__(self, *args, reallocation_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.reallocation_path = Path(reallocation_path).resolve()
        self.reallocation = validate_reallocation(self.reallocation_path, self.root)
        self.reallocation_sha256 = cost_recovery.sha(self.reallocation_path)

    def totals(self):
        if cost_recovery.sha(self.reallocation_path) != self.reallocation_sha256:
            raise ValueError("Second operational amendment changed")
        result = super().totals()
        result.update(guard_budget_by_arm=CAPS.copy(), reallocation_sha256=self.reallocation_sha256)
        for arm, value in result["arms"].items():
            value["guard_budget_usd"] = CAPS[arm]
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
        headroom = cost_recovery.ALLOWANCE_USD if task_calls else self.reserve_usd
        if (
            totals["budget_accounted_usd"] + totals["reserved_inflight_usd"] + headroom > self.global_usd
            or totals["arms"][arm]["budget_accounted_usd"] + sum(e["reserved_usd"] for e in active) + headroom
            > CAPS[arm]
        ):
            raise runtime.BudgetStop("Study or reallocated arm spend guard reached")
        return totals


def evaluate_reallocated(samples, prompt, arm, output, budget, protocol_sha256, split, **kwargs):
    if not isinstance(budget, ReallocatedBudget):
        raise TypeError("Reallocated evaluation requires the approved second amendment")
    if protocol_sha256 != budget.amendment["original_protocol_sha256"]:
        raise ValueError("Recovery protocol identity changed")
    before = budget.approve_failures()["unknown_cost_count"]
    while True:
        try:
            return runtime.evaluate(samples, prompt, arm, output, budget, protocol_sha256, split, **kwargs)
        except runtime.BudgetStop as exc:
            if not str(exc).startswith("Unknown request cost"):
                raise
            totals = budget.approve_failures()
            if totals["unknown_cost_count"] > before:
                raise runtime.BudgetStop(
                    "New connection failure batch retained; verify network before manual resume"
                ) from exc
            records = runtime.read_ledger(output)
            by_id = {row["sample"]: row for row in records}
            if all(sample["id"] in by_id for sample in samples):
                if (
                    totals["budget_accounted_usd"] > budget.global_usd
                    or totals["arms"][arm]["budget_accounted_usd"] > CAPS[arm]
                ):
                    raise runtime.BudgetStop("Study or reallocated arm spend guard reached")
                metric_samples = [
                    {**sample, "labels": {"ai_generated": bool(sample["gold"])}} for sample in samples
                ]
                result = summarize("ai_detection", metric_samples, [by_id[s["id"]] for s in samples])
                result["summary"].update(track="metadata_v3", metadata_arm=arm)
                return {**result, "ledger_path": str(output), "ledger_records": records}


def reallocation_summary(runs_root, reallocation_path):
    """Read-only report API for both amendments and true unknown actual costs."""
    second = validate_reallocation(reallocation_path, runs_root)
    first = cost_recovery.validate_amendment(second["first_amendment_path"], runs_root)
    protocol = json.loads(Path(first["original_protocol_path"]).read_text())
    budget = ReallocatedBudget(
        runs_root,
        max_calls=protocol["max_task_calls_per_arm"],
        reserve_usd=protocol["reserve_per_inflight_usd"],
        amendment_path=second["first_amendment_path"],
        reallocation_path=reallocation_path,
    )
    return {"amendment": first, "reallocation": second, "costs": budget.totals()}
