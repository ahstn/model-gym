"""Third operational amendment permits exact unresolved Decider read timeouts."""

import json
from datetime import UTC, datetime
from pathlib import Path

from bench_protocol import digest
from tracks.metadata_v3 import budget_reallocation, cost_recovery, runtime
from tracks.metrics import summarize

AMENDMENT_NAME = "cost-recovery-timeouts.json"
CODE = ("timeout_recovery.py", "resume_timeouts.py")
ERRORS = ("ConnectionError", "ReadTimeout")


def eligible(row, protocol_sha256):
    """Reuse strict transport fields on a copy; never relabel the retained raw attempt."""
    if row.get("transport_error") not in ERRORS:
        return False
    fields = {**row, "transport_error": "ConnectionError"}
    return cost_recovery.eligible(fields, protocol_sha256)


def freeze_extension(reallocation_path, runs_root, reason):
    reallocation_path, runs_root = Path(reallocation_path).resolve(), Path(runs_root).resolve()
    second = budget_reallocation.validate_reallocation(reallocation_path, runs_root)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Timeout extension requires an explicit review reason")
    if runtime.StudyBudget(runs_root).pending():
        raise runtime.BudgetStop("Resolve in-flight requests before the timeout amendment")
    prefixes = {}
    for ledger in sorted(runs_root.rglob("*.jsonl")):
        rows = runtime.read_ledger(ledger)
        if any(row.get("split") in ("test", "external") for row in rows):
            raise ValueError("Timeout amendment must precede every held-out attempt")
        for row in rows:
            if row.get("effective_cost_usd") is None and not eligible(
                row, second["original_protocol_sha256"]
            ):
                raise runtime.BudgetStop("Unsupported unknown cost; timeout amendment cannot apply")
        if rows:
            prefixes[str(ledger.relative_to(runs_root))] = {
                "prefix_bytes": ledger.stat().st_size,
                "prefix_sha256": cost_recovery.sha(ledger),
            }
    here = Path(__file__).resolve().parent
    extension = {
        "version": 3,
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_reason": reason,
        "reallocation_path": str(reallocation_path),
        "reallocation_sha256": cost_recovery.sha(reallocation_path),
        "original_protocol_sha256": second["original_protocol_sha256"],
        "runs_root": str(runs_root),
        "eligible_transport_errors": list(ERRORS),
        "added_transport_error": "ReadTimeout",
        "allowance_usd": cost_recovery.ALLOWANCE_USD,
        "actual_cost_remains_unknown": True,
        "global_budget_usd": 5.0,
        "guard_budget_by_arm": budget_reallocation.CAPS.copy(),
        "no_retry": True,
        "new_unknown_batch_action": "Record allowances and stop; verify network before manual resume",
        "initial_ledger_prefixes": prefixes,
        "initial_failure_receipts": {
            str(path.relative_to(runs_root)): cost_recovery.sha(path)
            for path in sorted((runs_root / "recovery-receipts").glob("*.json"))
        },
        "timeout_code_sha256": {name: cost_recovery.sha(here / name) for name in CODE},
    }
    path = reallocation_path.parent / AMENDMENT_NAME
    runtime.write_frozen(path, extension)
    return path


def validate_extension(path, runs_root):
    path, runs_root = Path(path).resolve(), Path(runs_root).resolve()
    extension = json.loads(path.read_text())
    if (
        extension.get("version") != 3
        or extension.get("runs_root") != str(runs_root)
        or extension.get("eligible_transport_errors") != list(ERRORS)
        or extension.get("added_transport_error") != "ReadTimeout"
        or extension.get("allowance_usd") != cost_recovery.ALLOWANCE_USD
        or extension.get("actual_cost_remains_unknown") is not True
        or extension.get("no_retry") is not True
        or extension.get("global_budget_usd") != 5.0
        or extension.get("guard_budget_by_arm") != budget_reallocation.CAPS
        or extension.get("new_unknown_batch_action")
        != "Record allowances and stop; verify network before manual resume"
        or not extension.get("review_reason")
    ):
        raise ValueError("Invalid timeout operational amendment")
    if cost_recovery.sha(extension["reallocation_path"]) != extension["reallocation_sha256"]:
        raise ValueError("Frozen reallocation amendment changed")
    second = budget_reallocation.validate_reallocation(extension["reallocation_path"], runs_root)
    if extension["original_protocol_sha256"] != second["original_protocol_sha256"]:
        raise ValueError("Timeout extension protocol changed")
    here = Path(__file__).resolve().parent
    if set(extension["timeout_code_sha256"]) != set(CODE):
        raise ValueError("Incomplete timeout code pins")
    for name, expected in extension["timeout_code_sha256"].items():
        if cost_recovery.sha(here / name) != expected:
            raise ValueError("Approved timeout recovery code changed")
    for relative, entry in extension["initial_ledger_prefixes"].items():
        ledger = (runs_root / relative).resolve()
        if (
            not ledger.is_relative_to(runs_root)
            or cost_recovery.prefix(ledger, entry["prefix_bytes"]) != entry["prefix_sha256"]
        ):
            raise ValueError("Timeout amendment ledger prefix changed")
    for relative, expected in extension["initial_failure_receipts"].items():
        receipt = (runs_root / relative).resolve()
        if not receipt.is_relative_to(runs_root) or cost_recovery.sha(receipt) != expected:
            raise ValueError("Previous failure receipt changed")
    return extension


class TimeoutBudget(budget_reallocation.ReallocatedBudget):
    """Use existing guard caps and keep receipt approval versions explicit."""

    def __init__(self, *args, extension_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.extension_path = Path(extension_path).resolve()
        self.extension = validate_extension(self.extension_path, self.root)
        self.extension_sha256 = cost_recovery.sha(self.extension_path)

    def approve_failures(self):
        for ledger, row in self.unknown_rows():
            self.ledger_path(ledger, row.get("metadata_arm"))
            if not eligible(row, self.amendment["original_protocol_sha256"]):
                raise runtime.BudgetStop("Unsupported unknown cost; timeout recovery policy does not apply")
            path = self.receipt_path(ledger, row)
            if not path.exists():
                runtime.write_frozen(
                    path,
                    {
                        "version": 3,
                        "created_at": datetime.now(UTC).isoformat(),
                        "amendment_sha256": self.amendment_sha256,
                        "reallocation_sha256": self.reallocation_sha256,
                        "timeout_extension_sha256": self.extension_sha256,
                        "ledger_path": str(ledger.relative_to(self.root)),
                        "ledger_prefix_bytes": ledger.stat().st_size,
                        "ledger_prefix_sha256": cost_recovery.sha(ledger),
                        "record_sha256": digest(row),
                        "sample": row["sample"],
                        "request_sha256": row["request_sha256"],
                        "metadata_arm": row["metadata_arm"],
                        "requested_model": row["requested_model"],
                        "transport_error": row["transport_error"],
                        "allowance_usd": cost_recovery.ALLOWANCE_USD,
                        "actual_cost_usd": None,
                        "no_retry": True,
                        "cost_interpretation": "Unverified guard allowance, not a provider charge or proven ceiling",
                    },
                )
        return self.totals()

    def validate_receipt(self, ledger, row):
        receipt = json.loads(self.receipt_path(ledger, row).read_text())
        if receipt.get("version") == 1:
            return super().validate_receipt(ledger, row)
        if not eligible(row, self.amendment["original_protocol_sha256"]):
            raise runtime.BudgetStop("Unsupported unknown cost; timeout recovery policy does not apply")
        expected = {
            "version": 3,
            "amendment_sha256": self.amendment_sha256,
            "reallocation_sha256": self.reallocation_sha256,
            "timeout_extension_sha256": self.extension_sha256,
            "ledger_path": str(ledger.relative_to(self.root)),
            "record_sha256": digest(row),
            "sample": row["sample"],
            "request_sha256": row["request_sha256"],
            "metadata_arm": row["metadata_arm"],
            "requested_model": row["requested_model"],
            "transport_error": row["transport_error"],
            "allowance_usd": cost_recovery.ALLOWANCE_USD,
            "actual_cost_usd": None,
            "no_retry": True,
        }
        if any(receipt.get(key, "missing") != value for key, value in expected.items()):
            raise ValueError("Timeout failure receipt changed")
        if cost_recovery.prefix(ledger, receipt["ledger_prefix_bytes"]) != receipt["ledger_prefix_sha256"]:
            raise ValueError("Timeout raw ledger prefix changed")
        return receipt

    def totals(self):
        if cost_recovery.sha(self.extension_path) != self.extension_sha256:
            raise ValueError("Timeout operational amendment changed")
        result = super().totals()
        result["timeout_extension_sha256"] = self.extension_sha256
        return result


def evaluate_timeouts(samples, prompt, arm, output, budget, protocol_sha256, split, **kwargs):
    """Allow cached failures as zero-reward rows; stop each new unknown batch."""
    if not isinstance(budget, TimeoutBudget):
        raise TypeError("Timeout evaluation requires the approved third amendment")
    if protocol_sha256 != budget.amendment["original_protocol_sha256"]:
        raise ValueError("Timeout recovery protocol identity changed")
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
            or totals["arms"][arm]["budget_accounted_usd"] > budget_reallocation.CAPS[arm]
        ):
            raise runtime.BudgetStop("Study or reallocated arm spend guard reached") from exc
        metric_samples = [{**sample, "labels": {"ai_generated": bool(sample["gold"])}} for sample in samples]
        result = summarize("ai_detection", metric_samples, [by_id[s["id"]] for s in samples])
        result["summary"].update(track="metadata_v3", metadata_arm=arm)
        return {**result, "ledger_path": str(output), "ledger_records": records}


def timeout_summary(runs_root, extension_path):
    """Read-only report API validates the complete three-amendment chain."""
    third = validate_extension(extension_path, runs_root)
    second = budget_reallocation.validate_reallocation(third["reallocation_path"], runs_root)
    first = cost_recovery.validate_amendment(second["first_amendment_path"], runs_root)
    protocol = json.loads(Path(first["original_protocol_path"]).read_text())
    budget = TimeoutBudget(
        runs_root,
        max_calls=protocol["max_task_calls_per_arm"],
        reserve_usd=protocol["reserve_per_inflight_usd"],
        amendment_path=second["first_amendment_path"],
        reallocation_path=third["reallocation_path"],
        extension_path=extension_path,
    )
    return {"amendment": first, "reallocation": second, "timeout_extension": third, "costs": budget.totals()}
