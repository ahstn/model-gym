"""Fifth before-test amendment covers strict unresolved transient Decider HTTP errors."""

import json
from datetime import UTC, datetime
from pathlib import Path

from bench_protocol import digest
from tracks.metadata_v3 import cost_recovery, runtime, shared_allowance_pool, timeout_recovery

AMENDMENT_NAME = "cost-recovery-http-errors.json"
CODE = ("http_error_recovery.py", "resume_http_errors.py")
STATUSES = (500, 502, 503, 504)


def http_eligible(row, protocol_sha256):
    response = row.get("response")
    if type(row.get("status")) is not int or row["status"] not in STATUSES:
        return False
    if row.get("transport_error") is not None or not isinstance(response, dict) or set(response) != {"error"}:
        return False
    error = response["error"]
    if (
        not isinstance(error, dict)
        or set(error) != {"code", "message"}
        or type(error.get("code")) is not int
        or error["code"] != row["status"]
        or not isinstance(error.get("message"), str)
        or not error["message"]
        or any(error.get(key) is not None for key in ("id", "generation_id", "usage", "cost", "cost_usd"))
    ):
        return False
    fields = {**row, "status": None, "response": {}, "transport_error": "ConnectionError"}
    return cost_recovery.eligible(fields, protocol_sha256)


def eligible(row, protocol_sha256):
    return timeout_recovery.eligible(row, protocol_sha256) or http_eligible(row, protocol_sha256)


def freeze_http_extension(shared_pool_path, runs_root, reason):
    shared_pool_path, runs_root = Path(shared_pool_path).resolve(), Path(runs_root).resolve()
    fourth = shared_allowance_pool.validate_shared_pool(shared_pool_path, runs_root)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("HTTP extension requires an explicit operational review reason")
    if runtime.StudyBudget(runs_root).pending():
        raise runtime.BudgetStop("Resolve in-flight requests before the HTTP amendment")
    prefixes = {}
    for ledger in sorted(runs_root.rglob("*.jsonl")):
        rows = runtime.read_ledger(ledger)
        if any(row.get("split") in ("test", "external") for row in rows):
            raise ValueError("HTTP amendment must precede held-out attempts")
        if any(
            row.get("effective_cost_usd") is None and not eligible(row, fourth["original_protocol_sha256"])
            for row in rows
        ):
            raise runtime.BudgetStop("Unsupported unknown cost; HTTP amendment cannot apply")
        if rows:
            prefixes[str(ledger.relative_to(runs_root))] = {
                "prefix_bytes": ledger.stat().st_size,
                "prefix_sha256": cost_recovery.sha(ledger),
            }
    here = Path(__file__).resolve().parent
    extension = {
        "version": 5,
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_reason": reason,
        "shared_pool_path": str(shared_pool_path),
        "shared_pool_sha256": cost_recovery.sha(shared_pool_path),
        "original_protocol_sha256": fourth["original_protocol_sha256"],
        "runs_root": str(runs_root),
        "added_http_statuses": list(STATUSES),
        "error_body_required": True,
        "response_contract": "Exactly one error object with only code and message, matching integer HTTP code and nonempty message; no ID, usage or known cost",
        "allowance_usd": cost_recovery.ALLOWANCE_USD,
        "actual_cost_remains_unknown": True,
        "no_retry": True,
        "global_budget_usd": 5.0,
        "known_spend_budget_per_arm_usd": 0.65,
        "unknown_allowance_scope": "shared_global_only",
        "circuit_stop_unchanged": True,
        "initial_ledger_prefixes": prefixes,
        "initial_failure_receipts": {
            str(path.relative_to(runs_root)): cost_recovery.sha(path)
            for path in sorted((runs_root / "recovery-receipts").glob("*.json"))
        },
        "http_code_sha256": {name: cost_recovery.sha(here / name) for name in CODE},
    }
    path = shared_pool_path.parent / AMENDMENT_NAME
    runtime.write_frozen(path, extension)
    return path


def validate_http_extension(path, runs_root):
    path, runs_root = Path(path).resolve(), Path(runs_root).resolve()
    extension = json.loads(path.read_text())
    if (
        extension.get("version") != 5
        or extension.get("runs_root") != str(runs_root)
        or extension.get("added_http_statuses") != list(STATUSES)
        or extension.get("error_body_required") is not True
        or extension.get("allowance_usd") != cost_recovery.ALLOWANCE_USD
        or extension.get("actual_cost_remains_unknown") is not True
        or extension.get("no_retry") is not True
        or extension.get("global_budget_usd") != 5.0
        or extension.get("known_spend_budget_per_arm_usd") != 0.65
        or extension.get("unknown_allowance_scope") != "shared_global_only"
        or extension.get("circuit_stop_unchanged") is not True
        or not extension.get("review_reason")
    ):
        raise ValueError("Invalid HTTP operational amendment")
    if cost_recovery.sha(extension["shared_pool_path"]) != extension["shared_pool_sha256"]:
        raise ValueError("Previous shared-pool amendment changed")
    fourth = shared_allowance_pool.validate_shared_pool(extension["shared_pool_path"], runs_root)
    if extension["original_protocol_sha256"] != fourth["original_protocol_sha256"]:
        raise ValueError("HTTP extension original protocol changed")
    here = Path(__file__).resolve().parent
    if set(extension["http_code_sha256"]) != set(CODE):
        raise ValueError("Incomplete HTTP recovery code pins")
    for name, expected in extension["http_code_sha256"].items():
        if cost_recovery.sha(here / name) != expected:
            raise ValueError("Approved HTTP recovery code changed")
    for relative, entry in extension["initial_ledger_prefixes"].items():
        ledger = (runs_root / relative).resolve()
        if (
            not ledger.is_relative_to(runs_root)
            or cost_recovery.prefix(ledger, entry["prefix_bytes"]) != entry["prefix_sha256"]
        ):
            raise ValueError("HTTP amendment raw ledger prefix changed")
    for relative, expected in extension["initial_failure_receipts"].items():
        receipt = (runs_root / relative).resolve()
        if not receipt.is_relative_to(runs_root) or cost_recovery.sha(receipt) != expected:
            raise ValueError("Previous failure receipt changed")
    return extension


class HTTPBudget(shared_allowance_pool.SharedPoolBudget):
    """Reuse shared-pool caps and distinguish HTTP approval from old transport receipts."""

    def __init__(self, *args, http_extension_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.http_extension_path = Path(http_extension_path).resolve()
        self.http_extension = validate_http_extension(self.http_extension_path, self.root)
        self.http_extension_sha256 = cost_recovery.sha(self.http_extension_path)

    def approve_failures(self):
        for ledger, row in self.unknown_rows():
            self.ledger_path(ledger, row.get("metadata_arm"))
            if not eligible(row, self.amendment["original_protocol_sha256"]):
                raise runtime.BudgetStop("Unsupported unknown cost; HTTP recovery policy does not apply")
            path = self.receipt_path(ledger, row)
            if path.exists():
                continue
            runtime.write_frozen(
                path,
                {
                    "version": 5,
                    "created_at": datetime.now(UTC).isoformat(),
                    "amendment_sha256": self.amendment_sha256,
                    "reallocation_sha256": self.reallocation_sha256,
                    "timeout_extension_sha256": self.extension_sha256,
                    "shared_pool_sha256": self.shared_pool_sha256,
                    "http_extension_sha256": self.http_extension_sha256,
                    "ledger_path": str(ledger.relative_to(self.root)),
                    "ledger_prefix_bytes": ledger.stat().st_size,
                    "ledger_prefix_sha256": cost_recovery.sha(ledger),
                    "record_sha256": digest(row),
                    "sample": row["sample"],
                    "request_sha256": row["request_sha256"],
                    "metadata_arm": row["metadata_arm"],
                    "requested_model": row["requested_model"],
                    "status": row.get("status"),
                    "transport_error": row.get("transport_error"),
                    "response_sha256": digest(row.get("response")),
                    "allowance_usd": cost_recovery.ALLOWANCE_USD,
                    "actual_cost_usd": None,
                    "no_retry": True,
                    "cost_interpretation": "Unverified shared guard allowance, not a provider charge or proven ceiling",
                },
            )
        return self.totals()

    def validate_receipt(self, ledger, row):
        receipt = json.loads(self.receipt_path(ledger, row).read_text())
        if receipt.get("version") in (1, 3):
            return super().validate_receipt(ledger, row)
        if not eligible(row, self.amendment["original_protocol_sha256"]):
            raise runtime.BudgetStop("Unsupported unknown HTTP/transport failure")
        expected = {
            "version": 5,
            "amendment_sha256": self.amendment_sha256,
            "reallocation_sha256": self.reallocation_sha256,
            "timeout_extension_sha256": self.extension_sha256,
            "shared_pool_sha256": self.shared_pool_sha256,
            "http_extension_sha256": self.http_extension_sha256,
            "ledger_path": str(ledger.relative_to(self.root)),
            "record_sha256": digest(row),
            "sample": row["sample"],
            "request_sha256": row["request_sha256"],
            "metadata_arm": row["metadata_arm"],
            "requested_model": row["requested_model"],
            "status": row.get("status"),
            "transport_error": row.get("transport_error"),
            "response_sha256": digest(row.get("response")),
            "allowance_usd": cost_recovery.ALLOWANCE_USD,
            "actual_cost_usd": None,
            "no_retry": True,
        }
        if any(receipt.get(key, "missing") != value for key, value in expected.items()):
            raise ValueError("HTTP failure receipt changed")
        if cost_recovery.prefix(ledger, receipt["ledger_prefix_bytes"]) != receipt["ledger_prefix_sha256"]:
            raise ValueError("HTTP raw ledger prefix changed")
        return receipt

    def totals(self):
        if cost_recovery.sha(self.http_extension_path) != self.http_extension_sha256:
            raise ValueError("HTTP operational amendment changed")
        result = super().totals()
        result["http_extension_sha256"] = self.http_extension_sha256
        return result


def evaluate_http(*args, **kwargs):
    return shared_allowance_pool.evaluate_shared_pool(*args, **kwargs)


def http_summary(runs_root, http_extension_path):
    fifth = validate_http_extension(http_extension_path, runs_root)
    fourth = shared_allowance_pool.validate_shared_pool(fifth["shared_pool_path"], runs_root)
    third = timeout_recovery.validate_extension(fourth["timeout_extension_path"], runs_root)
    from tracks.metadata_v3 import budget_reallocation

    second = budget_reallocation.validate_reallocation(third["reallocation_path"], runs_root)
    first = cost_recovery.validate_amendment(second["first_amendment_path"], runs_root)
    protocol = json.loads(Path(first["original_protocol_path"]).read_text())
    budget = HTTPBudget(
        runs_root,
        max_calls=protocol["max_task_calls_per_arm"],
        reserve_usd=protocol["reserve_per_inflight_usd"],
        amendment_path=second["first_amendment_path"],
        reallocation_path=third["reallocation_path"],
        extension_path=fourth["timeout_extension_path"],
        shared_pool_path=fifth["shared_pool_path"],
        http_extension_path=http_extension_path,
    )
    return {
        "amendment": first,
        "reallocation": second,
        "timeout_extension": third,
        "shared_pool_amendment": fourth,
        "http_extension": fifth,
        "costs": budget.totals(),
    }
