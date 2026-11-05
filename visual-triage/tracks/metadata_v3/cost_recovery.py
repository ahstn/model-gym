"""Approved guard allowances for unresolved Decider transport failures, never actual charges."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from bench_protocol import DECISIONS, MODELS, digest
from tracks.metadata_v3 import runtime
from tracks.metrics import summarize

ALLOWANCE_USD = 0.05
AMENDMENT_NAME = "cost-recovery-amendment.json"
CODE = ("cost_recovery.py", "recover_study.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefix(path, size):
    if type(size) is not int or size < 1:
        raise ValueError("Invalid reserved ledger prefix length")
    with Path(path).open("rb") as handle:
        value = handle.read(size)
    if len(value) != size:
        raise ValueError("Reserved ledger prefix was truncated")
    return hashlib.sha256(value).hexdigest()


def eligible(row, protocol_sha256):
    return (
        row.get("record_type") == "task"
        and row.get("task") == "ai"
        and row.get("arm") == "decider"
        and row.get("metadata_arm") in runtime.ARMS
        and row.get("endpoint") == DECISIONS
        and row.get("requested_model") == MODELS["decider"]
        and row.get("transport_error") == "ConnectionError"
        and row.get("status") is None
        and not row.get("response")
        and not row.get("generation_id")
        and not row.get("usage")
        and row.get("effective_cost_usd") is None
        and row.get("cost_usd") is None
        and row.get("normalized") is None
        and row.get("protocol_sha256") == protocol_sha256
        and isinstance(row.get("sample"), str)
        and isinstance(row.get("request_sha256"), str)
        and len(row["request_sha256"]) == 64
    )


def freeze_amendment(protocol_path, runs_root, pricing_capture, reason):
    """Approve a prospective policy before any held-out attempt; original files stay intact."""
    protocol_path, runs_root, pricing_capture = (
        Path(value).resolve() for value in (protocol_path, runs_root, pricing_capture)
    )
    protocol = json.loads(protocol_path.read_text())
    if protocol["global_budget_usd"] != 5.0 or protocol["arm_budget_usd"] != 0.65:
        raise ValueError("Original study caps must remain $5 and $0.65")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("An explicit operational review reason is required")
    check_pricing(pricing_capture)
    initial = {}
    for path in sorted(runs_root.rglob("*.jsonl")):
        rows = runtime.read_ledger(path)
        if any(row.get("split") in ("test", "external") for row in rows):
            raise ValueError("Freeze the amendment before any held-out attempt")
        if rows:
            initial[str(path.relative_to(runs_root))] = {
                "prefix_bytes": path.stat().st_size,
                "prefix_sha256": sha(path),
            }
    here = Path(__file__).resolve().parent
    amendment = {
        "version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "original_protocol_path": str(protocol_path),
        "original_protocol_sha256": sha(protocol_path),
        "runs_root": str(runs_root),
        "pricing_capture_path": str(pricing_capture),
        "pricing_capture_sha256": sha(pricing_capture),
        "recovery_code_sha256": {name: sha(here / name) for name in CODE},
        "review_reason": reason,
        "allowance_usd": ALLOWANCE_USD,
        "eligible_model": MODELS["decider"],
        "eligible_transport_error": "ConnectionError",
        "no_retry": True,
        "actual_cost_remains_unknown": True,
        "prospective_policy": True,
        "initial_ledger_prefixes": initial,
        "pricing_assumption": {
            "input_usd_per_token": 0.00000002,
            "context_tokens": 262144,
            "completion_usd_per_token": 0,
            "question_multiplier_assumption": 3,
            "estimated_three_context_cost_usd": 0.01572864,
            "proven_billing_ceiling": False,
        },
        "cost_interpretation": "Unverified budget allowance; not a provider charge or proven upper bound",
        "global_budget_usd": 5.0,
        "arm_budget_usd": 0.65,
    }
    path = protocol_path.parent / AMENDMENT_NAME
    runtime.write_frozen(path, amendment)
    return path


def check_pricing(path):
    capture = json.loads(Path(path).read_text())
    models = [model for model in capture["models"] if model.get("id") == MODELS["decider"]]
    if (
        capture.get("url") != "https://openrouter.ai/api/v1/models?output_modalities=decisions"
        or len(models) != 1
        or models[0].get("context_length") != 262144
        or float(models[0]["pricing"]["prompt"]) != 0.00000002
        or float(models[0]["pricing"]["completion"]) != 0
    ):
        raise ValueError("Pricing capture does not support the approved allowance assumption")


def validate_amendment(path, runs_root):
    path, runs_root = Path(path).resolve(), Path(runs_root).resolve()
    amendment = json.loads(path.read_text())
    if (
        amendment.get("version") != 1
        or amendment.get("runs_root") != str(runs_root)
        or amendment.get("allowance_usd") != ALLOWANCE_USD
        or amendment.get("eligible_model") != MODELS["decider"]
        or amendment.get("eligible_transport_error") != "ConnectionError"
        or amendment.get("no_retry") is not True
        or amendment.get("prospective_policy") is not True
        or amendment.get("actual_cost_remains_unknown") is not True
        or amendment.get("global_budget_usd") != 5.0
        or amendment.get("arm_budget_usd") != 0.65
        or not amendment.get("review_reason")
    ):
        raise ValueError("Invalid operational cost amendment")
    for key, expected_key in (
        ("original_protocol_path", "original_protocol_sha256"),
        ("pricing_capture_path", "pricing_capture_sha256"),
    ):
        if sha(amendment[key]) != amendment[expected_key]:
            raise ValueError("Original protocol or pricing capture changed")
    check_pricing(amendment["pricing_capture_path"])
    here = Path(__file__).resolve().parent
    if set(amendment["recovery_code_sha256"]) != set(CODE):
        raise ValueError("Incomplete recovery code pins")
    for name, expected in amendment["recovery_code_sha256"].items():
        if sha(here / name) != expected:
            raise ValueError("Approved recovery code changed")
    for relative, entry in amendment["initial_ledger_prefixes"].items():
        ledger = (runs_root / relative).resolve()
        if (
            not ledger.is_relative_to(runs_root)
            or prefix(ledger, entry["prefix_bytes"]) != entry["prefix_sha256"]
        ):
            raise ValueError("Original study ledger prefix changed")
    return amendment


class RecoveryBudget(runtime.StudyBudget):
    """Keep unknown actual costs intact and count reviewed allowances in the spend guard."""

    def __init__(
        self, runs_root, global_usd=5.0, arm_usd=0.65, max_calls=4000, reserve_usd=0.01, *, amendment_path
    ):
        super().__init__(runs_root, global_usd, arm_usd, max_calls, reserve_usd)
        if global_usd != 5.0 or arm_usd != 0.65:
            raise ValueError("Recovery cannot change the original cost caps")
        self.amendment_path = Path(amendment_path).resolve()
        self.amendment = validate_amendment(self.amendment_path, self.root)
        self.amendment_sha256 = sha(self.amendment_path)
        self._unknown_cache = {}

    def unknown_rows(self):
        for path in sorted(self.root.rglob("*.jsonl")):
            totals = self.ledger_totals(path)
            if not totals["unknown_cost_count"]:
                continue
            stat = path.stat()
            identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            cached = self._unknown_cache.get(path)
            if cached is None or cached[0] != identity:
                rows = [row for row in runtime.read_ledger(path) if row.get("effective_cost_usd") is None]
                self._unknown_cache[path] = identity, rows
            for row in self._unknown_cache[path][1]:
                yield path, row

    def receipt_path(self, ledger, row):
        key = digest(
            {
                "ledger": str(ledger.relative_to(self.root)),
                "sample": row["sample"],
                "request_sha256": row["request_sha256"],
            }
        )
        return self.root / "recovery-receipts" / (key + ".json")

    def approve_failures(self):
        """Apply only the explicit prospective amendment, retaining each raw record."""
        for ledger, row in self.unknown_rows():
            self.ledger_path(ledger, row.get("metadata_arm"))
            if not eligible(row, self.amendment["original_protocol_sha256"]):
                raise runtime.BudgetStop("Unapproved unknown request cost; recovery policy does not apply")
            path = self.receipt_path(ledger, row)
            if not path.exists():
                runtime.write_frozen(
                    path,
                    {
                        "version": 1,
                        "created_at": datetime.now(UTC).isoformat(),
                        "amendment_sha256": self.amendment_sha256,
                        "ledger_path": str(ledger.relative_to(self.root)),
                        "ledger_prefix_bytes": ledger.stat().st_size,
                        "ledger_prefix_sha256": sha(ledger),
                        "sample": row["sample"],
                        "request_sha256": row["request_sha256"],
                        "record_sha256": digest(row),
                        "metadata_arm": row["metadata_arm"],
                        "requested_model": row["requested_model"],
                        "allowance_usd": ALLOWANCE_USD,
                        "actual_cost_usd": None,
                        "no_retry": True,
                        "cost_interpretation": "Unverified guard allowance, not a provider charge",
                    },
                )
        return self.totals()

    def validate_receipt(self, ledger, row):
        if not eligible(row, self.amendment["original_protocol_sha256"]):
            raise runtime.BudgetStop("Unapproved unknown request cost; recovery policy does not apply")
        receipt = json.loads(self.receipt_path(ledger, row).read_text())
        expected = {
            "version": 1,
            "amendment_sha256": self.amendment_sha256,
            "ledger_path": str(ledger.relative_to(self.root)),
            "sample": row["sample"],
            "request_sha256": row["request_sha256"],
            "record_sha256": digest(row),
            "metadata_arm": row["metadata_arm"],
            "requested_model": row["requested_model"],
            "allowance_usd": ALLOWANCE_USD,
            "actual_cost_usd": None,
            "no_retry": True,
        }
        if any(receipt.get(key, "missing") != value for key, value in expected.items()):
            raise ValueError("Failure allowance receipt changed")
        if prefix(ledger, receipt["ledger_prefix_bytes"]) != receipt["ledger_prefix_sha256"]:
            raise ValueError("Reserved raw ledger prefix changed")
        return receipt

    def totals(self):
        if sha(self.amendment_path) != self.amendment_sha256:
            raise ValueError("Operational amendment changed")
        result = super().totals()
        reserved = 0.0
        for value in result["arms"].values():
            value["reserved_unverified_usd"] = 0.0
        for ledger, row in self.unknown_rows():
            try:
                self.validate_receipt(ledger, row)
            except FileNotFoundError as exc:
                raise runtime.BudgetStop(
                    "Unapproved unknown request cost; a frozen allowance receipt is required"
                ) from exc
            result["arms"][row["metadata_arm"]]["reserved_unverified_usd"] += ALLOWANCE_USD
            reserved += ALLOWANCE_USD
        for value in result["arms"].values():
            value["budget_accounted_usd"] = value["known_reported_usd"] + value["reserved_unverified_usd"]
        result.update(
            reserved_unverified_usd=reserved,
            budget_accounted_usd=result["known_reported_usd"] + reserved,
            amendment_sha256=self.amendment_sha256,
        )
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
        headroom = ALLOWANCE_USD if task_calls else self.reserve_usd
        if (
            totals["budget_accounted_usd"] + totals["reserved_inflight_usd"] + headroom > self.global_usd
            or totals["arms"][arm]["budget_accounted_usd"] + sum(e["reserved_usd"] for e in active) + headroom
            > self.arm_usd
        ):
            raise runtime.BudgetStop("Study or arm spend guard reached")
        return totals

    def begin(self, token, arm, kind, ledger):
        super().begin(token, arm, kind, ledger)
        if kind == "task":
            pending = self.pending()
            pending[token]["reserved_usd"] = ALLOWANCE_USD
            self.set_pending(pending)


def evaluate_recovered(samples, prompt, arm, output, budget, protocol_sha256, split, **kwargs):
    """Reuse frozen request checks and skip failed IDs; only the cost guard is amended."""
    if not isinstance(budget, RecoveryBudget):
        raise TypeError("Recovery evaluation requires the approved recovery budget")
    if protocol_sha256 != budget.amendment["original_protocol_sha256"]:
        raise ValueError("Recovery protocol identity changed")
    budget.approve_failures()
    while True:
        try:
            return runtime.evaluate(samples, prompt, arm, output, budget, protocol_sha256, split, **kwargs)
        except runtime.BudgetStop as exc:
            if not str(exc).startswith("Unknown request cost"):
                raise
            totals = budget.approve_failures()
            # Original evaluate already validated request/feature/prompt identities.
            records = runtime.read_ledger(output)
            by_id = {row["sample"]: row for row in records}
            if all(sample["id"] in by_id for sample in samples):
                if (
                    totals["budget_accounted_usd"] > budget.global_usd
                    or totals["arms"][arm]["budget_accounted_usd"] > budget.arm_usd
                ):
                    raise runtime.BudgetStop("Study or arm spend guard reached")
                metric_samples = [
                    {**sample, "labels": {"ai_generated": bool(sample["gold"])}} for sample in samples
                ]
                result = summarize("ai_detection", metric_samples, [by_id[s["id"]] for s in samples])
                result["summary"].update(track="metadata_v3", metadata_arm=arm)
                return {**result, "ledger_path": str(output), "ledger_records": records}


def allowance_summary(runs_root, amendment_path):
    """Read-only validator shared with reports; never creates failure receipts."""
    amendment = validate_amendment(amendment_path, runs_root)
    protocol = json.loads(Path(amendment["original_protocol_path"]).read_text())
    budget = RecoveryBudget(
        runs_root,
        max_calls=protocol["max_task_calls_per_arm"],
        reserve_usd=protocol["reserve_per_inflight_usd"],
        amendment_path=amendment_path,
    )
    return {"amendment": amendment, "costs": budget.totals()}
