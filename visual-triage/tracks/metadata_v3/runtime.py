"""Append-only Decider transport and shared cost guards for the metadata study."""

import copy
import fcntl
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from bench_protocol import digest, image_part, make_request, number, validate_prompt
from bench_runner import effective_cost, post, response_record
from tracks.metrics import summarize

ARMS = ("image_only", "resolution", "camera", "tone", "texture", "provenance", "all_context")


class BudgetStop(RuntimeError):
    """Stop before another attempt when costs or available funds are unknown."""


def read_ledger(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_frozen(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def context_for(sample, arm):
    from tracks.metadata_v3.features import assemble_context, validate_features

    if arm not in ARMS:
        raise ValueError("Unknown metadata arm")
    features = sample["features"]
    validate_features(features)
    if digest(features) != sample["features_sha256"]:
        raise ValueError("Frozen feature hash changed")
    return assemble_context(features, arm)


def feature_text(sample, arm):
    from tracks.metadata_v3.features import context_text

    context_for(sample, arm)
    return context_text(sample["features"], arm)


class StudyBudget:
    """Count all raw task and reflection ledgers, with one lock across study arms."""

    def __init__(self, runs_root, global_usd=5.0, arm_usd=0.65, max_calls=4000, reserve_usd=0.01):
        self.root = Path(runs_root).resolve()
        self.global_usd = number(global_usd, 0.01, 5.0)
        self.arm_usd = number(arm_usd, 0.01, self.global_usd)
        self.reserve_usd = number(reserve_usd, 0.000001, self.arm_usd)
        if type(max_calls) is not int or not 1 <= max_calls <= 10000:
            raise ValueError("Invalid task call cap")
        self.max_calls = max_calls
        self._ledger_cache = {}

    def ledger_path(self, path, arm):
        path = Path(path).resolve()
        if arm not in ARMS or not path.is_relative_to(self.root / arm) or path.suffix != ".jsonl":
            raise ValueError("Ledger must be inside this study arm")
        return path

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".budget.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def coordinator(self):
        """Only one process schedules requests; HTTP workers never hold the budget lock."""
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".coordinator.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BudgetStop("Another process owns the study coordinator") from exc
            try:
                if self.pending():
                    raise BudgetStop(
                        "Unresolved in-flight requests require explicit cost reconciliation before resume"
                    )
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def pending(self):
        path = self.root / ".inflight.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def set_pending(self, pending):
        path = self.root / ".inflight.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            handle.write(json.dumps(pending, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def begin(self, token, arm, kind, ledger):
        pending = self.pending()
        if token in pending:
            raise ValueError("Duplicate active request identity")
        pending[token] = {
            "metadata_arm": arm,
            "record_type": kind,
            "ledger": str(ledger),
            "reserved_usd": self.reserve_usd,
        }
        self.set_pending(pending)

    def finish(self, token):
        pending = self.pending()
        del pending[token]
        self.set_pending(pending)

    def ledger_totals(self, path):
        """Reparse changed raw files; each new budget instance audits every ledger."""
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._ledger_cache.get(path)
        if cached is not None and cached[0] == identity:
            return cached[1]
        result = {
            "arm": None,
            "task_calls": 0,
            "reflection_calls": 0,
            "known_reported_usd": 0.0,
            "unknown_cost_count": 0,
        }
        for row in read_ledger(path):
            arm = row.get("metadata_arm")
            self.ledger_path(path, arm)
            if result["arm"] not in (None, arm):
                raise ValueError("A ledger must contain one metadata arm")
            result["arm"] = arm
            kind = row.get("record_type")
            if kind not in ("task", "reflection"):
                raise ValueError("Unknown ledger record type")
            result[kind + "_calls"] += 1
            cost = row.get("effective_cost_usd")
            if cost is None:
                result["unknown_cost_count"] += 1
            else:
                result["known_reported_usd"] += number(cost, 0, float("inf"))
        self._ledger_cache[path] = identity, result
        return result

    def totals(self):
        arms = {arm: {"task_calls": 0, "reflection_calls": 0, "known_reported_usd": 0.0} for arm in ARMS}
        unknown = 0
        for path in sorted(self.root.rglob("*.jsonl")):
            totals = self.ledger_totals(path)
            if totals["arm"] is None:
                continue
            for field in ("task_calls", "reflection_calls", "known_reported_usd"):
                arms[totals["arm"]][field] += totals[field]
            unknown += totals["unknown_cost_count"]
        known = math.fsum(value["known_reported_usd"] for value in arms.values())
        pending = self.pending()
        for entry in pending.values():
            self.ledger_path(entry["ledger"], entry["metadata_arm"])
            number(entry["reserved_usd"], 0, self.global_usd)
        return {
            "arms": arms,
            "known_reported_usd": known,
            "effective_cost_usd": None if unknown else known,
            "unknown_cost_count": unknown,
            "global_budget_usd": self.global_usd,
            "arm_budget_usd": self.arm_usd,
            "inflight": pending,
            "reserved_inflight_usd": math.fsum(entry["reserved_usd"] for entry in pending.values()),
        }

    def check(self, arm, task_calls=0):
        if arm not in ARMS:
            raise ValueError("Unknown metadata arm")
        totals = self.totals()
        if totals["unknown_cost_count"]:
            raise BudgetStop("Unknown request cost; no automatic retry or further spending")
        current = totals["arms"][arm]
        active = [entry for entry in totals["inflight"].values() if entry["metadata_arm"] == arm]
        active_tasks = sum(entry["record_type"] == "task" for entry in active)
        if current["task_calls"] + active_tasks + task_calls > self.max_calls:
            raise BudgetStop("Arm task call cap reached")
        if (
            totals["known_reported_usd"] + totals["reserved_inflight_usd"] + self.reserve_usd
            > self.global_usd
            or current["known_reported_usd"]
            + sum(entry["reserved_usd"] for entry in active)
            + self.reserve_usd
            > self.arm_usd
        ):
            raise BudgetStop("Study or arm spend guard reached")
        return totals


def require_all_selections(budget, protocol_sha256):
    """No test calls until every arm has a frozen selection for this protocol."""
    for arm in ARMS:
        receipt = budget.root / arm / "selection.json"
        selected = json.loads(receipt.read_text())
        if selected.get("metadata_arm") != arm or selected.get("protocol_sha256") != protocol_sha256:
            raise ValueError("All arm selections must match the frozen protocol")
        prompt_path = Path(selected["selected_prompt_path"]).resolve()
        if not prompt_path.is_relative_to(budget.root / arm):
            raise ValueError("Selected prompt is outside its arm")
        prompt = json.loads(prompt_path.read_text())
        if digest(prompt) != selected["selected_prompt_sha256"]:
            raise ValueError("Selected prompt changed")


def evaluate(samples, prompt, arm, output, budget, protocol_sha256, split, post_fn=post, concurrency=1):
    """Evaluate one attempt per image; failures retain zero reward and cannot be retried."""
    output = budget.ledger_path(output, arm)
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise ValueError("Use 1..4 request workers")
    validate_prompt(prompt, "ai")
    if split not in ("train", "dev", "test", "external") or not samples:
        raise ValueError("Expected a populated train/dev/test/external split")
    if split in ("test", "external"):
        require_all_selections(budget, protocol_sha256)
        if not any(output.is_relative_to(budget.root / arm / name) for name in ("final", "controls")):
            raise ValueError("Final ledgers must be under arm/final or arm/controls")
    elif any(output.is_relative_to(budget.root / arm / name) for name in ("final", "controls")):
        raise ValueError("Training ledgers cannot use the final directory")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    requests = {}
    metric_samples = []
    for sample in samples:
        if (
            sample["split"] != split
            or type(sample["gold"]) not in (bool, int)
            or sample["gold"] not in (0, 1)
        ):
            raise ValueError("Wrong split or binary gold")
        context = context_for(sample, arm)
        request_prompt = copy.deepcopy(prompt)
        text = feature_text(sample, arm)
        if text:
            request_prompt["context"] += "\n\n" + text
        endpoint, payload = make_request("decider", image_part(sample), request_prompt, "low", 4096)
        requests[sample["id"]] = (endpoint, payload, digest(context))
        metric_samples.append({**sample, "labels": {"ai_generated": bool(sample["gold"])}})
    with budget.coordinator():
        records = read_ledger(output)
        done = {}
        for row in records:
            identifier = row["sample"]
            if identifier in done:
                raise ValueError("Duplicate ledger attempt")
            if row.get("metadata_arm") != arm or row.get("protocol_sha256") != protocol_sha256:
                raise ValueError("Ledger arm or protocol changed")
            if row.get("prompt_sha256") != digest(prompt) or row.get("split") != split:
                raise ValueError("Ledger prompt or split changed")
            if identifier in requests:
                endpoint, payload, context_hash = requests[identifier]
                if row["request_sha256"] != digest({"endpoint": endpoint, "payload": payload}):
                    raise ValueError("Frozen request changed")
                if row["context_sha256"] != context_hash:
                    raise ValueError("Frozen arm context changed")
                sample = next(sample for sample in samples if sample["id"] == identifier)
                if (
                    row.get("features_sha256") != sample["features_sha256"]
                    or row.get("delivered_sha256") != sample["delivered_sha256"]
                ):
                    raise ValueError("Frozen feature or image hash changed")
            done[identifier] = row
        queue = iter(sample for sample in samples if sample["id"] not in done)
        active, exhausted, stopped = {}, False, None

        def send(endpoint, payload):
            started = time.perf_counter()
            try:
                return post_fn(endpoint, payload)
            except Exception as exc:  # noqa: BLE001 - retain unexpected worker failures as unknown-cost attempts
                return {"transport_error": type(exc).__name__, "seconds": time.perf_counter() - started}

        with ThreadPoolExecutor(max_workers=concurrency) as workers:
            while not exhausted or active:
                while not exhausted and stopped is None and len(active) < concurrency:
                    sample = next(queue, None)
                    if sample is None:
                        exhausted = True
                        break
                    try:
                        budget.check(arm, task_calls=1)
                    except BudgetStop as exc:
                        stopped = exc
                        break
                    endpoint, payload, context_hash = requests[sample["id"]]
                    request_hash = digest({"endpoint": endpoint, "payload": payload})
                    token = digest({"arm": arm, "sample": sample["id"], "request": request_hash})
                    budget.begin(token, arm, "task", output)
                    future = workers.submit(send, endpoint, payload)
                    active[future] = (sample, context_hash, request_hash, token)
                if not active:
                    break
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    sample, context_hash, request_hash, token = active.pop(future)
                    row = future.result()
                    row.update(
                        record_type="task",
                        metadata_arm=arm,
                        task="ai",
                        arm="decider",
                        split=split,
                        sample=sample["id"],
                        timestamp=datetime.now(UTC).isoformat(),
                        protocol_sha256=protocol_sha256,
                        prompt_sha256=digest(prompt),
                        features_sha256=sample["features_sha256"],
                        context_sha256=context_hash,
                        delivered_sha256=sample["delivered_sha256"],
                        request_sha256=request_hash,
                        requested_model=requests[sample["id"]][1]["model"],
                    )
                    response_record(row, "decider", prompt, "low")
                    append_record(output, row)
                    budget.finish(token)
                    records.append(row)
                    done[sample["id"]] = row
                    if row["effective_cost_usd"] is None:
                        stopped = BudgetStop("Unknown request cost; recorded attempt will not be retried")
                if stopped is not None:
                    exhausted = True
            if stopped is not None:
                raise stopped
        # Check unknown costs even when all requested samples were already cached.
        if budget.totals()["unknown_cost_count"]:
            raise BudgetStop("Unknown request cost; no automatic retry or further spending")
    result = summarize("ai_detection", metric_samples, [done[s["id"]] for s in samples])
    result["summary"].update(track="metadata_v3", metadata_arm=arm)
    return {**result, "ledger_path": str(output), "ledger_records": records}


def reflection_cost(record):
    response = record.get("response")
    response = response if isinstance(response, dict) else {}
    usage = response.get("usage") or {}
    usage = usage if isinstance(usage, dict) else {}
    record.update(usage=usage, generation_id=response.get("id"), resolved_model=response.get("model"))
    try:
        cost, basis = effective_cost(usage)
    except (TypeError, ValueError):
        cost, basis = None, "invalid_cost"
    record.update(cost_usd=usage.get("cost"), effective_cost_usd=cost, cost_basis=basis)
    return record
