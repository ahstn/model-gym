"""Continue missing Lite requests, retaining every attempt and frozen reference admission."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01")
ROOT = DATA / "results/lite"
OPS = DATA / "lite-operations"
URL = "https://ns8sd5o8i0qkal-8888.proxy.runpod.net"
TERMINAL = {"valid", "partial", "invalid"}


def histories():
    grouped = defaultdict(list)
    for path in ROOT.glob("*/attempts/*.json"):
        row = json.loads(path.read_text())
        grouped[row["case_id"]].append(row)
    return grouped


def retryable(rows):
    if len(rows) >= 3 or any(row["status"] in TERMINAL for row in rows):
        return False
    return all(
        row["status"] == "transport_or_parse_error"
        or (
            row["status"] == "http_error"
            and row.get("http_status") in {429, 500, 502, 503, 504, 520, 522, 524}
        )
        for row in rows
    )


def main():
    deadline = time.monotonic() + 8 * 3600
    initial = ROOT / "rehearsal-continuation-01/summary.json"
    while not initial.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Initial Lite continuation did not finish")
        time.sleep(10)
    manifest = json.loads((DATA / "manifest.json").read_text())
    for partition in ("rehearsal", "benchmark"):
        cases = [
            c
            for c in manifest["cases"]
            if int(c["case_id"][1:]) > 4000 and (bool(c.get("benchmark_order")) == (partition == "benchmark"))
        ]
        if partition == "benchmark":
            while not (DATA / "references-freeze.json").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Independent reference freeze not ready")
                time.sleep(10)
        sequence = 1
        while True:
            history = histories()
            pending = [c["case_id"] for c in cases if retryable(history[c["case_id"]])]
            if not pending:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Lite queue deadline reached")
            while (ROOT / f"queued-{partition}-{sequence:02}").exists():
                sequence += 1
            name = f"queued-{partition}-{sequence:02}"
            selection = OPS / f"{name}-ids.json"
            selection.write_text(json.dumps(pending) + "\n")
            command = [
                sys.executable,
                str(HERE / "lite_run.py"),
                "--manifest",
                str(DATA / "manifest.json"),
                "--output",
                str(ROOT / name),
                "--partition",
                partition,
                "--case-ids",
                str(selection),
                "--concurrency",
                "3",
                "--url",
                URL,
            ]
            if partition == "benchmark":
                command += ["--references-freeze", str(DATA / "references-freeze.json")]
            print(
                json.dumps({"phase": name, "selected": len(pending), "at": datetime.now(UTC).isoformat()}),
                flush=True,
            )
            with (OPS / f"{name}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode and not (ROOT / name / "summary.json").exists():
                raise RuntimeError(f"Lite phase failed before a durable summary: {name}")
            time.sleep(5)
        history = histories()
        status = {
            "partition": partition,
            "cases": len(cases),
            "valid": sum(any(r["status"] == "valid" for r in history[c["case_id"]]) for c in cases),
            "unresolved_case_ids": [
                c["case_id"] for c in cases if not any(r["status"] == "valid" for r in history[c["case_id"]])
            ],
            "updated_at": datetime.now(UTC).isoformat(),
        }
        (OPS / f"{partition}-queue-summary.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
