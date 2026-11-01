"""Archive stopped-run receipts with hash and credential checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "rehearsal-4000-2026-11-01"
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-expanded-120-2026-11-01")
sys.path.insert(0, str(OLD))
import package_records as p


def main():
    for file in (HERE / "operations").glob("*.status.txt"):
        s = file.read_text()
        assert "ActiveState=inactive" in s and "Result=success" in s and "ExecMainStatus=0" in s
    assert len(list((HERE / "operations").glob("*.status.txt"))) == 3
    x = p.package(
        {"expanded": DATA / "attempts"},
        HERE / "execution-records",
        all_runs_stopped=True,
        secrets=p.credentials(),
    )
    print(
        json.dumps(
            {
                "files": x["credentials_scan_status"]["files_scanned"],
                "credential_scan": x["credentials_scan_status"]["status"],
                "archive_bytes": x["archives"][0]["archive_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
