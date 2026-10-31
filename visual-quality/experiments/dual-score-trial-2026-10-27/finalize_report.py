"""Assemble frozen quantitative results and separate post-inference audits."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    names = [
        "results.json",
        "explanation-audit-summary.json",
        "billing-summary.json",
        "usage-audit.json",
        "manifest.json",
        "references.jsonl",
        "references-freeze.json",
        "comparisons.csv",
        "disagreement-review.json",
    ]
    loaded = {name: json.loads((HERE / name).read_text()) for name in names if name.endswith(".json")}
    result = {
        "assembled_at": datetime.now(UTC).isoformat(),
        "reference_kind": "frozen_assistant_judgments",
        "human_gold": False,
        "training_ready": False,
        "quantitative": loaded["results.json"],
        "explanation_audit": loaded["explanation-audit-summary.json"],
        "billing": loaded["billing-summary.json"],
        "generation_metadata_audit": loaded["usage-audit.json"],
        "posthoc_disagreement_review": loaded["disagreement-review.json"],
        "inputs": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names},
    }
    (HERE / "complete-report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"saved": str(HERE / "complete-report.json"), "inputs": len(names)}))


if __name__ == "__main__":
    main()
