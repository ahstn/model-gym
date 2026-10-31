"""Publish all attempts, selected responses and immutable recovery evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import continuation

from visual_quality.storage import atomic_json, file_sha256, write_jsonl

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / HERE.name


def main():
    manifest = {"schema_version": "dual-high-2000-publication-v2", "routes": {}}
    for route in ("deepseek", "luna"):
        run, report, rows, pins = continuation.load_verified_route(route)
        if report["status"] != "complete" or report["selected_response_calls"] != 2000:
            raise ValueError("Only complete 2000-response routes can be published")
        attempts = report["all_attempts"]
        selected = [attempt for attempt in attempts if attempt["selected"]]
        if (
            len(rows) != 2000
            or len(selected) != 2000
            or len({a["attempt_id"] for a in attempts}) != len(attempts)
            or {a["attempt_id"] for a in selected} != {row["attempt_id"] for row in rows}
            or {a["case_id"] for a in selected} != {c["case_id"] for c in run["cases"]}
        ):
            raise ValueError("Selected responses must cover exactly the 2000 unique frozen cases")
        target = HERE / "results" / route
        target.mkdir(parents=True, exist_ok=True)
        atomic_json(target / "report.json", report)
        write_jsonl(target / "observations.jsonl", rows)
        # Keep the actual source snapshots as well as normalized aggregate exports.
        for filename, expected in pins.items():
            source = Path(filename)
            if file_sha256(source) != expected:
                raise ValueError("Verified source changed before export")
            relative = source.relative_to(DATA / route)
            destination = target / "source-evidence" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if file_sha256(destination) != expected:
                raise ValueError("Source snapshot export differs")
        with (target / "attempts.jsonl").open("w") as handle:
            for attempt in attempts:
                request = json.loads(Path(attempt["request_path"]).read_text())
                result = json.loads(Path(attempt["result_path"]).read_text())
                if "messages" in request or "Authorization" in request:
                    raise ValueError("Unexpected full payload or credentials in journal")
                handle.write(
                    json.dumps(
                        {"ledger": attempt, "request": request, "result": result},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        with (target / "attempts.jsonl").open() as handle:
            for line, attempt in zip(handle, attempts, strict=True):
                exported = json.loads(line)
                original = json.loads(Path(attempt["result_path"]).read_text())
                if exported["result"] != original or exported["ledger"] != attempt:
                    raise ValueError("Raw response or attempt ledger export differs")
        files = {
            str(path.relative_to(target)): file_sha256(path)
            for path in sorted(target.rglob("*"))
            if path.is_file()
        }
        manifest["routes"][route] = {
            "http_attempts": len(attempts),
            "selected_responses": len(selected),
            "unpriced_attempts": report["unpriced_attempts"],
            "files_sha256": files,
        }
    manifest.update(
        total_http_attempts=sum(route["http_attempts"] for route in manifest["routes"].values()),
        total_selected_responses=sum(route["selected_responses"] for route in manifest["routes"].values()),
        raw_response_strings_unchanged=True,
        originals="The 72 selected delivered originals are in collage/images; bulk inputs remain locally cached.",
        metadata="Absolute source paths remain as provenance; gallery image links are checkout-relative.",
    )
    atomic_json(HERE / "publication-manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "routes"}, indent=2))


if __name__ == "__main__":
    main()
