"""Export derived compact results; restore the original evidence before running."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_archive import require_restored

TRACKS = ("quality", "ai_detection", "quality_v2", "ai_detection_v2")
NOTICE = "Derived export, not raw API evidence. Restore the versioned archive for response validation or resume. Empty CSV cells mean unavailable, not zero."
FIELDS = (
    "track",
    "phase",
    "panel",
    "evaluation",
    "sample",
    "group_id",
    "split",
    "source_dataset",
    "source_id",
    "gold_field",
    "gold",
    "gold_aesthetics",
    "technical_quality",
    "aesthetics",
    "ai_probability",
    "medium",
    "uncertainty",
    "attempted",
    "valid",
    "status",
    "validation_error",
    "transport_error",
    "reward",
    "correct",
    "error",
    "cost_usd",
    "effective_cost_usd",
    "cost_basis",
    "seconds",
    "arm",
    "model",
    "provider",
    "prompt_id",
    "prompt_sha256",
    "request_sha256",
    "delivered_sha256",
    "raw_ledger",
    "raw_line",
    "raw_record_sha256",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def compact(value):
    """Retain all aggregate values, including numeric metrics and intervals."""
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items() if key not in {"per_image", "feedback"}}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


class Sources:
    def __init__(self, root):
        self.root = root.resolve()
        self.inputs = {}

    def path(self, relative):
        relative = relative.removeprefix("visual-triage/")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Input outside experiment directory: {relative}")
        return path

    def read(self, relative, expected=None):
        path = self.path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Restore the evidence archive first; missing input: {relative}")
        raw = path.read_bytes()
        digest = sha(raw)
        if expected is not None and digest != expected:
            raise ValueError(f"Input hash mismatch: {relative}")
        self.inputs[str(path.relative_to(self.root))] = {"sha256": digest, "bytes": len(raw)}
        return raw

    def json(self, relative):
        return json.loads(self.read(relative))

    def ledgers(self, inputs):
        records = []
        for source in inputs:
            relative = source.get("ledger", source.get("path"))
            if not relative.endswith(".jsonl"):
                self.read(relative, source["sha256"])
                continue
            for line, raw in enumerate(self.read(relative, source["sha256"]).splitlines(), 1):
                if raw.strip():
                    records.append((json.loads(raw), relative.removeprefix("visual-triage/"), line, sha(raw)))
        return records


def csv_row(row, records, track, phase, panel, evaluation):
    prediction = row.get("prediction") or {}
    source = row.get("source") or {}
    sample = row.get("id", row.get("sample"))
    request = row.get("request_sha256")
    matches = [
        entry
        for entry in records
        if (
            entry[0].get("request_sha256") == request
            if request
            else entry[0].get("sample") == sample
            and entry[0].get("arm") == row.get("arm")
            and entry[0].get("prompt_id") == row.get("prompt_id")
            and entry[0].get("provider") == row.get("provider")
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one raw attempt for {track}/{phase}/{panel}/{evaluation}/{sample}, found {len(matches)}"
        )
    raw, ledger, line, record_sha = matches[0]
    human = row.get("human_reference") or {}
    result = {key: row.get(key) for key in FIELDS if key in row}
    result.update(
        {
            "track": track,
            "phase": phase,
            "panel": panel,
            "evaluation": evaluation,
            "sample": sample,
            "group_id": row.get("group_id", row.get("sample_group")),
            "split": row.get("split", row.get("partition")),
            "source_dataset": source.get("dataset", row.get("source_stratum")),
            "source_id": source.get("id", source.get("image_id", row.get("sample_group"))),
            "gold_field": row.get(
                "gold_field", "ai_generated" if row.get("task") == "ai" else "technical_quality"
            ),
            "gold": row.get(
                "gold", row.get("ai_generated") if row.get("task") == "ai" else human.get("technical_quality")
            ),
            "gold_aesthetics": human.get("aesthetics"),
            "attempted": row.get("attempted", True),
            "model": row.get("model", raw.get("resolved_model")),
            "raw_ledger": ledger,
            "raw_line": line,
            "raw_record_sha256": record_sha,
        }
    )
    for key in ("arm", "prompt_id", "prompt_sha256", "request_sha256", "status", "transport_error"):
        result[key] = raw.get(key)
    for key in ("technical_quality", "aesthetics", "ai_probability", "medium", "uncertainty"):
        result[key] = prediction.get(key)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_report(sources, relative, track, phase):
    report = sources.json(relative)
    rows = []
    panels = report.get("panels", {"test": report})
    if "external_validation" in report:
        panels = {**panels, "external": report["external_validation"]}
    for panel, content in panels.items():
        records = sources.ledgers(content["inputs"])
        for evaluation, values in content["evaluations"].items():
            rows.extend(csv_row(row, records, track, phase, panel, evaluation) for row in values["per_image"])
    return compact(report), rows


def export(root, output):
    require_restored(root)
    output.mkdir(parents=True, exist_ok=True)
    for track in TRACKS:
        sources = Sources(root)
        main, rows = export_report(sources, f"tracks/{track}/report/report.json", track, "primary")
        aggregate = {"primary": main}
        if track.endswith("_v2"):
            metadata, extra = export_report(
                sources, f"tracks/{track}/report/metadata-report.json", track, "metadata"
            )
            aggregate["metadata"] = metadata
            rows.extend(extra)
        aggregate["optimisation"] = compact(sources.json(f"tracks/{track}/optimisation/summary.json"))
        aggregate.update(
            {"export_version": 1, "track": track, "notice": NOTICE, "source_inputs": sources.inputs}
        )
        write_json(output / f"{track}.json", aggregate)
        write_csv(output / f"{track}.csv", rows)
    sources = Sources(root)
    benchmark = sources.json("evidence/benchmarks/summary.json")
    records = sources.ledgers(benchmark["inputs"])
    rows = [
        csv_row(row, records, "initial_benchmark", "primary", row["benchmark"], row["arm"])
        for row in benchmark["per_image"]
    ]
    write_json(
        output / "initial_benchmark.json",
        {
            "export_version": 1,
            "notice": NOTICE,
            "source_inputs": sources.inputs,
            "results": compact(benchmark),
        },
    )
    write_csv(output / "initial_benchmark.csv", rows)
    history = []
    sources = Sources(root)
    for track in TRACKS:
        directory = root / "tracks" / track / "optimisation" / "candidates"
        if not directory.is_dir():
            raise FileNotFoundError(f"Restore the candidate archive first: {directory}")
        for path in sorted(directory.glob("*/prompt.json")):
            relative = str(path.relative_to(root))
            raw = sources.read(relative)
            history.append(
                {
                    "track": track,
                    "original_path": relative,
                    "original_sha256": sha(raw),
                    "prompt": json.loads(raw),
                }
            )
    (output / "candidate-prompts.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in history)
    )
    write_json(
        output / "candidate-prompts-index.json",
        {"export_version": 1, "notice": NOTICE, "count": len(history), "source_inputs": sources.inputs},
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        export(args.root.resolve(), args.output or args.root / "summaries")
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")


if __name__ == "__main__":
    main()
