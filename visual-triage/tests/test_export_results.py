"""Compact exports retain metrics and an exact link to each original attempt."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_archive import EvidenceError
from scripts.export_results import Sources, compact, export, export_report, sha, write_csv


def test_compact_preserves_unknown_costs_and_nested_intervals():
    original = {
        "summary": {"mae": 0.42, "ci": [0.1, 0.6], "actual_total_usd": None, "reserved_unverified_usd": 0.1},
        "panels": [{"summary": {"count": 80}, "per_image": [{"feedback": "large"}]}],
        "per_image": [{"prediction": 3}],
    }
    result = compact(original)
    assert result["summary"] == original["summary"]
    assert result["panels"] == [{"summary": {"count": 80}}]
    assert "per_image" in original


def test_export_keeps_failed_attempt_and_raw_reference(tmp_path):
    raw_record = {
        "sample": "x",
        "request_sha256": "request",
        "prompt_sha256": "prompt",
        "status": None,
        "transport_error": "ConnectionError",
        "arm": "decider",
    }
    raw = json.dumps(raw_record).encode()
    (tmp_path / "raw.jsonl").write_bytes(raw + b"\n")
    report = {
        "inputs": [{"ledger": "raw.jsonl", "sha256": sha(raw + b"\n")}],
        "evaluations": {
            "selected": {
                "summary": {"invalid": 1},
                "per_image": [
                    {
                        "id": "x",
                        "group_id": "g",
                        "request_sha256": "request",
                        "valid": False,
                        "attempted": True,
                        "gold": 1.5,
                        "prediction": None,
                        "cost_usd": None,
                        "feedback": {"large": "data"},
                        "source": {"dataset": "source", "id": "native"},
                    }
                ],
            }
        },
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    sources = Sources(tmp_path)
    aggregate, rows = export_report(sources, "report.json", "quality_v2", "primary")
    assert aggregate["evaluations"]["selected"] == {"summary": {"invalid": 1}}
    assert rows[0]["transport_error"] == "ConnectionError"
    assert rows[0]["raw_record_sha256"] == sha(raw)
    assert rows[0]["raw_line"] == 1
    assert rows[0]["gold"] == 1.5
    assert rows[0]["source_id"] == "native"
    assert rows[0]["cost_usd"] is None
    write_csv(tmp_path / "compact.csv", rows)
    assert "ConnectionError" in (tmp_path / "compact.csv").read_text()
    assert "large" not in (tmp_path / "compact.csv").read_text()
    assert b"\r\n" not in (tmp_path / "compact.csv").read_bytes()
    assert set(sources.inputs) == {"raw.jsonl", "report.json"}


def test_inputs_require_restore_and_matching_hash(tmp_path):
    sources = Sources(tmp_path)
    with pytest.raises(FileNotFoundError, match="Restore the evidence archive"):
        sources.read("missing.json")
    (tmp_path / "input").write_bytes(b"frozen")
    with pytest.raises(ValueError, match="hash mismatch"):
        sources.read("input", "changed")
    with pytest.raises(ValueError, match="outside experiment"):
        sources.read("../outside")


def test_export_rejects_missing_candidate_from_real_archive_index(tmp_path):
    relative = "tracks/quality_v2/optimisation/candidates/example/prompt.json"
    (tmp_path / relative).parent.mkdir(parents=True)
    original = b'{"context":"Original candidate prompt"}\n'
    index = {
        "version": 1,
        "archive_id": "evidence-v1",
        "source_commit": "b" * 40,
        "repository": "https://github.com/ahstn/model-gym",
        "image_bytes_included": False,
        "archive": {
            "url": "https://github.com/ahstn/model-gym/releases/download/evidence-v1/evidence.tar.gz",
            "size_bytes": 1,
            "sha256": "0" * 64,
        },
        "members": [
            {"path": relative, "size_bytes": len(original), "sha256": sha(original), "restore": True}
        ],
    }
    (tmp_path / "archive-index.json").write_text(json.dumps(index))
    output = tmp_path / "summaries"
    with pytest.raises(EvidenceError, match="python evidence_archive.py restore"):
        export(tmp_path, output)
    assert not output.exists()
