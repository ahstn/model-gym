"""Test selection and immutable-export boundaries without paid API calls."""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import corpus  # noqa: F401 -- establishes the frozen adapter import context
import selection

spec = importlib.util.spec_from_file_location("expansion_export", HERE / "export.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_balancing_accounts_for_existing_content():
    pool = [{"candidate_id": f"{k}-{i}", "image_id": f"{k}-{i}"} for k in range(2) for i in range(10)]
    clusters = {r["candidate_id"]: int(r["candidate_id"][0]) for r in pool}
    old = [{"content_cluster": 0} for _ in range(100)]
    picked = selection.quotas(pool, 8, old, clusters, "test")
    assert len(picked) == 8 and all(clusters[r["candidate_id"]] == 1 for r in picked)
    assert picked == selection.quotas(list(reversed(pool)), 8, old, clusters, "test")


def test_quality_scores_do_not_change_selection():
    pool = [{"candidate_id": str(i), "image_id": str(i)} for i in range(20)]
    clusters = {str(i): i % 3 for i in range(20)}
    first = selection.quotas(pool, 10, [], clusters, "test")
    for i, row in enumerate(pool):
        row.update(technical_score=1 + i % 5, aesthetic_score=5 - i % 5)
    second = selection.quotas(pool, 10, [], clusters, "test")
    assert [r["image_id"] for r in first] == [r["image_id"] for r in second]


def test_append_preserves_original_bytes(tmp_path):
    old = tmp_path / "old.jsonl"
    old.write_bytes(b'{"old": 1, "score": 3.0000}\n')
    output = tmp_path / "combined.jsonl"
    export.append_export(old, [{"new": 2}], output)
    assert output.read_bytes().startswith(old.read_bytes())
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"old": 1, "score": 3.0},
        {"new": 2},
    ]


def test_expansion_has_no_lite_route():
    import score

    assert set(score.POLICY) == {"luna", "deepseek", "liquid", "jev"}
    assert export.MODELS == ("luna", "deepseek", "liquid", "jev")
