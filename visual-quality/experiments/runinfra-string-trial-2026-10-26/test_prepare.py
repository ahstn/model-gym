"""Checks for image-budget invariants and fresh-family exclusion."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("trial_prepare", Path(__file__).with_name("prepare.py"))
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_budget_rerenders_all_members_from_original_bytes(monkeypatch):
    calls = []

    def render(payload, edge):
        calls.append((payload, edge))
        size = edge * (2 if payload == b"noisy" else 1)
        return b"p" * size, {"output_dimensions": [edge, edge // 2]}

    monkeypatch.setattr(prepare.views, "_render", render)
    edge, rendered = prepare.render_family(
        [b"parent", b"noisy", b"blur", b"jpeg"], max_edge=100, max_bytes=180
    )
    assert edge == 90
    assert all(len(data) <= 180 for data, _ in rendered)
    assert calls == [
        (payload, edge) for edge in (100, 90) for payload in (b"parent", b"noisy", b"blur", b"jpeg")
    ]
    assert {tuple(metadata["output_dimensions"]) for _, metadata in rendered} == {(90, 45)}


def test_budget_does_not_resize_a_family_that_already_fits(monkeypatch):
    monkeypatch.setattr(prepare.views, "_render", lambda payload, edge: (payload, {"edge": edge}))
    edge, rendered = prepare.render_family([b"one", b"two"], max_edge=1536, max_bytes=3)
    assert edge == 1536
    assert [data for data, _ in rendered] == [b"one", b"two"]


def test_impossible_minimum_budget_fails(monkeypatch):
    monkeypatch.setattr(prepare.views, "_render", lambda payload, edge: (b"xx", {}))
    with pytest.raises(ValueError, match="minimum-size"):
        prepare.render_family([b"source"], max_edge=64, max_bytes=1)


def test_exclusion_checks_family_exact_pixels_and_near_duplicate():
    old = {"split_group": "old", "sha256": "a", "decoded_sha256": "b", "dhash": "0000"}
    fresh = {"split_group": "new", "sha256": "c", "decoded_sha256": "d", "dhash": "ffff"}
    assert not prepare.overlaps([fresh], [old])
    for key in prepare.IDENTITY_KEYS:
        assert prepare.overlaps([{**fresh, key: old[key]}], [old])
    assert prepare.overlaps([{**fresh, "dhash": "0003"}], [old])
    assert not prepare.overlaps([{**fresh, "dhash": "0007"}], [old])


def test_freeze_reuses_exact_bytes_and_refuses_overwrite(tmp_path):
    path = tmp_path / "frozen.json"
    prepare.freeze(path, b"first")
    prepare.freeze(path, b"first")
    with pytest.raises(ValueError, match="Frozen artifact differs"):
        prepare.freeze(path, b"second")
    assert path.read_bytes() == b"first"


def test_packets_are_anonymous_complete_and_stably_shuffled(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "OUTPUT", tmp_path)
    cases = [
        {
            "case_id": f"T{i:03d}",
            "image_path": f"/views/{i}.png",
            "view_sha256": str(i),
            "width": 800,
            "height": 600,
            "stratum": "secret",
        }
        for i in range(1, 201)
    ]
    paths = prepare.packets(cases)
    rows = [prepare.json.loads(Path(path).read_text()) for path in paths]
    assert [len(packet) for packet in rows] == [67, 67, 66]
    flat = [row for packet in rows for row in packet]
    assert {row["case_id"] for row in flat} == {case["case_id"] for case in cases}
    assert all(set(row) == {"case_id", "path", "view_sha256", "width", "height"} for row in flat)
    assert flat[0]["case_id"] != cases[0]["case_id"]
    assert prepare.packets(cases) == paths
