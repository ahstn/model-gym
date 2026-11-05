"""Checks for source-family/view preservation and content-allocation edge cases."""

import hashlib
import importlib.util
from pathlib import Path

import corpus_finalize
import numpy as np
import pytest
from corpus_select import sample
from PIL import Image

from visual_quality.views import _render


def test_sampling_handles_zero_and_capacity_without_duplicates():
    pool = [{"candidate_id": f"C{i}", "image_id": f"image-{i}"} for i in range(9)]
    clusters = {r["candidate_id"]: 0 if i < 8 else 1 for i, r in enumerate(pool)}
    assert sample(pool, 0, clusters, "zero", sqrt=True) == []
    picked = sample(pool, 9, clusters, "all", sqrt=True)
    assert len({r["candidate_id"] for r in picked}) == 9
    assert any(clusters[r["candidate_id"]] == 1 for r in sample(pool, 4, clusters, "balanced", sqrt=True))
    with pytest.raises(ValueError, match="insufficient"):
        sample(pool, 10, clusters, "overflow")


def test_remote_renderer_matches_canonical_function():
    module_path = Path(__file__).with_name("corpus_render_exact.py")
    spec = importlib.util.spec_from_file_location("render_exact_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import io

    pixels = np.random.default_rng(77).integers(0, 256, (320, 480, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    assert module._render(stream.getvalue(), 384) == _render(stream.getvalue(), 384)


def test_controlled_family_keeps_original_and_common_viewing_scale(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_finalize, "DATA", tmp_path)
    (tmp_path / "views").mkdir()
    source = tmp_path / "original.png"
    Image.fromarray(np.random.default_rng(12).integers(0, 256, (400, 520, 3), dtype=np.uint8)).save(source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    payload, _prep = _render(source.read_bytes(), 1536)
    parent = {
        "image_id": "source-123",
        "source_id": "123",
        "source": "example",
        "source_revision": "fixed",
        "source_path": str(source),
        "source_sha256": source_sha,
        "source_group": "family-123",
        "family_id": "family-123",
        "candidate_id": "C123",
        "view_sha256": hashlib.sha256(payload).hexdigest(),
        "license": {"id": "CC0-1.0"},
        "component": "megalith",
        "category": "megalith",
        "task_masks": {"technical_quality": False, "aesthetics": False, "photo_domain": False},
    }
    rows = corpus_finalize.family(parent)
    assert len(rows) == 4 and len({r["view_sha256"] for r in rows}) == 4
    assert len({(r["width"], r["height"]) for r in rows}) == 1
    assert {r["family_id"] for r in rows} == {"family-123"}
    assert {r["benchmark_partition"] for r in rows} == {"rehearsal"}
    assert all(r["benchmark_order"] is None for r in rows)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    assert all(r["delivered_bytes"] <= 2650000 for r in rows)
