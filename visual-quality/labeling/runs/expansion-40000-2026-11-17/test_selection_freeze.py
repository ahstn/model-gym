"""Check compatible family parents and restartable publication without inference."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def modules(monkeypatch):
    loaded = {}
    for name in ("corpus", "selection"):
        spec = importlib.util.spec_from_file_location(f"freeze_40000_{name}", HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def record(identity, **changes):
    return {
        "image_id": identity,
        "candidate_id": identity,
        "case_id": identity,
        "family_id": identity,
        "source_id": identity,
        "source_sha256": "source-" + identity,
        "view_sha256": "view-" + identity,
        "decoded_sha256": "pixels-" + identity,
        "dhash": "0123456789abcdef",
        "category": "megalith",
        "source_metadata": {"icc": False, "alpha": False},
        **changes,
    }


def test_family_parent_filter_preserves_all_base_images(modules, monkeypatch, tmp_path):
    selection = modules["selection"]
    chosen = [
        record("icc", source_metadata={"icc": True, "alpha": False}),
        record("alpha", source_metadata={"icc": False, "alpha": True}),
        record("unknown", source_metadata={}),
        record("plain"),
    ]
    monkeypatch.setattr(selection, "DATA", tmp_path)
    monkeypatch.setattr(selection, "sample", lambda pool, count, *args, **kwargs: pool[:count])
    monkeypatch.setattr(selection, "historical_matcher", lambda old: lambda row: None)
    rendered = []

    def family(parent):
        rendered.append(parent["image_id"])
        return [dict(parent)] + [record(f"variant{i}", family_id=parent["family_id"]) for i in range(3)]

    monkeypatch.setattr(selection.legacy, "family", family)
    final = selection.generate_families(chosen, {}, [], count=1)
    assert rendered == ["plain"]
    assert len(final) == 7
    for name in ("icc", "alpha", "unknown"):
        assert next(r for r in final if r["image_id"] == name) == next(
            r for r in chosen if r["image_id"] == name
        )


@pytest.fixture
def prepared_stage(modules, monkeypatch, tmp_path):
    corpus, selection = modules["corpus"], modules["selection"]
    data, old_root = tmp_path / "new", tmp_path / "old"
    data.mkdir()
    old_root.mkdir()
    monkeypatch.setattr(corpus, "DATA", data)
    monkeypatch.setattr(selection, "DATA", data)
    monkeypatch.setattr(selection, "OLD", old_root)
    monkeypatch.setattr(corpus, "pin_initial_evidence", dict)
    candidate = data / "candidate-manifest.json"
    candidate.write_text('{"cases": ["frozen candidate input"]}\n')
    (old_root / "cluster-prototypes.npz").write_bytes(b"unchanged frozen prototypes")
    vector = np.zeros(768, dtype=np.float32)
    vector[0] = 1
    cache = data / "embedding-cache"
    cache.mkdir()
    np.save(cache / "new-view.npy", vector)
    old = [{"case_id": "old", "view_sha256": "old-view"}]
    manifest = {
        "status": "frozen",
        "old_manifest_sha256": "frozen-old-manifest",
        "candidate_manifest_sha256": corpus.digest(candidate),
        "cases": [{"case_id": "new", "view_sha256": "new-view", "embedding_view_sha256": "new-view"}],
    }
    args = (manifest, old, {"clusters": []}, {"pairs": []}, vector[None, :], np.asarray(["old-view"]))
    return selection, corpus, data, old_root, args


def test_failed_embedding_archive_leaves_selection_unpublished(prepared_stage, monkeypatch):
    selection, _, data, _, args = prepared_stage

    def fail(*args, **kwargs):
        raise OSError("simulated archive write failure")

    monkeypatch.setattr(selection.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="archive write failure"):
        selection.publish_selection(*args)
    assert not (data / "manifest.json").exists()
    assert not (data / "combined-manifest.json").exists()
    assert not list(data.glob(".selection-*"))


def test_candidate_advance_rejects_staged_selection(prepared_stage, monkeypatch):
    selection, _, data, _, args = prepared_stage
    original = selection.np.savez_compressed

    def advance(*values, **kwargs):
        original(*values, **kwargs)
        (data / "candidate-manifest.json").write_text('{"cases": ["newer frame"]}\n')

    monkeypatch.setattr(selection.np, "savez_compressed", advance)
    with pytest.raises(ValueError, match="Candidate frame changed"):
        selection.publish_selection(*args)
    assert not (data / "manifest.json").exists()
    assert not (data / "combined-manifest.json").exists()


def test_partial_publication_can_restart_and_preserves_old_prefix(prepared_stage, monkeypatch):
    selection, corpus, data, old_root, args = prepared_stage
    original = Path.replace

    def fail_mid_commit(path, target):
        if path.name == "content-coverage.json" and path.parent.name.startswith(".selection-"):
            raise OSError("simulated publication interruption")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_mid_commit)
    with pytest.raises(OSError, match="publication interruption"):
        selection.publish_selection(*args)
    assert (data / "combined-manifest.json").exists()
    assert not (data / "manifest.json").exists()
    monkeypatch.setattr(Path, "replace", original)
    selection.publish_selection(*args)
    combined = json.loads((data / "combined-manifest.json").read_text())
    assert combined["cases"] == args[1] + args[0]["cases"]
    assert combined["increment_manifest_sha256"] == corpus.digest(data / "manifest.json")
    assert (data / "cluster-prototypes.npz").read_bytes() == (
        old_root / "cluster-prototypes.npz"
    ).read_bytes()
    with np.load(data / "corpus-embeddings.npz", allow_pickle=False) as saved:
        assert saved["case_ids"].tolist() == ["old", "new"]
        np.testing.assert_array_equal(saved["embeddings"][:1], args[4])
        np.testing.assert_array_equal(saved["embedding_view_sha256"][:1], args[5])


def test_manifest_is_published_after_every_dependent_artifact(prepared_stage, monkeypatch):
    selection, _, data, _, args = prepared_stage
    original, published = Path.replace, []
    dependencies = {
        "combined-manifest.json",
        "content-coverage.json",
        "candidate-related-review.json",
        "cluster-prototypes.npz",
        "corpus-embeddings.npz",
    }

    def observe(path, target):
        target = Path(target)
        if target.parent == data and path.parent.name.startswith(".selection-"):
            if target.name == "manifest.json":
                assert set(published) == dependencies
                assert all((data / name).is_file() for name in dependencies)
            published.append(target.name)
        return original(path, target)

    monkeypatch.setattr(Path, "replace", observe)
    selection.publish_selection(*args)
    assert published[-1] == "manifest.json"


def test_inflight_prepare_rechecks_freeze_before_candidate_write(modules, monkeypatch, tmp_path):
    corpus = modules["corpus"]
    monkeypatch.setattr(corpus, "DATA", tmp_path)
    monkeypatch.setattr(corpus, "old_cases", list)
    monkeypatch.setattr(corpus, "pin_initial_evidence", dict)
    candidate = tmp_path / "candidate-manifest.json"
    candidate.write_bytes(b"previous frame")
    input_row = {"id": "fresh", "image_path": "unused"}
    monkeypatch.setattr(
        corpus, "read_lines", lambda path: [input_row] if path.name == "extra-megalith.jsonl" else []
    )

    def prepared(row):
        (tmp_path / "manifest.json").write_text('{"status": "frozen"}\n')
        return record("fresh")

    monkeypatch.setattr(corpus.preparation, "prepare", prepared)
    with pytest.raises(ValueError, match="froze while preparation"):
        corpus.prepare(1)
    assert candidate.read_bytes() == b"previous frame"
