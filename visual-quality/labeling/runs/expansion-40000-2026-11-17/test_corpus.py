"""Protect the expansion's immutable boundaries and score-blind selection."""

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
    for name in ("corpus", "embed", "selection"):
        spec = importlib.util.spec_from_file_location(f"expansion_40000_{name}", HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def record(identity, dhash="0123456789abcdef", **changes):
    return {
        "image_id": identity,
        "candidate_id": identity,
        "case_id": identity,
        "family_id": identity,
        "source_id": identity,
        "source_sha256": "source-" + identity,
        "decoded_sha256": "pixels-" + identity,
        "view_sha256": "view-" + identity,
        "category": "megalith",
        "dhash": dhash,
        **changes,
    }


def test_preparation_excludes_all_historical_identities_and_protected_families(
    modules, monkeypatch, tmp_path
):
    corpus = modules["corpus"]
    old = record("old", dhash="0000000000000000")
    protected = {
        "id": "protected",
        "split_group": "protected-family",
        "source_id": "protected",
        "sha256": "protected-source",
        "decoded_sha256": "protected-pixels",
        "dhash": "ffffffffffffffff",
    }
    candidates = [
        record("family", family_id="old"),
        record("source", source_sha256=old["source_sha256"]),
        record("pixels", decoded_sha256=old["decoded_sha256"]),
        record("view", view_sha256=old["view_sha256"]),
        record("near", dhash="0000000000000003"),
        record("protected-candidate", family_id="protected-family"),
        record("fresh"),
    ]
    monkeypatch.setattr(corpus, "DATA", tmp_path)
    monkeypatch.setattr(corpus, "old_cases", lambda: [old])
    monkeypatch.setattr(corpus, "pin_initial_evidence", dict)
    monkeypatch.setattr(corpus, "digest", lambda path: "pinned")
    original_read = corpus.read_lines

    def read(path):
        if str(path).endswith("development.jsonl"):
            return [protected]
        if str(path).endswith("extra-megalith.jsonl"):
            return [{"id": r["image_id"], **r} for r in candidates]
        return []

    monkeypatch.setattr(corpus, "read_lines", read)
    monkeypatch.setattr(corpus.preparation, "prepare", lambda row: dict(row))
    corpus.prepare(2)
    manifest = json.loads((tmp_path / "candidate-manifest.json").read_text())
    assert [r["image_id"] for r in manifest["cases"]] == ["fresh"]
    assert len(original_read(tmp_path / "candidate-rejections.jsonl")) == 6


def test_frozen_manifest_cannot_be_reprepared(modules, monkeypatch, tmp_path):
    corpus = modules["corpus"]
    monkeypatch.setattr(corpus, "DATA", tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="already frozen"):
        corpus.prepare(1)


def test_duplicate_detection_checks_beyond_top_six(modules):
    selection = modules["selection"]
    # All seven prior images are semantically indistinguishable; only the first
    # passes the conservative perceptual threshold. A top-six gate can miss it.
    old = [record("old0", "0000000000000000")]
    old += [record(f"old{i}", "ffffffffffffffff") for i in range(1, 8)]
    fresh = [record("new", "0000000000000001")]
    old_vectors = np.ones((len(old), 1), dtype=np.float32)
    candidates, excluded, _ = selection.exclude_semantic_duplicates(old, fresh, old_vectors, np.ones((1, 1)))
    assert not candidates
    assert excluded == ["new"]


def test_semantic_similarity_alone_does_not_exclude(modules):
    selection = modules["selection"]
    old = [record("old", "0000000000000000")]
    fresh = [record("new", "ffffffffffffffff")]
    candidates, excluded, _ = selection.exclude_semantic_duplicates(
        old, fresh, np.ones((1, 1)), np.ones((1, 1))
    )
    assert candidates == fresh
    assert excluded == []


def test_quota_offsets_overrepresented_subjects_and_is_stable(modules):
    selection = modules["selection"]
    rows = [record(f"c{i}") for i in range(20)]
    clusters = {r["candidate_id"]: i // 10 for i, r in enumerate(rows)}
    old = [{"content_cluster": 0} for _ in range(100)]
    selected = selection.quotas(rows, 5, old, clusters, "test")
    assert all(clusters[row["candidate_id"]] == 1 for row in selected)
    assert selected == selection.quotas(list(reversed(rows)), 5, old, clusters, "test")
    with pytest.raises(ValueError, match="available"):
        selection.quotas(rows, 21, old, clusters, "test")


def test_invalid_vectors_and_prototypes_fail_closed(modules, tmp_path):
    embed, selection = modules["embed"], modules["selection"]
    with pytest.raises(ValueError, match="dimensions"):
        embed.validate_vector(np.ones(767))
    with pytest.raises(AssertionError):
        embed.validate_vector(np.ones(768))
    file = tmp_path / "prototypes.npz"
    np.savez(file, centroids=np.ones((63, 768)))
    with pytest.raises(ValueError, match="64 original"):
        selection.load_prototypes(file)


def test_evidence_pins_are_verified_not_rewritten(modules, monkeypatch, tmp_path):
    corpus = modules["corpus"]
    original = tmp_path / "historical.json"
    original.write_text("original")
    pins = {str(original): corpus.digest(original)}
    target = tmp_path / "initial-evidence-pins.json"
    target.write_text(json.dumps(pins))
    monkeypatch.setattr(corpus, "DATA", tmp_path)
    assert corpus.pin_initial_evidence() == pins
    before = target.read_bytes()
    original.write_text("changed")
    with pytest.raises(ValueError, match="Historical evidence changed"):
        corpus.pin_initial_evidence()
    assert target.read_bytes() == before


def test_failed_variant_parent_is_replaced_without_losing_base_image(modules, monkeypatch, tmp_path):
    selection = modules["selection"]
    chosen = [record("bad"), record("good")]
    monkeypatch.setattr(selection, "DATA", tmp_path)
    monkeypatch.setattr(selection, "sample", lambda pool, count, *args, **kwargs: pool[:count])
    monkeypatch.setattr(
        selection,
        "historical_matcher",
        lambda old: lambda row: "protected overlap" if row["image_id"] == "bad-v1" else None,
    )

    def family(parent):
        return [dict(parent)] + [
            record(parent["image_id"] + f"-v{i}", family_id=parent["family_id"], category="distortions")
            for i in range(1, 4)
        ]

    monkeypatch.setattr(selection.legacy, "family", family)
    final = selection.generate_families(chosen, {}, [], count=1)
    assert {r["image_id"] for r in final} == {"bad", "good", "good-v1", "good-v2", "good-v3"}
    rejects = selection.corpus.read_lines(tmp_path / "family-generation-rejections.jsonl")
    assert rejects == [{"image_id": "bad", "reason": "protected overlap"}]


def test_embedding_receipt_pins_snapshot_when_prepare_advances(modules, monkeypatch, tmp_path):
    embed, corpus = modules["embed"], modules["corpus"]
    frame = tmp_path / "candidate-manifest.json"
    frame.write_text('{"cases": []}')
    initial_sha = corpus.digest(frame)
    monkeypatch.setattr(embed, "DATA", tmp_path)
    monkeypatch.setattr(corpus, "pin_initial_evidence", dict)
    monkeypatch.setattr(embed, "previous_vectors", lambda: ({}, []))
    monkeypatch.setattr(embed, "read_lines", lambda path: [])
    for field in ("INITIAL", "OLD"):
        directory = tmp_path / field
        (directory / "embedding-operations").mkdir(parents=True)
        (directory / "embedding-operations/provenance.json").write_text("{}")
        monkeypatch.setattr(embed, field, directory)

    def overlap(endpoint, prior, previous):
        frame.write_text('{"cases": [], "newer": true}')
        (tmp_path / "embedding-operations/cpu-gpu-overlap.json").write_text("{}")
        return {"worker_identity": {"pinned": True}}

    monkeypatch.setattr(embed.legacy, "check_overlap", overlap)
    embed.main("unused")
    provenance = json.loads((tmp_path / "embedding-operations/provenance.json").read_text())
    assert provenance["candidate_manifest_sha256"] == initial_sha
    assert corpus.digest(Path(provenance["candidate_snapshot"])) == initial_sha
    assert corpus.digest(frame) != initial_sha
