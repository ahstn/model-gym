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
        "image_path": identity,
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


def test_relocated_source_requires_matching_historical_bytes(modules, monkeypatch, tmp_path):
    corpus = modules["corpus"]
    monkeypatch.setattr(corpus, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(corpus, "DATA", tmp_path / "run")
    source = tmp_path / "sources/photo.png"
    source.parent.mkdir()
    source.write_bytes(b"exact historical image")
    row = {
        "id": "migrated",
        "image_path": "/Users/ahstn/git/model-gym/visual-quality/data/sources/photo.png",
        "sha256": corpus.digest(source),
    }
    monkeypatch.setattr(corpus.preparation, "prepare", lambda item: dict(item))
    result = corpus.prepare_input(row)
    assert result["image_path"] == str(source)
    assert result["source_path_relocation"]["original_path"] == row["image_path"]
    source.write_bytes(b"different bytes")
    result = corpus.prepare_input(row)
    assert result["image_path"] == row["image_path"]
    assert "source_path_relocation" not in result


def test_local_embedding_splits_actual_bytes_and_preserves_order(modules, monkeypatch, tmp_path):
    from types import SimpleNamespace

    embed, corpus = modules["embed"], modules["corpus"]
    monkeypatch.setattr(embed, "MAX_REQUEST_BYTES", 450)
    rows, expected, calls = [], [], []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        path.write_bytes(bytes([index]) * 80)
        row = {"image_path": str(path), "view_sha256": corpus.digest(path)}
        rows.append(row)
        vector = np.zeros(768, dtype=np.float32)
        vector[index] = 1
        expected.append(vector)
    by_sha = {row["view_sha256"]: vector.tolist() for row, vector in zip(rows, expected, strict=True)}

    def post(endpoint, *, data, headers, timeout):
        assert len(data) <= embed.MAX_REQUEST_BYTES
        assert headers == {"Content-Type": "application/json"}
        items = json.loads(data)["images"]
        calls.append(items)
        result = {
            "sha256": [item["sha256"] for item in items],
            "embeddings": [by_sha[item["sha256"]] for item in items],
            "seconds": 0.25,
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: result)

    monkeypatch.setattr(embed.requests, "post", post)
    actual, timing = embed.encode_local(rows, "unused")
    np.testing.assert_array_equal(actual, np.asarray(expected))
    assert [len(items) for items in calls] == [2, 2]
    assert timing["server_seconds"] == 0.5
    assert timing["transport_requests"] == 2
    assert timing["wall_seconds"] >= 0
    assert timing["verified_local_bytes"]


def test_local_embedding_rejects_oversized_single_before_any_http(modules, monkeypatch, tmp_path):
    embed, corpus = modules["embed"], modules["corpus"]
    monkeypatch.setattr(embed, "MAX_REQUEST_BYTES", 450)
    rows = []
    for name, size in (("small", 40), ("large", 1000)):
        path = tmp_path / f"{name}.png"
        path.write_bytes(b"a" * size)
        rows.append({"image_path": str(path), "view_sha256": corpus.digest(path)})

    def forbidden(*args, **kwargs):
        pytest.fail("Oversized single image must be rejected before any HTTP")

    monkeypatch.setattr(embed.requests, "post", forbidden)
    with pytest.raises(ValueError, match="Single embedding image"):
        embed.encode_local(rows, "unused")


def test_license_path_relocation_requires_independent_revision_and_hash_pin(modules, monkeypatch, tmp_path):
    corpus = modules["corpus"]
    monkeypatch.setattr(corpus, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(corpus, "DATA", tmp_path / "run")
    evidence = tmp_path / "sources/megalith/metadata/README.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("pinned source card")
    policy = {"revision": "known-revision", "metadata_pins": {str(evidence): corpus.digest(evidence)}}
    corpus.write_json(corpus.DATA / "acquisition/megalith-candidates.provenance.json", policy)
    record = {
        "id": "CC0-1.0",
        "evidence_path": "/Users/ahstn/git/model-gym/visual-quality/data/sources/megalith/metadata/README.md",
    }
    result = corpus.resolve_license_evidence(record, "known-revision")
    assert result["evidence_path"] == str(evidence)
    assert result["evidence_original_path"] == record["evidence_path"]
    assert result["evidence_sha256"] == corpus.digest(evidence)
    assert result["evidence_path_status"] == "verified_local_copy"
    evidence.write_text("different source card")
    assert (
        corpus.resolve_license_evidence(record, "known-revision")["evidence_path_status"]
        == "historical_path_not_resolved"
    )


def test_unpinned_historical_license_path_is_explicitly_unresolved(modules, monkeypatch, tmp_path):
    corpus = modules["corpus"]
    monkeypatch.setattr(corpus, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(corpus, "DATA", tmp_path / "run")
    evidence = tmp_path / "sources/vizwiz/evidence/provenance.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}")
    original = "/Users/ahstn/git/model-gym/visual-quality/data/sources/vizwiz/evidence/provenance.json"
    result = corpus.resolve_license_evidence({"evidence_path": original}, "vizwiz-revision")
    assert result["evidence_path"] == result["evidence_original_path"] == original
    assert result["evidence_path_status"] == "historical_path_not_resolved"
    assert "evidence_sha256" not in result


def test_current_vizwiz_evidence_is_verified_without_claiming_old_byte_equivalence(
    modules, monkeypatch, tmp_path
):
    corpus = modules["corpus"]
    monkeypatch.setattr(corpus, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(corpus, "DATA", tmp_path / "run")
    config = corpus.acquisition.vizwiz
    source = tmp_path / "sources/vizwiz"
    (source / "evidence").mkdir(parents=True)
    (source / "train.json").write_text("[]")
    monkeypatch.setattr(config, "ANNOTATION_SHA256", corpus.digest(source / "train.json"))
    current = {
        "annotation_revision": config.REPO_REVISION,
        "annotation_sha256": config.ANNOTATION_SHA256,
        "archive_url": config.ARCHIVE_URL,
        "archive_etag": config.ARCHIVE_ETAG,
        "archive_size": config.ARCHIVE_SIZE,
        "media_license": "CC-BY-4.0",
        "media_license_url": "https://creativecommons.org/licenses/by/4.0/",
    }
    corpus.write_json(source / "evidence/provenance.json", current)
    (source / "evidence/caption-media-license.html").write_text(
        "creativecommons.org/licenses/by/4.0 " + config.ARCHIVE_URL
    )
    record = {
        "id": "CC-BY-4.0",
        "url": current["media_license_url"],
        "evidence_path": "/Users/ahstn/git/model-gym/visual-quality/data/sources/vizwiz/evidence/provenance.json",
    }
    revision = f"annotations:{config.REPO_REVISION};media-etag:{config.ARCHIVE_ETAG}"
    result = corpus.resolve_license_evidence(record, revision)
    evidence = result["current_source_evidence"]
    assert result["evidence_path_status"] == "historical_path_not_resolved"
    assert evidence["relation"] == "current source evidence; historical byte equivalence unknown"
    assert evidence["sha256"] == corpus.digest(source / "evidence/provenance.json")
    assert "current_source_evidence" not in corpus.resolve_license_evidence(record, "wrong revision")
