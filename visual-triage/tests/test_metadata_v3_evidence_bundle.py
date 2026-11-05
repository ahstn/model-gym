"""Local bundles preserve raw metadata and refuse unsafe or incomplete restoration."""

import ast
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_archive import EvidenceError
from tracks.metadata_v3 import evidence_bundle as bundle


@pytest.fixture
def study(tmp_path, monkeypatch):
    for relative in (bundle.DATA, bundle.TRACK, "research/metadata-2026-10-12"):
        (tmp_path / relative).mkdir(parents=True)
    for relative in bundle.REPRODUCTION_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Reproduction input\n")
    source = tmp_path / bundle.TRACK / "evidence_bundle.py"
    source.write_text("# Bundle source is included without generated outputs.\n")
    (tmp_path / bundle.TRACK / "protocol.json").write_text('{"code_sha256":{}}\n')
    files = {
        "tracks/metadata_v3/dataset.json": b'{"samples":[]}\n',
        "tracks/metadata_v3/controls.json": b'{"cases":[]}\n',
        "tracks/metadata_v3/results/summary.json": b'{"native":{}}\n',
        "tracks/metadata_v3/report.md": b"Archived report\n",
        "data/metadata-v3/runs/image_only/selection.json": b'{"selected":1}\n',
        "data/metadata-v3/runs/image_only/final/baseline.jsonl": b'{"cost_usd":null}\n',
        "data/metadata-v3/cache/features/one.json": b'{"seconds":0.1}\n',
        "data/metadata-v3/research/public-license": b"Public licence record\n",
        "research/metadata-2026-10-12/source.md": b"Source terms\n",
        "data/metadata-v3/native/one.jpg": b"native pixels",
        "data/metadata-v3/delivered/one.png": b"delivered pixels",
        "data/metadata-v3/contact-sheet.png": b"contact sheet pixels",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(bundle, "_complete", lambda root: {"selection_count": 7, "attempts_complete": True})
    return tmp_path, files


def test_bundle_roundtrip_exact_metadata_and_local_image_exclusions(study):
    root, files = study
    result = bundle.create(root)
    receipt = root / bundle.TRACK / "evidence-bundle.json"
    index = json.loads((root / result["index"]["path"]).read_text())
    assert result["excluded_image_count"] == 3
    assert {row["path"] for row in index["excluded_images"]} == {
        name for name in files if name.endswith((".jpg", ".png"))
    }
    for row in index["excluded_images"]:
        assert (root / row["path"]).read_bytes() == files[row["path"]]
        assert row["sha256"] == hashlib.sha256(files[row["path"]]).hexdigest()
    assert bundle.verify(receipt, root)["verified_members"] == len(index["members"])
    destination = root / "restored"
    bundle.restore(receipt, root, destination=destination)
    assert bundle.restore(receipt, root, destination=destination)["restored_members"] == 0
    for member in index["members"]:
        assert (destination / member["path"]).read_bytes() == (root / member["path"]).read_bytes()
    assert not (destination / bundle.DATA / "native/one.jpg").exists()
    assert all("evidence-bundles" not in member["path"] for member in index["members"])
    with pytest.raises(EvidenceError, match="immutable"):
        bundle.create(root)


def test_evidence_only_restore_preserves_edited_reports_and_restores_ignored_inputs(study):
    root, files = study
    bundle.create(root)
    receipt = root / bundle.TRACK / "evidence-bundle.json"
    destination = root / "clone"
    report = destination / bundle.TRACK / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text("Reviewed report with new wording\n")
    result = bundle.restore(receipt, root, destination=destination, evidence_only=True)
    expected = {
        name
        for name in files
        if name in bundle.EVIDENCE_FILES
        or (name.startswith(bundle.DATA + "/") and not name.endswith((".jpg", ".png")))
    }
    assert result["restored_members"] == len(expected)
    assert report.read_text() == "Reviewed report with new wording\n"
    for relative in expected:
        assert (destination / relative).read_bytes() == files[relative]
    assert {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()} == (
        expected | {str(report.relative_to(destination))}
    )
    assert bundle.restore(receipt, root, destination=destination, evidence_only=True)["restored_members"] == 0


@pytest.mark.parametrize(
    "relative", ["tracks/metadata_v3/dataset.json", "data/metadata-v3/cache/features/one.json"]
)
def test_evidence_only_restore_rejects_selected_conflicts_before_writing(study, relative):
    root, _ = study
    bundle.create(root)
    destination = root / "clone"
    conflicting = destination / relative
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text("Local data must remain unchanged\n")
    with pytest.raises(EvidenceError, match="checksum|size"):
        bundle.restore(
            root / bundle.TRACK / "evidence-bundle.json", root, destination=destination, evidence_only=True
        )
    assert conflicting.read_text() == "Local data must remain unchanged\n"
    assert [path for path in destination.rglob("*") if path.is_file()] == [conflicting]


def test_evidence_only_restore_checks_unselected_archive_members(study):
    root, _ = study
    result = bundle.create(root)
    receipt = root / bundle.TRACK / "evidence-bundle.json"
    archive_path = root / result["archive"]["path"]
    index_path = root / result["index"]["path"]
    index = json.loads(index_path.read_text())
    # Keep the outer archive checksums correct, but corrupt an unselected report member.
    with tarfile.open(archive_path, "r:gz") as archive:
        contents = [(entry, archive.extractfile(entry).read()) for entry in archive.getmembers()]
    with tarfile.open(archive_path, "w:gz") as archive:
        for entry, content in contents:
            if entry.name == bundle.TRACK + "/report.md":
                content = b"X" + content[1:]
            archive.addfile(entry, io.BytesIO(content))
    index["archive"] = bundle._descriptor(root, archive_path)
    index_path.write_text(json.dumps(index))
    result.update(archive=index["archive"], index=bundle._descriptor(root, index_path))
    receipt.write_text(json.dumps(result))
    destination = root / "clone"
    with pytest.raises(EvidenceError, match="hash|checksum"):
        bundle.restore(receipt, root, destination=destination, evidence_only=True)
    assert not destination.exists()


def test_archive_bytes_are_deterministic_and_outputs_do_not_recurse(study):
    root, _ = study
    first = bundle.create(root)
    second = bundle.create(root, root / bundle.DATA / "evidence-bundles/second", root / "second-receipt.json")
    assert first["archive"]["sha256"] == second["archive"]["sha256"]


@pytest.mark.parametrize("failure", ["conflict", "symlink", "archive_checksum", "traversal"])
def test_restore_rejects_conflicts_or_unsafe_bundle_before_writing(study, failure):
    root, _ = study
    result = bundle.create(root)
    receipt = root / bundle.TRACK / "evidence-bundle.json"
    destination = root / "restored"
    index_path = root / result["index"]["path"]
    index = json.loads(index_path.read_text())
    target = destination / index["members"][-1]["path"]
    if failure == "conflict":
        target.parent.mkdir(parents=True)
        target.write_bytes(b"Unrelated local work")
    elif failure == "symlink":
        destination.symlink_to(root / bundle.DATA, target_is_directory=True)
    elif failure == "archive_checksum":
        path = root / result["archive"]["path"]
        path.write_bytes(b"X" + path.read_bytes()[1:])
    else:
        path = root / result["archive"]["path"]
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo("../escape")
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
        descriptor = bundle._descriptor(root, path)
        index["archive"] = descriptor
        index_path.write_text(json.dumps(index))
        result.update(archive=descriptor, index=bundle._descriptor(root, index_path))
        receipt.write_text(json.dumps(result))
    with pytest.raises(EvidenceError):
        bundle.restore(receipt, root, destination=destination)
    if failure == "conflict":
        assert target.read_bytes() == b"Unrelated local work"
        assert len(list(destination.rglob("*.*"))) == 1
    elif failure != "symlink":
        assert not destination.exists()


def test_create_rejects_unknown_types_and_private_keys(study):
    root, _ = study
    unsafe = root / bundle.DATA / "research/key.txt"
    unsafe.write_text("-----BEGIN " + "PRIVATE KEY-----\n")
    with pytest.raises(EvidenceError, match="private-key"):
        bundle.create(root)
    unsafe.unlink()
    unsafe.with_suffix(".bin").write_bytes(b"unknown")
    with pytest.raises(EvidenceError, match="Unknown evidence file type"):
        bundle.create(root)
    assert not (root / bundle.DATA / "evidence-bundles").exists()


def test_completion_requires_all_expected_native_and_control_attempts(tmp_path, monkeypatch):
    from tracks.metadata_v3 import report
    from tracks.metadata_v3.features import ARMS

    summary = {"expected": 2, "unattempted": 0}
    result = {
        "protocol_sha256": "a" * 64,
        "selection": dict.fromkeys(ARMS),
        "native": {arm: {"baseline": dict(summary), "selected": dict(summary)} for arm in ARMS},
        "controls": {"intervention": {"provenance": dict(summary)}},
    }
    monkeypatch.setattr(report, "build_report", lambda *args, **kwargs: result)
    assert bundle._complete(tmp_path)["selection_count"] == 7
    result["controls"]["intervention"]["provenance"]["unattempted"] = 1
    with pytest.raises(EvidenceError, match="Complete every"):
        bundle._complete(tmp_path)
    result["controls"]["intervention"]["provenance"]["unattempted"] = 0
    result["selection"].pop(ARMS[0])
    with pytest.raises(EvidenceError, match="All seven"):
        bundle._complete(tmp_path)


def test_pending_requests_and_output_outside_ignored_data_are_rejected(tmp_path):
    runs = tmp_path / bundle.DATA / "runs"
    runs.mkdir(parents=True)
    (runs / ".inflight.json").write_text('{"unfinished":{}}')
    with pytest.raises(EvidenceError, match="in-flight"):
        bundle._complete(tmp_path)
    with pytest.raises(EvidenceError, match="ignored data"):
        bundle.create(tmp_path, output=tmp_path / "tracked-output")


def test_inventory_keeps_dependency_lock_and_skips_runtime_locks(study):
    root, _ = study
    (root / "uv.lock").write_text("version = 1\n")
    (root / bundle.TRACK / "protocol.json").write_text('{"code_sha256":{"uv.lock":"frozen"}}')
    for name in (".coordinator.lock", ".budget.lock"):
        (root / bundle.DATA / "runs" / name).write_bytes(b"")
    result = bundle.create(root)
    index = json.loads((root / result["index"]["path"]).read_text())
    names = {member["path"] for member in index["members"]}
    assert "uv.lock" in names
    assert not any(name.endswith((".coordinator.lock", ".budget.lock")) for name in names)


def test_documentation_key_placeholder_is_not_key_material():
    header = b"-----BEGIN " + b"PRIVATE KEY-----"
    placeholder = b'"private_key":"' + header + b'\\nMIIExample...\\n-----END PRIVATE KEY-----"'
    assert not bundle._has_private_key(placeholder)
    assert bundle._has_private_key(placeholder + b"\n" + header + b"\nactual material")


def test_inventory_preserves_reproduction_dependencies_tests_and_bundle_source(study):
    root, _ = study
    test = root / "tests/test_metadata_v3_example.py"
    test.parent.mkdir()
    test.write_text("def test_example():\n    assert True\n")
    result = bundle.create(root)
    index = json.loads((root / result["index"]["path"]).read_text())
    names = {member["path"] for member in index["members"]}
    assert set(bundle.REPRODUCTION_FILES) <= names
    assert "tracks/metadata_v3/evidence_bundle.py" in names
    assert "tests/test_metadata_v3_example.py" in names
    assert "tracks/metadata_v3/evidence-bundle.json" not in names
    assert not any("evidence-bundles/" in name for name in names)
    destination = root / "snapshot"
    bundle.restore(root / bundle.TRACK / "evidence-bundle.json", root, destination=destination)
    for relative in (
        *bundle.REPRODUCTION_FILES,
        "tracks/metadata_v3/evidence_bundle.py",
        str(test.relative_to(root)),
    ):
        assert (destination / relative).read_bytes() == (root / relative).read_bytes()


def test_standalone_restored_code_has_complete_local_imports_and_cli_help(study):
    root, _ = study
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads((project / bundle.TRACK / "protocol.json").read_text())
    sources = {project / relative for relative in (*bundle.REPRODUCTION_FILES, *protocol["code_sha256"])}
    sources.update((project / bundle.TRACK).glob("*.py"))
    sources.update((project / "tests").glob("test_metadata_v3*.py"))
    for source in sources:
        destination = root / source.relative_to(project)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (root / bundle.TRACK / "protocol.json").write_text(json.dumps({"code_sha256": protocol["code_sha256"]}))
    result = bundle.create(root)
    index = json.loads((root / result["index"]["path"]).read_text())
    names = {member["path"] for member in index["members"]}
    # Check every project-local static import, including function-local imports and test fixtures.
    for relative in names:
        source = project / relative
        if source.suffix != ".py" or relative.startswith((bundle.DATA + "/", "research/")):
            continue
        for node in ast.walk(ast.parse(source.read_text(), filename=relative)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert not node.level, "Review new relative imports in the bundle closure test"
                base = node.module or ""
                modules = [base, *(base + "." + alias.name for alias in node.names if alias.name != "*")]
            for module in modules:
                for directory in (project, source.parent):
                    path = directory.joinpath(*module.split("."))
                    for dependency in (path.with_suffix(".py"), path / "__init__.py"):
                        if dependency.is_file():
                            assert str(dependency.relative_to(project)) in names, (relative, module)
    restored = root / "standalone"
    bundle.restore(root / bundle.TRACK / "evidence-bundle.json", root, destination=restored)
    modules = (
        "report",
        "request_audit",
        "study",
        "resume_http_errors",
        "evidence_bundle",
        "acquire_real",
        "acquire_ai",
        "build_dataset",
        "build_controls",
    )
    for module in modules:
        # Isolated Python ignores PYTHONPATH and the current checkout; only this restored tree is added.
        script = (
            "import runpy,sys; sys.path.insert(0,sys.argv[1]); "
            "sys.argv=['restored-cli','--help']; runpy.run_module(sys.argv_module,run_name='__main__')"
        )
        script = script.replace("sys.argv_module", repr("tracks.metadata_v3." + module))
        process = subprocess.run(
            [sys.executable, "-I", "-B", "-c", script, str(restored)],
            cwd=restored,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        assert process.returncode == 0, (module, process.stderr)
        assert "usage:" in process.stdout.lower(), module
