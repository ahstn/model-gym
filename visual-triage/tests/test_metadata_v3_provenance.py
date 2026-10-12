import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracks.metadata_v3 import provenance


def manifest(source="trainedAlgorithmicMedia"):
    return {
        "title": "private filename",
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "softwareAgent": "private generator",
                            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/" + source,
                        }
                    ]
                },
            }
        ],
    }


class FakeReader:
    def __init__(self, active=None, successes=None, failures=None, state="Valid"):
        self.active = active or manifest()
        self.successes = (
            successes if successes is not None else ["claimSignature.validated", "assertion.dataHash.match"]
        )
        self.failures = failures if failures is not None else ["signingCredential.untrusted"]
        self.state = state

    def get_active_manifest(self):
        return self.active

    def get_validation_results(self):
        return {
            "activeManifest": {
                "success": [{"code": code} for code in self.successes],
                "failure": [{"code": code} for code in self.failures],
            }
        }

    def get_validation_state(self):
        return self.state


def test_signature_binding_and_trust_are_separate():
    result = provenance._validated_summary(FakeReader())
    assert result == {
        "status": "valid",
        "manifest_present": True,
        "signature_valid": True,
        "hard_binding_valid": True,
        "trust_status": "untrusted",
        "ai_generation_claim": True,
    }
    assert "private" not in str(result)


def test_trusted_requires_explicit_trust_success():
    reader = FakeReader(
        successes=["claimSignature.validated", "assertion.dataHash.match", "signingCredential.trusted"],
        failures=[],
        state="Trusted",
    )
    assert provenance._validated_summary(reader)["trust_status"] == "trusted"
    reader.successes.remove("signingCredential.trusted")
    assert provenance._validated_summary(reader)["trust_status"] == "unknown"


@pytest.mark.parametrize(
    "failure,signature,binding",
    [
        ("claimSignature.mismatch", False, True),
        ("assertion.dataHash.mismatch", True, False),
        ("assertion.hashedURI.mismatch", True, True),
        ("signingCredential.expired", True, True),
    ],
)
def test_invalid_manifest_cannot_supply_origin_claim(failure, signature, binding):
    result = provenance._validated_summary(FakeReader(failures=[failure], state="Invalid"))
    assert result["status"] == "invalid"
    assert result["signature_valid"] is signature
    assert result["hard_binding_valid"] is binding
    assert result["ai_generation_claim"] is None


def test_missing_verification_results_remain_unknown():
    result = provenance._validated_summary(FakeReader(successes=[], failures=[]))
    assert result["status"] == "unavailable"
    assert result["signature_valid"] is None
    assert result["hard_binding_valid"] is None


@pytest.mark.parametrize(
    "source,expected",
    [
        ("trainedAlgorithmicMedia", True),
        ("digitalCapture", False),
        ("computationalCapture", False),
        ("digitalCreation", None),
        ("unknown", None),
    ],
)
def test_declared_source_is_not_inferred_from_producer_name(source, expected):
    assert provenance._declared_origin(manifest(source)) is expected


def test_conflicting_or_edit_only_claims_remain_unknown():
    candidate = manifest()
    candidate["assertions"] += manifest("digitalCapture")["assertions"]
    assert provenance._declared_origin(candidate) is None
    candidate = manifest()
    candidate["assertions"][0]["data"]["actions"][0]["action"] = "c2pa.edited"
    assert provenance._declared_origin(candidate) is None


@pytest.mark.parametrize(
    "assertions",
    [
        None,
        "untrusted text",
        [None],
        [manifest()["assertions"][0]] * 129,
        [{"label": "c2pa.actions.v2", "data": "untrusted text"}],
        [{"label": "c2pa.actions.v2", "data": {"actions": "not a list"}}],
        [{"label": "c2pa.actions.v2", "data": {"actions": [None]}}],
    ],
)
def test_malformed_and_oversized_action_claims_remain_unknown(assertions):
    assert provenance._declared_origin({"assertions": assertions}) is None


def test_unknown_provenance_is_not_inferred_from_reader_handle():
    reader = FakeReader()
    reader.active = None
    assert provenance._validated_summary(reader) == provenance._summary("no_manifest", False)


def test_offline_policy_and_stream_mime_use_bytes_not_extension(tmp_path, monkeypatch):
    expected_policy = copy.deepcopy(provenance.OFFLINE_SETTINGS)
    calls = []

    class Manager:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Context(Manager):
        @classmethod
        def from_dict(cls, policy):
            calls.append(policy)
            return cls()

    class Reader(Manager, FakeReader):
        def __init__(self, mime, stream, *, context):
            assert mime == "image/jpeg"
            assert stream.tell() == 0
            assert isinstance(context, Context)
            FakeReader.__init__(self)

    class Error(Exception):
        pass

    Error.ManifestNotFound = type("ManifestNotFound", (Error,), {})
    Error.RemoteManifest = type("RemoteManifest", (Error,), {})
    sdk = SimpleNamespace(
        __version__="0.38.0", sdk_version=lambda: "0.91.0", Context=Context, Reader=Reader, C2paError=Error
    )
    monkeypatch.setitem(sys.modules, "c2pa", sdk)
    image = tmp_path / "misleading-generator-name.png"
    image.write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)
    assert provenance.extract_c2pa(image)["status"] == "valid"
    assert calls == [expected_policy]
    assert expected_policy["verify"]["remote_manifest_fetch"] is False
    assert expected_policy["verify"]["ocsp_fetch"] is False
    assert expected_policy["core"]["allowed_network_hosts"] == []
    assert "trust" not in expected_policy
    sdk.__version__ = "0.39.0"
    assert provenance.extract_c2pa(image)["status"] == "unavailable"


def test_missing_sdk_returns_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "c2pa", None)
    assert provenance.extract_c2pa(tmp_path / "image.jpg") == provenance._summary("unavailable")


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("control-firefly-stripped.jpg", provenance._summary("no_manifest", False)),
        ("control-firefly-swapped.jpg", provenance._summary("invalid", True, True, False, "untrusted")),
        (
            "control-firefly-invalid-signature.jpg",
            provenance._summary("invalid", True, False, True, "untrusted"),
        ),
    ],
)
def test_local_byte_control_fixtures(filename, expected):
    sdk = pytest.importorskip("c2pa")
    if sdk.__version__ != provenance.PYTHON_SDK_VERSION or sdk.sdk_version() != provenance.RUST_SDK_VERSION:
        pytest.skip("byte controls require the frozen SDK versions")
    path = Path(__file__).resolve().parents[1] / "data/metadata-v3/candidates/c2pa" / filename
    if not path.exists():
        pytest.skip("local research byte controls are not distributed with the tests")
    assert provenance.extract_c2pa(path) == expected
