"""The retained candidate preserves provenance trust and keeps quality diagnostic."""

import math

import pytest

from PIL import Image, PngImagePlugin

from image_stats.authenticity import decide, inspect_image
from image_stats.authenticity.c2pa import _validated_summary, credential_state
from image_stats.authenticity.provenance import explicit_declarations, inspect_provenance, xmp_fields
from image_stats.authenticity.smaug import image_crops


def evidence(state="missing", *, ai=False, unsigned=False):
    return {
        "c2pa": {"state": state, "declared_ai": ai},
        "unsigned_ai_declarations": [{"source": "XMP"}] if unsigned else [],
    }


@pytest.mark.parametrize("state", ["invalid", "untrusted"])
def test_declared_ai_is_review_evidence_without_verified_status(state):
    result = decide(0.01, evidence(state, ai=True))
    assert result["review_flag"] and result["ai_declaration_flag"]
    assert not result["verified_ai_declaration"] and not result["smaug_flag"]
    assert result["credential_state"] == state


@pytest.mark.parametrize("state", ["missing", "error", "unavailable"])
def test_missing_or_failed_provenance_does_not_determine_origin(state):
    assert not decide(0.1, evidence(state))["review_flag"]
    assert decide(0.5, evidence(state))["review_flag"]
    assert decide(0.1, evidence(state, unsigned=True))["review_flag"]
    assert not decide(0.1, evidence(state, unsigned=True))["verified_ai_declaration"]


@pytest.mark.parametrize("score", [-0.01, 1.01, math.nan, math.inf])
def test_invalid_probabilities_are_rejected(score):
    with pytest.raises(ValueError):
        decide(score, evidence())


def test_stripped_copy_does_not_inherit_a_declaration(tmp_path):
    original, stripped = tmp_path / "native.png", tmp_path / "stripped.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_itxt("XML:com.adobe.xmp", '<r Model="AI Generated"/>')
    Image.new("RGB", (40, 30), "blue").save(original, pnginfo=metadata)
    Image.new("RGB", (40, 30), "blue").save(stripped)
    first, second = inspect_provenance(original), inspect_provenance(stripped)
    assert decide(0.1, first)["ai_declaration_flag"]
    assert not decide(0.1, second)["ai_declaration_flag"]
    assert first["sha256"] != second["sha256"]


def test_xmp_parsing_and_explicit_only_detection():
    uri = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    fields, error = xmp_fields(f'<r xmlns:i="urn:iptc" i:DigitalSourceType="{uri}"/>'.encode())
    assert error is None and explicit_declarations({}, [fields])
    assert xmp_fields(b"<broken>")[1]
    assert not explicit_declarations({"Software": "Adobe Photoshop", "Model": "Real Camera"}, [])


def test_signature_validity_is_preserved_separately_from_trust():
    class Reader:
        def get_active_manifest(self):
            return {"assertions": []}

        def get_validation_results(self):
            return {
                "activeManifest": {
                    "success": [{"code": "claimSignature.validated"}, {"code": "assertion.dataHash.match"}],
                    "failure": [{"code": "signingCredential.untrusted"}],
                }
            }

        def get_validation_state(self):
            return "Valid"

    summary = _validated_summary(Reader())
    assert summary["signature_valid"] and summary["hard_binding_valid"]
    assert credential_state(summary) == "untrusted"


def test_three_crops_cover_long_axis_and_preserve_size():
    landscape = image_crops(Image.new("RGB", (1200, 600)))
    portrait = image_crops(Image.new("RGB", (600, 1200)))
    tiny = image_crops(Image.new("RGB", (40, 30)))
    assert all(
        len(crops) == 3 and all(image.size == (528, 528) for image in crops) for crops in (landscape, portrait, tiny)
    )


def test_statistics_are_not_an_authenticity_vote(tmp_path):
    path = tmp_path / "uniform.png"
    Image.new("RGB", (100, 100), "white").save(path)

    class Model:
        def predict(self, path):
            return {"probability_ai": 0.01}

    result = inspect_image(path, Model())
    assert result["image_stats"]["analysis"]["flags"]
    assert not result["decision"]["review_flag"]
    assert result["decision"]["image_stats_role"] == "diagnostic_only"
