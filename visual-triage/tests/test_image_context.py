import copy
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, TiffImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracks import image_context as context


def image_bytes(size=(8, 8), colour="white", *, exif=None, mode="RGB", format="PNG"):
    image = Image.new(mode, size, colour)
    buffer = io.BytesIO()
    image.save(buffer, format=format, **({"exif": exif} if exif is not None else {}))
    return buffer.getvalue()


def test_deterministic_context_and_model_whitelist(tmp_path):
    data = image_bytes()
    filename = tmp_path / "real-photo-dataset-label.png"
    filename.write_bytes(data)
    first = context.extract_image_context(data)
    second = context.extract_image_context(filename)
    assert context.canonical_context_json(first) == context.canonical_context_json(second)
    assert first["native_source"] == {"status": "not_supplied", "exif_availability": "unknown"}
    first["native_source"]["label"] = "AI"
    first["delivered_view"]["statistics"]["dataset"] = "secret"
    first["delivered_view"]["statistics"]["channel_moments"]["r"]["caption"] = "private"
    model = context.model_context(first)
    encoded = context.canonical_context_json(model)
    assert all(token not in encoded for token in (str(filename), "secret", "private", "native_source"))
    assert len(encoded) < 2500
    assert (
        hashlib.sha256(encoded.encode()).hexdigest()
        == hashlib.sha256(context.canonical_context_json(json.loads(encoded)).encode()).hexdigest()
    )


@pytest.mark.parametrize("colour,value", [("black", 0), ("white", 1)])
def test_constant_tiny_image(colour, value):
    view = context.extract_image_context(image_bytes((1, 1), colour))["delivered_view"]
    stats = view["statistics"]
    assert stats["encoded_rgb_luma_quantiles"] == dict.fromkeys(("p05", "p50", "p95"), value)
    assert stats["mean_adjacent_luma_difference"] is None
    assert stats["channel_moments"]["r"] == {"mean": value, "std": 0}
    assert stats["luma_le_1_over_255_fraction"] == 1 - value
    assert stats["luma_ge_254_over_255_fraction"] == value


def test_downsample_and_native_dimensions_remain_separate():
    result = context.extract_image_context(image_bytes((512, 128)), native_source=image_bytes((800, 600)))
    assert result["native_source"]["encoded_width"] == 800
    assert result["delivered_view"]["width"] == 512
    assert result["delivered_view"]["measurement_width"] == 256
    assert result["delivered_view"]["measurement_height"] == 64
    assert "encoded_width" not in context.canonical_context_json(context.model_context(result))


def test_camera_allowlist_excludes_identifiers_and_missingness():
    exif = Image.Exif()
    for tag, value in {
        34855: 200,
        33434: TiffImagePlugin.IFDRational(1, 125),
        33437: 2.8,
        37386: 35,
        271: "private make",
        272: "private model",
        305: "generator label",
        306: "2026:10:11 10:00:00",
        42033: "private serial",
        270: "private caption",
    }.items():
        exif[tag] = value
    source = image_bytes(exif=exif)
    default = context.extract_image_context(image_bytes(), native_source=source)
    assert default["native_source"]["exif_availability"] == "present"
    assert "camera" not in default["native_source"]
    result = context.extract_image_context(image_bytes(), native_source=source, include_camera=True)
    assert result["native_source"]["camera"] == {
        "iso": 200,
        "exposure_seconds": 0.008,
        "f_number": 2.8,
        "focal_length_mm": 35,
    }
    model = context.model_context(result, "camera")
    encoded = context.canonical_context_json(model)
    assert all(
        token not in encoded for token in ("private", "generator label", "exif_availability", "native_source")
    )
    absent = context.extract_image_context(image_bytes(), native_source=image_bytes(), include_camera=True)
    assert context.model_context(absent, "camera")["camera"] == dict.fromkeys(context.CAMERA_TAGS)


def test_explicit_camera_field_subset_and_invalid_fields():
    exif = Image.Exif()
    exif[34855] = 100
    result = context.extract_image_context(
        image_bytes(), native_source=image_bytes(exif=exif), include_camera=True, camera_fields=("iso",)
    )
    assert result["native_source"]["camera"] == {"iso": 100}
    with pytest.raises(ValueError):
        context.extract_image_context(image_bytes(), camera_fields=("model",))
    with pytest.raises(ValueError):
        context.model_context(result, "metadata")


def test_invalid_numeric_camera_values_are_unknown():
    exif = Image.Exif()
    exif[33434] = TiffImagePlugin.IFDRational(0, 0)
    exif[33437] = 0
    assert context._camera(exif, ("exposure_seconds", "f_number")) == {
        "exposure_seconds": None,
        "f_number": None,
    }


@pytest.mark.parametrize("payload", [b"", b"not an image", image_bytes()[:30]])
def test_corrupt_inputs_return_bounded_status(payload):
    result = context.extract_image_context(payload, native_source=payload)
    assert result["native_source"]["exif_availability"] == "unknown"
    assert result["delivered_view"] == {"status": "unreadable"}
    assert len(context.canonical_context_json(result)) < 900


def test_resource_limits(monkeypatch):
    monkeypatch.setattr(context, "MAX_PIXELS", 2)
    assert context.extract_image_context(image_bytes())["delivered_view"]["status"] == "input_limit"
    monkeypatch.setattr(context, "MAX_INPUT_BYTES", 2)
    assert context.extract_image_context(image_bytes())["delivered_view"]["status"] == "unreadable"


def test_orientation_and_alpha_policies_are_explicit():
    exif = Image.Exif()
    exif[274] = 6
    view = context.extract_image_context(image_bytes((6, 3), exif=exif))["delivered_view"]
    assert (view["width"], view["height"]) == (3, 6)
    alpha = context.extract_image_context(image_bytes(colour=(0, 0, 0, 0), mode="RGBA"))["delivered_view"]
    assert alpha["alpha_policy"] == "composite_white"
    assert alpha["statistics"]["encoded_rgb_luma_quantiles"]["p50"] == 1


def test_animated_image_is_rejected():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(
        buffer, format="GIF", save_all=True, append_images=[Image.new("RGB", (2, 2), "blue")]
    )
    assert (
        context.extract_image_context(buffer.getvalue())["delivered_view"]["status"]
        == "unsupported_multiframe"
    )


def test_edge_measurement_is_not_an_inferred_blur_score():
    image = Image.new("RGB", (2, 1), "black")
    image.putpixel((1, 0), (255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    stats = context.extract_image_context(buffer.getvalue())["delivered_view"]["statistics"]
    assert stats["mean_adjacent_luma_difference"] == 1
    assert "blur" not in stats and "noise" not in stats and "quality" not in stats


def test_nested_exif_and_gps_are_not_passed_to_model():
    exif = Image.Exif()
    exif[34665] = {34855: 800, 33434: TiffImagePlugin.IFDRational(1, 50), 42033: "serial-secret"}
    exif[34853] = {
        1: "N",
        2: (TiffImagePlugin.IFDRational(54), TiffImagePlugin.IFDRational(0), TiffImagePlugin.IFDRational(0)),
    }
    result = context.extract_image_context(
        image_bytes(), native_source=image_bytes(exif=exif), include_camera=True
    )
    assert result["native_source"]["camera"]["iso"] == 800
    assert result["native_source"]["camera"]["exposure_seconds"] == 0.02
    serialized = context.canonical_context_json(context.model_context(result, "camera"))
    assert "serial-secret" not in serialized and "GPS" not in serialized and '"N"' not in serialized


@pytest.mark.parametrize("mutation", ["unknown", "nested_unknown", "text", "nan", "bool", "policy"])
def test_validation_rejects_context_injection(mutation):
    candidate = copy.deepcopy(context.model_context(context.extract_image_context(image_bytes())))
    view = candidate["delivered_view"]
    if mutation == "unknown":
        candidate["dataset"] = "label-secret"
    elif mutation == "nested_unknown":
        view["statistics"]["channel_moments"]["r"]["gps"] = "secret"
    elif mutation == "text":
        view["statistics"]["luma_p95_minus_p05"] = "generated"
    elif mutation == "nan":
        view["statistics"]["luma_p95_minus_p05"] = float("nan")
    elif mutation == "bool":
        view["width"] = True
    else:
        view["measurement_policy"] = "dataset secret"
    with pytest.raises((ValueError, TypeError)):
        context.validate_model_context(candidate)


def test_malformed_exif_does_not_prevent_delivered_measurement():
    broken_source = image_bytes(format="JPEG", exif=b"Exif\x00\x00broken")
    result = context.extract_image_context(image_bytes(), native_source=broken_source, include_camera=True)
    assert result["delivered_view"]["status"] == "ok"
    assert result["native_source"]["exif_availability"] in ("unknown", "absent")
    assert context.model_context(result, "camera")["camera"] == dict.fromkeys(context.CAMERA_TAGS)
