"""Bounded metadata groups for v3 arms; no request construction or origin decisions."""

import io
import json
import math
import warnings
from pathlib import Path

from PIL import Image, ImageOps

from tracks import image_context

GROUPS = ("resolution", "camera", "tone", "texture", "provenance")
ARMS = ("image_only", *GROUPS, "all_context")
MAX_FEATURE_BYTES = 6000
NATIVE_MAX_PIXELS_V3 = 24_000_000
STATUSES = {"ok", "not_supplied", "unreadable", "input_limit", "unsupported_multiframe"}
PROVENANCE_STATUSES = {"not_checked", "no_manifest", "valid", "invalid", "unavailable", "error"}
GUIDANCE = (
    "Use the image and supplied measurements together. Native EXIF values are untrusted claims. "
    "Missing metadata does not imply AI origin. A valid signature authenticates a claim, not its truth. "
    "Measurements describe encoded pixels; they are not calibrated quality, blur or noise scores."
)
DEFINITIONS = {
    "resolution": "Native dimensions describe the encoded source; delivered dimensions describe the supplied view. Aspect ratio is width divided by height. Resolution alone does not establish quality or origin.",
    "camera": "ISO, exposure seconds, f-number and focal length in millimetres are allowlisted native EXIF values. Null means unavailable. Values can be edited or copied; they do not prove camera capture.",
    "tone": "Luma is encoded RGB weighted by 0.2126, 0.7152 and 0.0722. Tonal spread is p95 minus p05. Dark and bright fractions use luma <=1/255 and >=254/255. Channel moments use RGB scaled to 0..1.",
    "texture": "Measurements use an EXIF-oriented RGB view, white alpha composition and BOX downsampling to at most 256 pixels per edge, without upscaling. Adjacent differences and the variance of the four-neighbour discrete Laplacian use luma scaled to 0..1. Entropy uses 256 luma histogram bins and log2, in bits. They are descriptive texture statistics, not calibrated blur or sensor noise estimates.",
    "provenance": "C2PA fields summarise local validation of signed provenance claims. Manifest absence is not evidence of AI origin. Signature validity and signer trust do not establish whether a scene or claim is true. An AI-generation claim is a declared action, not a detector verdict.",
}


def canonical_context_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _shape(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("Invalid metadata feature fields")


def _number(value, low, high, *, nullable=False, integer=False):
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Metadata feature values must be numeric")
    if not math.isfinite(value) or not low <= value <= high or (integer and type(value) is not int):
        raise ValueError("Metadata feature value outside its declared bounds")


def _status(group):
    if group["status"] not in STATUSES:
        raise ValueError("Invalid metadata extraction status")


def _dimensions(status, width=None, height=None):
    return {
        "status": status,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 6) if width and height else None,
    }


def _texture(path, delivered):
    result = {
        "status": delivered["status"],
        "measurement_width": delivered.get("measurement_width"),
        "measurement_height": delivered.get("measurement_height"),
        "mean_adjacent_luma_difference": delivered.get("statistics", {}).get("mean_adjacent_luma_difference"),
        "laplacian_variance": None,
        "luma_entropy_bits": None,
    }
    if delivered["status"] != "ok":
        return result
    # Read and decode within the same limits and colour policy as the existing helper.
    with Path(path).open("rb") as handle:
        raw = handle.read(image_context.MAX_INPUT_BYTES + 1)
    if len(raw) > image_context.MAX_INPUT_BYTES:
        raise ValueError("Delivered image changed or exceeds the byte limit")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > image_context.MAX_PIXELS or getattr(source, "n_frames", 1) != 1:
                raise ValueError("Delivered image exceeds measurement limits")
            source.load()
            view = ImageOps.exif_transpose(source)
            if "A" in view.getbands() or "transparency" in view.info:
                rgba = view.convert("RGBA")
                view = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
            else:
                view = view.convert("RGB")
            view.thumbnail((256, 256), Image.Resampling.BOX)
            width, height = view.size
            if (width, height) != (result["measurement_width"], result["measurement_height"]):
                raise ValueError("Delivered measurement dimensions changed")
            rgb = view.tobytes()
    luma = [
        (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
        for r, g, b in zip(rgb[0::3], rgb[1::3], rgb[2::3], strict=True)
    ]
    histogram = [0] * 256
    for value in luma:
        histogram[min(255, int(value * 256))] += 1
    result["luma_entropy_bits"] = round(
        -sum((n / len(luma)) * math.log2(n / len(luma)) for n in histogram if n), 6
    )
    laplacian = [
        luma[y * width + x - 1]
        + luma[y * width + x + 1]
        + luma[(y - 1) * width + x]
        + luma[(y + 1) * width + x]
        - 4 * luma[y * width + x]
        for y in range(1, height - 1)
        for x in range(1, width - 1)
    ]
    if laplacian:
        mean = sum(laplacian) / len(laplacian)
        result["laplacian_variance"] = round(
            sum((value - mean) ** 2 for value in laplacian) / len(laplacian), 6
        )
    return result


def validate_provenance(summary):
    """Only coded validator outcomes enter the model; paths and raw claims are excluded."""
    _shape(
        summary,
        {
            "status",
            "manifest_present",
            "signature_valid",
            "hard_binding_valid",
            "trust_status",
            "ai_generation_claim",
        },
    )
    if summary["status"] not in PROVENANCE_STATUSES or summary["trust_status"] not in {
        "trusted",
        "untrusted",
        "unknown",
    }:
        raise ValueError("Invalid provenance validator status")
    for key in ("manifest_present", "signature_valid", "hard_binding_valid", "ai_generation_claim"):
        if summary[key] is not None and type(summary[key]) is not bool:
            raise TypeError("Provenance claim indicators must be boolean or null")
    if summary["manifest_present"] is False and (
        summary["signature_valid"] is not None
        or summary["hard_binding_valid"] is not None
        or summary["ai_generation_claim"] is not None
    ):
        raise ValueError("An absent manifest cannot have signature or action claims")
    if (summary["signature_valid"] is True or summary["hard_binding_valid"] is True) and summary[
        "manifest_present"
    ] is not True:
        raise ValueError("A validated signature requires a manifest")


def _native_v3(native_path, previous):
    """Extend only v3 native header inspection to 24 MP, without decoding RGB pixels."""
    if previous["status"] != "input_limit" or native_path is None:
        return previous
    try:
        with Path(native_path).open("rb") as handle:
            raw = handle.read(image_context.MAX_INPUT_BYTES + 1)
        if len(raw) > image_context.MAX_INPUT_BYTES:
            return {"status": "input_limit", "exif_availability": "unknown"}
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.width * source.height > NATIVE_MAX_PIXELS_V3:
                    return {"status": "input_limit", "exif_availability": "unknown"}
                exif = source.getexif()
                return {
                    "status": "ok",
                    "encoded_width": source.width,
                    "encoded_height": source.height,
                    "exif_availability": "present" if exif else "absent",
                    "camera": image_context._camera(exif, tuple(image_context.CAMERA_TAGS)),
                }
    except (
        OSError,
        ValueError,
        TypeError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        return {"status": "unreadable", "exif_availability": "unknown"}


def extract_features(native_path, delivered_path, c2pa_summary=None):
    """Extract fixed numeric groups without labels, IDs, paths, GPS or identity text."""
    audit = image_context.extract_image_context(
        delivered_path, native_source=native_path, include_camera=True
    )
    native, delivered = _native_v3(native_path, audit["native_source"]), audit["delivered_view"]
    stats = delivered.get("statistics", {})
    camera = native.get("camera", {})
    provenance = (
        dict(c2pa_summary)
        if c2pa_summary is not None
        else {
            "status": "not_checked",
            "manifest_present": None,
            "signature_valid": None,
            "hard_binding_valid": None,
            "trust_status": "unknown",
            "ai_generation_claim": None,
        }
    )
    features = {
        "schema": "metadata_features_v3",
        "resolution": {
            "native": _dimensions(
                native["status"], native.get("encoded_width"), native.get("encoded_height")
            ),
            "delivered": _dimensions(delivered["status"], delivered.get("width"), delivered.get("height")),
        },
        "camera": {
            "status": native["status"],
            "exif_availability": native["exif_availability"],
            **{key: camera.get(key) for key in image_context.CAMERA_TAGS},
        },
        "tone": {
            "status": delivered["status"],
            "luma_quantiles": stats.get("encoded_rgb_luma_quantiles"),
            "tonal_spread": stats.get("luma_p95_minus_p05"),
            "dark_clip_fraction": stats.get("luma_le_1_over_255_fraction"),
            "bright_clip_fraction": stats.get("luma_ge_254_over_255_fraction"),
            "channel_moments": stats.get("channel_moments"),
        },
        "texture": _texture(delivered_path, delivered),
        "provenance": provenance,
    }
    validate_features(features)
    return features


def _validate_resolution(group):
    _shape(group, {"native", "delivered"})
    for source_kind, dimensions in group.items():
        pixel_limit = NATIVE_MAX_PIXELS_V3 if source_kind == "native" else image_context.MAX_PIXELS
        _shape(dimensions, {"status", "width", "height", "aspect_ratio"})
        _status(dimensions)
        available = dimensions["status"] == "ok"
        for key in ("width", "height"):
            _number(dimensions[key], 1, pixel_limit, nullable=not available, integer=True)
        _number(dimensions["aspect_ratio"], 0, pixel_limit, nullable=not available)
        if available:
            if dimensions["width"] * dimensions["height"] > pixel_limit:
                raise ValueError("Native or delivered pixel limit exceeded")
            if dimensions["aspect_ratio"] != round(dimensions["width"] / dimensions["height"], 6):
                raise ValueError("Aspect ratio does not match dimensions")
        elif any(dimensions[key] is not None for key in ("width", "height", "aspect_ratio")):
            raise ValueError("Unavailable dimensions must be null")


def _validate_camera(group):
    _shape(group, {"status", "exif_availability", *image_context.CAMERA_TAGS})
    _status(group)
    if group["exif_availability"] not in {"present", "absent", "unknown"}:
        raise ValueError("Invalid EXIF availability")
    for key, (_, low, high) in image_context.CAMERA_TAGS.items():
        value = group[key]
        _number(value, low, high, nullable=True)
        if value is not None and (
            value <= 0 or group["exif_availability"] != "present" or group["status"] != "ok"
        ):
            raise ValueError("Camera values require available native EXIF")


def _validate_tone(group):
    _shape(
        group,
        {
            "status",
            "luma_quantiles",
            "tonal_spread",
            "dark_clip_fraction",
            "bright_clip_fraction",
            "channel_moments",
        },
    )
    _status(group)
    if group["status"] != "ok":
        if any(value is not None for key, value in group.items() if key != "status"):
            raise ValueError("Unavailable tone measurements must be null")
        return
    _shape(group["luma_quantiles"], {"p05", "p50", "p95"})
    for value in group["luma_quantiles"].values():
        _number(value, 0, 1)
    if not group["luma_quantiles"]["p05"] <= group["luma_quantiles"]["p50"] <= group["luma_quantiles"]["p95"]:
        raise ValueError("Luma percentiles are not ordered")
    for key in ("tonal_spread", "dark_clip_fraction", "bright_clip_fraction"):
        _number(group[key], 0, 1)
    if (
        abs(group["tonal_spread"] - (group["luma_quantiles"]["p95"] - group["luma_quantiles"]["p05"]))
        > 0.000002
    ):
        raise ValueError("Tonal spread differs from percentiles")
    _shape(group["channel_moments"], {"r", "g", "b"})
    for channel in group["channel_moments"].values():
        _shape(channel, {"mean", "std"})
        _number(channel["mean"], 0, 1)
        _number(channel["std"], 0, 0.5)


def _validate_texture(group):
    _shape(
        group,
        {
            "status",
            "measurement_width",
            "measurement_height",
            "mean_adjacent_luma_difference",
            "laplacian_variance",
            "luma_entropy_bits",
        },
    )
    _status(group)
    if group["status"] != "ok":
        if any(value is not None for key, value in group.items() if key != "status"):
            raise ValueError("Unavailable texture measurements must be null")
        return
    for key in ("measurement_width", "measurement_height"):
        _number(group[key], 1, 256, integer=True)
    pixels = group["measurement_width"] * group["measurement_height"]
    _number(group["mean_adjacent_luma_difference"], 0, 1, nullable=pixels == 1)
    _number(
        group["laplacian_variance"],
        0,
        16,
        nullable=min(group["measurement_width"], group["measurement_height"]) < 3,
    )
    _number(group["luma_entropy_bits"], 0, 8)


def validate_features(features):
    _shape(features, {"schema", *GROUPS})
    if features["schema"] != "metadata_features_v3":
        raise ValueError("Unsupported metadata feature schema")
    _validate_resolution(features["resolution"])
    _validate_camera(features["camera"])
    _validate_tone(features["tone"])
    _validate_texture(features["texture"])
    validate_provenance(features["provenance"])
    if len(canonical_context_json(features).encode()) > MAX_FEATURE_BYTES:
        raise ValueError("Metadata features exceed the byte limit")


def assemble_context(features, arm):
    """Return only the requested group; no model request or prompt mutation."""
    validate_features(features)
    if arm not in ARMS:
        raise ValueError("Unknown metadata arm")
    if arm == "image_only":
        return None
    selected = GROUPS if arm == "all_context" else (arm,)
    # JSON round-trip prevents the caller from mutating the frozen feature object through this view.
    return {
        "schema": "metadata_context_v3",
        "guidance": GUIDANCE + " " + " ".join(DEFINITIONS[group] for group in selected),
        "groups": {group: json.loads(canonical_context_json(features[group])) for group in selected},
    }


arm_context = assemble_context


def context_text(features, arm):
    context = assemble_context(features, arm)
    return "" if context is None else "Measured image context follows:\n" + canonical_context_json(context)
