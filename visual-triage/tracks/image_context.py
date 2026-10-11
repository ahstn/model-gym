"""Bounded, descriptive image context. This module makes no quality or origin decisions."""

import io
import json
import math
import warnings
from pathlib import Path
from struct import error as struct_error

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 20_000_000
MEASUREMENT_EDGE = 256
MAX_CONTEXT_BYTES = 2500
CAMERA_TAGS = {
    "iso": (34855, 1, 1_000_000),
    "exposure_seconds": (33434, 0, 86_400),
    "f_number": (33437, 0, 256),
    "focal_length_mm": (37386, 0, 10_000),
}
CONTEXT_GUIDANCE = (
    "Resolution is not a quality score. Missing EXIF does not imply AI origin. "
    "Camera values are unverified claims and may be dataset shortcuts. "
    "Pixel statistics describe the measurement view, not native sensor data; "
    "they do not measure calibrated noise or prove blur, poor exposure, or AI origin."
)


def canonical_context_json(context: dict) -> str:
    """Return deterministic JSON suitable for a frozen request and external hash."""
    return json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read(source: bytes | str | Path) -> bytes:
    if isinstance(source, bytes):
        data = source
    else:
        with Path(source).open("rb") as handle:
            data = handle.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("input_limit")
    return data


def _camera(exif, fields: tuple[str, ...]) -> dict:
    # Camera numeric tags may be in either the root IFD or the nested Exif IFD.
    nested = exif.get_ifd(34665) if 34665 in exif else {}
    result = {}
    for field in fields:
        tag, low, high = CAMERA_TAGS[field]
        value = nested.get(tag, exif.get(tag))
        try:
            # Text is excluded, including strings that resemble a numeric value.
            number = float(value) if not isinstance(value, (str, bytes, bool)) else math.nan
            valid = math.isfinite(number) and low <= number <= high and number > 0
            result[field] = round(number, 6) if valid else None
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            result[field] = None
    return result


def _native(source, camera_fields: tuple[str, ...]) -> dict:
    if source is None:
        return {"status": "not_supplied", "exif_availability": "unknown"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(_read(source))) as image:
                if image.width * image.height > MAX_PIXELS:
                    return {"status": "input_limit", "exif_availability": "unknown"}
                dimensions = {"encoded_width": image.width, "encoded_height": image.height}
                image.load()
                try:
                    exif = image.getexif()
                    metadata = {"exif_availability": "present" if exif else "absent"}
                    if camera_fields:
                        metadata["camera"] = _camera(exif, camera_fields)
                except (OSError, ValueError, TypeError, SyntaxError, KeyError, struct_error):
                    metadata = {"exif_availability": "unknown"}
                    if camera_fields:
                        metadata["camera"] = dict.fromkeys(camera_fields)
                return {"status": "ok", **dimensions, **metadata}
    except _DECODE_ERRORS:
        return {"status": "unreadable", "exif_availability": "unknown"}


def _quantile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return round(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower), 6)


def _statistics(image: Image.Image) -> dict:
    width, height = image.size
    raw = image.tobytes()
    pixels = list(zip(raw[0::3], raw[1::3], raw[2::3], strict=True))
    # Encoded RGB luma, not linear-light or calibrated physical luminance.
    luma = [(0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 for r, g, b in pixels]
    ordered = sorted(luma)
    quantiles = {name: _quantile(ordered, q) for name, q in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))}
    channels = {}
    for index, name in enumerate(("r", "g", "b")):
        values = [pixel[index] / 255 for pixel in pixels]
        mean = sum(values) / len(values)
        channels[name] = {
            "mean": round(mean, 6),
            "std": round(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)), 6),
        }
    differences = [
        abs(luma[y * width + x] - luma[y * width + x - 1]) for y in range(height) for x in range(1, width)
    ]
    differences.extend(
        abs(luma[y * width + x] - luma[(y - 1) * width + x]) for y in range(1, height) for x in range(width)
    )
    return {
        "encoded_rgb_luma_quantiles": quantiles,
        "luma_p95_minus_p05": round(quantiles["p95"] - quantiles["p05"], 6),
        "luma_le_1_over_255_fraction": round(sum(value <= 1 / 255 for value in luma) / len(luma), 6),
        "luma_ge_254_over_255_fraction": round(sum(value >= 254 / 255 for value in luma) / len(luma), 6),
        "mean_adjacent_luma_difference": round(sum(differences) / len(differences), 6)
        if differences
        else None,
        "channel_moments": channels,
    }


def _delivered(source) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(_read(source))) as image:
                if image.width * image.height > MAX_PIXELS:
                    return {"status": "input_limit"}
                if getattr(image, "n_frames", 1) != 1:
                    return {"status": "unsupported_multiframe"}
                image.load()
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                alpha = "A" in image.getbands() or "transparency" in image.info
                if alpha:
                    rgba = image.convert("RGBA")
                    image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
                else:
                    image = image.convert("RGB")
                image.thumbnail((MEASUREMENT_EDGE, MEASUREMENT_EDGE), Image.Resampling.BOX)
                return {
                    "status": "ok",
                    "width": width,
                    "height": height,
                    "measurement_width": image.width,
                    "measurement_height": image.height,
                    "measurement_policy": "exif_transpose_rgb_box_max256_no_upscale",
                    "alpha_policy": "composite_white" if alpha else "no_alpha",
                    "colour_policy": "pillow_rgb_no_icc_transform",
                    "statistics": _statistics(image),
                }
    except _DECODE_ERRORS:
        return {"status": "unreadable"}


_DECODE_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    SyntaxError,
    struct_error,
    UnidentifiedImageError,
    Image.DecompressionBombError,
    Image.DecompressionBombWarning,
)


def extract_image_context(
    delivered_image: bytes | str | Path,
    *,
    native_source: bytes | str | Path | None = None,
    include_camera: bool = False,
    camera_fields: tuple[str, ...] = tuple(CAMERA_TAGS),
) -> dict:
    """Extract an allowlisted context without paths, IDs, captions or origin labels.

    Native source must be supplied explicitly; never infer it from delivered pixels.
    Camera settings require an explicit field allowlist and remain unverified.
    Input limits return an unavailable status; programmer field errors raise ValueError.
    """
    if len(set(camera_fields)) != len(camera_fields) or set(camera_fields) - CAMERA_TAGS.keys():
        raise ValueError("camera_fields must contain unique allowlisted field names")
    return {
        "schema": "image_context_v1",
        "guidance": CONTEXT_GUIDANCE,
        "native_source": _native(native_source, tuple(sorted(camera_fields)) if include_camera else ()),
        "delivered_view": _delivered(delivered_image),
    }


def model_context(context: dict, mode: str = "pixel_stats") -> dict:
    """Return only fields permitted in a frozen model request.

    Native dimensions and EXIF availability are audit data, even in camera mode.
    Camera mode requires balanced source coverage; null means unknown, not generated.
    """
    if mode not in ("pixel_stats", "camera"):
        raise ValueError("mode must be pixel_stats or camera")
    view = context.get("delivered_view", {})
    fields = (
        "status",
        "width",
        "height",
        "measurement_width",
        "measurement_height",
        "measurement_policy",
        "alpha_policy",
        "colour_policy",
    )
    result = {
        "schema": "image_context_v1",
        "guidance": CONTEXT_GUIDANCE,
        "delivered_view": {key: view[key] for key in fields if key in view},
    }
    stats = view.get("statistics", {})
    scalar_fields = (
        "luma_p95_minus_p05",
        "luma_le_1_over_255_fraction",
        "luma_ge_254_over_255_fraction",
        "mean_adjacent_luma_difference",
    )
    safe_stats = {key: stats[key] for key in scalar_fields if key in stats}
    if "encoded_rgb_luma_quantiles" in stats:
        safe_stats["encoded_rgb_luma_quantiles"] = {
            key: stats["encoded_rgb_luma_quantiles"][key]
            for key in ("p05", "p50", "p95")
            if key in stats["encoded_rgb_luma_quantiles"]
        }
    if "channel_moments" in stats:
        safe_stats["channel_moments"] = {
            channel: {
                key: stats["channel_moments"][channel][key]
                for key in ("mean", "std")
                if key in stats["channel_moments"][channel]
            }
            for channel in ("r", "g", "b")
            if channel in stats["channel_moments"]
        }
    result["delivered_view"]["statistics"] = safe_stats
    if mode == "camera":
        camera = context.get("native_source", {}).get("camera", {})
        result["camera"] = {key: camera.get(key) for key in CAMERA_TAGS}
    validate_model_context(result)
    return result


def _require_keys(value: dict, keys: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("invalid context field shape")


def _require_number(value, low: float, high: float, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("context values must be finite numbers")
    if not low <= value <= high or not math.isfinite(value):
        raise ValueError("context number outside allowed range")


def validate_model_context(context: dict) -> None:
    """Reject unknown fields, free text and invalid values before freezing a request."""
    if not isinstance(context, dict):
        raise TypeError("model context must be a mapping")
    required = {"schema", "guidance", "delivered_view"}
    _require_keys(context, required | ({"camera"} if "camera" in context else set()))
    if context["schema"] != "image_context_v1" or context["guidance"] != CONTEXT_GUIDANCE:
        raise ValueError("invalid context version or guidance")
    view = context["delivered_view"]
    if not isinstance(view, dict):
        raise TypeError("delivered_view must be a mapping")
    if view.get("status") in ("unreadable", "input_limit", "unsupported_multiframe"):
        _require_keys(view, {"status", "statistics"})
        _require_keys(view["statistics"], set())
    elif view.get("status") == "ok":
        _validate_view(view)
    else:
        raise ValueError("invalid delivered status")
    if "camera" in context:
        _require_keys(context["camera"], set(CAMERA_TAGS))
        for field, (_, low, high) in CAMERA_TAGS.items():
            value = context["camera"][field]
            _require_number(value, low, high, nullable=True)
            if value is not None and value <= 0:
                raise ValueError("camera values must be positive")
    if len(canonical_context_json(context).encode("utf-8")) > MAX_CONTEXT_BYTES:
        raise ValueError("model context exceeds byte limit")


def _validate_view(view: dict) -> None:
    dimensions = {"width", "height", "measurement_width", "measurement_height"}
    _require_keys(
        view, dimensions | {"status", "measurement_policy", "alpha_policy", "colour_policy", "statistics"}
    )
    for key in dimensions:
        limit = MEASUREMENT_EDGE if key.startswith("measurement_") else MAX_PIXELS
        _require_number(view[key], 1, limit)
        if not isinstance(view[key], int):
            raise TypeError("dimensions must be integers")
    if view["width"] * view["height"] > MAX_PIXELS:
        raise ValueError("delivered dimensions exceed pixel limit")
    if view["measurement_width"] > view["width"] or view["measurement_height"] > view["height"]:
        raise ValueError("measurement cannot upscale delivered pixels")
    if view["measurement_policy"] != "exif_transpose_rgb_box_max256_no_upscale":
        raise ValueError("invalid measurement policy")
    if view["alpha_policy"] not in ("composite_white", "no_alpha"):
        raise ValueError("invalid alpha policy")
    if view["colour_policy"] != "pillow_rgb_no_icc_transform":
        raise ValueError("invalid colour policy")
    stats = view["statistics"]
    scalar_fields = {
        "luma_p95_minus_p05",
        "luma_le_1_over_255_fraction",
        "luma_ge_254_over_255_fraction",
        "mean_adjacent_luma_difference",
    }
    _require_keys(stats, scalar_fields | {"encoded_rgb_luma_quantiles", "channel_moments"})
    for key in scalar_fields:
        _require_number(stats[key], 0, 1, nullable=key == "mean_adjacent_luma_difference")
    _require_keys(stats["encoded_rgb_luma_quantiles"], {"p05", "p50", "p95"})
    for value in stats["encoded_rgb_luma_quantiles"].values():
        _require_number(value, 0, 1)
    _require_keys(stats["channel_moments"], {"r", "g", "b"})
    for channel in stats["channel_moments"].values():
        _require_keys(channel, {"mean", "std"})
        for value in channel.values():
            _require_number(value, 0, 1)
