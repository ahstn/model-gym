"""Deterministic damage variants and locally drawn domain controls.

Configured severity describes an operation, not a human quality score or a
guaranteed preference order. Reproduction requires the recorded library versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from PIL import __version__ as pillow_version

from visual_quality.storage import atomic_json, file_sha256

DISTORTION_VERSION = "photo-distortions-v1"
CONTROL_VERSION = "owned-controls-v1"
MAX_VARIANT_EDGE = 1536
OPERATORS = ("gaussian_blur", "gaussian_noise", "jpeg_compression")
CONTROL_KINDS = ("chart", "flowchart", "circuit_diagram", "mock_interface", "isometric_render")
PROVENANCE_FIELDS = (
    "source_url",
    "license",
    "creator",
    "attribution",
    "content_kind",
    "source_origin",
    "photo_domain_eligible",
    "source_split",
    "source_group",
    "split",
)


def _seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def _save(image: Image.Image, path: Path, *, jpeg_quality: int | None = None) -> dict:
    """Publish a complete image atomically and return its byte-level identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        if jpeg_quality is None:
            image.save(temporary, format="PNG", compress_level=6)
        else:
            image.save(temporary, format="JPEG", quality=jpeg_quality, subsampling=2, optimize=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "image_path": str(path),
        "width": image.width,
        "height": image.height,
        "sha256": file_sha256(path),
    }


def _variant(image: Image.Image, operator: str, level: int, seed: int) -> tuple[Image.Image, dict]:
    if operator == "gaussian_blur":
        radius = round((1.0, 3.0, 8.0)[level] * max(1.0, min(image.size) / 512), 4)
        return image.filter(ImageFilter.GaussianBlur(radius)), {"radius_pixels": radius}
    if operator == "gaussian_noise":
        sigma = (5.0, 15.0, 35.0)[level]
        # Reusing the family seed scales the same noise field for all three variants.
        noise = np.random.default_rng(seed).normal(0, sigma, (image.height, image.width, 3))
        pixels = np.clip(np.asarray(image, dtype=np.float64) + noise, 0, 255).round().astype(np.uint8)
        return Image.fromarray(pixels), {"sigma_8bit": sigma}
    return image, {"quality": (70, 30, 8)[level], "subsampling": 2}


def _cached_record(path: Path, expected: dict) -> dict | None:
    """Reuse only complete records whose identity and current image bytes match."""
    try:
        cached = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(cached, dict) or cached.get("identity") != expected:
        return None
    record = cached.get("record")
    if not isinstance(record, dict) or record.get("image_path") != str(path) or not path.is_file():
        return None
    if record.get("sha256") != file_sha256(path):
        return None
    return record


def _parent_view(path: Path) -> tuple[Image.Image, dict]:
    with Image.open(path) as opened:
        # Copy decoded RGB pixels, excluding source metadata from the output image.
        original = Image.fromarray(np.asarray(ImageOps.exif_transpose(opened).convert("RGB")))
    source_width, source_height = original.size
    resized = max(original.size) > MAX_VARIANT_EDGE
    if resized:
        original.thumbnail((MAX_VARIANT_EDGE, MAX_VARIANT_EDGE), Image.Resampling.LANCZOS)
    return original, {
        "orientation": "exif_transpose",
        "source_dimensions": {"width": source_width, "height": source_height},
        "delivered_dimensions": {"width": original.width, "height": original.height},
        "max_edge": MAX_VARIANT_EDGE,
        "resized": resized,
        "resampling": "lanczos" if resized else "none",
        "aspect_ratio_policy": "preserve_with_integer_rounding",
        "source_aspect_ratio": source_width / source_height,
        "delivered_aspect_ratio": original.width / original.height,
    }


def make_variants(root: Path, parents: list[dict], seed: int) -> list[dict]:
    """Make three variants per unique parent, retaining the parent scene group.

    Operator choice and random draws depend on parent ID, never input order.
    Original files and records are unchanged. Large parents share one resized
    view before the three transforms. No source ratings are copied. Existing
    variants are reused only after hashing both their original and output bytes.
    """
    if len({parent["id"] for parent in parents}) != len(parents):
        raise ValueError("Distortion parents must have unique IDs")
    root = Path(root).resolve()
    records = []
    for parent in sorted(parents, key=lambda row: row["id"]):
        family_seed = _seed(seed, f"{DISTORTION_VERSION}/{parent['id']}")
        operator = OPERATORS[family_seed % len(OPERATORS)]
        provenance = {key: deepcopy(parent[key]) for key in PROVENANCE_FIELDS if key in parent}
        parent_source = {
            key: deepcopy(parent.get(key))
            for key in ("source", "source_id", "source_revision", "source_url", "download_url")
        }
        # Verify the actual input, even when the caller supplies a cached SHA.
        parent_source["sha256"] = file_sha256(Path(parent["image_path"]))
        software = {"pillow": pillow_version, "numpy": np.__version__}
        original = None
        preprocessing = None
        for level in range(3):
            identity = hashlib.sha256(
                f"{DISTORTION_VERSION}/{seed}/{parent['id']}/{level}".encode()
            ).hexdigest()[:24]
            identifier = f"distortion-{identity}"
            extension = "jpg" if operator == "jpeg_compression" else "png"
            path = root / "images" / "generated_distortion" / f"{identifier}.{extension}"
            cache_identity = {
                "id": identifier,
                "parent_id": parent["id"],
                "seed": seed,
                "family_seed": family_seed,
                "version": DISTORTION_VERSION,
                "max_edge": MAX_VARIANT_EDGE,
                "software": software,
                "operator": operator,
                "severity_index": level + 1,
                "parent_source": parent_source,
                "provenance": provenance,
            }
            cached = _cached_record(path, cache_identity)
            if cached is not None:
                records.append(cached)
                continue
            if original is None:
                original, preprocessing = _parent_view(Path(parent["image_path"]))
            image, parameters = _variant(original, operator, level, family_seed)
            record = deepcopy(provenance)
            record.update(
                {
                    "id": identifier,
                    "source": "generated_distortion",
                    "source_id": identifier,
                    "source_revision": DISTORTION_VERSION,
                    "download_url": None,
                    "parent_id": parent["id"],
                    "source_group": parent["source_group"],
                    "parent_source": deepcopy(parent_source),
                    "source_labels": {},
                    "technical_quality": None,
                    "aesthetics": None,
                    "preprocessing": deepcopy(preprocessing),
                    "transform": {
                        "operator": operator,
                        "parameters": parameters,
                        "severity_index": level + 1,
                        "seed": family_seed,
                        "version": DISTORTION_VERSION,
                        "software": software,
                    },
                }
            )
            record.update(_save(image, path, jpeg_quality=parameters.get("quality")))
            atomic_json(
                path.with_suffix(path.suffix + ".json"), {"identity": cache_identity, "record": record}
            )
            records.append(record)
    return records


def _chart(draw: ImageDraw.ImageDraw, rng: np.random.Generator, accent: tuple, template: int) -> None:
    draw.line([(70, 100), (70, 425), (585, 425)], fill="#263746", width=3)
    for y in range(125, 425, 75):
        draw.line([(72, y), (585, y)], fill="#dae1e7")
    count = 6 + template % 4
    step = 480 // count
    values = rng.integers(60, 295, size=count)
    if template // 4:
        points = [(95 + i * step, 425 - int(value)) for i, value in enumerate(values)]
        if template // 4 == 1:
            draw.line(points, fill=accent, width=5)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=accent)
    else:
        for i, value in enumerate(values):
            x = 85 + i * step
            draw.rectangle((x, 425 - int(value), x + step - 15, 424), fill=accent)


def _flowchart(draw: ImageDraw.ImageDraw, rng: np.random.Generator, accent: tuple, template: int) -> None:
    font = ImageFont.load_default(size=18)
    columns = 2 + template % 2
    count = 4 + template // 4
    nodes = [
        (90 + 190 * (i % columns), 120 + (110 if columns == 2 else 190) * (i // columns))
        for i in range(count)
    ]
    for index, (x, y) in enumerate(nodes):
        if index + 1 < len(nodes):
            end_x, end_y = nodes[index + 1]
            draw.line([(x + 60, y + 30), (end_x + 60, end_y + 30)], fill="#798999", width=3)
        if (index + template) % 3 == 0:
            draw.polygon(
                [(x + 60, y - 10), (x + 130, y + 30), (x + 60, y + 70), (x - 10, y + 30)],
                fill=accent,
                outline="#253746",
            )
        else:
            draw.rounded_rectangle((x, y, x + 120, y + 60), radius=12, fill=accent, outline="#253746")
        draw.text((x + 25, y + 20), f"Step {index + 1}", fill="white", font=font)


def _circuit(draw: ImageDraw.ImageDraw, rng: np.random.Generator, accent: tuple, template: int) -> None:
    font = ImageFont.load_default(size=16)
    rows, columns = 2 + template % 3, 2 + template // 3 % 2
    row_step = 260 // rows
    column_step = 380 // columns
    for row in range(rows):
        y = 140 + row * row_step
        draw.line([(70, y), (560, y)], fill=accent, width=3)
        for column in range(columns):
            x = 160 + column * column_step
            if (row + column + template) % 2:
                draw.rectangle((x - 28, y - 15, x + 28, y + 15), fill="white", outline=accent, width=3)
            else:
                draw.rectangle((x - 10, y - 20, x + 10, y + 20), fill="white")
                draw.line([(x - 6, y - 22), (x - 6, y + 22)], fill=accent, width=3)
                draw.line([(x + 6, y - 22), (x + 6, y + 22)], fill=accent, width=3)
            draw.text((x - 15, y - 42), f"X{row * columns + column}", fill="#263746", font=font)
        draw.line([(70, y), (70, min(y + row_step, 140 + (rows - 1) * row_step))], fill=accent, width=3)
    draw.line([(560, 140), (560, 140 + (rows - 1) * row_step)], fill=accent, width=3)


def _interface(draw: ImageDraw.ImageDraw, rng: np.random.Generator, accent: tuple, template: int) -> None:
    font = ImageFont.load_default(size=17)
    draw.rounded_rectangle((35, 90, 605, 445), radius=12, fill="white", outline="#c4ced8", width=2)
    draw.rectangle((36, 91, 150, 444), fill=accent)
    for row in range(3 + template % 3):
        draw.text((52, 118 + row * 48), f"Panel {row + 1}", fill="white", font=font)
    columns = 2 + template // 3 % 2
    card_width = 390 // columns
    rows = 1 + template // 6
    for row in range(rows):
        for column in range(columns):
            x, y = 170 + column * (card_width + 8), 120 + row * 140
            draw.rounded_rectangle(
                (x, y, x + card_width, y + 115), radius=8, fill="#eef2f6", outline="#cad5df"
            )
            draw.text((x + 12, y + 14), f"Item {1 + row * 3 + column}", fill="#263746", font=font)
            width = int(rng.integers(25, 93))
            draw.rectangle((x + 12, y + 61, x + width, y + 75), fill=accent)


def _render(draw: ImageDraw.ImageDraw, rng: np.random.Generator, accent: tuple, template: int) -> None:
    count, columns = 2 + template % 4, 2 + template // 4 % 2
    for index in range(count):
        x, y = 100 + (440 // columns) * (index % columns), 170 + 105 * (index // columns)
        size = int(rng.integers(35, 65))
        draw.ellipse((x - size, y + 55, x + size * 2, y + 80), fill="#d5dce4")
        top = [(x, y - size), (x + size, y - size // 2), (x, y), (x - size, y - size // 2)]
        left = [(x - size, y - size // 2), (x, y), (x, y + size), (x - size, y + size // 2)]
        right = [(x, y), (x + size, y - size // 2), (x + size, y + size // 2), (x, y + size)]
        draw.polygon(top, fill=tuple(min(255, channel + 55) for channel in accent))
        draw.polygon(left, fill=accent)
        draw.polygon(right, fill=tuple(max(0, channel - 40) for channel in accent))


def _control_image(kind: str, template: int, seed: int, disrupted: bool) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (640, 480), "#f8fafc")
    draw = ImageDraw.Draw(image)
    accent = tuple(int(value) for value in rng.integers(35, 170, size=3))
    draw.text(
        (35, 28),
        f"{kind.replace('_', ' ').title()} {template + 1}",
        fill="#233443",
        font=ImageFont.load_default(size=24),
    )
    renderer = dict(zip(CONTROL_KINDS, (_chart, _flowchart, _circuit, _interface, _render), strict=True))[
        kind
    ]
    renderer(draw, rng, accent, template)
    if disrupted:
        # These are construction flags, not claims about quality or aesthetics.
        for _ in range(12):
            x, y = (int(value) for value in rng.integers([0, 60], [580, 450]))
            draw.line([(x, y), (x + 60, y - 40)], fill="#8b3444", width=3)
        pixels = np.asarray(image, dtype=np.int16)
        noise = rng.integers(-18, 19, size=pixels.shape, dtype=np.int16)
        image = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
    return image


def make_owned_controls(root: Path, count: int, seed: int) -> list[dict]:
    """Draw original controls without external image assets or API calls.

    The first 240 rows cover 60 template groups with four variants each. Extending
    the count preserves prior IDs and pixels. Related templates keep one group.
    """
    if count < 0:
        raise ValueError("Control count must be non-negative")
    root = Path(root).resolve()
    records = []
    for index in range(count):
        kind = CONTROL_KINDS[index % len(CONTROL_KINDS)]
        template = (index // len(CONTROL_KINDS)) % 12
        template_id = f"{kind}-{template:02d}"
        variant = index // (len(CONTROL_KINDS) * 12)
        item_seed = _seed(seed, f"{CONTROL_VERSION}/{template_id}/{variant}")
        identifier = f"owned-{seed}-{index:06d}"
        disrupted = variant % 4 == 3
        image = _control_image(kind, template, item_seed, disrupted)
        record = {
            "id": identifier,
            "source": "owned_controls",
            "source_id": identifier,
            "source_url": f"local-generator://{CONTROL_VERSION}/{template_id}/{variant}",
            "download_url": None,
            "source_revision": CONTROL_VERSION,
            "source_split": "generated",
            "source_group": f"{CONTROL_VERSION}/{template_id}",
            "parent_id": None,
            "license": {
                "id": "LicenseRef-Project-Owned",
                "url": None,
                "evidence": "Original geometry and text generated by this project; no external image assets.",
                "font": {
                    "name": "Pillow embedded Aileron Regular",
                    "terms": "No Rights Reserved",
                    "url": "https://dotcolon.net/fonts/aileron/",
                },
                "renderer_license_url": f"https://github.com/python-pillow/Pillow/blob/{pillow_version}/LICENSE",
            },
            "source_labels": {},
            "technical_quality": None,
            "aesthetics": None,
            "content_kind": kind,
            "source_origin": "render",
            "photo_domain_eligible": False,
            "generation": {
                "version": CONTROL_VERSION,
                "template_id": template_id,
                "variant": variant,
                "seed": item_seed,
                "layout_disrupted": disrupted,
                "software": {"pillow": pillow_version, "numpy": np.__version__},
            },
        }
        record.update(_save(image, root / "images" / "owned_controls" / f"{identifier}.png"))
        records.append(record)
    return records
