"""Freeze full-image, colour-managed views without changing source manifests."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageCms, ImageOps, PngImagePlugin, features
from PIL import __version__ as pillow_version

from visual_quality.storage import atomic_json, file_sha256, read_jsonl, write_jsonl

_COMPONENTS = ("megalith", "vizwiz", "distortions")
_VERSION = "fixed-view-preparation-v1"
_VIEW_KEYS = {
    "view_id",
    "image_id",
    "source_sha256",
    "view_sha256",
    "image_path",
    "width",
    "height",
    "mime_type",
    "split",
    "split_group",
    "component",
    "photo_domain_eligible",
    "preprocessing",
}


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Expected a nonempty relative image or manifest path")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("Path escapes its declared root")
    return candidate


def _dataset(dataset_dir: Path) -> tuple[dict, Path, list[dict]]:
    dataset_path = dataset_dir / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    if dataset.get("schema_version") != 1 or dataset.get("image_root") != "..":
        raise ValueError("Expected assembled dataset v1 with image_root '..'")
    manifest = _inside(dataset_dir, dataset["manifests"]["train"])
    rows = read_jsonl(manifest)
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("Expected a nonempty training-only manifest")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Training manifest contains duplicate image IDs")
    if any(
        row.get("photo_domain_eligible") is not None and type(row["photo_domain_eligible"]) is not bool
        for row in rows
    ):
        raise ValueError("Photo eligibility must be true, false, or null")
    source = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "image_root": str(dataset_dir.parent),
    }
    return source, dataset_dir.parent, rows


def _select(rows: list[dict], limit: int, seed: int) -> list[dict]:
    pools = {
        component: sorted(
            (
                row
                for row in rows
                if row["component"] == component and row.get("photo_domain_eligible") is not False
            ),
            key=lambda row: _digest([seed, row["id"]]),
        )
        for component in _COMPONENTS
    }
    selected = []
    for index in range(max((len(pool) for pool in pools.values()), default=0)):
        for component in _COMPONENTS:
            if index < len(pools[component]):
                selected.append(pools[component][index])
                if len(selected) == limit:
                    return selected
    raise ValueError(f"Only {len(selected)} eligible candidate rows exist; requested {limit}")


def _request(limit: int, seed: int, max_edge: int) -> dict:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("View preparation requires a limit from 1 to 100")
    if type(seed) is not int or type(max_edge) is not int or not 64 <= max_edge <= 4096:
        raise ValueError("Seed must be an integer and max_edge must be from 64 to 4096")
    return {"limit": limit, "seed": seed, "max_edge": max_edge}


def _policy(max_edge: int) -> dict:
    return {
        "version": "full-image-srgb-png-v1",
        "max_edge": max_edge,
        "orientation": "EXIF transpose",
        "colour": "embedded ICC to sRGB; missing ICC assumes sRGB",
        "colour_rendering_intent": "perceptual (ICC intent 0)",
        "alpha": "composite on white after colour conversion",
        "resize": "LANCZOS fit; no upscale",
        "crop": "none",
        "encoding": "RGB PNG with sRGB chunk; no EXIF",
        "autocorrection": False,
        "selection": {
            "split": "train",
            "components": list(_COMPONENTS),
            "method": "seeded hash order within component; round robin",
            "exclude_domain_false": True,
        },
        "software": {
            "pillow": pillow_version,
            "littlecms": features.version_module("littlecms2"),
            "module_sha256": file_sha256(Path(__file__)),
        },
    }


def _render(payload: bytes, max_edge: int) -> tuple[bytes, dict]:
    with Image.open(io.BytesIO(payload)) as original:
        original.load()
        if getattr(original, "n_frames", 1) != 1:
            raise ValueError("Fixed image views do not accept animated or multi-page inputs")
        native_size, native_mode = list(original.size), original.mode
        orientation = original.getexif().get(274, 1)
        icc = original.info.get("icc_profile")
        oriented = ImageOps.exif_transpose(original)
        decoded_size = list(oriented.size)
        has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
        alpha = oriented.convert("RGBA").getchannel("A") if has_alpha else None
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        if icc is not None:
            try:
                profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                cms_input = (
                    oriented if oriented.mode in ("RGB", "CMYK", "LAB", "L") else oriented.convert("RGB")
                )
                rgb = ImageCms.profileToProfile(cms_input, profile, srgb, outputMode="RGB")
            except (ImageCms.PyCMSError, OSError, ValueError) as error:
                raise ValueError("Cannot convert embedded ICC profile to sRGB") from error
            colour_action = "embedded ICC converted to sRGB"
        else:
            rgb = oriented.convert("RGB")
            colour_action = "missing ICC assumed sRGB; Pillow mode conversion to RGB"
        if alpha is not None:
            rgb = Image.composite(rgb, Image.new("RGB", rgb.size, "white"), alpha)
        rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        rgb.info.clear()
        buffer = io.BytesIO()
        # A standard sRGB chunk avoids the creation timestamp in generated ICC profiles.
        png_info = PngImagePlugin.PngInfo()
        png_info.add(b"sRGB", b"\x00")
        rgb.save(buffer, format="PNG", pnginfo=png_info, optimize=False, compress_level=6)
        preprocessing = {
            "native_dimensions": native_size,
            "decoded_dimensions": decoded_size,
            "output_dimensions": list(rgb.size),
            "native_mode": native_mode,
            "exif_orientation": orientation,
            "orientation_applied": orientation not in (None, 1),
            "source_icc_sha256": hashlib.sha256(icc).hexdigest() if icc is not None else None,
            "colour_action": colour_action,
            "alpha_composited_on_white": has_alpha,
            "resized": list(rgb.size) != decoded_size,
            "crop_bounds": None,
        }
        return buffer.getvalue(), preprocessing


def _view(row: dict, image_root: Path, destination: Path, policy: dict) -> dict:
    source_path = _inside(image_root, row["image_path"])
    payload = source_path.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    if source_hash != row["sha256"]:
        raise ValueError(f"Source image hash changed: {row['id']}")
    data, preprocessing = _render(payload, policy["max_edge"])
    view_hash, policy_hash = hashlib.sha256(data).hexdigest(), _digest(policy)
    relative = f"images/{view_hash}.png"
    (destination / relative).write_bytes(data)
    preprocessing.update(policy_sha256=policy_hash, software=policy["software"])
    return {
        "view_id": _digest([row["id"], source_hash, policy_hash, view_hash]),
        "image_id": row["id"],
        "source_sha256": source_hash,
        "view_sha256": view_hash,
        "image_path": relative,
        "width": preprocessing["output_dimensions"][0],
        "height": preprocessing["output_dimensions"][1],
        "mime_type": "image/png",
        "split": "train",
        "split_group": row["split_group"],
        "component": row["component"],
        "photo_domain_eligible": row.get("photo_domain_eligible"),
        "preprocessing": preprocessing,
    }


def prepare_views(
    dataset_dir: Path,
    output_dir: Path,
    *,
    limit: int = 12,
    seed: int = 20261018,
    max_edge: int = 1536,
) -> dict:
    """Create a bounded training sample, or verify and reuse an identical frozen sample."""
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    request = _request(limit, seed, max_edge)
    if output_dir.is_relative_to(dataset_dir) or dataset_dir.is_relative_to(output_dir):
        raise ValueError("View output must be separate from the assembled dataset directory")
    source, image_root, rows = _dataset(dataset_dir)
    policy = _policy(max_edge)
    if output_dir.exists():
        load_views(output_dir)
        preparation = json.loads((output_dir / "preparation.json").read_text())
        if any(
            preparation[key] != value
            for key, value in (("request", request), ("source", source), ("policy", policy))
        ):
            raise ValueError("Existing views have a different frozen request, source, or preparation policy")
        return preparation
    selected = _select(rows, limit, seed)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (staging / "images").mkdir()
        views = [_view(row, image_root, staging, policy) for row in selected]
        write_jsonl(staging / "views.jsonl", views)
        preparation = {
            "schema_version": _VERSION,
            "complete": True,
            "source": source,
            "request": request,
            "policy": policy,
            "policy_sha256": _digest(policy),
            "count": len(views),
            "counts_by_component": dict(Counter(view["component"] for view in views)),
            "views_sha256": file_sha256(staging / "views.jsonl"),
            "purpose": "technical pipeline smoke test; teacher observations require review",
        }
        atomic_json(staging / "preparation.json", preparation)
        load_views(staging)
        if output_dir.exists():
            raise ValueError("View output appeared during preparation; refusing to replace it")
        staging.rename(output_dir)
        return preparation
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_view(view: dict, row: dict, output_dir: Path, image_root: Path, preparation: dict) -> None:
    if set(view) != _VIEW_KEYS:
        raise ValueError("Unexpected fixed-view record fields")
    expected = {
        "image_id": row["id"],
        "source_sha256": row["sha256"],
        "split": "train",
        "split_group": row["split_group"],
        "component": row["component"],
        "photo_domain_eligible": row.get("photo_domain_eligible"),
        "mime_type": "image/png",
    }
    if any(view[key] != value for key, value in expected.items()):
        raise ValueError("Fixed-view metadata does not match its source manifest")
    if view["image_path"] != f"images/{view['view_sha256']}.png":
        raise ValueError("Expected a content-addressed PNG view path")
    source_path = _inside(image_root, row["image_path"])
    view_path = _inside(output_dir, view["image_path"])
    if file_sha256(source_path) != view["source_sha256"]:
        raise ValueError("Source image hash changed")
    if file_sha256(view_path) != view["view_sha256"]:
        raise ValueError("Frozen image view hash changed")
    policy_hash = preparation["policy_sha256"]
    identity = _digest([view["image_id"], view["source_sha256"], policy_hash, view["view_sha256"]])
    if view["view_id"] != identity or view["preprocessing"].get("policy_sha256") != policy_hash:
        raise ValueError("Fixed-view identity or preprocessing policy changed")
    if view["preprocessing"].get("software") != preparation["policy"]["software"]:
        raise ValueError("Fixed-view software metadata differs from its policy")
    if view["preprocessing"].get("crop_bounds") is not None:
        raise ValueError("Fixed-view policy does not allow cropping")
    with Image.open(view_path) as image:
        image.load()
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or list(image.size) != [view["width"], view["height"]]
        ):
            raise ValueError("Frozen image format or dimensions changed")
        if list(image.size) != view["preprocessing"].get("output_dimensions"):
            raise ValueError("Frozen image dimensions differ from preprocessing metadata")
        if max(image.size) > preparation["request"]["max_edge"]:
            raise ValueError("Frozen view exceeds its maximum edge")


def load_views(output_dir: Path) -> list[dict]:
    """Verify frozen files and their original training provenance before returning views."""
    output_dir = Path(output_dir).resolve()
    preparation_path = output_dir / "preparation.json"
    if not preparation_path.is_file() or not (output_dir / "views.jsonl").is_file():
        raise ValueError("Fixed-view preparation is incomplete")
    preparation = json.loads(preparation_path.read_text())
    if preparation.get("schema_version") != _VERSION or preparation.get("complete") is not True:
        raise ValueError("Fixed-view preparation is incomplete or has an unsupported version")
    request = _request(**preparation["request"])
    if _digest(preparation["policy"]) != preparation["policy_sha256"]:
        raise ValueError("Fixed-view policy hash changed")
    if preparation["policy"].get("max_edge") != request["max_edge"]:
        raise ValueError("Fixed-view request differs from its policy")
    source, image_root, rows = _dataset(Path(preparation["source"]["dataset_path"]).parent)
    if source != preparation["source"]:
        raise ValueError("Source dataset or training manifest changed")
    if file_sha256(output_dir / "views.jsonl") != preparation["views_sha256"]:
        raise ValueError("Frozen views manifest hash changed")
    views = read_jsonl(output_dir / "views.jsonl")
    selected = _select(rows, request["limit"], request["seed"])
    if len(views) != request["limit"] or preparation["count"] != len(views):
        raise ValueError("Frozen views count differs from the request")
    if preparation["counts_by_component"] != dict(Counter(view["component"] for view in views)):
        raise ValueError("Frozen component counts changed")
    for view, row in zip(views, selected, strict=True):
        _validate_view(view, row, output_dir, image_root, preparation)
    return views
