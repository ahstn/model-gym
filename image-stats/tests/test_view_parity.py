"""Run with `mise run parity` (or `uv run --with ../visual-quality pytest tests/test_view_parity.py`)."""

import io

import numpy as np
import pytest

from PIL import Image, ImageCms

from image_stats.loading import decode, fit
from tests.images import encode, texture

views = pytest.importorskip("visual_quality.views")


def srgb_icc():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def rotated_jpeg_with_icc():
    exif = Image.Exif()
    exif[274] = 6
    return encode(texture(900, 1300), "JPEG", quality=90, exif=exif, icc_profile=srgb_icc())


def transparent_png():
    rgba = np.dstack([texture(700, 500), np.linspace(0, 255, 500, dtype=np.uint8)[None, :].repeat(700, 0)])
    return encode(rgba)


@pytest.mark.parametrize("payload", [rotated_jpeg_with_icc(), transparent_png(), encode(texture(2000, 3000))])
def test_fitted_pixels_match_frozen_views(payload):
    view_png, _ = views._render(payload, 1536)
    expected = np.asarray(Image.open(io.BytesIO(view_png)).convert("RGB"))
    assert np.array_equal(fit(decode(payload).rgb, 1536), expected)
