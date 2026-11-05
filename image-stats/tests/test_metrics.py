import io
import math

import cv2
import numpy as np
import pytest

from PIL import Image

from image_stats import METRIC_NAMES, analyse
from tests.images import add_noise, blur, encode, gradient, solid, texture


def metrics(rgb):
    return analyse(rgb).metrics


def test_every_registered_metric_is_finite():
    values = metrics(texture())
    assert tuple(values) == METRIC_NAMES
    assert all(math.isfinite(value) for value in values.values())


def test_noise_sigma_recovers_known_luma_noise():
    clean = metrics(gradient())
    noisy = metrics(add_noise(gradient(), 8.0))
    assert clean["noise_sigma"] < 0.5
    assert noisy["noise_sigma"] == pytest.approx(8.0, rel=0.1)
    # The same noise on every channel is luma noise, not colour noise.
    assert noisy["chroma_noise_sigma"] < 1.0


def test_blur_lowers_every_sharpness_measure():
    series = [metrics(texture()), *(metrics(blur(texture(), sigma)) for sigma in (1, 2, 4))]
    for name in ("laplacian_var", "sharpness_peak", "tenengrad", "high_frequency_ratio"):
        values = [item[name] for item in series]
        assert values == sorted(values, reverse=True), name


def test_peak_and_focus_area_separate_shallow_focus_from_global_blur():
    sharp = texture()
    shallow = sharp.copy()
    shallow[:, 288:] = blur(sharp, 4)[:, 288:]
    full, partial, blurred = metrics(sharp), metrics(shallow), metrics(blur(sharp, 4))
    assert partial["sharpness_peak"] == pytest.approx(full["sharpness_peak"], rel=0.25)
    assert blurred["sharpness_peak"] < 0.05 * full["sharpness_peak"]
    assert 0.35 < partial["focus_area"] < 0.65
    assert full["focus_area"] > 0.9


def test_motion_blur_raises_gradient_anisotropy():
    kernel = np.zeros((15, 15), np.float32)
    kernel[7, :] = 1 / 15
    smeared = cv2.filter2D(texture(), -1, kernel)
    assert metrics(smeared)["gradient_anisotropy"] > 0.6
    assert metrics(blur(texture(), 2))["gradient_anisotropy"] < 0.2


def test_clipping_and_haze():
    split = np.concatenate([solid(0, width=288), solid(255, width=288)], axis=1)
    values = metrics(split)
    assert values["shadow_clip"] == pytest.approx(0.5, abs=0.01)
    assert values["highlight_clip"] == pytest.approx(0.5, abs=0.01)
    hazy = (0.4 * texture() + 0.6 * 255).astype(np.uint8)
    assert metrics(hazy)["dark_channel_mean"] > metrics(texture())["dark_channel_mean"] + 0.3


def test_colour_measures_and_monochrome():
    grey = np.repeat(texture()[..., :1], 3, axis=2)
    grey_values, colour_values = metrics(grey), metrics(texture())
    assert grey_values["colourfulness"] < 0.5
    assert grey_values["chroma_p99"] < 1.0
    assert colour_values["colourfulness"] > 20
    assert "monochrome" in analyse(grey).flags
    assert "monochrome" not in analyse(texture()).flags


def test_blockiness_tracks_jpeg_quality_at_native_resolution():
    def decoded(quality):
        payload = encode(texture(), "JPEG", quality=quality)
        return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))

    assert metrics(decoded(10))["blockiness"] > 1.3
    assert metrics(decoded(95))["blockiness"] < 1.1
    assert metrics(texture())["blockiness"] == pytest.approx(1.0, abs=0.05)
    # Rotation and an off-grid crop move the 8x8 phase; the measure must still find it.
    shifted = np.rot90(decoded(10)[3:, 5:]).copy()
    assert metrics(shifted)["blockiness"] > 1.3


def test_blockiness_ignores_near_flat_frames():
    payload = encode(
        np.clip(np.rint(solid(1) + np.random.default_rng(0).normal(0, 0.4, (384, 576, 3))), 0, 255).astype(np.uint8),
        "JPEG",
        quality=30,
    )
    assert metrics(np.asarray(Image.open(io.BytesIO(payload))))["blockiness"] == 1.0


def test_content_measures_for_a_solid_frame():
    values = metrics(solid((200, 30, 30)))
    assert values["palette_size"] == 1
    assert values["flat_area"] == 1.0
    assert values["top_colour_share"] == 1.0
    assert metrics(texture())["palette_size"] > 100


def test_rejects_images_below_minimum_size():
    with pytest.raises(ValueError, match="at least 16"):
        analyse(solid(0, height=8, width=64))
