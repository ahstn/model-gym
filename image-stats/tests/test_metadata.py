import io

import numpy as np
import pytest

from PIL import Image

from image_stats import read_metadata
from tests.images import encode, texture


@pytest.mark.parametrize("quality", [30, 75, 95])
def test_jpeg_quality_estimate(quality):
    metadata = read_metadata(encode(texture(), "JPEG", quality=quality))
    assert metadata["jpeg_quality_estimate"] == pytest.approx(quality, abs=3)
    assert metadata["jpeg_subsampling"] == "4:2:0"


def test_mpo_files_get_jpeg_fields():
    buffer = io.BytesIO()
    frame = Image.fromarray(texture())
    frame.save(buffer, "MPO", save_all=True, append_images=[frame], quality=60)
    metadata = read_metadata(buffer.getvalue())
    assert metadata["format"] == "MPO"
    assert metadata["jpeg_quality_estimate"] == pytest.approx(60, abs=3)


def test_camera_exif_fields_without_location_or_time():
    exif = Image.Exif()
    exif[271], exif[272], exif[305] = "Acme\x00", "Lens 9", "Firmware 1"
    details = exif.get_ifd(0x8769)
    details[33434], details[33437], details[34855], details[37386] = 1 / 125, 2.8, 800, 26.0
    metadata = read_metadata(encode(texture(), "JPEG", quality=90, exif=exif))
    assert metadata["exif_present"] is True
    assert (metadata["camera_make"], metadata["camera_model"], metadata["software"]) == (
        "Acme",
        "Lens 9",
        "Firmware 1",
    )
    assert metadata["exposure_time_s"] == pytest.approx(0.008)
    assert metadata["f_number"] == pytest.approx(2.8)
    assert metadata["iso"] == 800
    assert metadata["focal_length_mm"] == 26
    assert not any("gps" in key or "date" in key for key in metadata)


def test_png_without_exif(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(encode(np.zeros((40, 50, 4), np.uint8)))
    metadata = read_metadata(path)
    assert metadata["format"] == "PNG"
    assert metadata["stored_dimensions"] == [50, 40]
    assert metadata["has_alpha"] is True
    assert metadata["exif_present"] is False
    assert metadata["jpeg_quality_estimate"] is None
    assert metadata["camera_make"] is None
