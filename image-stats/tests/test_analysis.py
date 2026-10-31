import numpy as np
import pytest

from PIL import Image

from image_stats import ALGORITHM_VERSION, Analysis, analyse, analyse_file
from tests.images import blur, encode, solid, texture


@pytest.mark.parametrize("colour", [0, (200, 20, 20)])
def test_uniform_frames_are_gated(colour):
    analysis = analyse(solid(colour))
    assert analysis.gate == "uniform_frame"
    assert analysis.to_dict()["gate"] == "uniform_frame"


def test_black_frame_is_also_near_black_and_texture_has_no_flags():
    assert {"uniform_frame", "near_black", "shadow_clipping"} <= set(analyse(solid(0)).flags)
    plain = analyse(texture())
    assert plain.flags == ()
    assert plain.gate is None


def test_graphic_like_needs_bright_content():
    logo = np.full((384, 576, 3), 255, np.uint8)
    logo[150:230, 200:380] = (20, 60, 200)
    assert "graphic_like" in analyse(logo).flags
    dark_photo = analyse((blur(texture(), 3) * 0.12).astype(np.uint8))
    # Few colour bins and flat tiles, like a graphic, but with no bright content.
    assert dark_photo.metrics["palette_size"] <= 24 and dark_photo.metrics["flat_area"] >= 0.5
    assert "graphic_like" not in dark_photo.flags


def test_array_bytes_path_and_pil_inputs_agree(tmp_path):
    rgb = texture()
    path = tmp_path / "image.png"
    path.write_bytes(encode(rgb))
    results = [
        analyse(rgb).to_dict(),
        analyse(encode(rgb)).to_dict(),
        analyse(path).to_dict(),
        analyse(Image.fromarray(rgb)).to_dict(),
    ]
    assert all(result["metrics"] == results[0]["metrics"] for result in results)
    assert analyse(rgb).to_dict() == results[0]


def test_downscales_to_max_edge_without_upscaling():
    large = analyse(texture(400, 800), max_edge=256)
    assert large.native_dimensions == (800, 400)
    assert large.analysed_dimensions == (256, 128)
    small = analyse(texture(100, 200), max_edge=256)
    assert small.analysed_dimensions == (200, 100)


def test_exif_orientation_is_applied():
    exif = Image.Exif()
    exif[274] = 6
    payload = encode(texture(200, 300), "JPEG", quality=95, exif=exif)
    analysis = analyse(payload)
    assert analysis.native_dimensions == (200, 300)
    assert analysis.exif_orientation == 6


def test_transparent_pixels_composite_on_white():
    rgba = np.zeros((64, 64, 4), np.uint8)
    assert analyse(encode(rgba)).metrics["luma_mean"] == pytest.approx(1.0)


def test_dictionary_round_trip_and_version():
    data = analyse(texture()).to_dict()
    assert data["version"] == ALGORITHM_VERSION
    assert Analysis.from_dict(data).to_dict() == data


def test_analyse_file_reads_metadata_from_the_original(tmp_path):
    original = tmp_path / "original.jpg"
    original.write_bytes(encode(texture(), "JPEG", quality=20))
    view = tmp_path / "view.png"
    view.write_bytes(encode(np.asarray(Image.open(original).resize((288, 192)))))
    record = analyse_file(view, source_path=original)
    assert record["metadata"]["format"] == "JPEG"
    assert record["metadata"]["jpeg_quality_estimate"] == pytest.approx(20, abs=3)
    assert record["metadata"]["source_blockiness"] > 1.2
    assert record["analysis"]["analysed_dimensions"] == [288, 192]
    assert "source_blockiness" not in analyse_file(original)["metadata"]
