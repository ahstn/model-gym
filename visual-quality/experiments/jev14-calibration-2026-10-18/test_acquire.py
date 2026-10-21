"""Checks for the calibration acquisition's view and leakage controls."""

import importlib.util
import io
import math
import statistics
from pathlib import Path

from PIL import Image, ImageOps

spec = importlib.util.spec_from_file_location("calibration_acquire", Path(__file__).with_name("acquire.py"))
acquire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)


def test_delivery_matches_historical_jpeg_policy():
    image = Image.new("RGB", (640, 480), "red")
    image.paste("blue", (120, 100, 500, 400))
    source = io.BytesIO()
    image.save(source, format="PNG")
    actual, record = acquire.image_record(source.getvalue())
    expected = io.BytesIO()
    ImageOps.fit(image, (512, 512), Image.Resampling.LANCZOS, centering=(0.5, 0.5)).save(
        expected, format="JPEG", quality=95, subsampling=0
    )
    assert actual == expected.getvalue()
    assert record["source_dimensions"] == [640, 480]
    assert record["source_sha256"] == acquire.sha(source.getvalue())


def test_caption_grouping_ignores_case_and_whitespace():
    assert acquire.family_id("  A CAT\n in a room ") == acquire.family_id("a cat in a room")
    assert acquire.family_id("a dog in a room") != acquire.family_id("a cat in a room")


def test_phash_matches_prior_origin_audit_definition():
    image = Image.new("L", (32, 32))
    image.putdata([(x * 13 + y * 19 + x * y) % 256 for y in range(32) for x in range(32)])
    pixels = image.tobytes()
    cosines = [
        [math.cos(math.pi * (2 * x + 1) * frequency / 64) for x in range(32)] for frequency in range(8)
    ]
    horizontal = [
        [sum(pixels[y * 32 + x] * cosines[u][x] for x in range(32)) for u in range(8)] for y in range(32)
    ]
    coefficients = [
        sum(horizontal[y][u] * cosines[v][y] for y in range(32)) for v in range(8) for u in range(8)
    ]
    median = statistics.median(coefficients[1:])
    expected = sum((value > median) << i for i, value in enumerate(coefficients))
    assert acquire.phash(image) == expected


def test_duplicate_rule_checks_both_views_and_pilot_hashes():
    base = {
        "source_sha256": "a",
        "decoded_rgb_sha256": "b",
        "delivered_sha256": "c",
        "source_phash": "0000000000000000",
        "delivered_phash": "0000000000000000",
    }
    other = {
        "source_sha256": "d",
        "decoded_rgb_sha256": "e",
        "delivered_sha256": "f",
        "source_phash": "000000000000007f",
        "delivered_phash": "000000000000007f",
    }
    assert acquire.collisions(base, {"prior": other}, set()) is None
    other["delivered_phash"] = "000000000000003f"
    assert acquire.collisions(base, {"prior": other}, set())["distance"] == 6
    assert acquire.collisions(base, {}, {"a"})["kind"] == "pilot_exact_source_hash"
