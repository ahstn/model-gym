"""Check split isolation and duplicate rejection for native-photo acquisition."""

from PIL import Image

from tracks.metadata_v3.acquire_real import duplicate, perceptual_hash, source_devices


def test_selection_keeps_camera_models_disjoint_and_ignores_social_versions():
    urls = []
    for device in range(50):
        model = device // 2
        base = f"https://example.invalid/dataset/D{device:02d}_Brand_Model{model}/images"
        urls += [f"{base}/nat/a.jpg", f"{base}/natFBL/b.jpg", f"{base}/flat/c.jpg"]
    devices, assignments = source_devices(urls)
    assert len(assignments) == 20
    assert len({name.split("_", 1)[1] for name in assignments}) == 20
    assert {split: list(assignments.values()).count(split) for split in ("train", "dev", "test")} == {
        "train": 5,
        "dev": 5,
        "test": 10,
    }
    assert all(len(rows) == 1 and "/nat/" in rows[0] for rows in devices.values())
    assert source_devices(list(reversed(urls)))[1] == assignments


def test_duplicate_check_flags_nearby_phash_without_matching_bytes():
    image = Image.new("RGB", (64, 64))
    image.putdata([(x * 4, y * 4, (x + y) * 2) for y in range(64) for x in range(64)])
    fingerprint = perceptual_hash(image)
    row = {
        "id": "current",
        "original_sha256": "a",
        "decoded_rgb_sha256": "b",
        "sized_rgb_sha256": "c",
        "phash": f"{fingerprint:016x}",
    }
    other = {
        "id": "prior",
        "original_sha256": "d",
        "decoded_rgb_sha256": "e",
        "sized_rgb_sha256": "f",
        "phash": f"{fingerprint ^ 7:016x}",
    }
    assert duplicate(row, [other]) == {"kind": "phash", "distance": 3, "other": "prior"}
    assert duplicate(row, [{**other, "phash": f"{fingerprint ^ ((1 << 64) - 1):016x}"}]) is None
