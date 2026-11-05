import io
import zipfile

import pytest
from PIL import Image

from visual_quality import controls


def png_bytes(color="red"):
    output = io.BytesIO()
    Image.new("RGB", (12, 8), color).save(output, format="PNG")
    return output.getvalue()


def plotqa_row(index=10, split="train", revision=controls.PLOTQA_REVISION):
    return {
        "row_idx": index,
        "row": {
            "image": {
                "src": (
                    "https://datasets-server.huggingface.co/cached-assets/achang/plot_qa/--/"
                    f"{revision}/--/default/{split}/{index}/image/image.png"
                ),
                "width": 12,
                "height": 8,
            },
            "text": "<s_y>1<sep/>2</s_y><s_name>example</s_name>",
        },
    }


@pytest.mark.parametrize(
    "change",
    [
        {"split": "test"},
        {"split": "validation"},
        {"revision": "different"},
        {"index": -1},
        {"index": controls.PLOTQA_TRAIN_COUNT},
    ],
)
def test_plotqa_rejects_other_splits_revisions_and_invalid_rows(change):
    with pytest.raises(ValueError):
        controls._validate_plotqa_row(plotqa_row(**change))


def test_plotqa_manifest_preserves_provenance_without_claiming_original_pixels(tmp_path, monkeypatch):
    raw = png_bytes()

    def download_stub(url, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    monkeypatch.setattr(controls, "download", download_stub)
    evidence = tmp_path / "license.md"
    records = [
        controls._plotqa_record(tmp_path, plotqa_row(index=i), "viewer-window", evidence) for i in (10, 11)
    ]
    assert len({r["id"] for r in records}) == 2
    assert {r["source_split"] for r in records} == {"train"}
    assert {r["source_revision"] for r in records} == {controls.PLOTQA_REVISION}
    assert all(r["license"]["id"] == "CC-BY-4.0" for r in records)
    assert all(r["photo_domain_eligible"] is False for r in records)
    assert all(r["delivery"]["original_pixels_verified"] is False for r in records)
    assert records[0]["decoded_sha256"] == records[1]["decoded_sha256"]
    assert "technical_quality" not in records[0]["source_labels"]


def test_plotqa_group_ignores_recorded_rendering_properties():
    series = "<s_y>1<sep/>2</s_y><s_name>example</s_name>"
    a = controls._plotqa_group(series + "<s_color>red</s_color><s_bboxes>10</s_bboxes>", 1)
    b = controls._plotqa_group(series + "<s_color>blue</s_color><s_bboxes>99</s_bboxes>", 2)
    assert a[0] == b[0]
    assert "canonical template group unknown" in a[1]
    assert "unknown-group" in controls._plotqa_group("", 5)[0]


def test_clevr_selects_only_train_members_deterministically():
    train = [f"CLEVR_v1.0/images/train/CLEVR_train_{i:06d}.png" for i in range(70000)]
    mixed = train + [
        "CLEVR_v1.0/images/test/CLEVR_test_000001.png",
        "CLEVR_v1.0/images/val/CLEVR_val_000002.png",
    ]
    selected = controls._clevr_members(mixed, 20260927)
    assert len(set(selected)) == 70000
    assert all("/images/train/" in entry for entry in selected)
    assert selected[:500] == controls._clevr_members(list(reversed(mixed)), 20260927)[:500]


def test_clevr_png_bytes_and_source_ids_are_preserved(tmp_path):
    raw = png_bytes()
    records = [
        controls._clevr_record(
            tmp_path, f"CLEVR_v1.0/images/train/CLEVR_train_{i:06d}.png", raw, tmp_path / "COPYRIGHT.txt"
        )
        for i in (1, 2)
    ]
    assert len({r["id"] for r in records}) == 2
    assert len({r["source_group"] for r in records}) == 2
    assert {r["source_split"] for r in records} == {"train"}
    assert all(r["delivery"]["original_bytes_verified"] for r in records)
    assert (tmp_path / "images" / "CLEVR_train_000001.png").read_bytes() == raw
    with pytest.raises(ValueError, match="restricted"):
        controls._clevr_record(
            tmp_path, "CLEVR_v1.0/images/test/CLEVR_test_000001.png", raw, tmp_path / "COPYRIGHT.txt"
        )


def test_changed_acquisition_seed_is_rejected(tmp_path):
    controls._checkpoint(tmp_path, [], {"seed": 1, "source_revision": "pinned"}, complete=True)
    with pytest.raises(ValueError, match="different seed or revision"):
        controls._cached(tmp_path, 10, 2, "pinned")


def test_original_zip_member_integrity_is_verified():
    raw = png_bytes()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("image.png", raw)
    with zipfile.ZipFile(io.BytesIO(output.getvalue())) as archive:
        info = archive.getinfo("image.png")
        assert controls._decode_zip_member(output.getvalue(), info) == raw
        info.CRC = 0
        with pytest.raises(ValueError, match="integrity"):
            controls._decode_zip_member(output.getvalue(), info)
