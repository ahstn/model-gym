"""The AI panel can be restored without silently replacing frozen evidence."""

import copy
import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "ai_builder", Path(__file__).resolve().parents[1] / "benchmark_data/ai_builder.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_frozen_sample_comparison_rejects_label_changes():
    record = {"id": "ai-000", "labels": {"ai_generated": True}, "size": [32, 20]}
    builder.verify_frozen_samples({"samples": [record]}, [record])
    altered = copy.deepcopy(record)
    altered["labels"]["ai_generated"] = False
    with pytest.raises(ValueError, match="frozen sample records"):
        builder.verify_frozen_samples({"samples": [record]}, [altered])


@pytest.mark.parametrize("changed_field", [None, "original_sha256", "delivered_sha256"])
def test_cached_reproduction_validates_before_writes(tmp_path, monkeypatch, changed_field):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    stream = io.BytesIO()
    Image.new("RGB", (32, 20), "navy").save(stream, format="PNG")
    raw = stream.getvalue()
    encoded, original_size, size, _ = builder.deliver(raw)
    (tmp_path / "source-001.bin").write_bytes(raw)
    sample = {
        "id": "ai-000",
        "image": "delivered.jpg",
        "source": {"row_id": "source-001"},
        "original_sha256": builder.digest(raw),
        "delivered_sha256": builder.digest(encoded),
        "original_size": list(original_size),
        "delivered_size": list(size),
        "delivered_bytes": len(encoded),
    }
    if changed_field:
        sample[changed_field] = "0" * 64
        with pytest.raises(ValueError):
            builder.cached_deliveries([sample], tmp_path)
    else:
        assert builder.cached_deliveries([sample], tmp_path) == [(tmp_path / "delivered.jpg", encoded)]
    assert not (tmp_path / "delivered.jpg").exists()
