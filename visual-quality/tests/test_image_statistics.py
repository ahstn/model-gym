from copy import deepcopy

import pytest

from visual_quality.image_statistics import extract_metadata


def test_metadata_is_separate_and_entropy_is_an_explicit_alias():
    case = {"case_id": "R0001", "family_id": "family", "view_sha256": "view"}
    record = {
        "sha256": "view",
        "analysis": {
            "metrics": {
                "sharpness_peak": 10.5,
                "luma_entropy": 7.2,
                "focus_area": 0.8,
                "channel_clip": 0.05,
                "colourfulness": 22.0,
            },
            "analysed_dimensions": [640, 480],
            "native_dimensions": [640, 480],
        },
    }
    original = deepcopy(record)
    result = extract_metadata(case, record, {"algorithm_version": "image-stats-v2"})
    assert result["image_stats"]["histogram_entropy"] == 7.2
    assert result["image_stats"]["definitions"]["histogram_entropy"]["alias_of"] == "luma_entropy"
    assert result["image_stats"]["used_in_inference"] is False
    assert result["image_stats"]["training_features_enabled"] is False
    assert "technical_score" not in result
    assert record == original
    with pytest.raises(ValueError, match="hashes differ"):
        extract_metadata({**case, "view_sha256": "different-view"}, record, {})
