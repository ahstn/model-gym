import pytest

from image_stats import METRIC_NAMES, Reference, analyse, observations, render
from image_stats.describe import band
from tests.images import blur, solid, texture


def reference(values, name="test"):
    rows = [dict.fromkeys(METRIC_NAMES, float(value)) for value in values]
    return Reference.build(rows, name=name, max_edge=1536)


def test_percentile_interpolates_and_clamps():
    ref = reference(range(100))
    assert ref.percentile("noise_sigma", 49.5) == pytest.approx(50, abs=0.6)
    assert ref.percentile("noise_sigma", -1) == 0
    assert ref.percentile("noise_sigma", 1000) == 100
    ranks = [ref.percentile("noise_sigma", value) for value in (5, 20, 60, 90)]
    assert ranks == sorted(ranks)


def test_ties_take_the_middle_rank():
    ref = reference([0] * 60 + list(range(1, 41)))
    assert ref.percentile("shadow_clip", 0) == pytest.approx(29.5, abs=0.6)


def test_small_corpus_is_rejected():
    with pytest.raises(ValueError, match="at least 20"):
        reference(range(5))


def test_save_load_round_trip_and_version_check(tmp_path):
    ref = reference(range(30))
    ref.save(tmp_path / "ref.json")
    loaded = Reference.load(tmp_path / "ref.json")
    assert loaded == ref
    loaded.check(ref.version, 1536)
    with pytest.raises(ValueError, match="max_edge"):
        loaded.check(ref.version, 1024)


def test_bands():
    assert [band(p) for p in (2, 20, 50, 90, 99)] == ["very low", "low", "typical", "high", "very high"]


def test_render_without_reference_keeps_only_absolute_lines_and_flags():
    text = render(analyse(solid(0)))
    assert "Pixels crushed to black: 100%" in text
    assert "percentile" not in text
    assert "almost no image content" in text


def test_render_ranks_a_blurred_image_low_against_sharp_references():
    corpus = [analyse(texture(seed=seed)).metrics for seed in range(10)]
    corpus += [analyse(blur(texture(seed=seed), 0.6)).metrics for seed in range(10)]
    ref = Reference.build(corpus, name="synthetic", max_edge=1536)
    analysis = analyse(blur(texture(seed=99), 3))
    found = {item.metric: item for item in observations(analysis, ref)}
    assert found["sharpness_peak"].band == "very low"
    text = render(analysis, ref)
    assert "ranked against 20 images in 'synthetic'" in text
    assert "Detail in the sharpest region: very low (0th percentile)" in text
