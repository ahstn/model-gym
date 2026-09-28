import numpy as np

from vq.data import SOURCES, frame_paths, rows_for
from vq.metrics import correlations


def test_kadid_score_comes_from_id_when_gt_score_is_absent():
    raw = {
        "id": "I66_01_01.png->4.63",
        "image": "kadid10k/I66_01_01.png",
        "conversations": [{"from": "human", "value": "q\n<|image|>"}, {"from": "gpt", "value": "a"}],
    }
    (row,) = rows_for(SOURCES["test_kadid"], [raw])
    assert row["mos"] == 4.63
    assert row["media"] == ["img/kadid10k/I66_01_01.png"]


def test_train_rows_keep_verbatim_prompt_without_image_marker():
    raw = {
        "image": "koniq/1.jpg",
        "conversations": [
            {"from": "human", "value": "Rate the quality of this image.\n<|image|>"},
            {"from": "gpt", "value": "The quality of the image is average."},
        ],
    }
    (row,) = rows_for(SOURCES["koniq"], [raw])
    assert row["prompt"] == "Rate the quality of this image."
    assert row["answer"] == "The quality of the image is average."


def test_lsvq_frame_dirs_do_not_collide_across_batches():
    a = frame_paths("lsvq/yfcc-batch1/747.mp4")
    b = frame_paths("lsvq/ia-batch1/747.mp4")
    assert len(a) == 8
    assert set(a).isdisjoint(b)


def test_dmos_sets_report_positive_correlation_for_a_good_predictor():
    dmos = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([0.9, 0.7, 0.4, 0.1])  # higher score = better quality = lower DMOS
    m = correlations(pred, dmos, higher_better=SOURCES["live"].higher_better)
    assert m["srcc"] == 1.0
    assert m["plcc"] > 0.99
