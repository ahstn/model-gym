"""Reference admission must reject changed inputs and duplicate packets."""

import json

import api_queue_benchmarks as queue
import pytest


def fixture(tmp_path, case_ids):
    directory = tmp_path / "examples"
    directory.mkdir()
    packet = directory / "packet-001.json"
    packet.write_text("{}\n")
    refs = directory / "packet-001.jsonl"
    refs.write_text("{}\n")
    freeze = directory / "packet-001-freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "packet_path": str(packet),
                "packet_sha256": queue.runner.sha(packet),
                "references_path": str(refs),
                "references_sha256": queue.runner.sha(refs),
                "case_ids": case_ids,
            }
        )
    )
    return freeze, packet, refs


def test_packet_admission_identity_and_overlap(tmp_path):
    freeze, _, _ = fixture(tmp_path, ["R4001", "R4002"])
    _, ids, phase = queue.validate_packet(freeze, {"admitted_case_ids": []})
    assert ids == ["R4001", "R4002"] and phase == "ref-examples-packet-001"
    with pytest.raises(ValueError, match="Overlapping"):
        queue.validate_packet(freeze, {"admitted_case_ids": ["R4001"]})


@pytest.mark.parametrize("changed", ["packet", "labels"])
def test_packet_admission_rejects_changed_inputs(tmp_path, changed):
    freeze, packet, refs = fixture(tmp_path, ["R4501"])
    (packet if changed == "packet" else refs).write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        queue.validate_packet(freeze, {"admitted_case_ids": []})


@pytest.mark.parametrize("ids", [[], ["R4001", "R4001"], ["R5001"], ["../../secret"]])
def test_packet_admission_rejects_bad_id_sets(tmp_path, ids):
    freeze, _, _ = fixture(tmp_path, ids)
    with pytest.raises(ValueError):
        queue.validate_packet(freeze, {"admitted_case_ids": []})
