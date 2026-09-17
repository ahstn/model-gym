"""Export helpers: quantization targets, parity sampling, and the runtime contract."""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from model_gym.config import Config, load_config
from model_gym.data.schema import RiskRecord, write_jsonl
from model_gym.export import (
    build_metadata,
    min_argmax_agreement,
    quantization_config,
    quantization_params,
    sample_commands,
)
from model_gym.labels import LABELS, NUM_LABELS

pytest.importorskip("optimum")

SEED = Path("data/seed/seed_commands.jsonl")


@pytest.mark.parametrize(
    ("precision", "expected"),
    [("fp32", 1.0), ("fp16", 1.0), ("int8", 0.95)],
)
def test_exact_precisions_must_agree_exactly(precision: str, expected: float) -> None:
    assert min_argmax_agreement(precision, 0.95) == expected


@pytest.mark.parametrize("target", ["arm64", "avx2", "avx512", "avx512_vnni"])
def test_quantization_targets_produce_dynamic_configs(target: str) -> None:
    params = quantization_params(quantization_config(target))
    assert params["is_static"] is False
    assert "MatMul" in params["operators_to_quantize"]


def test_unknown_quantization_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown quantization target"):
        quantization_config("mips")


def test_sample_commands_spreads_across_the_corpus(tmp_path: Path) -> None:
    records = [RiskRecord(command=f"cmd {index}", level=index % 5 + 1, group=f"g{index % 7}") for index in range(100)]
    corpus = tmp_path / "test.jsonl"
    write_jsonl(corpus, records)
    config = load_config(None, [f"data.output_dir={tmp_path}"])
    sampled = sample_commands(config, 10)
    assert len(sampled) == 10
    assert len(set(sampled)) == 10
    assert sampled[-1] == "cmd 99"
    assert sample_commands(config, 500) == [record.command for record in records]


def test_sample_commands_reports_a_missing_corpus(tmp_path: Path) -> None:
    config = load_config(None, [f"data.output_dir={tmp_path}", "data.sources=missing.jsonl"])
    with pytest.raises(FileNotFoundError, match="build-data"):
        sample_commands(config, 4)


def test_metadata_is_the_runtime_contract(tmp_path: Path) -> None:
    config = load_config("configs/base.yaml", [f"export.output_dir={tmp_path}"])
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "model_quantized.onnx").write_bytes(b"not really onnx")
    model_path = output_dir / "model_quantized.onnx"
    parity = {"decisive_argmax_agreement": 0.97, "samples": 32}

    metadata = build_metadata(
        config,
        model_dir=tmp_path / "best",
        output_dir=output_dir,
        precision="int8",
        quant_target="avx512_vnni",
        model_path=model_path,
        parity=parity,
    )

    assert metadata["task"] == "command_risk_5level"
    assert [label["slug"] for label in metadata["labels"]] == list(LABELS)
    assert [label["level"] for label in metadata["labels"]] == [1, 2, 3, 4, 5]
    assert metadata["runtime_contract"]["argmax_index_to_level"] == "level = index + 1"
    assert len(metadata["decision_bands"]) == 4
    assert metadata["decision_bands"][0]["decision"] == "block"
    assert metadata["risk_score"]["min"] == 1.0
    assert metadata["max_seq_length"] == config.model.max_seq_length
    assert metadata["quantization"]["target"] == "avx512_vnni"
    assert metadata["quantization"]["file"] == "model_quantized.onnx"
    assert metadata["parity"] == parity
    assert metadata["metrics"] is None
    # Artifact hashes let a transfer be verified on the training host.
    files = {entry["name"]: entry for entry in metadata["files"]}
    assert "model_quantized.onnx" in files
    assert files["model_quantized.onnx"]["bytes"] == len(b"not really onnx")
    assert len(files["model_quantized.onnx"]["sha256"]) == 64
    assert json.loads(json.dumps(metadata))


def test_metadata_records_absent_parity_when_verification_is_skipped(tmp_path: Path) -> None:
    config = load_config(None)
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    model_path = output_dir / "model.onnx"
    model_path.write_bytes(b"x")
    metadata = build_metadata(
        config,
        model_dir=tmp_path / "best",
        output_dir=output_dir,
        precision="fp32",
        quant_target="arm64",
        model_path=model_path,
        parity=None,
    )
    assert metadata["parity"] is None
    assert metadata["quantization"] is None
    assert metadata["labels"][0]["decision"] == "block"


def test_int8_config_defaults_are_gated() -> None:
    config: Config = load_config(None)
    assert config.export.precision == "int8"
    assert config.export.parity_min_argmax_agreement == 0.95
    assert config.export.parity_tie_epsilon == 0.05
    assert NUM_LABELS == 5
