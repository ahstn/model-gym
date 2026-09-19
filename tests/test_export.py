"""Export helpers: quantization targets, parity sampling, and the runtime contract."""

from __future__ import annotations

import hashlib

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from transformers import BertTokenizerFast

from model_gym.config import load_config
from model_gym.data.schema import RiskRecord, write_jsonl
from model_gym.export import (
    _bake_temperature,
    build_metadata,
    check_parity,
    export_onnx,
    min_argmax_agreement,
    quantization_config,
    sample_commands,
)
from model_gym.labels import NUM_LABELS
from model_gym.metrics import softmax

pytest.importorskip("optimum")


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_parity_only_tolerates_undecided_flips_for_int8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, precision: str
) -> None:
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ndecisive\ntie\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary))
    reference = np.zeros((7, NUM_LABELS), dtype=np.float32)
    reference[5] = [8, 0, 0, 0, 0]
    reference[6] = [0, 0, 0, 2, 2.0001]

    class LookupClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.table = torch.nn.Embedding.from_pretrained(torch.from_numpy(reference))

        def forward(self, input_ids: torch.Tensor, **_: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=self.table(input_ids[:, 1]))

    model = LookupClassifier()
    monkeypatch.setattr("model_gym.export.load_checkpoint", lambda *_args, **_kwargs: (model, tokenizer))
    exported = reference.copy()
    exported[6, 3], exported[6, 4] = exported[6, 4], exported[6, 3]
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Gather", ["input_ids", "position"], ["word"], axis=1),
            onnx.helper.make_node("Gather", ["table", "word"], ["logits"], axis=0),
        ],
        "near-tie-parity",
        [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.INT64, [None, None])
            for name in ("input_ids", "attention_mask")
        ],
        [onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [None, NUM_LABELS])],
        [
            onnx.numpy_helper.from_array(np.asarray(1, dtype=np.int64), "position"),
            onnx.numpy_helper.from_array(exported, "table"),
        ],
    )
    path = tmp_path / "model.onnx"
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9), path)
    kwargs = {
        "max_seq_length": 16,
        "mean_prob_delta": 0.05,
        "min_argmax_agreement": min_argmax_agreement(precision, 0.95),
        "device": torch.device("cpu"),
    }
    if precision == "int8":
        parity = check_parity(tmp_path, path, ["decisive", "tie"], **kwargs)
        assert parity["raw_argmax_agreement"] == 0.5
        assert parity["decisive_argmax_agreement"] == 1
    else:
        with pytest.raises(RuntimeError, match="exact argmax agreement required"):
            check_parity(tmp_path, path, ["decisive", "tie"], **kwargs)


def test_unknown_quantization_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown quantization target"):
        quantization_config("mips")


def test_sample_commands_spreads_across_the_corpus(tmp_path: Path) -> None:
    records = [RiskRecord(command=f"cmd {index}", level=index % 5 + 1, group=f"g{index % 7}") for index in range(100)]
    corpus = tmp_path / "validation.jsonl"
    write_jsonl(corpus, records)
    config = load_config(None, [f"data.output_dir={tmp_path}"])
    sampled = sample_commands(config, 10)
    assert len(sampled) == 10
    assert len(set(sampled)) == 10
    assert sampled[-1] == "cmd 99"
    assert sample_commands(config, 500) == [record.command for record in records]


def test_parity_sampling_prefers_validation_and_never_reads_test(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "validation.jsonl", [RiskRecord(command="validation", level=1)])
    write_jsonl(tmp_path / "train.jsonl", [RiskRecord(command="train", level=2)])
    (tmp_path / "test.jsonl").write_text("not valid JSON\n", encoding="utf-8")
    config = load_config(None, [f"data.output_dir={tmp_path}"])
    assert sample_commands(config, 4) == ["validation"]
    (tmp_path / "validation.jsonl").unlink()
    assert sample_commands(config, 4) == ["train"]


def test_sample_commands_reports_a_missing_corpus(tmp_path: Path) -> None:
    config = load_config(None, [f"data.output_dir={tmp_path}", "data.sources=missing.jsonl"])
    with pytest.raises(FileNotFoundError, match="build-data"):
        sample_commands(config, 4)


def test_metadata_hash_detects_an_artifact_changed_after_export(tmp_path: Path) -> None:
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
        input_names=["input_ids", "attention_mask"],
    )

    files = {entry["name"]: entry for entry in metadata["files"]}
    expected_hash = files["model_quantized.onnx"]["sha256"]
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == expected_hash
    model_path.write_bytes(b"damaged transfer")
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() != expected_hash


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_exported_logits_apply_temperature_exactly_once(tmp_path: Path, dtype: type) -> None:
    path = tmp_path / "model.onnx"
    onnx_type = onnx.TensorProto.FLOAT if dtype is np.float32 else onnx.TensorProto.FLOAT16
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["logits"])],
        "temperature-parity",
        [onnx.helper.make_tensor_value_info("input", onnx_type, [None, NUM_LABELS])],
        [onnx.helper.make_tensor_value_info("logits", onnx_type, [None, NUM_LABELS])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, path)
    temperature = 2.0
    _bake_temperature(path, temperature)
    logits = np.asarray([[6.0, -3.0, 0.0, 1.0, -1.0], [0.0, 1.0, 2.0, 3.0, 5.0]], dtype=dtype)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    exported = softmax(session.run(None, {"input": logits})[0])
    np.testing.assert_allclose(exported, softmax(logits / temperature), atol=1e-7)
    assert not np.allclose(exported, softmax(logits / temperature**2))
    with pytest.raises(ValueError, match="already contains"):
        _bake_temperature(path, temperature)


def test_export_refuses_to_overwrite_existing_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "export"
    output.mkdir()
    artifact = output / "important.bin"
    artifact.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        export_onnx(load_config(None), checkpoint, output_dir=output, verify=False)
    assert artifact.read_bytes() == b"preserve me"
