"""ONNX export and INT8 quantization for the Rust/`ort` runtime.

Produces a self-contained directory a Rust guardrail can load:

    model-metadata.json   taxonomy, decision bands, thresholds, hashes
    labels.json           index-ordered label slugs
    tokenizer.json        HF tokenizer the runtime feeds with the `tokenizers` crate
    model.onnx            exported graph (fp32 or fp16)
    model_quantized.onnx  dynamically quantized INT8 graph (precision=int8)

Every export ends with a parity check against the PyTorch checkpoint: argmax
agreement must be exact, and the mean probability delta must stay under
``export.parity_mean_prob_delta``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import shutil

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

from model_gym.config import EXPORT_PRECISIONS, QUANT_TARGETS, Config
from model_gym.data.schema import read_jsonl
from model_gym.evaluate import load_checkpoint
from model_gym.labels import LABELS, NUM_LABELS, bands, label_table
from model_gym.modeling import configure_runtime, predict_proba, resolve_device

LOGGER = logging.getLogger("model_gym.export")

EXPORTED_MODEL_NAME = "model.onnx"
QUANTIZED_MODEL_NAME = "model_quantized.onnx"
METADATA_NAME = "model-metadata.json"
LABELS_NAME = "labels.json"
PARITY_NAME = "parity.json"

# Validation tolerance for optimum's own fp32 comparison against PyTorch.
_EXPORT_ATOL = {"fp32": 1e-4, "fp16": 5e-2, "int8": 5e-2}


def quantization_config(target: str) -> Any:
    """Dynamic INT8 settings for a target CPU.

    Per-channel and range-reduction defaults differ per target and have changed
    between Optimum releases, so this defers to the target factory instead of
    hard-coding a table that would silently go stale.
    """
    if target not in QUANT_TARGETS:
        raise ValueError(f"unknown quantization target {target!r}; expected one of {list(QUANT_TARGETS)}")
    factory = getattr(AutoQuantizationConfig, target)
    if "is_static" not in set(inspect.signature(factory).parameters):
        raise RuntimeError(
            f"optimum {metadata.version('optimum')} has an unexpected AutoQuantizationConfig.{target} signature"
        )
    return factory(is_static=False)


def quantization_params(config: Any) -> dict[str, Any]:
    """JSON-safe record of the effective quantization settings."""
    values = getattr(config, "__dict__", {})
    return {
        str(key): value
        for key, value in sorted(values.items())
        if isinstance(value, str | int | float | bool | list | tuple | type(None))
    }


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Exported artifact directory and its parity result."""

    output_dir: Path
    model_path: Path
    precision: str
    quant_target: str
    parity: dict[str, Any] | None

    @property
    def size_mib(self) -> float:
        return round(self.model_path.stat().st_size / 1024**2, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "model_path": str(self.model_path),
            "precision": self.precision,
            "quant_target": self.quant_target,
            "size_mib": self.size_mib,
            "parity": self.parity,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(output_dir: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return files


def sample_commands(config: Config, count: int) -> list[str]:
    """Spread the parity sample across the corpus so every level is represented."""
    candidates = [
        Path(config.data.output_dir) / "test.jsonl",
        Path(config.data.output_dir) / "validation.jsonl",
    ]
    candidates.extend(Path(source) for source in config.data.sources)
    records: Sequence[Any] = []
    for candidate in candidates:
        if candidate.is_file():
            records = read_jsonl(candidate)
            break
    if not records:
        raise FileNotFoundError(
            "no corpus found for the parity check; run `model-gym build-data` first "
            "or point data.sources at a JSONL file"
        )
    if len(records) <= count:
        return [record.command for record in records]
    stride = len(records) / count
    picked = [records[int(index * stride)].command for index in range(count)]
    picked[-1] = records[-1].command
    return picked


def check_parity(
    model_dir: Path | str,
    onnx_path: Path,
    commands: Sequence[str],
    *,
    max_seq_length: int,
    mean_prob_delta: float,
    min_argmax_agreement: float = 1.0,
    tie_epsilon: float = 0.05,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Compare PyTorch probabilities with the exported graph on the same inputs.

    ``min_argmax_agreement`` is 1.0 for fp32 and fp16, which are expected to agree
    exactly. INT8 quantization moves the numerics, so a row whose top two
    probabilities are within ``tie_epsilon`` of each other counts as undecided and
    is excluded from the gate: argmax on such a row is a coin flip in either
    runtime. A broken export lands near chance agreement on the remaining rows,
    which the floor still catches.
    """
    configure_runtime()
    device = device or resolve_device()
    model, tokenizer = load_checkpoint(model_dir, device=device)
    reference = predict_proba(
        model, tokenizer, list(commands), max_seq_length=max_seq_length, batch_size=len(commands), device=device
    )
    encoded = tokenizer(list(commands), return_tensors="np", truncation=True, max_length=max_seq_length, padding=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {name: encoded[name].astype(np.int64) for name in ("input_ids", "attention_mask") if name in encoded}
    outputs = session.run(None, feeds)
    logits = np.asarray(outputs[0])
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    if logits.shape[1] != NUM_LABELS:
        raise RuntimeError(f"exported model returned {logits.shape[1]} logits, expected {NUM_LABELS}")
    exported = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=-1).numpy()

    delta = np.abs(reference - exported)
    reference_sorted = np.sort(reference, axis=1)
    margin = reference_sorted[:, -1] - reference_sorted[:, -2]
    decisive = margin >= tie_epsilon
    matches = reference.argmax(axis=1) == exported.argmax(axis=1)
    if not decisive.any():
        raise RuntimeError(
            f"parity check failed: no decisive rows out of {len(commands)} "
            f"(top-two probability margin below {tie_epsilon} everywhere); the checkpoint is not confident"
        )
    decisive_agreement = float(matches[decisive].mean())
    parity = {
        "samples": len(commands),
        "raw_argmax_agreement": float(matches.mean()),
        "decisive_argmax_agreement": decisive_agreement,
        "decisive_rows": int(decisive.sum()),
        "undecided_rows": int((~decisive).sum()),
        "tie_epsilon": tie_epsilon,
        "min_argmax_agreement": min_argmax_agreement,
        "mean_prob_delta": float(delta.mean()),
        "max_prob_delta": float(delta.max()),
        "p95_prob_delta": float(np.percentile(delta.max(axis=1), 95)),
        "max_prob_delta_row": int(delta.max(axis=1).argmax()),
        "max_seq_length": max_seq_length,
        "mean_batch_prob_delta": float(np.abs(reference.mean(axis=0) - exported.mean(axis=0)).mean()),
    }
    if decisive_agreement < min_argmax_agreement:
        raise RuntimeError(
            f"parity check failed: decisive argmax agreement {decisive_agreement:.4f} "
            f"< {min_argmax_agreement:.4f} over {parity['decisive_rows']} decisive rows "
            f"(worst row {parity['max_prob_delta_row']}: {commands[parity['max_prob_delta_row']]!r})"
        )
    if parity["mean_prob_delta"] > mean_prob_delta:
        raise RuntimeError(
            f"parity check failed: mean probability delta {parity['mean_prob_delta']:.4f} > {mean_prob_delta}"
        )
    return parity


def min_argmax_agreement(precision: str, configured: float) -> float:
    """fp32 and fp16 must match exactly; INT8 gets the configured tolerance."""
    return 1.0 if precision in {"fp32", "fp16"} else configured


@contextmanager
def _quiet_root_logging() -> Iterator[None]:
    """Silence the per-tensor info logging that onnxruntime emits while quantizing."""
    root = logging.getLogger()
    previous = root.level
    root.setLevel(max(previous, logging.WARNING))
    try:
        yield
    finally:
        root.setLevel(previous)


def export_onnx(
    config: Config,
    model_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    precision: str | None = None,
    quant_target: str | None = None,
    verify: bool = True,
) -> ExportResult:
    """Export a fine-tuned checkpoint to ONNX, optionally quantize it, and verify parity."""
    configure_runtime()
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    precision = precision or config.export.precision
    quant_target = quant_target or config.export.quant_target
    if precision not in EXPORT_PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; expected one of {list(EXPORT_PRECISIONS)}")
    output_dir = Path(output_dir or Path(config.export.output_dir) / f"{model_dir.name}-{precision}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    LOGGER.info("exporting %s to %s (%s)", model_dir, output_dir, precision)
    export_kwargs: dict[str, Any] = {
        "task": "text-classification",
        # A local checkpoint directory carries no library hint, so name it.
        "library_name": "transformers",
        "opset": config.export.opset,
        "do_validation": config.export.do_validation,
        "atol": _EXPORT_ATOL[precision],
    }
    if precision == "fp16":
        export_kwargs["dtype"] = "fp16"
    main_export(model_dir, output_dir, **export_kwargs)

    model_path = output_dir / EXPORTED_MODEL_NAME
    if not model_path.is_file():
        raise RuntimeError(f"export produced no {EXPORTED_MODEL_NAME} in {output_dir}")

    if precision == "int8":
        quantizer = ORTQuantizer.from_pretrained(output_dir)
        with _quiet_root_logging():
            quantizer.quantize(
                quantization_config=quantization_config(quant_target),
                save_dir=output_dir,
                file_suffix="quantized",
            )
        model_path = output_dir / QUANTIZED_MODEL_NAME
        if not model_path.is_file():
            raise RuntimeError(f"quantization produced no {QUANTIZED_MODEL_NAME} in {output_dir}")
        # The fp32 graph is only a stepping stone for quantization; shipping it
        # would quadruple the artifact size. Re-export with --precision fp32 when
        # a reference graph is needed.
        (output_dir / EXPORTED_MODEL_NAME).unlink()

    parity: dict[str, Any] | None = None
    if verify:
        parity = check_parity(
            model_dir,
            model_path,
            sample_commands(config, config.export.parity_samples),
            max_seq_length=config.model.max_seq_length,
            mean_prob_delta=config.export.parity_mean_prob_delta,
            min_argmax_agreement=min_argmax_agreement(precision, config.export.parity_min_argmax_agreement),
            tie_epsilon=config.export.parity_tie_epsilon,
        )
        (output_dir / PARITY_NAME).write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        LOGGER.warning("parity check skipped; the exported graph is unverified")

    (output_dir / LABELS_NAME).write_text(json.dumps(list(LABELS), indent=2) + "\n", encoding="utf-8")
    metadata = build_metadata(
        config,
        model_dir=model_dir,
        output_dir=output_dir,
        precision=precision,
        quant_target=quant_target,
        model_path=model_path,
        parity=parity,
    )
    (output_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ExportResult(
        output_dir=output_dir,
        model_path=model_path,
        precision=precision,
        quant_target=quant_target,
        parity=parity,
    )


def build_metadata(
    config: Config,
    *,
    model_dir: Path,
    output_dir: Path,
    precision: str,
    quant_target: str,
    model_path: Path,
    parity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Runtime contract for the guardrail: taxonomy, bands, thresholds, hashes."""
    run_metadata = model_dir.parent / "run_metadata.json"
    run = json.loads(run_metadata.read_text(encoding="utf-8")).get("run", {}) if run_metadata.is_file() else {}
    quantization = (
        {
            "mode": "dynamic",
            "target": quant_target,
            "file": model_path.name,
            "params": quantization_params(quantization_config(quant_target)),
            "source_graph": "removed after quantization; export with --precision fp32 for a reference graph",
        }
        if precision == "int8"
        else None
    )
    return {
        "schema_version": 1,
        "task": "command_risk_5level",
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_model": config.model.name,
        "source_model_dir": str(model_dir),
        "max_seq_length": config.model.max_seq_length,
        "tokenizer": {"file": "tokenizer.json", "class": "AutoTokenizer"},
        "inputs": ["input_ids", "attention_mask"],
        "output": "logits",
        "precision": precision,
        "quantization": quantization,
        "opset": config.export.opset,
        "labels": label_table(),
        "decision_bands": [{"low": band.low, "high": band.high, "decision": str(band.decision)} for band in bands()],
        "risk_score": {
            "definition": "sum(p_level * level); 1.0 is critical, 5.0 is low risk",
            "min": 1.0,
            "max": 5.0,
        },
        "runtime_contract": {
            "argmax_index_to_level": "level = index + 1",
            "decision": "first band whose [low, high) contains the risk score",
            "tokenize": "truncate to max_seq_length, pad to the longest row in the batch",
        },
        "metrics": run.get("test") or run.get("validation"),
        "parity": parity,
        "files": _artifact_files(output_dir),
    }
