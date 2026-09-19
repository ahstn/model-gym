"""Command line interface.

model-gym build-data            # assemble the corpus into data/processed
model-gym train                 # fine-tune ModernBERT
model-gym eval --model DIR      # score a checkpoint and write a report
model-gym calibrate --model DIR # fit temperature using validation only
model-gym fit-policy --model DIR --out FILE # fit action thresholds on validation
model-gym export --model DIR    # ONNX + INT8 for the Rust/ort runtime
model-gym config --print        # show the defaults for every setting
model-gym info                  # versions, device, GPU memory
"""

from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

from model_gym import __version__
from model_gym.config import Config, ConfigError, config_to_dict, describe_types, load_config

LOGGER = logging.getLogger("model_gym")


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (see configs/); omitted uses the built-in defaults",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one setting, repeatable (example: --set train.learning_rate=2e-5)",
    )


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-gym", description=description)
    parser.add_argument("--version", action="version", version=f"model-gym {__version__}")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    parser.add_argument("--quiet", action="store_true", help="warnings and errors only")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = _common_parser(__doc__ or "")
    parser.set_defaults(handler=None)
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    data = subparsers.add_parser("build-data", help="assemble and split the corpus")
    _add_config_arguments(data)
    data.add_argument("--no-synthetic", action="store_true", help="exclude the templated bootstrap corpus")
    data.add_argument("--out", type=Path, default=None, help="output directory for the splits")
    data.set_defaults(handler=_cmd_build_data)

    train = subparsers.add_parser("train", help="fine-tune ModernBERT")
    _add_config_arguments(train)
    train.add_argument("--resume", type=str, default=None, help="checkpoint directory to resume from")
    train.set_defaults(handler=_cmd_train)

    evaluate = subparsers.add_parser("eval", help="evaluate a checkpoint")
    _add_config_arguments(evaluate)
    evaluate.add_argument("--model", type=Path, required=True, help="fine-tuned model directory")
    evaluate.add_argument("--split", default="test", help="split to score (default: test)")
    evaluate.add_argument("--report-dir", type=Path, default=None, help="where to write the report")
    evaluate.add_argument("--dump-predictions", type=Path, default=None, help="write per-row predictions as JSONL")
    evaluate.set_defaults(handler=_cmd_eval)

    calibrate = subparsers.add_parser("calibrate", help="fit checkpoint temperature using validation only")
    _add_config_arguments(calibrate)
    calibrate.add_argument("--model", type=Path, required=True, help="fine-tuned model directory")
    calibrate.set_defaults(handler=_cmd_calibrate)

    policy = subparsers.add_parser("fit-policy", help="fit risk thresholds using calibrated validation predictions")
    _add_config_arguments(policy)
    policy.add_argument("--model", type=Path, required=True, help="calibrated model directory")
    policy.add_argument("--out", type=Path, required=True, help="new policy JSON file; existing files are refused")
    policy.set_defaults(handler=_cmd_fit_policy)

    export = subparsers.add_parser("export", help="export ONNX and quantize for the Rust runtime")
    _add_config_arguments(export)
    export.add_argument("--model", type=Path, required=True, help="fine-tuned model directory")
    export.add_argument("--out", type=Path, default=None, help="output directory for the artifacts")
    export.add_argument("--precision", choices=("fp32", "fp16", "int8"), default=None)
    export.add_argument("--quant-target", choices=("arm64", "avx2", "avx512", "avx512_vnni"), default=None)
    export.add_argument("--no-verify", action="store_true", help="skip the PyTorch/ONNX parity check")
    export.set_defaults(handler=_cmd_export)

    config = subparsers.add_parser("config", help="inspect the resolved configuration")
    _add_config_arguments(config)
    config.add_argument("--print", dest="print_defaults", action="store_true", help="print every default setting")
    config.set_defaults(handler=_cmd_config)

    info = subparsers.add_parser("info", help="environment report")
    info.set_defaults(handler=_cmd_info)
    return parser


def _resolve_config(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.overrides)


def _cmd_build_data(args: argparse.Namespace) -> int:
    from model_gym.data.build import build_corpus

    config = _resolve_config(args)
    summary = build_corpus(
        sources=config.data.sources,
        output_dir=args.out or config.data.output_dir,
        synthetic_output=None if args.no_synthetic else config.data.synthetic_output,
        ratios=config.data.ratios,
        seed=config.data.seed,
        include_synthetic=config.data.include_synthetic and not args.no_synthetic,
    )
    print(f"wrote {summary.total} records to {summary.output_dir}")
    print(summary.format())
    print(
        f"\ngroups: {summary.group_counts} | duplicates dropped: {summary.duplicates_dropped} "
        f"| generated rows: {summary.synthetic_rows}"
    )
    for warning in summary.warnings:
        LOGGER.warning("%s", warning)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from model_gym.train import checkpoint_selection, run_training

    config = _resolve_config(args)
    if args.resume:
        from dataclasses import replace

        config = replace(config, train=replace(config.train, resume_from_checkpoint=args.resume))
    result = run_training(config)
    print(f"\ntrained for {result.train_seconds:.1f}s at step {result.global_step} -> {result.best_model_dir}")
    print(f"trainable parameters: {result.trainable_parameters:,}")
    if result.class_weights:
        rendered = ", ".join(f"L{index + 1}={value:.3f}" for index, value in enumerate(result.class_weights))
        print(f"class weights: {rendered}")
    for label, evaluation in (("validation", result.validation), ("test", result.test)):
        if evaluation is None:
            continue
        metrics = evaluation.metrics
        policy = metrics.runtime_policy
        print(
            f"{label}: accuracy {metrics.accuracy:.4f} "
            f"| {metrics.policy_kind} accuracy {policy['decision_accuracy']:.4f} "
            f"| qwk {metrics.qwk:.4f} | runtime critical misses {policy['critical_miss_rate']:.4f} "
            f"| runtime unsafe allows {policy['unsafe_allow_rate']:.4f} "
            f"| runtime unnecessary interventions {policy['unnecessary_intervention_rate']:.4f}"
        )
    selection = checkpoint_selection(
        result.validation.metrics.headline(), config.train, policy_kind=result.validation.policy.kind
    )
    print(f"validation safety gates: {'PASS' if selection['eligible'] else 'FAIL'} | score {selection['score']:.6f}")
    if result.test is None:
        print("test not scored; use `model-gym eval` only after freezing model selection")
    print(f"reports: {result.output_dir / 'reports'}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from model_gym.evaluate import evaluate_model, write_predictions, write_report

    config = _resolve_config(args)
    result = evaluate_model(
        args.model,
        config.data.output_dir,
        split=args.split,
        batch_size=config.train.eval_batch_size,
        max_seq_length=config.model.max_seq_length,
        model_config=config.model,
    )
    report_dir = args.report_dir or (Path(args.model).parent / "reports")
    markdown_path, json_path = write_report(result, report_dir, title=f"{args.model} {args.split}")
    print(result.metrics.format(title=f"{args.model} | {args.split}"))
    print(f"report: {markdown_path}\nraw: {json_path}")
    if args.dump_predictions:
        count = write_predictions(result, args.dump_predictions)
        print(f"predictions: {args.dump_predictions} ({count} rows)")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from model_gym.export import export_onnx

    config = _resolve_config(args)
    result = export_onnx(
        config,
        args.model,
        output_dir=args.out,
        precision=args.precision,
        quant_target=args.quant_target,
        verify=not args.no_verify,
    )
    parity = result.parity
    print(f"exported {result.precision} to {result.output_dir} ({result.size_mib} MiB)")
    if parity is None:
        print("parity: not checked (--no-verify); treat this artifact as unverified")
        print(f"runtime contract: {result.output_dir / 'model-metadata.json'}")
        return 0
    print(
        f"parity: decisive argmax agreement {parity['decisive_argmax_agreement']:.4f} "
        f"over {parity['decisive_rows']}/{parity['samples']} decisive rows "
        f"(floor {parity['min_argmax_agreement']:.2f}, raw {parity['raw_argmax_agreement']:.4f}) "
        f"| mean prob delta {parity['mean_prob_delta']:.5f} "
        f"| p95 {parity['p95_prob_delta']:.5f} | max {parity['max_prob_delta']:.5f} "
    )
    print(f"parity detail: {result.output_dir / 'parity.json'}")
    print(f"runtime contract: {result.output_dir / 'model-metadata.json'}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import json

    from model_gym.evaluate import calibrate_checkpoint

    config = _resolve_config(args)
    result = calibrate_checkpoint(
        args.model,
        config.data.output_dir,
        max_seq_length=config.model.max_seq_length,
        batch_size=config.train.eval_batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"calibration: {args.model / 'calibration.json'}")
    return 0


def _cmd_fit_policy(args: argparse.Namespace) -> int:
    import hashlib
    import json

    from model_gym.calibration import load_calibration
    from model_gym.evaluate import evaluate_model
    from model_gym.policy import DEFAULT_LIMITS, fit_threshold_policy, save_policy

    config = _resolve_config(args)
    if args.out.exists():
        raise FileExistsError(f"policy output already exists: {args.out}")
    calibration = load_calibration(args.model)
    validation_path = Path(config.data.output_dir) / "validation.jsonl"
    digest = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    if calibration is None or calibration["validation_sha256"] != digest:
        raise ValueError("run calibrate on this validation split before fitting its policy")
    result = evaluate_model(
        args.model,
        config.data.output_dir,
        split="validation",
        batch_size=config.train.eval_batch_size,
        max_seq_length=config.model.max_seq_length,
    )
    limits = {key: getattr(config.train, f"max_{key}") for key in DEFAULT_LIMITS}
    policy, report = fit_threshold_policy(result.probabilities, result.labels, limits=limits)
    save_policy(
        args.out, policy, provenance={"validation_sha256": digest, "temperature": calibration["temperature"], **report}
    )
    print(json.dumps({"policy": policy.to_dict(), **report}, indent=2, sort_keys=True))
    print(f"policy: {args.out}")
    print("These are fitting-set estimates, not an independent safety certification.")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    import json

    config = _resolve_config(args)
    print(json.dumps(config_to_dict(config), indent=2, sort_keys=True))
    if args.print_defaults:
        print("\n# defaults")
        print(json.dumps(describe_types(), indent=2, sort_keys=True))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from model_gym.env import format_info

    print(format_info())
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for the ``model-gym`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.handler is None:
        parser.print_help()
        return 2
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.ERROR if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    try:
        return int(args.handler(args))
    except (ConfigError, ValueError, TypeError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
