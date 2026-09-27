"""Command line entry point. Heavy modules are imported inside each command."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pathlib import Path

# Same as decision.modeling.DEFAULT_MODEL; kept here so the CLI does not import torch at startup.
DEFAULT_MODEL = "openbmb/MiniCPM5-2B"


def _info(_: argparse.Namespace) -> int:
    import peft
    import torch
    import transformers

    print(f"torch {torch.__version__} (cuda {torch.version.cuda})")
    print(f"transformers {transformers.__version__}")
    print(f"peft {peft.__version__}")
    if not torch.cuda.is_available():
        print("cuda: not available")
        return 0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"cuda:{i} {name} sm_{cap[0]}{cap[1]} total {total / 2**30:.1f} GiB free {free / 2**30:.1f} GiB")
    return 0


def _check_readout(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from decision.prompt import MAX_OPTIONS, Readout
    from decision.schema import Decision

    readout = Readout(AutoTokenizer.from_pretrained(args.model))
    print(f"codes: {len(readout.codes)} first {readout.codes[:5]} last {readout.codes[-5:]}")
    sample = Decision(
        id="sample",
        dataset="sample",
        source="sample",
        kind="choice",
        state="It is raining and you have to walk to work.",
        question="What do you take?",
        options=tuple(f"Option {i}" for i in range(MAX_OPTIONS)),
        target=(1.0,) + (0.0,) * (MAX_OPTIONS - 1),
        group_id="sample",
    )
    small = Decision(
        id="small",
        dataset="sample",
        source="sample",
        kind="choice",
        state=sample.state,
        question=sample.question,
        options=("An umbrella", "Sunglasses", "Nothing"),
        target=(1.0, 0.0, 0.0),
        group_id="sample",
    )
    print(readout.render(small))
    prompt = readout.render(sample)
    prompt_ids = readout.tokenizer.encode(prompt, add_special_tokens=False)
    bad = [
        code
        for code, token_id in zip(readout.codes, readout.code_token_ids, strict=True)
        if readout.answer_id(prompt_ids, prompt, code) != token_id
    ]
    if len(set(readout.code_token_ids)) != MAX_OPTIONS or bad:
        print(f"FAIL: {len(bad)} codes change token at the answer position: {bad[:20]}")
        return 1
    print(f"OK: all {MAX_OPTIONS} codes are one unique token at the answer position")
    return 0


def _build_data(args: argparse.Namespace) -> int:
    from decision import data

    manifest = data.build(args.out, pool_size=args.pool_size, seed=args.seed, suite_rows=args.suite_rows)
    print(json.dumps(manifest, indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    from decision import train

    print(train.run(args.config))
    return 0


def _eval(args: argparse.Namespace) -> int:
    from decision import evaluate

    metrics = evaluate.run(
        args.model,
        adapter=args.adapter,
        panel=args.panel,
        calib=args.calib,
        out_dir=args.out,
        max_tokens=args.max_tokens,
        batch_tokens=args.batch_tokens,
        limit=args.limit,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if argv and argv[0] == "suite":
        from decision import suite

        return suite.main(argv[1:])

    parser = argparse.ArgumentParser(prog="decision")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="versions, CUDA device, memory").set_defaults(func=_info)

    p = sub.add_parser("check-readout", help="verify option-code tokenization")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.set_defaults(func=_check_readout)

    p = sub.add_parser("build-data", help="build training pool and panels")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--pool-size", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite-rows", type=Path, default=None)
    p.set_defaults(func=_build_data)

    p = sub.add_parser("train", help="train a LoRA adapter")
    p.add_argument("--config", type=Path, required=True)
    p.set_defaults(func=_train)

    p = sub.add_parser("eval", help="evaluate on a panel")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--calib", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-tokens", type=int, default=65536)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_eval)

    sub.add_parser("suite", help="run the decision-index suite (own arguments)")

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
