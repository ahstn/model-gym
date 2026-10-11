"""Counterbalance low/medium effort on fixed development images only."""

import argparse
import json
from types import SimpleNamespace

from bench_protocol import ROOT
from bench_runner import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["quality", "ai"], required=True)
    args = parser.parse_args()
    ids = json.loads((ROOT / "evidence/assessments/protocol.json").read_text())["effort_pilot_ids"]
    for index, sample_id in enumerate(ids):
        efforts = ["low", "medium"] if index % 2 == 0 else ["medium", "low"]
        models = ["gemini", "luna"] if index % 2 == 0 else ["luna", "gemini"]
        for effort in efforts:
            for model in models:
                run(
                    SimpleNamespace(
                        manifest=ROOT / "evidence/samples.json",
                        prompt=ROOT / f"prompts/trials/{args.task}-r1.json",
                        task=args.task,
                        models=[model],
                        output=ROOT / f"evidence/effort/{args.task}-{model}-{effort}.jsonl",
                        ids=[sample_id],
                        limit=None,
                        effort=effort,
                        max_tokens=4096,
                        max_calls=12,
                        budget_usd=2.0,
                        reserve_usd=0.10,
                    )
                )


if __name__ == "__main__":
    main()
