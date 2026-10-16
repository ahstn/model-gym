"""Run frozen panels with model order rotated across images."""

import argparse
import hashlib
import json
from types import SimpleNamespace

from bench_protocol import ROOT
from bench_runner import run


def validate_preflight(task, freeze, manifest, reference, prompt_bytes):
    if hashlib.sha256(prompt_bytes).hexdigest() != freeze.get("prompt_hashes", {}).get(task):
        raise ValueError("Final prompt bytes differ from the frozen prompt SHA256")
    hashes = []
    for name, document in [("manifest", manifest), ("reference", reference)]:
        samples = document["samples"]
        if len(samples) != 60:
            raise ValueError(f"Final {name} must contain exactly 60 samples")
        sample_hashes = {sample["id"]: sample["delivered_sha256"] for sample in samples}
        if len(sample_hashes) != 60:
            raise ValueError(f"Final {name} must contain 60 unique sample IDs")
        hashes.append(sample_hashes)
    if hashes[0] != hashes[1]:
        raise ValueError("Assessment sample IDs and delivered SHA256 values must match the manifest")
    allowed = {"gemini": {"low", "medium", "high"}, "luna": {"low", "medium"}}
    for arm, efforts in allowed.items():
        if freeze.get("reasoning", {}).get(arm) not in efforts:
            raise ValueError(f"Frozen reasoning effort is missing or unsupported for {arm}")
    cap = freeze.get("max_output_tokens")
    if type(cap) is not int or not 128 <= cap <= 16384:
        raise ValueError("Frozen output-token cap must be an integer from 128 to 16384")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["quality", "ai"], required=True)
    args = parser.parse_args()
    freeze = json.loads((ROOT / "evidence/benchmarks/freeze.json").read_text())
    manifest_path = ROOT / f"evidence/benchmarks/{args.task}.json"
    manifest = json.loads(manifest_path.read_text())
    reference = json.loads((ROOT / f"evidence/assessments/{args.task}-60.json").read_text())
    prompt_path = ROOT / f"prompts/final/{args.task}.json"
    validate_preflight(args.task, freeze, manifest, reference, prompt_path.read_bytes())
    models = ["decider", "gemini", "luna"]
    for index, sample in enumerate(manifest["samples"]):
        for arm in models[index % 3 :] + models[: index % 3]:
            run(
                SimpleNamespace(
                    manifest=manifest_path,
                    prompt=ROOT / f"prompts/final/{args.task}.json",
                    task=args.task,
                    models=[arm],
                    output=ROOT / f"evidence/final/{args.task}.jsonl",
                    ids=[sample["id"]],
                    limit=None,
                    effort=freeze["reasoning"].get(arm, "low") if arm != "decider" else "low",
                    max_tokens=freeze["max_output_tokens"],
                    max_calls=180,
                    budget_usd=2.0,
                    reserve_usd=0.10,
                )
            )


if __name__ == "__main__":
    main()
