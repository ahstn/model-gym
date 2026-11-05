"""Freeze and execute the native metadata study without using test labels for search."""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench_protocol import MODELS, digest, validate_prompt
from tracks.metadata_v3.features import ARMS, validate_features
from tracks.metadata_v3.optimise import optimise_arm
from tracks.metadata_v3.provenance import OFFLINE_SETTINGS, TRUST_POLICY
from tracks.metadata_v3.runtime import (
    StudyBudget,
    evaluate,
    read_ledger,
    require_all_selections,
    write_frozen,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = ROOT / "data/metadata-v3/runs"
CODE = (
    "bench_protocol.py",
    "bench_runner.py",
    "bench_analyze.py",
    "tracks/image_context.py",
    "tracks/metrics.py",
    "tracks/run.py",
    "tracks/optimise.py",
    "tracks/metadata_v3/features.py",
    "tracks/metadata_v3/provenance.py",
    "tracks/metadata_v3/runtime.py",
    "tracks/metadata_v3/optimise.py",
    "tracks/metadata_v3/build_dataset.py",
    "tracks/metadata_v3/build_controls.py",
    "tracks/metadata_v3/acquire_ai.py",
    "tracks/metadata_v3/acquire_real.py",
    "tracks/metadata_v3/study.py",
    "uv.lock",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def check_dataset(dataset):
    counts, ids, groups, views = Counter(), set(), defaultdict(set), set()
    natives = set()
    group_sizes = Counter()
    for sample in dataset["samples"]:
        if sample["id"] in ids or type(sample["gold"]) not in (int, bool) or sample["gold"] not in (0, 1):
            raise ValueError("Duplicate ID or invalid gold")
        ids.add(sample["id"])
        counts[sample["split"], sample["gold"]] += 1
        groups[sample["group_id"]].add((sample["split"], sample["gold"]))
        group_sizes[sample["group_id"]] += 1
        if sample["delivered_sha256"] in views or sample["native_sha256"] in natives:
            raise ValueError("Duplicate delivered view or native source file")
        views.add(sample["delivered_sha256"])
        natives.add(sample["native_sha256"])
        validate_features(sample["features"])
        if digest(sample["features"]) != sample["features_sha256"]:
            raise ValueError("Feature hash mismatch")
        for key, hash_key in (("image", "delivered_sha256"), ("native_image", "native_sha256")):
            path = (ROOT / sample[key]).resolve()
            if not path.is_relative_to(ROOT) or sha(path) != sample[hash_key]:
                raise ValueError("Frozen image path or hash mismatch")
    expected = {
        (split, gold): n for split, n in (("train", 40), ("dev", 40), ("test", 80)) for gold in (0, 1)
    }
    if counts != expected or any(len(values) != 1 for values in groups.values()):
        raise ValueError("Class balance or source-group isolation changed")
    if set(group_sizes.values()) != {8}:
        raise ValueError("Expected equal eight-image groups and isolated delivered views")


def freeze():
    dataset, controls, baseline = (
        read(HERE / "dataset.json"),
        read(HERE / "controls.json"),
        read(HERE / "baseline.json"),
    )
    check_dataset(dataset)
    validate_prompt(baseline, "ai")
    samples = {sample["id"]: sample for sample in dataset["samples"]}
    control_ids = set()
    for control in controls["cases"]:
        base = samples[control["base_sample_id"]]
        if control["id"] in control_ids or control["id"] in samples:
            raise ValueError("Control IDs must be unique and separate from native samples")
        control_ids.add(control["id"])
        if (
            base["split"] != "test"
            or control["split"] != "test"
            or type(control["gold"]) not in (int, bool)
            or control["gold"] != base["gold"]
            or control["group_id"] != base["group_id"]
        ):
            raise ValueError("Controls must retain held-out labels")
        if (
            control["delivered_sha256"] != base["delivered_sha256"]
            or control["image"] != base["image"]
            or sha(ROOT / control["image"]) != control["delivered_sha256"]
        ):
            raise ValueError("Metadata controls must retain the delivered pixels")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", control["condition"]) or not control["arms"]:
            raise ValueError("Invalid control condition or arm list")
        if len(set(control["arms"])) != len(control["arms"]) or not set(control["arms"]) <= set(ARMS):
            raise ValueError("Unknown or duplicate control arm")
        if "native_image" in control:
            path = (ROOT / control["native_image"]).resolve()
            if not path.is_relative_to(ROOT) or sha(path) != control["native_sha256"]:
                raise ValueError("Frozen native control file changed")
        validate_features(control["features"])
        if digest(control["features"]) != control["features_sha256"]:
            raise ValueError("Control features changed")
    protocol = {
        "version": 1,
        "study": "native-exif-c2pa-decider-v3",
        "frozen_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": sha(HERE / "dataset.json"),
        "controls_sha256": sha(HERE / "controls.json"),
        "baseline_prompt_path": "tracks/metadata_v3/baseline.json",
        "baseline_prompt_sha256": digest(baseline),
        "hash_conventions": "This protocol hashes dataset, controls and code as raw files. Prompts, features, requests, controls.base_dataset_sha256 and optimisation run.dataset_sha256 use bench_protocol.digest canonical JSON",
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "arms": list(ARMS),
        "model": MODELS["decider"],
        "reflection_model": MODELS["luna"],
        "reflection_reasoning": "low",
        "reflection_max_images": 4,
        "seed": 20261012,
        "max_proposals": 20,
        "minimum_dev_reward_gain": 0.005,
        "objective": "Mean 1-Brier reward over equal eight-image, single-class source groups",
        "ai_threshold": 0.5,
        "test_used_for_selection": False,
        "global_budget_usd": 5.0,
        "arm_budget_usd": 0.65,
        "max_task_calls_per_arm": 5000,
        "reserve_per_inflight_usd": 0.01,
        "concurrency": 4,
        "trust_policy": TRUST_POLICY,
        "c2pa_settings": OFFLINE_SETTINGS,
        "native_coverage": "Class-balanced images; naturally imbalanced EXIF and C2PA availability. Observational cohort, not balanced provenance proof.",
        "controls": "Context stripping and donor swaps are experimental. C2PA byte controls use fresh validator results. Control cases are excluded from native accuracy.",
        "selection_rule": "Freeze all seven GEPA dev selections before any test or control calls. Retain baseline unless dev reward improves by at least 0.005.",
        "final_order": "Each forty-image batch runs all arms in fixed order, baseline then selected; unchanged prompts reuse baseline calls. Native evaluation precedes controls.",
        "limitations": [
            "Source collection can confound labels",
            "No wholly unseen generator family",
            "Phone model names differ across splits but hardware families and scenes may overlap",
            "Untrusted EXIF and signed claims do not prove scene truth",
        ],
    }
    write_frozen(HERE / "protocol.json", protocol)
    return protocol


def load_study():
    protocol = read(HERE / "protocol.json")
    for key, filename in (("dataset_sha256", "dataset.json"), ("controls_sha256", "controls.json")):
        if sha(HERE / filename) != protocol[key]:
            raise ValueError("Frozen study input changed: " + filename)
    for path, expected in protocol["code_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen execution code changed: " + path)
    baseline = read(HERE / "baseline.json")
    if digest(baseline) != protocol["baseline_prompt_sha256"]:
        raise ValueError("Frozen common prompt changed")
    budget = StudyBudget(
        RUNS,
        protocol["global_budget_usd"],
        protocol["arm_budget_usd"],
        protocol["max_task_calls_per_arm"],
        protocol["reserve_per_inflight_usd"],
    )
    return protocol, read(HERE / "dataset.json"), baseline, budget


def require_native_completion(dataset, baseline, budget, protocol_hash):
    """Every native test attempt must finish before any control request starts."""
    expected = {sample["id"]: sample for sample in dataset["samples"] if sample["split"] == "test"}
    for arm in ARMS:
        selection = read(RUNS / arm / "selection.json")
        selected = read(selection["selected_prompt_path"])
        for variant, prompt in (("baseline", baseline), ("selected", selected)):
            if variant == "selected" and digest(prompt) == digest(baseline):
                continue
            rows = read_ledger(budget.root / arm / "final" / f"{variant}.jsonl")
            if len(rows) != len(expected) or {row["sample"] for row in rows} != set(expected):
                raise ValueError("Complete all native baseline and selected attempts before controls")
            if any(
                row.get("metadata_arm") != arm
                or row.get("split") != "test"
                or row.get("protocol_sha256") != protocol_hash
                or row.get("prompt_sha256") != digest(prompt)
                or row.get("delivered_sha256") != expected[row["sample"]]["delivered_sha256"]
                for row in rows
            ):
                raise ValueError("Native final ledger identity changed")


def run(mode):
    protocol, dataset, baseline, budget = load_study()
    protocol_hash = sha(HERE / "protocol.json")
    if mode == "optimise":
        for arm in ARMS:
            output = RUNS / arm
            if (output / "selection.json").exists():
                continue
            print(f"Optimising {arm}", flush=True)
            result = optimise_arm(
                dataset,
                baseline,
                arm,
                output,
                budget,
                protocol_hash,
                resume=(output / "run.json").exists(),
                concurrency=protocol["concurrency"],
            )
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "selected": result["selected_index"],
                        "dev_reward": result["selected_dev_reward"],
                        "costs": budget.totals(),
                    }
                ),
                flush=True,
            )
        require_all_selections(budget, protocol_hash)
        return
    require_all_selections(budget, protocol_hash)
    test = sorted(
        (sample for sample in dataset["samples"] if sample["split"] == "test"),
        key=lambda sample: digest(sample["id"]),
    )
    if mode == "final":
        for start in range(0, len(test), 40):
            for arm in ARMS:
                selection = read(RUNS / arm / "selection.json")
                selected = read(selection["selected_prompt_path"])
                for variant, prompt in (("baseline", baseline), ("selected", selected)):
                    if variant == "selected" and digest(prompt) == digest(baseline):
                        continue
                    print(f"Final {arm}/{variant} cases {start}..{min(start + 40, len(test))}", flush=True)
                    evaluate(
                        test[start : start + 40],
                        prompt,
                        arm,
                        RUNS / arm / "final" / f"{variant}.jsonl",
                        budget,
                        protocol_hash,
                        "test",
                        concurrency=protocol["concurrency"],
                    )
    elif mode == "controls":
        require_native_completion(dataset, baseline, budget, protocol_hash)
        cases = read(HERE / "controls.json")["cases"]
        for condition in sorted({case["condition"] for case in cases}):
            for arm in ARMS:
                subset = [case for case in cases if case["condition"] == condition and arm in case["arms"]]
                if not subset:
                    continue
                selected = read(read(RUNS / arm / "selection.json")["selected_prompt_path"])
                print(f"Controls {condition}/{arm}: {len(subset)}", flush=True)
                evaluate(
                    subset,
                    selected,
                    arm,
                    RUNS / arm / "controls" / f"{condition}.jsonl",
                    budget,
                    protocol_hash,
                    "test",
                    concurrency=protocol["concurrency"],
                )
    else:
        raise ValueError("Unknown run mode")
    print(json.dumps(budget.totals()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "optimise", "final", "controls"))
    args = parser.parse_args()
    if args.mode == "freeze":
        print(json.dumps(freeze(), indent=2))
    else:
        run(args.mode)


if __name__ == "__main__":
    main()
