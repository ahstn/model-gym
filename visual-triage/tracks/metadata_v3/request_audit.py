"""Reconstruct frozen task payloads offline; export text context and hashes, never image data."""

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench_protocol import digest, image_part, make_request
from tracks.metadata_v3.evidence_bundle import _complete
from tracks.metadata_v3.features import ARMS
from tracks.metadata_v3.runtime import context_for, feature_text

ROOT = Path(__file__).resolve().parents[2]
HERE = "tracks/metadata_v3"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Reconstruction:
    """Cache verified delivered images, prompt files, contexts and repeated payload hashes."""

    def __init__(self, root, runs):
        self.root, self.runs = root, runs
        self.images, self.prompts, self.contexts, self.requests = {}, {}, {}, {}

    def prompt(self, path):
        path = path if path.is_absolute() else self.root / path
        path = path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Audit prompt must remain within the project")
        if path not in self.prompts:
            self.prompts[path] = json.loads(path.read_bytes())
        return self.prompts[path]

    def request(self, sample, arm, prompt):
        feature_key = (arm, sample["features_sha256"])
        if feature_key not in self.contexts:
            # The allowlisted projection has no gold, ID, native path or source identity.
            projected = {key: sample[key] for key in ("features", "features_sha256")}
            context, text = context_for(projected, arm), feature_text(projected, arm)
            if context != context_for(sample, arm) or text != feature_text(sample, arm):
                raise ValueError("Gold or source fields affected model context")
            self.contexts[feature_key] = (digest(context), text)
        context_hash, text = self.contexts[feature_key]
        image_key = (sample["image"], sample["delivered_sha256"])
        if image_key not in self.images:
            self.images[image_key] = image_part(sample, root=self.root)
        key = (arm, digest(prompt), feature_key, image_key)
        if key not in self.requests:
            request_prompt = copy.deepcopy(prompt)
            if text:
                request_prompt["context"] += "\n\n" + text
            endpoint, payload = make_request("decider", self.images[image_key], request_prompt, "low", 4096)
            if (
                set(payload) != {"model", "state", "questions"}
                or payload["state"][0] != request_prompt["context"]
            ):
                raise ValueError("Unexpected Decider payload fields")
            self.requests[key] = (
                digest({"endpoint": endpoint, "payload": payload}),
                endpoint,
                payload["model"],
            )
        request_hash, endpoint, model = self.requests[key]
        return {
            "request_sha256": request_hash,
            "context_sha256": context_hash,
            "features_sha256": sample["features_sha256"],
            "delivered_sha256": sample["delivered_sha256"],
            "prompt_sha256": digest(prompt),
            "endpoint": endpoint,
            "requested_model": model,
        }, text


def _location(path, runs):
    parts = path.relative_to(runs).parts
    if parts[0] not in ARMS:
        raise ValueError("Task ledger is outside a recognised arm")
    arm = parts[0]
    if len(parts) == 4 and parts[1] == "candidates" and path.stem in {"train", "dev"}:
        return arm, path.stem, path.parent / "prompt.json"
    if len(parts) == 3 and parts[1] == "final" and path.stem in {"baseline", "selected"}:
        selection = json.loads((runs / arm / "selection.json").read_bytes())
        return arm, "native_final", Path(selection[path.stem + "_prompt_path"])
    if len(parts) == 3 and parts[1] == "controls":
        selection = json.loads((runs / arm / "selection.json").read_bytes())
        return arm, "controls:" + path.stem, Path(selection["selected_prompt_path"])
    raise ValueError("Unexpected task ledger path")


def audit(root=ROOT):
    root = Path(root).resolve()
    _complete(root)
    track, runs = root / HERE, root / "data/metadata-v3/runs"
    protocol_path = track / "protocol.json"
    protocol, protocol_hash = json.loads(protocol_path.read_bytes()), _sha(protocol_path)
    for name in ("dataset", "controls"):
        path = track / (name + ".json")
        if _sha(path) != protocol[name + "_sha256"]:
            raise ValueError("Frozen " + name + " changed")
    for relative, expected in protocol["code_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or _sha(path) != expected:
            raise ValueError("Frozen request execution code changed: " + relative)
    samples = json.loads((track / "dataset.json").read_bytes())["samples"]
    controls = json.loads((track / "controls.json").read_bytes())["cases"]
    native = {sample["id"]: sample for sample in samples}
    cases = {sample["id"]: sample for sample in controls}
    if len(native) != len(samples) or len(cases) != len(controls):
        raise ValueError("Duplicate frozen case IDs")
    example_ids = {
        "held_out_real": min(
            sample["id"] for sample in samples if sample["split"] == "test" and not sample["gold"]
        ),
        "held_out_firefly": min(
            sample["id"]
            for sample in samples
            if sample["split"] == "test"
            and str(sample.get("source", {}).get("generator", "")).lower() == "firefly"
        ),
    }
    cache, counts, examples, inputs = Reconstruction(root, runs), Counter(), {}, []
    skipped = 0
    for ledger in sorted(runs.rglob("*.jsonl")):
        raw_ledger = ledger.read_bytes()
        inputs.append(
            {"path": str(ledger.relative_to(root)), "sha256": hashlib.sha256(raw_ledger).hexdigest()}
        )
        seen = set()
        for number, raw in enumerate(raw_ledger.splitlines(), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("record_type") == "reflection":
                skipped += 1
                continue
            if row.get("record_type") != "task":
                raise ValueError("Unknown ledger record type")
            arm, phase, prompt_path = _location(ledger, runs)
            sample = (cases if phase.startswith("controls:") else native)[row["sample"]]
            if row["sample"] in seen:
                raise ValueError("Duplicate raw task attempt in one ledger")
            seen.add(row["sample"])
            if phase.startswith("controls:") and (
                arm not in sample["arms"] or sample["condition"] != ledger.stem
            ):
                raise ValueError("Control record is in the wrong arm or condition")
            prompt = cache.prompt(prompt_path)
            expected, text = cache.request(sample, arm, prompt)
            expected.update(
                metadata_arm=arm,
                protocol_sha256=protocol_hash,
                task="ai",
                arm="decider",
                split=sample["split"],
            )
            if phase in {"train", "dev"} and sample["split"] != phase:
                raise ValueError("Candidate ledger contains the wrong split")
            if phase == "native_final" and sample["split"] != "test":
                raise ValueError("Native final ledger contains a non-test image")
            for key, value in expected.items():
                if row.get(key) != value:
                    raise ValueError(f"Raw request {key} differs: {ledger}:{number}")
            counts[(arm, phase)] += 1
            for label, identifier in example_ids.items():
                if phase == "native_final" and row["sample"] == identifier and (label, arm) not in examples:
                    examples[(label, arm)] = {
                        "example": label,
                        "metadata_arm": arm,
                        "phase": phase,
                        "feature_text": text,
                        "request_context_text": prompt["context"] + ("\n\n" + text if text else ""),
                        "ledger": str(ledger.relative_to(root)),
                        "line": number,
                        "raw_record_sha256": hashlib.sha256(raw).hexdigest(),
                        **{
                            key: row[key]
                            for key in (
                                "request_sha256",
                                "context_sha256",
                                "features_sha256",
                                "delivered_sha256",
                                "prompt_sha256",
                            )
                        },
                    }
    if set(examples) != {(label, arm) for label in example_ids for arm in ARMS}:
        raise ValueError("Both held-out examples need audited native records in all seven arms")
    return {
        "version": 1,
        "protocol_sha256": protocol_hash,
        "all_task_payload_hashes_match": True,
        "task_attempts": sum(counts.values()),
        "reflection_records_not_audited": skipped,
        "counts": [
            {"metadata_arm": arm, "phase": phase, "attempts": count}
            for (arm, phase), count in sorted(counts.items())
        ],
        "examples": list(examples.values()),
        "inputs": inputs,
        "unique_delivered_images_verified": len(cache.images),
        "gold_and_source_fields_used_to_construct_context": False,
        "limits": "Reconstructed payload hashes prove recorded request construction, including failed attempts. They do not prove server receipt or model use. Only allowlisted frozen features enter context; native metadata can still encode source differences. Reflection payloads are outside this task audit. Image base64 is omitted.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    output = args.output or args.root / HERE / "results/request-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"task_attempts": result["task_attempts"], "output": str(output)}))


if __name__ == "__main__":
    main()
