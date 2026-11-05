"""Independent read-only audit. Run only after the parent reports final data ready."""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def dt(text):
    return datetime.fromisoformat(text)


def audit(track):
    from bench_protocol import digest, image_part, make_request, normalize_decisions
    from bench_runner import effective_cost, sample_context
    from tracks.metrics import paired_reward_delta, summarize

    directory = ROOT / "tracks" / track
    protocol = read(directory / "protocol.json")
    dataset = read(ROOT / protocol["dataset_path"])
    assert sha(ROOT / protocol["dataset_path"]) == protocol["dataset_sha256"]
    samples = {s["id"]: s for s in dataset["samples"]}
    for sample in samples.values():
        assert sha(ROOT / sample["image"]) == sample["delivered_sha256"]
    owner, views = {}, {}
    for s in samples.values():
        for mapping, key in ((owner, s["group_id"]), (views, s["delivered_sha256"])):
            assert mapping.get(key, s["split"]) == s["split"]
            mapping[key] = s["split"]
    reflection = rows(directory / "optimisation/reflection.jsonl")
    assert len(reflection) == 20
    assert [r["proposal_number"] for r in reflection] == list(range(1, 21))
    for record in reflection:
        images = record.get("reflection_images", [])
        assert len(images) <= 4
        for image in images:
            sample = samples[image["sample"]]
            assert sample["split"] == "train"
            assert sample["delivered_sha256"] == image["delivered_sha256"]
    selection = read(directory / "prompt-selection.json")
    context_plan = read(directory / "ablation-plan.json")
    assert selection["protocol_sha256"] == sha(directory / "protocol.json")
    assert selection["selection_split"] == "dev"
    assert sha(ROOT / selection["selected_prompt_path"]) == selection["selected_prompt_sha256"]
    assert selection["minimum_gain"] == 0.005
    gain = selection["evaluations"]["candidate"]["score"] - selection["evaluations"]["baseline"]["score"]
    assert math.isclose(gain, selection["candidate_gain"], abs_tol=1e-12)
    assert selection["selected"] == ("candidate" if gain >= 0.005 else "baseline")
    for entry in selection["evaluations"].values():
        assert sha(ROOT / entry["prompt_path"]) == entry["prompt_sha256"]
        assert sha(ROOT / entry["ledger_path"]) == entry["ledger_sha256"]
        assert all(samples[r["sample"]]["split"] == "dev" for r in rows(ROOT / entry["ledger_path"]))
    costs = []
    all_requests = []
    raw_validated = 0
    count = 0

    def inspect(ledger, manifest, prompt, expected_split, mode, final=False):
        nonlocal count, raw_validated
        data = {s["id"]: s for s in manifest["samples"]}
        records = rows(ledger)
        assert len({r["sample"] for r in records}) == len(records)
        for record in records:
            sample = data[record["sample"]]
            assert sample["split"] == expected_split
            assert record["manifest_sha256"] == digest(manifest)
            assert record["prompt_sha256"] == digest(prompt)
            assert record["delivered_sha256"] == sample["delivered_sha256"]
            endpoint, payload = make_request(
                protocol["model_alias"],
                image_part(sample),
                prompt,
                "low",
                4096,
                extra_context=sample_context(sample, mode),
            )
            assert digest({"endpoint": endpoint, "payload": payload}) == record["request_sha256"]
            assert record["requested_model"] == protocol["model_id"]
            all_requests.append(record["request_sha256"])
            if record.get("status") == 200:
                assert record["normalized"] == normalize_decisions(record["response"], prompt["questions"])
                raw_validated += 1
            if final:
                assert dt(selection["selected_at"]) <= dt(record["timestamp"])
                assert dt(context_plan["created_at"]) <= dt(record["timestamp"])
            else:
                assert dt(record["timestamp"]) <= dt(selection["selected_at"])
            usage = record.get("usage") or record.get("response", {}).get("usage") or {}
            cost, _ = effective_cost(usage, record.get("cost_usd"))
            costs.append((ledger, record, cost))
            count += 1
        return records

    for ledger in (directory / "optimisation/candidates").glob("*/*.jsonl"):
        inspect(ledger, dataset, read(ledger.parent / "prompt.json"), ledger.stem, "image_only")
    final_report = read(directory / "report/report.json")
    for name, expected in final_report["evaluations"].items():
        path = ROOT / (
            protocol["baseline_prompt_path"] if name == "baseline" else selection["selected_prompt_path"]
        )
        records = inspect(
            directory / "final" / (name + ".jsonl"), dataset, read(path), "test", "image_only", True
        )
        recomputed = summarize(track, [s for s in samples.values() if s["split"] == "test"], records, 500)
        assert recomputed == expected
        assert recomputed["summary"]["unattempted"] == 0
    if "external_validation" in final_report:
        for name, expected in final_report["external_validation"]["evaluations"].items():
            path = ROOT / (
                protocol["baseline_prompt_path"] if name == "baseline" else selection["selected_prompt_path"]
            )
            records = inspect(
                directory / "final" / ("external-" + name + ".jsonl"),
                dataset,
                read(path),
                "external",
                "image_only",
                True,
            )
            assert (
                summarize(track, [s for s in samples.values() if s["split"] == "external"], records, 500)
                == expected
            )
            assert expected["summary"]["unattempted"] == 0
        ext = final_report["external_validation"]
        a = ext["evaluations"]["baseline"]["per_image"]
        b = ext["evaluations"].get("selected", ext["evaluations"]["baseline"])["per_image"]
        delta = paired_reward_delta(a, b, 0 if final_report["selected_reuses_baseline"] else 500)
        for key, value in delta.items():
            if key != "note":
                assert ext["paired_comparison"][key] == value
    primary = final_report["evaluations"]["baseline"]["per_image"]
    selected = final_report["evaluations"].get("selected", final_report["evaluations"]["baseline"])[
        "per_image"
    ]
    paired = paired_reward_delta(primary, selected, 0 if final_report["selected_reuses_baseline"] else 500)
    for key, value in paired.items():
        if key != "note":
            assert final_report["paired_comparison"][key] == value
    plan = read(directory / "ablation-plan.json")
    context = read(ROOT / plan["context_dataset_path"])
    assert sha(ROOT / plan["context_dataset_path"]) == plan["context_dataset_sha256"]
    metadata = read(directory / "report/metadata-report.json")
    assert metadata["selected_prompt_sha256"] == selection["selected_prompt_sha256"]
    for split, panel in metadata["panels"].items():
        prompt = read(ROOT / selection["selected_prompt_path"])
        records = inspect(
            directory / "final-metadata" / (split + ".jsonl"), context, prompt, split, "pixel_stats", True
        )
        cohort = [s for s in context["samples"] if s["split"] == split]
        assert {r["sample"] for r in records} == set(plan["sample_ids"][split])
        assert summarize(track, cohort, records, 500) == panel["evaluations"]["pixel_stats"]
        assert panel["evaluations"]["pixel_stats"]["summary"]["unattempted"] == 0
        delta = paired_reward_delta(
            panel["evaluations"]["image_only"]["per_image"],
            panel["evaluations"]["pixel_stats"]["per_image"],
            500,
        )
        assert delta == panel["paired_comparison"]
    for record in reflection:
        usage = record.get("response", {}).get("usage") or {}
        cost, _ = effective_cost(usage)
        costs.append((directory / "optimisation/reflection.jsonl", record, cost))
    known = sum(cost for _, _, cost in costs if cost is not None)
    unknown = [(p, r) for p, r, c in costs if c is None]
    reservation = 0
    if unknown:
        assert track == "quality_v2" and len(unknown) == 1
        plan = read(directory / "cost-reservations.json")
        assert plan["no_retry"] is True and len(plan["reservations"]) == 1
        entry = plan["reservations"][0]
        ledger, record = unknown[0]
        assert entry["ledger_path"] == str(ledger.relative_to(directory))
        assert all_requests.count(record["request_sha256"]) == 1
        assert entry["record_sha256"] == digest(record)
        assert entry["request_sha256"] == record["request_sha256"]
        assert entry["sample"] == record["sample"]
        assert entry["reservation_usd"] == 0.1 and entry["actual_cost_usd"] is None
        assert record["normalized"] is None and record["effective_cost_usd"] is None
        assert (
            sha(ROOT / (str(directory.relative_to(ROOT)) + "/" + plan["evidence_path"]))
            == plan["evidence_sha256"]
        )
        assert (
            hashlib.sha256(ledger.read_bytes()[: entry["ledger_prefix_bytes"]]).hexdigest()
            == entry["ledger_prefix_sha256"]
        )
        reservation = 0.1
    assert known + reservation <= 5.0
    spend = metadata["spending"]
    assert math.isclose(known, spend.get("known_reported_usd", spend["total_usd"]), abs_tol=1e-12)
    assert math.isclose(reservation, spend.get("reserved_unverified_usd", 0), abs_tol=1e-12)
    assert spend.get("unknown_cost_count", 0) == len(unknown)
    summary = read(directory / "optimisation/summary.json")
    pool = len(summary["gepa"]["candidates"])
    return {
        "track": track,
        "task_attempts": count,
        "reflection_attempts": len(reflection),
        "valid_proposals": sum(r.get("proposal_outcome") == "accepted" for r in reflection),
        "rejected_proposals": sum(r.get("proposal_outcome") == "rejected" for r in reflection),
        "train_improved_candidates": pool - 1,
        "full_dev_candidates_including_baseline": pool,
        "raw_response_normalisations_verified": raw_validated,
        "known_reported_usd": known,
        "reservation_usd": reservation,
        "unknown_cost_count": len(unknown),
        "all_checks_passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-ready", action="store_true")
    parser.add_argument("--track", choices=["quality_v2", "ai_detection_v2"], required=True)
    args = parser.parse_args()
    if not args.final_ready:
        parser.error("Do not inspect outcomes until final data notice; pass --final-ready afterwards.")
    snapshot = read(Path(__file__).with_name("final-audit-baseline.json"))
    for relative, expected in snapshot.items():
        path = ROOT / relative
        assert sha(path) == expected, "Frozen file changed: " + path
    print(json.dumps(audit(args.track), indent=2))
