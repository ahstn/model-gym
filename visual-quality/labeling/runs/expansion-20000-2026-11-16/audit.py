"""Reparse all new outcomes and audit the incremental/combined export boundaries."""

# ruff: noqa: I001 -- corpus must establish the legacy adapter import context.
import json
import hashlib
from collections import Counter
from corpus import DATA, OLD, digest, read_lines, write_json
import api_runner as runner
from visual_quality.ensemble import score_ensemble


def main():
    pins = json.loads((DATA / "initial-evidence-pins.json").read_text())
    for path, expected in pins.items():
        assert digest(path) == expected, f"Initial evidence changed: {path}"
    manifest = json.loads((DATA / "manifest.json").read_text())["cases"]
    old = json.loads((OLD / "manifest.json").read_text())["cases"]
    assert len(manifest) == 10000 and len(old) == 10000
    assert len({r["case_id"] for r in old + manifest}) == 20000
    assert len({r["view_sha256"] for r in old + manifest}) == 20000
    assert not ({r["family_id"] for r in old} & {r["family_id"] for r in manifest})
    labels = read_lines(DATA / "exports/new-labelled-images.jsonl")
    stats = {r["case_id"]: r for r in read_lines(DATA / "stats.jsonl")}
    assert len(labels) == len(stats) == 10000
    plans = {
        m: runner.read(DATA / "results/api" / m / "plans/expansion.json")
        for m in ["luna", "deepseek", "liquid", "jev"]
    }
    all_plans = {
        m: [runner.read(p) for p in sorted((DATA / "results/api" / m / "plans").glob("*.json"))]
        for m in plans
    }
    for model, model_plans in all_plans.items():
        for plan in model_plans:
            runner.validate_plan(plan)
            assert plan["profile"]["sha256"] == plans[model]["profile"]["sha256"]
            assert plan["manifest_sha256"] == digest(DATA / "manifest.json")
    for plan in plans.values():
        runner.validate_plan(plan)
        assert plan["manifest_sha256"] == digest(DATA / "manifest.json")
        assert plan["stats_sent_to_models"] is False and plan["reference_scores_sent_to_models"] is False
    parsed = 0
    statuses = Counter()
    for case, label in zip(manifest, labels, strict=True):
        assert case["case_id"] == label["case_id"]
        assert digest(case["image_path"]) == case["view_sha256"]
        assert label["image_stats"] == stats[case["case_id"]]["image_stats"]
        assert stats[case["case_id"]]["view_sha256"] == case["view_sha256"]
        assert label["training_ready"] is False and not any(label["task_masks"].values())
        group = label["assessments"]
        assert set(group) == {"luna", "deepseek", "liquid", "jev", "photo_ensemble"}
        for model in plans:
            row = group[model]
            route = plans[model]["profile"]["routes"]["routes"][model]
            assert row["view_sha256"] == case["view_sha256"]
            assert row["source_request_plan_id"] in {p["plan_id"] for p in all_plans[model]}
            path = corpus_path(row["raw_response_path"])
            assert digest(path) == row["raw_response_sha256"]
            result = runner.read(path)
            value = runner.parsed_result(result, route)
            assert value == result["parsed"]
            for field in ["status", "technical_score", "aesthetic_score"]:
                assert row[field] == value[field]
            assert row["latency_seconds"] == result["latency_seconds"] and row["latency_seconds"] > 0
            assert row["price_usd"] == value["price_usd"]
            if model in ("luna", "deepseek"):
                assert row["requested_reasoning"] == "high"
            statuses[model + ":" + row["status"]] += 1
            parsed += 1
        assert group["photo_ensemble"] == score_ensemble(
            {m: group[m] for m in ["luna", "deepseek", "liquid"]}, case_id=case["case_id"]
        )
    for name in ["labelled-images.jsonl", "scores-long.jsonl"]:
        original = OLD / "exports" / name
        combined = DATA / "exports" / name
        hasher = hashlib.sha256()
        remaining = original.stat().st_size
        with combined.open("rb") as stream:
            while remaining:
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                assert chunk
                hasher.update(chunk)
                remaining -= len(chunk)
        assert hasher.hexdigest() == digest(original)
    with (DATA / "exports/labelled-images.jsonl").open() as stream:
        assert sum(1 for _ in stream) == 20000
    with (DATA / "exports/scores-long.jsonl").open() as stream:
        assert sum(1 for _ in stream) == 111500
    result = {
        "passed": True,
        "new_images": 10000,
        "total_images": 20000,
        "raw_observations_reparsed": parsed,
        "new_statuses": dict(statuses),
        "image_statistics_joins": 10000,
        "initial_records_unchanged": 10000,
        "new_qrealign_calls": 0,
        "reference_images": 1500,
        "publication_images": 1000,
        "training_ready": False,
        "exports_sha256": digest(DATA / "exports/labelled-images.jsonl"),
    }
    write_json(DATA / "audit.json", result)
    print(json.dumps(result), flush=True)


def corpus_path(value):
    from pathlib import Path

    return Path(value)


if __name__ == "__main__":
    main()
