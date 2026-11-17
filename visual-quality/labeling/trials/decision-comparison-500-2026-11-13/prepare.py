"""Freeze a diverse disagreement-enriched subset before either new model runs."""

import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/decision-comparison-500-2026-11-13"
MODELS = {"perplexity": "perplexity/pplx-decider-v1.3:free", "gemini": "google/gemini-decision-1"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def quotas(capacities, total, weights):
    ideal = {k: total * weights[k] / sum(weights.values()) for k in capacities}
    result = {k: min(capacities[k], math.floor(ideal[k])) for k in capacities}
    while sum(result.values()) < total:
        key = max(
            (k for k in capacities if result[k] < capacities[k]), key=lambda k: (ideal[k] - result[k], -k)
        )
        result[key] += 1
    return result


def variance(row):
    return statistics.mean(
        statistics.pvariance(
            [row["assessments"][m][f"{d}_score"] for m in ("luna", "deepseek", "liquid", "jev")]
        )
        for d in ("technical", "aesthetic")
    )


def select(rows):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["content_cluster"]].append(row)
    capacity = {k: len(v) for k, v in clusters.items()}
    counts = quotas(capacity, 500, {k: math.sqrt(n) for k, n in capacity.items()})
    high_counts = quotas(counts, 375, counts)
    selected = []
    for cluster, members in sorted(clusters.items()):
        ranked = sorted(members, key=lambda r: (-variance(r), r["case_id"]))
        high = ranked[: high_counts[cluster]]
        rest = sorted(
            ranked[high_counts[cluster] :],
            key=lambda r: hashlib.sha256(("2026-11-13:" + r["case_id"]).encode()).hexdigest(),
        )
        for arm, group in [
            ("high_disagreement", high),
            ("diversity_sample", rest[: counts[cluster] - len(high)]),
        ]:
            selected.extend({**r, "selection_arm": arm, "selection_variance": variance(r)} for r in group)
    return sorted(
        selected, key=lambda r: hashlib.sha256(("request-order:" + r["case_id"]).encode()).hexdigest()
    )


def main():
    if (DATA / "plan.json").exists():
        raise SystemExit("Frozen plan exists; do not resample after seeing new scores")
    source = ROOT / "labeling/records/labelled-images.jsonl"
    rows = [json.loads(line) for line in source.open()]
    chosen = select(rows)
    assert len(chosen) == len({r["family_id"] for r in chosen}) == 500
    assert all(1 <= r["benchmark_order"] <= 1000 for r in chosen)
    images = DATA / "images"
    images.mkdir(exist_ok=True)
    for row in chosen:
        original = Path(row["image_path"])
        assert sha(original) == row["view_sha256"]
        target = images / (row["case_id"] + ".png")
        if not target.exists():
            os.link(original, target)
        assert sha(target) == row["view_sha256"]
        row["trial_image_path"] = str(target)
    write(DATA / "subset.json", chosen)
    source_prompt = ROOT / "labeling/prompts/decision.json"
    prompt = json.loads(source_prompt.read_text())
    catalogue = json.loads((DATA / "model-catalogue.json").read_text())
    routes = {}
    for key, model in MODELS.items():
        metadata = next(r for r in catalogue if r["id"] == model)
        routes[key] = {
            "requested_model": model,
            "kind": "decisions",
            "provider": "openrouter",
            "endpoint": "https://openrouter.ai/api/alpha/decisions",
            "accepted_models": list(
                dict.fromkeys([model, metadata["canonical_slug"], model.removesuffix(":free")])
            ),
        }
    plan = {
        "version": "decision-comparison-500-2026-11-13",
        "source_sha256": sha(source),
        "subset_sha256": sha(DATA / "subset.json"),
        "source_prompt_sha256": sha(source_prompt),
        "common": prompt["common"],
        "questions": prompt["models"]["jev"]["questions"],
        "prompt_policy": "Identical frozen Jev questions for both new models; no score-driven tuning",
        "routes": routes,
        "max_attempts": 3,
        "budget_per_model_usd": 5.0,
        "request_reservation_usd": 0.03,
        "timeout_seconds": 180,
        "source_pins": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("run.py"),
                ROOT / "src/visual_quality/scoring/api_contract.py",
                ROOT / "src/visual_quality/scoring/wire.py",
            ]
        },
    }
    write(DATA / "plan.json", plan)
    summary = {
        "images": len(chosen),
        "clusters": len({r["content_cluster"] for r in chosen}),
        "sources": dict(Counter(r["category"] for r in chosen)),
        "arms": dict(Counter(r["selection_arm"] for r in chosen)),
        "mean_variance_pool": statistics.mean(map(variance, rows)),
        "mean_variance_selected": statistics.mean(map(variance, chosen)),
        "cluster_counts": dict(sorted(Counter(r["content_cluster"] for r in chosen).items())),
        "plan_sha256": sha(DATA / "plan.json"),
        "warning": "Disagreement-enriched diagnostic sample, not an unbiased accuracy benchmark. References were not used in selection.",
    }
    write(DATA / "selection-summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
