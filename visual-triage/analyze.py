"""Summarize recorded results without a model or network call."""

import json
import math
import statistics
from collections import Counter
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    path = ROOT / "evidence" / name
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None


def metrics(rows):
    valid = [r for r in rows if r.get("normalized") is not None]
    called = [r for r in rows if not r.get("skipped")]
    lookup = {r["sample"]: r["normalized"] for r in valid}
    strict = ties = increases = missing = 0
    sequences = {}
    for start, operator in ((0, "blur"), (4, "noise"), (8, "jpeg")):
        seq = [lookup.get(f"sample-{i:03d}", {}).get("technical_quality") for i in range(start, start + 4)]
        sequences[operator] = seq
        for first, second in pairwise(seq):
            if first is None or second is None:
                missing += 1
            elif first > second:
                strict += 1
            elif first == second:
                ties += 1
            else:
                increases += 1
    ai = [lookup.get(f"sample-{i:03d}", {}) for i in range(24, 30)]
    controls = [lookup.get(f"sample-{i:03d}", {}) for i in range(18, 24)]
    cost = sum(r.get("cost_usd") or 0 for r in rows)
    return {
        "rows": len(rows),
        "called": len(called),
        "valid": len(valid),
        "http_200": sum(r.get("status") == 200 for r in called),
        "providers": dict(Counter(r.get("response", {}).get("provider") for r in called)),
        "cost_usd": cost,
        "cost_per_panel_image_usd": cost / len(rows) if rows else None,
        "p50_seconds": statistics.median(r["seconds"] for r in called) if called else None,
        "p95_seconds": p95([r["seconds"] for r in called]),
        "technical_sequences": sequences,
        "adjacent_distortion_pairs": {
            "strict_decrease": strict,
            "tie": ties,
            "increase": increases,
            "missing": missing,
            "total": 9,
        },
        "known_ai_labels": dict(Counter(r.get("origin", "invalid") for r in ai)),
        "control_correct_medium": sum(
            r.get("origin") == ("render" if i < 2 else "diagram") for i, r in enumerate(controls)
        ),
        "control_denominator": 6,
        "sexual_labels": dict(Counter(r["normalized"]["sexual_content"] for r in valid)),
        "shadow_routes": dict(Counter(r["shadow_route"] for r in rows)),
        "human_unrecognizable_severe_probabilities": {
            f"sample-{i:03d}": lookup.get(f"sample-{i:03d}", {}).get("severe_defect") for i in range(15, 18)
        },
    }


def main():
    base = read("run.jsonl")
    groups = {arm: [r for r in base if r["arm"] == arm] for arm in ("clef", "decider", "vision", "jev")}
    pinned = read("clef-cloudflare.jsonl")
    if pinned:
        groups["clef_cloudflare"] = pinned
    summary = {name: metrics(rows) for name, rows in groups.items()}
    vision = {r["sample"]: r for r in groups["vision"]}
    chained = []
    for row in groups["jev"]:
        v = vision[row["sample"]]
        chained.append(
            {
                **row,
                "seconds": row["seconds"] + v["seconds"],
                "cost_usd": (row.get("cost_usd") or 0) + (v.get("cost_usd") or 0),
                "skipped": False,
            }
        )
    summary["vision_plus_jev"] = metrics(chained)
    summary["vision_plus_jev"]["note"] = (
        "End-to-end time sums sequential calls. Called counts are pipeline attempts, not Jev requests. Invalid upstream rows remain failures."
    )
    all_calls = base + pinned + read("provider-probes.jsonl")
    summary["experiment"] = {
        "calls": sum(not r.get("skipped") for r in all_calls),
        "cost_usd": sum(
            r.get("cost_usd", r.get("response", {}).get("usage", {}).get("cost", 0)) or 0 for r in all_calls
        ),
        "nsfw_positive_gold": 0,
        "aligned_technical_or_aesthetics_gold": 0,
        "agiqa_source_perception_mos": 6,
    }
    (ROOT / "evidence/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Per-image results",
        "",
        "Model judgments at a 512-pixel maximum edge. Scores are 1–5; `—` means unscorable or invalid. AI labels are suspicions, not verified provenance.",
        "",
    ]
    for name, rows in groups.items():
        lines += [
            f"## {name}",
            "",
            "| Image | Quality | Aesthetics | Origin | Sexual content | Severe defect |",
            "|---|---:|---:|---|---|---:|",
        ]
        for r in rows:
            n = r.get("normalized") or {}
            score = lambda k, n=n: f"{n[k]:.2f}" if n.get(k) is not None else "—"
            lines.append(
                f"| {r['sample']} | {score('technical_quality')} | {score('aesthetics')} | {n.get('origin', 'invalid')} | {n.get('sexual_content', 'invalid')} | {score('severe_defect')} |"
            )
        lines.append("")
    (ROOT / "evidence/per-image.md").write_text("\n".join(lines).rstrip() + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
