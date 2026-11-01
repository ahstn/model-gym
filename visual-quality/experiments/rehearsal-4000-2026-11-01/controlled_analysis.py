"""Freeze blind controlled reviews, then supplement the unchanged all-family analysis.

The audit command never opens model output. Run analyze only on the final long export.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import analysis_metrics as metrics
import controlled_review as review

HERE = Path(__file__).resolve().parent
MODELS = (*metrics.BASE_MODELS, "luna_deepseek_mean")
RELATIONS = ("versus_parent", "versus_previous")
TOLERANCE = 1e-8
METHOD = (
    "Supplemental descriptive technical-score diagnostics; all 100 controlled families remain "
    "in the original analysis. Blind categorical stimulus reviews select additional comparisons, "
    "never numeric labels. A comparison is verified only for clear_degradation. Parent comparisons, "
    "severe-versus-parent comparisons and fully verified adjacent ladders have separate denominators. "
    "Positive mean drop is comparison technical score minus derivative technical score. Native "
    "precision is retained; absolute differences <=1e-8 are ties. Exact (1,1) floor pairs are "
    "reported separately and can be policy-correct for art/nonphotographic content; neither ties "
    "nor every severe score at least as high as its parent is automatically an error. No aesthetic "
    "monotonicity is required. This is not PACE ground-truth-threshold Holistic Override Rate."
)


def binding(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": review.sha(path)}


def controlled_families(cases: list[dict]) -> dict[str, list[dict]]:
    if len(cases) != 4000 or len({c["case_id"] for c in cases}) != 4000:
        raise ValueError("Expected the complete 4000-case manifest with unique case IDs")
    grouped = defaultdict(dict)
    for case in cases:
        controlled_role = case.get("role") in {"source_parent", "variant"}
        if not controlled_role and case.get("stratum") != "controlled":
            continue
        if not controlled_role or case.get("stratum") != "controlled" or case["benchmark_partition"] != "rehearsal":
            raise ValueError("Controlled members must have valid roles and remain outside benchmarks")
        severity = 0 if case["role"] == "source_parent" else case.get("severity")
        allowed = {0} if case["role"] == "source_parent" else {1, 2, 3}
        if type(severity) is not int or severity not in allowed or severity in grouped[case["family_id"]]:
            raise ValueError("Duplicate or invalid controlled severity")
        grouped[case["family_id"]][severity] = case
    if len(grouped) != 100 or any(set(rows) != {0, 1, 2, 3} for rows in grouped.values()):
        raise ValueError("Expected exactly 100 complete parent-plus-three-variant families")
    result = {family: [rows[i] for i in range(4)] for family, rows in sorted(grouped.items())}
    if any(not rows[0].get("operator") or len({r.get("operator") for r in rows}) != 1 for rows in result.values()):
        raise ValueError("Each controlled family must have one consistent operator")
    return result


def category_counts(entries: list[dict]) -> dict:
    counts = Counter(e["classification"] for e in entries)
    return {category: counts[category] for category in sorted(review.CATEGORIES)}


def unique_pairs(entries: list[dict]) -> list[dict]:
    pairs = {}
    for entry in entries:
        key = (entry["case_id"], entry["comparison_case_id"])
        if key in pairs:
            previous = pairs[key]
            if any(previous[k] != entry[k] for k in ("family_id", "classification", "evidence")):
                raise ValueError("Repeated image pair has inconsistent review")
        else:
            pairs[key] = entry
    return list(pairs.values())


def reviewed_entries(packet: dict, completed: dict, families: dict) -> tuple[list[dict], list[dict]]:
    entries, summaries = [], []
    completed_by_id = {f["family_id"]: f for f in completed["families"]}
    for family in packet["families"]:
        family_id = family["family_id"]
        if family_id not in families:
            raise ValueError("Packet contains a family outside the final manifest")
        members = families[family_id]
        if len(family["members"]) != 4:
            raise ValueError("Packet must retain all four family members")
        for actual, expected in zip(family["members"], members, strict=True):
            if review.identity(actual) != review.identity(expected):
                raise ValueError("Packet member identity differs from the final manifest")
            if review.sha(actual["source_path"]) != actual["source_sha256"]:
                raise ValueError("Controlled source bytes changed")
        inspected = completed_by_id[family_id]
        family_entries = []
        for severity, comparison in enumerate(inspected["comparisons"], 1):
            for relation in RELATIONS:
                judgment = comparison[relation]
                family_entries.append({
                    "family_id": family_id, "review_family_id": family["review_family_id"],
                    "operator": members[0]["operator"], "severity": severity,
                    "case_id": comparison["case_id"], "view_sha256": comparison["view_sha256"],
                    "relation": relation, **judgment,
                })
        entries.extend(family_entries)
        summaries.append({
            "family_id": family_id, "review_family_id": family["review_family_id"],
            "operator": members[0]["operator"], "case_ids": [r["case_id"] for r in members],
            "reviewer": completed["reviewer"], "reviewed_at": completed.get("reviewed_at"),
            "individual_views_opened": inspected["individual_views_opened"],
            "all_three_vs_parent_verified": all(e["classification"] == "clear_degradation" for e in family_entries if e["relation"] == "versus_parent"),
            "fully_adjacent_verified": all(e["classification"] == "clear_degradation" for e in family_entries if e["relation"] == "versus_previous"),
            "severe_vs_parent_verified": family_entries[4]["classification"] == "clear_degradation",
            "unverified_comparisons": [
                {k: e[k] for k in ("case_id", "comparison_case_id", "relation", "classification", "evidence")}
                for e in family_entries if e["classification"] != "clear_degradation"
            ],
        })
    return entries, summaries


def build_audit(manifest: Path, directory: Path) -> dict:
    """Validate manifest, packets, reviews and assets without accessing any scores."""
    families = controlled_families(json.loads(manifest.read_text())["cases"])
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text())
    manifest_hash = review.sha(manifest)
    if index["manifest_sha256"] != manifest_hash or len(index["packets"]) != 10:
        raise ValueError("Review index does not bind ten packets to the final manifest")
    expected_counts = {"families": 100, "derivatives": 300, "comparison_entries": 600, "unique_image_pairs": 500}
    if any(index.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("Review index coverage is incomplete")
    pinned, entries, summaries = [], [], []
    for number, indexed in enumerate(index["packets"], 1):
        packet_id = f"controlled-{number:02d}"
        packet_path = directory / f"{packet_id}.json"
        completed_path = directory / f"{packet_id}-reviewed.json"
        if indexed["packet_id"] != packet_id or Path(indexed["path"]).resolve() != packet_path.resolve() or indexed["sha256"] != review.sha(packet_path):
            raise ValueError("Review index packet identity/hash mismatch")
        packet = json.loads(packet_path.read_text())
        if packet.get("packet_id") != packet_id or packet["manifest_sha256"] != manifest_hash or len(packet["families"]) != 10:
            raise ValueError("Packet does not bind ten families to the final manifest")
        review.validate(packet_path, completed_path)
        completed = json.loads(completed_path.read_text())
        packet_entries, packet_summaries = reviewed_entries(packet, completed, families)
        entries.extend(packet_entries)
        summaries.extend(packet_summaries)
        pinned.append({"packet": binding(packet_path), "review": binding(completed_path)})
    if len(summaries) != 100 or {f["family_id"] for f in summaries} != set(families) or len({f["review_family_id"] for f in summaries}) != 100:
        raise ValueError("Review packets must cover every controlled family exactly once")
    pairs = unique_pairs(entries)
    if len(entries) != 600 or len(pairs) != 500:
        raise ValueError("Expected 600 entries representing exactly 500 unique image pairs")
    operators = sorted({f["operator"] for f in summaries})
    subsets = {
        name: [f["family_id"] for f in summaries if f[flag]]
        for name, flag in (
            ("all_three_vs_parent_verified_family_ids", "all_three_vs_parent_verified"),
            ("fully_adjacent_verified_family_ids", "fully_adjacent_verified"),
            ("severe_vs_parent_verified_family_ids", "severe_vs_parent_verified"),
        )
    }
    return {
        "schema_version": 1, "audited_at": datetime.now(timezone.utc).isoformat(),
        "method": METHOD, "model_outputs_read": False, "numeric_labels_modified": False,
        "inputs": {"manifest": binding(manifest), "index": binding(index_path),
                   "script": binding(Path(__file__)), "validator": binding(Path(review.__file__)),
                   "metrics": binding(Path(metrics.__file__)), "packets": pinned},
        "coverage": {"manifest_cases": 4000, "packets": 10, "families": 100, "members": 400,
                     "comparison_entries": len(entries), "unique_image_pairs": len(pairs)},
        "classifications": {"entries": category_counts(entries), "unique_pairs": category_counts(pairs),
                            **{relation: category_counts([e for e in entries if e["relation"] == relation]) for relation in RELATIONS}},
        "by_operator": [{"operator": operator, "families": sum(f["operator"] == operator for f in summaries),
                         "entries": category_counts([e for e in entries if e["operator"] == operator]),
                         "unique_pairs": category_counts([e for e in pairs if e["operator"] == operator]),
                         **{flag + "_families": sum(f["operator"] == operator and f[flag] for f in summaries)
                            for flag in ("all_three_vs_parent_verified", "fully_adjacent_verified", "severe_vs_parent_verified")}}
                        for operator in operators],
        "all_family_ids": sorted(families), "subsets": subsets,
        "families": summaries, "comparisons": entries,
    }


def check_bindings(audit: dict, manifest: Path) -> None:
    inputs = audit["inputs"]
    if inputs["manifest"] != binding(manifest) or inputs["script"] != binding(Path(__file__)):
        raise ValueError("Audit manifest/script binding changed")
    if inputs["validator"] != binding(Path(review.__file__)) or inputs["metrics"] != binding(Path(metrics.__file__)):
        raise ValueError("Audited validator/metrics dependency changed")
    pinned = [inputs[key] for key in ("manifest", "index", "script", "validator", "metrics")]
    pinned.extend(item[key] for item in inputs["packets"] for key in ("packet", "review"))
    for item in pinned:
        if review.sha(item["path"]) != item["sha256"]:
            raise ValueError(f"Audited input changed: {item['path']}")


def load_audit(audit_path: Path, manifest: Path) -> dict:
    """Check pins, revalidate assets and independently reconstruct all audit membership."""
    audit = json.loads(audit_path.read_text())
    check_bindings(audit, manifest)
    rebuilt = build_audit(manifest, Path(audit["inputs"]["index"]["path"]).parent)
    if {k: v for k, v in audit.items() if k != "audited_at"} != {k: v for k, v in rebuilt.items() if k != "audited_at"}:
        raise ValueError("Audit contents differ from independently reconstructed review evidence")
    return audit


def score_rows(cases: list[dict], rows: list[dict]) -> tuple[dict, list[dict]]:
    by_id = {c["case_id"]: c for c in cases}
    lookup = {}
    for row in rows:
        if row["case_id"] not in by_id or row["arm"] != "baseline" or row["model"] not in MODELS:
            continue
        key = (row["case_id"], row["model"])
        if key in lookup:
            raise ValueError("Duplicate controlled baseline score slot")
        case = by_id[row["case_id"]]
        if any(row.get(k) != case[k] for k in ("family_id", "view_sha256", "benchmark_partition", "operator", "severity")):
            raise ValueError("Controlled score row identity differs from audited manifest")
        score = row.get("technical_score")
        valid = type(score) in {int, float} and math.isfinite(score) and 1 <= score <= 5
        if (score is not None and not valid) or row.get("technical_score_valid") is not valid:
            raise ValueError("Invalid technical score or contradictory validity flag")
        lookup[key] = {**row, "technical_score": float(score) if valid else None, "slot_present": True}
    for case in cases:
        for model in MODELS:
            lookup.setdefault((case["case_id"], model), {"case_id": case["case_id"], "model": model,
                              "arm": "baseline", "technical_score": None, "slot_present": False})
    return lookup, list(lookup.values())


def scored_pairs(entries: list[dict], lookup: dict) -> list[dict]:
    result = []
    for entry in unique_pairs(entries):
        for model in MODELS:
            derivative = lookup[(entry["case_id"], model)]
            comparison = lookup[(entry["comparison_case_id"], model)]
            value, baseline = derivative["technical_score"], comparison["technical_score"]
            complete = value is not None and baseline is not None
            drop = baseline - value if complete else None
            outcome = "missing" if not complete else "strict_decrease" if drop > TOLERANCE else "strict_increase" if drop < -TOLERANCE else "tie"
            result.append({
                **entry, "model": model, "complete": complete, "outcome": outcome,
                "derivative_technical_score": value, "comparison_technical_score": baseline,
                "technical_drop": drop, "both_technical_scores_exactly_1": complete and value == baseline == 1,
                "both_scope_floor_applied": complete and derivative.get("scope_floor_applied") is True and comparison.get("scope_floor_applied") is True,
                "missing_case_ids": [r["case_id"] for r in (comparison, derivative) if r["technical_score"] is None],
                "absent_row_case_ids": [r["case_id"] for r in (comparison, derivative) if not r["slot_present"]],
            })
    return result


def pair_summaries(entries: list[dict], pairs: list[dict], operators: list[str]) -> list[dict]:
    identities = {(e["case_id"], e["comparison_case_id"]) for e in entries}
    selected = [p for p in pairs if (p["case_id"], p["comparison_case_id"]) in identities]
    summaries = []
    for model in MODELS:
        for operator in ["all", *operators]:
            group = [p for p in selected if p["model"] == model and (operator == "all" or p["operator"] == operator)]
            complete = [p for p in group if p["complete"]]
            counts = Counter(p["outcome"] for p in group)
            summaries.append({
                "model": model, "operator": operator, "expected_pairs": len(group),
                "complete_pairs": len(complete), "missing_pairs": counts["missing"],
                "pairs_with_absent_rows": sum(bool(p["absent_row_case_ids"]) for p in group),
                "expected_families": len({p["family_id"] for p in group}),
                "strict_decrease": counts["strict_decrease"], "ties": counts["tie"],
                "strict_increase": counts["strict_increase"],
                "floor_1_pairs": sum(p["both_technical_scores_exactly_1"] for p in complete),
                "other_ties": sum(p["outcome"] == "tie" and not p["both_technical_scores_exactly_1"] for p in complete),
                "floor_1_pairs_with_both_scope_floor_flags": sum(p["both_technical_scores_exactly_1"] and p["both_scope_floor_applied"] for p in complete),
                "mean_technical_drop": sum(p["technical_drop"] for p in complete) / len(complete) if complete else None,
            })
    return summaries


def ladder_subset(cases: list[dict], rows: list[dict], family_ids: list[str], *, verified: bool) -> dict:
    selected_ids = set(family_ids)
    subset = metrics.ladders([c for c in cases if c["family_id"] in selected_ids], rows)
    present = {(s["model"], s["operator"]) for s in subset["summary"]}
    for empty in metrics.ladders([], [])["summary"]:
        for operator in sorted({c["operator"] for c in cases}):
            if (empty["model"], operator) not in present:
                subset["summary"].append({**empty, "operator": operator})
    subset["scope"] = "Fully visually verified adjacent comparisons only; supplemental subset" if verified else "All 100 controlled families retained; unchanged original ladder definitions"
    subset["verified_subset_status"] = "All three adjacent comparisons independently classified clear_degradation" if verified else "All families retained; verified comparisons reported separately"
    for summary in subset["summary"]:
        summary["missing_families"] = summary["families"] - summary["complete_families"]
        complete = [f for f in subset["families"] if f["model"] == summary["model"] and f["complete"] and (summary["operator"] == "all" or f["operator"] == summary["operator"])]
        summary["strictly_decreasing_families"] = sum(all(a - b > TOLERANCE for a, b in zip(f["technical_scores"], f["technical_scores"][1:])) for f in complete)
        summary["adjacent_strict_decreases"] = sum(sum(a - b > TOLERANCE for a, b in zip(f["technical_scores"], f["technical_scores"][1:])) for f in complete)
        summary["adjacent_floor_1_pairs"] = sum(sum(a == b == 1 for a, b in zip(f["technical_scores"], f["technical_scores"][1:])) for f in complete)
    return subset


def analyze_controlled(cases: list[dict], rows: list[dict], audit: dict) -> dict:
    families = controlled_families(cases)
    controlled = [c for members in families.values() for c in members]
    lookup, normalized = score_rows(controlled, rows)
    entries = audit["comparisons"]
    pairs = scored_pairs(entries, lookup)
    operators = sorted({c["operator"] for c in controlled})
    comparisons = {}
    for name, selected in (
        ("parent_comparisons", [e for e in entries if e["relation"] == "versus_parent"]),
        ("severe_parent_comparisons", [e for e in entries if e["relation"] == "versus_parent" and e["severity"] == 3]),
        ("adjacent_comparisons", [e for e in entries if e["relation"] == "versus_previous"]),
    ):
        comparisons[name] = {"all": pair_summaries(selected, pairs, operators),
                             "visually_verified": pair_summaries([e for e in selected if e["classification"] == "clear_degradation"], pairs, operators)}
    verified_ids = audit["subsets"]["fully_adjacent_verified_family_ids"]
    return {
        "schema_version": 1, "method": METHOD, "tolerance": TOLERANCE,
        "all_families_preserved": len(families) == 100, "primary_analysis_modified": False,
        "review_coverage": audit["coverage"], "review_classifications": audit["classifications"],
        "review_subsets": audit["subsets"], "review_by_operator": audit["by_operator"],
        "review_families_with_reasons": audit["families"], **comparisons,
        "all_family_ladders": ladder_subset(controlled, normalized, list(families), verified=False),
        "fully_adjacent_verified_ladders": ladder_subset(controlled, normalized, verified_ids, verified=True),
        "pair_details": pairs,
    }


def render_report(result: dict) -> str:
    coverage = result["review_coverage"]
    subsets = result["review_subsets"]
    lines = ["## Supplemental blind review of controlled degradation", "", METHOD, "",
             f"The audit covers all {coverage['families']} families ({coverage['members']} images), "
             f"{coverage['comparison_entries']} recorded entries and {coverage['unique_image_pairs']} unique image pairs. "
             "V1’s parent and previous comparisons name the same pair. All 100 families remain in the primary diagnostics; no label changed.", "",
             f"All three derivatives are verified against the parent in {len(subsets['all_three_vs_parent_verified_family_ids'])} families; "
             f"{len(subsets['fully_adjacent_verified_family_ids'])} families have all three adjacent comparisons verified; "
             f"{len(subsets['severe_vs_parent_verified_family_ids'])} severe-versus-parent comparisons are verified. "
             "These are distinct subsets. Review classifications and per-pair reasons are preserved in `controlled-review/audit.json` and `controlled-analysis.json`.", ""]
    counts = result["review_classifications"]["unique_pairs"]
    lines += ["Unique-pair review classifications: " + "; ".join(f"{key}: {value}" for key, value in counts.items()) + ".", ""]
    for key, title in (("parent_comparisons", "All derivatives versus parent"), ("severe_parent_comparisons", "Severe derivative versus parent")):
        lines += [f"### {title}", "", "↓ / = / ↑ count strict decreases, ties and strict increases. Floor counts are included in ties. Drop uses all complete pairs, including ties.", "",
                  "| Model | Operator | Review subset | Complete/expected | Missing | ↓ / = / ↑ | Floor 1 | Mean technical drop |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for scope in ("all", "visually_verified"):
            for row in result[key][scope]:
                drop = "—" if row["mean_technical_drop"] is None else f"{row['mean_technical_drop']:.4f}"
                lines.append(f"| {row['model']} | {row['operator']} | {scope} | {row['complete_pairs']}/{row['expected_pairs']} | {row['missing_pairs']} | {row['strict_decrease']} / {row['ties']} / {row['strict_increase']} | {row['floor_1_pairs']} | {drop} |")
        lines.append("")
    lines += ["### Fully verified adjacent families", "", "Ladder counts require all four technical scores. Pair tables above require only their two endpoints, so missing intermediate scores do not erase a valid parent–severe comparison.", "",
              "| Model | Operator | Complete/expected families | Missing | Non-increasing | Strictly decreasing | Adjacent ↓ / = / ↑ | Adjacent floor 1 | Mean parent–severe drop |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["fully_adjacent_verified_ladders"]["summary"]:
        drop = "—" if row["mean_parent_minus_severity3"] is None else f"{row['mean_parent_minus_severity3']:.4f}"
        lines.append(f"| {row['model']} | {row['operator']} | {row['complete_families']}/{row['families']} | {row['missing_families']} | {row['nonincreasing_families']} | {row['strictly_decreasing_families']} | {row['adjacent_strict_decreases']} / {row['adjacent_ties']} / {row['strict_inversions']} | {row['adjacent_floor_1_pairs']} | {drop} |")
    lines += ["", "The JSON preserves all-family ladders, adjacent-pair coverage, both scope-floor flags, missing endpoint IDs and native score differences. Review verification concerns visible degradation and does not establish an independently scored quality target.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "analyze"))
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--reviews", type=Path, default=HERE / "controlled-review")
    parser.add_argument("--audit", type=Path, default=HERE / "controlled-review/audit.json")
    parser.add_argument("--scores", type=Path, default=HERE / "scores-long.jsonl")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    if args.command == "audit":
        audit = build_audit(args.manifest, args.reviews)
        review.write_new(args.audit, audit)
        print(json.dumps({"audit": str(args.audit), "coverage": audit["coverage"], "classifications": audit["classifications"], "subset_counts": {k: len(v) for k, v in audit["subsets"].items()}}, indent=2))
        return
    audit = load_audit(args.audit, args.manifest)
    score_binding = binding(args.scores)
    rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()]
    result = analyze_controlled(json.loads(args.manifest.read_text())["cases"], rows, audit)
    if score_binding != binding(args.scores):
        raise ValueError("Score export changed during supplemental analysis")
    result["inputs"] = {"audit": binding(args.audit), "scores": score_binding, "manifest": binding(args.manifest), "script": binding(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "controlled-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "controlled-report.md").write_text(render_report(result))
    print(f"Wrote controlled-analysis.json and controlled-report.md to {args.output}; all 100 families retained")


if __name__ == "__main__":
    main()
