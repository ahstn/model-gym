"""Render the documented eligibility trial from immutable result records."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-optimization-2026-11-01")


def read(p):
    return json.loads(Path(p).read_text())


def table(head, rows):
    def line(a):
        return "| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in a) + " |"

    return "\n".join([line(head), line(["---"] * len(head)), *[line(r) for r in rows]])


def cell(r):
    return f"{r['missed_exclusions']} / {r['false_floors']} ({r['valid']}/{r['expected']})"


def direction(p):
    b, c = p["baseline_paired"], p["candidate_paired"]
    a = c["missed_exclusions"] - b["missed_exclusions"]
    d = c["false_floors"] - b["false_floors"]
    if a <= 0 and d <= 0 and (a < 0 or d < 0):
        label = "Improved"
    elif a >= 0 and d >= 0 and (a > 0 or d > 0):
        label = "Worsened"
    elif a == 0 and d == 0:
        label = "Unchanged"
    else:
        label = "Mixed"
    return f"{label}: {a:+d} / {d:+d} ({p['paired_cases']} paired)"


def money():
    totals = {
        m: {"attempts": 0, "valid": 0, "errors": 0, "known": Decimal(0), "unknown": 0, "latencies": []}
        for m in ("liquid", "jev")
    }
    statuses = Counter()
    for req in sorted((DATA / "attempts").glob("*/*/*/*/request.json")):
        meta = read(req)
        m = meta["model"]
        v = totals[m]
        v["attempts"] += 1
        p = req.with_name("result.json")
        if not p.exists():
            v["unknown"] += 1
            statuses["unknown_no_response"] += 1
            continue
        r = read(p)
        statuses[r["status"]] += 1
        v["valid"] += r["status"] == "valid"
        v["errors"] += r["status"] != "valid"
        if r.get("price_usd") is None:
            v["unknown"] += 1
        else:
            v["known"] += Decimal(str(r["price_usd"]))
        if r["status"] == "valid" and r.get("latency_seconds") is not None:
            v["latencies"].append(r["latency_seconds"])
    return totals, statuses


def main():
    r = read(HERE / "results.json")
    w = read(HERE / "winners.json")
    parts = [
        "# Liquid and Jev: eligibility prompt optimisation",
        "Ten candidate prompts were tested on a 40-image development set, followed by the frozen model-specific candidates on a separate 40-image fresh holdout and a 20-image historical natural-image challenge. There were 20 excluded and 20 eligible cases in each fresh partition. These are assistant-reviewed labels, not human ground truth.",
        "The fresh cases are drawn from source families absent from the previous 4,000-image run, three separately sourced public-domain Commons works, and 19 synthetic browser screenshots paired with fresh photographs within the same split. The original views are hash-checked; exact hashes and source families differ from the prior run, but this does not prove that no visual near-duplicate exists. The screenshot controls share one UI template across splits. The held-out excluded class includes only two natural flat artwork/document cases; the rest are synthetic screenshots, simple charts or obvious renders. The sample is balanced by design and is not representative of deployment prevalence. A Wikimedia rate limit restricted acquisition of additional provenance-complete public-domain posters.",
        "The 20-image natural challenge (10 excluded, 10 eligible) was previously model-scored and includes two already identified disagreements. It supplies more natural poster/document cases, but is a selected historical stress set rather than fresh unbiased evidence. No challenge or held-out model result was used to choose prompts.",
        "[Fresh and challenge image/label manifest](reference-manifest.json) · [Raw case comparisons and all round statistics](results.json) · [Ten full prompt texts: rounds 1–5](rounds-first.json) · [rounds 6–10](rounds-second.json) · [protocol](protocol.md) · [comparison gallery](gallery/eligibility-trial.html)",
    ]
    rows = []
    for rid in [f"r{i:02d}" for i in range(1, 11)]:
        cells = [rid]
        for model in ("liquid", "jev"):
            item = r["development"][model]["rounds"][rid]
            cells.extend([cell(item["metrics"]), direction(item["paired"])])
        rows.append(cells)
    baseline = [cell(r["development"][m]["baseline"]) for m in ("liquid", "jev")]
    parts.extend(
        [
            "## Ten development rounds",
            f"Original baseline (missed exclusions / false floors, valid): Liquid **{baseline[0]}**; Jev **{baseline[1]}**. Each class has 20 expected cases. Each revision changes only the eligibility-question text, retaining the previous common instruction, both quality questions and 0.50 threshold. Positive and negative changes below are on the same paired cases; they are exploratory development results, not independent confirmations. A missing case reduces coverage and is never counted as a correct decision.",
            table(
                (
                    "Round",
                    "Liquid misses / false floors (valid)",
                    "Liquid paired impact: Δmiss / Δfalse",
                    "Jev misses / false floors (valid)",
                    "Jev paired impact: Δmiss / Δfalse",
                ),
                rows,
            ),
            "Rounds 6–10 were written after reviewing only rounds 1–5 development results. Their motivation is in [rounds-second-rationale.md](rounds-second-rationale.md). Prompt versions combine several edits, so the trial does not isolate the causal contribution of individual sentences.",
        ]
    )
    for part, title in [
        ("heldout", "Fresh held-out evaluation"),
        ("natural_challenge", "Previously scored natural challenge"),
    ]:
        rows = []
        for model in ("liquid", "jev"):
            d = r["evaluation"][part][model]
            for arm in ("baseline", "candidate"):
                x = d[arm]
                rows.append(
                    [
                        model.title(),
                        "Original" if arm == "baseline" else d["candidate_round"],
                        f"{x['missed_exclusions']} / {x['excluded_valid']} of {x['excluded_expected']}",
                        f"{x['false_floors']} / {x['eligible_valid']} of {x['eligible_expected']}",
                        f"{x['valid']} / {x['expected']}",
                    ]
                )
        parts.extend(
            [
                f"## {title}",
                table(
                    (
                        "Model",
                        "Prompt",
                        "Missed exclusions / valid of expected",
                        "Wrongly floored photos / valid of expected",
                        "All valid",
                    ),
                    rows,
                ),
            ]
        )
        if part == "heldout":
            details = []
            for model in ("liquid", "jev"):
                x = r["evaluation"][part][model]
                b = x["paired"]["baseline_paired"]
                c = x["paired"]["candidate_paired"]
                details.append(
                    f"**{model.title()} ({x['candidate_round']}):** paired {x['paired']['paired_cases']}/40; missed exclusions {b['missed_exclusions']} → {c['missed_exclusions']}; false floors {b['false_floors']} → {c['false_floors']}."
                )
            parts.append(
                " ".join(details)
                + " The 20 examples per class and shared screenshot template are too small and narrow to establish production safety or population-level rates. Binomial Wilson intervals are recorded in results.json as descriptive calculations; this purposive, family-paired sample is not an independent random deployment sample."
            )
        groups = []
        for model in ("liquid", "jev"):
            d = r["evaluation"][part][model]
            keys = sorted(set(d["baseline"]["by_group"]) | set(d["candidate"]["by_group"]))
            for group in keys:
                a = d["baseline"]["by_group"].get(group, {})
                b = d["candidate"]["by_group"].get(group, {})
                groups.append(
                    [
                        model.title(),
                        group,
                        f"{a.get('errors', 0)}/{a.get('valid', 0)}",
                        f"{b.get('errors', 0)}/{b.get('valid', 0)}",
                        a.get("expected", b.get("expected", 0)),
                    ]
                )
        parts.append(
            table(
                (
                    "Model",
                    "Reference category",
                    "Baseline errors / valid",
                    "Candidate errors / valid",
                    "Expected",
                ),
                groups,
            )
        )
    diagnostic = read(HERE / "resolution-retest-results.json")
    diagnostic_rows = []
    for x in diagnostic["rows"]:
        diagnostic_rows.append(
            [
                x["model"].title(),
                x["display_arm"],
                x["status"],
                f"{x['exclusion_probability']:.4f}" if x["exclusion_probability"] is not None else "—",
                "Floor" if x["scope_floor_applied"] else "Eligible",
            ]
        )
    parts.extend(
        [
            "## Post-hoc W0002 image-size diagnostic",
            "Both original Jev requests for W0002 returned HTTP 400, 'Invalid image dimensions', so they remain missing in the frozen holdout. After completing the holdout, a separate copy U0002 was enlarged from 482 × 236 to 572 × 280 pixels using Lanczos interpolation without a crop or padding. Both models accepted that copy under both unchanged prompts. This single post-hoc test suggests a size-related compatibility issue; it is not an independent held-out image and does not replace the original missing results. [Diagnostic freeze](resolution-retest-freeze.json) · [diagnostic results](resolution-retest-results.json).",
            table(("Model", "Prompt", "Status", "Exclude probability", "Decision"), diagnostic_rows),
            "Liquid's candidate wrongly floored C4864, an eligible photograph of a physical printed object with visible surroundings. Jev's candidate still missed C1546, a full-frame drawing. These image judgments were reviewed after results were available, so are explanatory rather than blind adjudication.",
        ]
    )
    parts.append("## Photographic quality score drift")
    rows = []
    for model in ("liquid", "jev"):
        for part in ("heldout", "natural_challenge"):
            drift = r["evaluation"][part][model]["paired"]["native_quality_drift"]
            for key, label in [("technical_native", "Technical"), ("aesthetic_native", "Aesthetic")]:
                d = drift[key]
                rows.append(
                    [
                        model.title(),
                        part.replace("_", " "),
                        label,
                        d["paired_photos"],
                        f"{d['mean_signed_change']:.3f}" if d["mean_signed_change"] is not None else "—",
                        f"{d['mean_absolute_change']:.3f}" if d["mean_absolute_change"] is not None else "—",
                        d["changes_at_least_half_point"],
                    ]
                )
    parts.extend(
        [
            "For eligible photos, these compare the native 1–5 scores before any scope floor. No independent technical or aesthetic reference scores were collected for this trial, so differences measure drift, not an improvement or degradation in quality accuracy. Provider stochasticity and the changed question may both contribute.",
            table(
                (
                    "Model",
                    "Panel",
                    "Task",
                    "Paired photos",
                    "Mean signed change",
                    "Mean absolute change",
                    "Changes ≥0.50",
                ),
                rows,
            ),
        ]
    )
    parts.extend(
        [
            "## Prompt selection and recommendation",
            "The candidate for each model was frozen before any held-out request. It was chosen from the ten development variants by the coverage and false-floor guard, then fewest paired errors; ties favored fewer false floors, shorter wording and lower round ID. A development candidate must reduce missed exclusions without increasing total paired errors to replace the original. The exact rule and all counts are in [winners.json](winners.json).",
        ]
    )
    for model in ("liquid", "jev"):
        x = w["models"][model]
        d = r["evaluation"]["heldout"][model]
        b = d["paired"]["baseline_paired"]
        c = d["paired"]["candidate_paired"]
        safe = c["false_floors"] <= b["false_floors"]
        better = c["missed_exclusions"] < b["missed_exclusions"]
        decision = (
            "promising for a wider independent trial"
            if safe and better and x["advance_from_development"]
            else "do not adopt as a default on this evidence"
        )
        parts.append(
            f"**{model.title()}:** candidate **{x['candidate_round']}**, development advancement {'passed' if x['advance_from_development'] else 'not met'}; fresh paired misses {b['missed_exclusions']} → {c['missed_exclusions']}, false floors {b['false_floors']} → {c['false_floors']}. Recommendation: **{decision}**. Even a zero count among 20 photos cannot rule out an appreciable false-floor rate."
        )
    parts.extend(
        [
            "The prompts were evaluated at the existing uncalibrated 0.50 cutoff. Do not lower or retune it using this holdout. Before a larger labeling run, validate the selected candidates on a new human-reviewed, family-held-out set with diverse real poster scans, screenshots, photographed physical packaging and art in scenes; include different UI designs, photographs with text, and random cases. Preserve the original baseline and retain the underlying native scores when the gate applies.",
            "## Latency and costs",
        ]
    )
    totals, statuses = money()
    rows = []
    for model in ("liquid", "jev"):
        x = totals[model]
        rows.append(
            [
                model.title(),
                x["attempts"],
                x["valid"],
                x["errors"],
                f"{statistics.median(x['latencies']):.2f}" if x["latencies"] else "—",
                "$" + str(x["known"]),
                x["unknown"],
                "Unknown" if x["unknown"] else "$" + str(x["known"]),
            ]
        )
    parts.extend(
        [
            "End-to-end latency runs from HTTP request through the complete response. The median below is for valid requests across all rounds; concurrency was four combined workers and the routes are not a controlled production latency benchmark. Known charges include all recorded attempts. An unpriced error or interruption remains unknown rather than being assigned $0. Raw parsed provider response bodies and their file hashes are retained; hashes of received wire text were recorded, but the exact wire text was not retained and cannot be independently reconstructed from the parsed object.",
            table(
                (
                    "Model",
                    "Attempts",
                    "Valid responses",
                    "Other outcomes",
                    "Median valid s",
                    "Known subtotal",
                    "Unpriced attempts",
                    "Total",
                ),
                rows,
            ),
            "Observed attempt statuses: "
            + ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items()))
            + ". No score, eligibility reference or original trial output was overwritten.",
        ]
    )
    (HERE / "report.md").write_text("\n\n".join(parts) + "\n")
    print(
        json.dumps(
            {
                "report": str(HERE / "report.md"),
                "statuses": dict(statuses),
                "known_usd": {m: str(v["known"]) for m, v in totals.items()},
                "unknown_attempts": {m: v["unknown"] for m, v in totals.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
