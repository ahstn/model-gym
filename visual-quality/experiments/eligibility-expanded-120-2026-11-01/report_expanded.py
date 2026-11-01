"""Write the expanded eligibility trial report from retained model results."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

import analyze_expanded as analysis

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-expanded-120-2026-11-01")


def read(p):
    return json.loads(Path(p).read_text())


def table(head, rows):
    def line(xs):
        return "| " + " | ".join(str(x).replace("|", "\\|") for x in xs) + " |"

    return "\n".join((line(head), line(["---"] * len(head)), *(line(r) for r in rows)))


def pc(n, d):
    return f"{100 * n / d:.1f}%" if d else "—"


def impact(p):
    b = p["baseline_paired"]
    c = p["candidate_paired"]
    return f"{c['missed_exclusions'] - b['missed_exclusions']:+d} / {c['false_floors'] - b['false_floors']:+d} ({p['paired_cases']})"


def main():
    w = read(HERE / "winner.json")
    r = read(HERE / "results.json")
    parts = [
        "# Expanded eligibility prompt optimisation: 120 + 120 views",
        "Ten additional eligibility questions were evaluated on 120 development views per model. Each model-specific selection was frozen before the 120-view held-out comparison. On the holdout, the original question, the previously selected prompt, and the new selection were evaluated separately. The quality questions, typed decision schema and 0.50 exclusion threshold were unchanged. Reference eligibility labels are assistant visual assessments, not human ground truth.",
        "Each panel contains 60 eligible photographs and 60 excluded views. Per panel: 30 physical-context photos, 30 other photo scenes, 10 historical flat works, two public-domain museum works, 10 charts, 10 renders, and 28 synthetic screenshots derived from photos in the same panel. Families are split between panels. Screenshot styles are generated from two UI templates per panel and are paired with their source photos; counts are not independent observations and neither the set nor its 50% exclusion share represents real-world prevalence.",
        "The ten development historical cases were part of a previous eligibility trial and include known failures. The ten held-out historical cases were model-scored in the previous 4,000-view corpus, but were not part of the earlier eligibility optimization; they were selected from prior reference classifications rather than model scores. The other sampled families were absent from those runs, but family separation does not prove absence of visually related images. [Met Open Access API](https://metmuseum.github.io/) retrieval encountered HTTP 403s, so only four distinct public-domain museum images were usable. See [protocol](protocol.md), [reference manifest](reference-manifest.json), [prompt variants](variants.json) and [previous prompts](previous-prompts.json).",
        "The first holdout coordinator matched the wrong partition name and scheduled zero jobs. After confirming no holdout requests had been made, the driver filter was corrected without changing the already frozen selection; the corrected stage then attempted all 120 expected images. See [the driver amendment](holdout-driver-amendment.json) and service logs.",
        "## Holdout including historical challenge",
        "The 120 held-out views include the ten previously scored natural cases; the table is intentionally split by category below. A missing or invalid result is never counted as correct.",
    ]
    rows = []
    for model in ("liquid", "jev"):
        z = r["models"][model]
        for arm, label in [
            ("original", "Original"),
            ("previous", "Previous"),
            ("selected", "Selected " + z["chosen_round"]),
        ]:
            m = z["arms"][arm]
            rows.append(
                [
                    model.title(),
                    label,
                    f"{m['valid']}/{m['expected']}",
                    f"{m['missed_exclusions']}/{m['excluded_valid']}",
                    pc(m["correct_exclusions"], m["excluded_valid"]),
                    f"{m['false_floors']}/{m['eligible_valid']}",
                    pc(m["false_floors"], m["eligible_valid"]),
                ]
            )
    parts.append(
        table(
            [
                "Model",
                "Question",
                "Valid",
                "Missed exclusions / valid",
                "Exclusion recall",
                "False floors / valid",
                "False-floor rate",
            ],
            rows,
        )
    )
    parts.append("### Paired change against the previous prompt")
    rows = []
    for model in ("liquid", "jev"):
        p = r["models"][model]["selected_vs"]["previous"]
        b = p["baseline_paired"]
        c = p["candidate_paired"]
        rows.append(
            [
                model.title(),
                p["paired_cases"],
                f"{b['missed_exclusions']} → {c['missed_exclusions']}",
                f"{b['false_floors']} → {c['false_floors']}",
                impact(p),
            ]
        )
    parts.append(
        table(
            ["Model", "Paired cases", "Missed exclusions", "False floors", "Δ misses / Δ false (paired)"],
            rows,
        )
    )
    for model in ("liquid", "jev"):
        if r["models"][model]["chosen_round"] == "previous":
            parts.append(
                f"For **{model.title()}**, the selected and previous arms use the same question in separate held-out requests. Any difference between them reflects repeat-request variability or provider behavior, not an improvement from a new prompt."
            )
    parts.append(
        "**Recommendation:** keep Liquid's previous revision for further validation. Jev's r14 improved development errors but did not improve the paired holdout over its previous revision, so retain the previous Jev prompt for now. Neither result is sufficient to deploy an automatic floor without broader independent review."
    )
    parts.append("### Held-out breakdown by reference category")
    rows = []
    for model in ("liquid", "jev"):
        d = r["models"][model]["arms"]
        groups = sorted(d["original"]["by_group"])
        for group in groups:
            vals = []
            for arm in ("original", "previous", "selected"):
                a = d[arm]["by_group"][group]
                vals.append(f"{a['errors']}/{a['valid']}")
            rows.append([model.title(), group, *vals, d["original"]["by_group"][group]["expected"]])
    parts.append(
        table(
            [
                "Model",
                "Category",
                "Original errors / valid",
                "Previous errors / valid",
                "Selected errors / valid",
                "Expected",
            ],
            rows,
        )
    )
    parts.append(
        "For excluded categories an error is a missed exclusion; for eligible photographs it is a false floor. Historical works are a selected previously scored challenge. Synthetic UI templates, chart and render families limit generalization. Even zero false floors among 60 eligible photos is not proof that the rate is negligible."
    )
    sensitivity = read(HERE / "boundary-sensitivity.json")
    assert sensitivity["reference_sha256"] == r["reference_sha256"]
    parts.append("### Boundary-label sensitivity")
    parts.append(
        "Before seeing holdout responses, three historical cases (R0226, R0317 and R0356) were flagged as potentially eligible photos of physical printed material with surrounding context. The main table preserves the prior excluded labels. This secondary calculation treats those three as eligible, leaving 57 excluded and 63 eligible reference cases; it is an alternative interpretation, not a changed ground truth. See [boundary review](boundary-sensitivity.json)."
    )
    sensitivity_rows = []
    for model in ("liquid", "jev"):
        adjusted = {}
        for arm in ("original", "previous", "selected"):
            rows_for_arm = [
                row | {"reference_eligibility": "eligible"} if row["case_id"] in sensitivity["cases"] else row
                for row in r["case_rows"]
                if row["model"] == model and row["display_arm"] == arm
            ]
            adjusted[arm] = rows_for_arm
            metrics = analysis.a.evaluate(rows_for_arm)
            sensitivity_rows.append(
                [
                    model.title(),
                    arm,
                    f"{metrics['missed_exclusions']}/{metrics['excluded_valid']}",
                    f"{metrics['false_floors']}/{metrics['eligible_valid']}",
                ]
            )
        comparison = analysis.a.comparison(adjusted["previous"], adjusted["selected"])
        sensitivity_rows.append([model.title(), "paired Δ selected vs previous", impact(comparison), "—"])
    parts.append(
        table(["Model", "Prompt", "Missed exclusions / valid", "False floors / valid"], sensitivity_rows)
    )
    parts.append("## Development: ten additional rounds")
    rows = []
    for rid in read(HERE / "variants.json"):
        row = [rid]
        for model in ("liquid", "jev"):
            x = w["development"][model]["rounds"][rid]
            m = x["metrics"]
            row.extend(
                [
                    f"{m['missed_exclusions']} / {m['false_floors']} ({m['valid']}/120)",
                    impact(x["paired_vs_previous"]),
                    "yes" if x["qualifies"] else "no",
                ]
            )
        rows.append(row)
    parts.append(
        table(
            [
                "Round",
                "Liquid misses / false (valid)",
                "Liquid Δ vs previous (paired)",
                "Liquid qualifies",
                "Jev misses / false (valid)",
                "Jev Δ vs previous (paired)",
                "Jev qualifies",
            ],
            rows,
        )
    )
    parts.append(
        "**Polarity failure:** r15 produced 59 missed exclusions and 59 false floors for Liquid, and 59 missed exclusions and 56 false floors for Jev. Its wording begins by asking whether the image is a photographic scene, then assigns YES to exclusion. The near reversal is consistent with the models following the opening question's polarity instead of the later mapping, but the trial does not establish the cause. Jev's r18 and r20 also regressed sharply. Keep the decision question's polarity explicit and test it; do not assume extra instructions are harmless."
    )
    for model in ("liquid", "jev"):
        x = w["models"][model]
        parts.append(
            f"**{model.title()}:** selected **{x['round']}**; {x['qualified_new_variants']} new variants met the development guard. Eligibility required at least 114 valid and 57 valid per class, no extra paired false floors, and fewer paired total errors than the previous prompt. Missing candidate responses on previous-valid cases count as errors for ranking. If no variant qualified the previous prompt was retained. The complete frozen selection and timings are in [winner.json](winner.json) and [winner-freeze.json](winner-freeze.json)."
        )
    parts.append("## Quality-score drift on eligible photos")
    rows = []
    for model in ("liquid", "jev"):
        d = r["models"][model]["selected_vs"]["previous"]["native_quality_drift"]
        for name in ("technical_native", "aesthetic_native"):
            x = d[name]
            rows.append(
                [
                    model.title(),
                    name,
                    x["paired_photos"],
                    f"{x['mean_signed_change']:.3f}" if x["mean_signed_change"] is not None else "—",
                    f"{x['mean_absolute_change']:.3f}" if x["mean_absolute_change"] is not None else "—",
                    x["changes_at_least_half_point"],
                ]
            )
    parts.append(
        "These are changes in native 1–5 scores before applying any floor. No independent technical or aesthetic quality scores were collected here; drift is not a measure of accuracy and may include model stochasticity."
    )
    parts.append(
        table(
            ["Model", "Score", "Paired photos", "Mean signed change", "Mean absolute change", "Changes ≥0.5"],
            rows,
        )
    )
    parts.append("## Latency and cost")
    rows = []
    statuses = Counter()
    for model in ("liquid", "jev"):
        known = Decimal(0)
        unknown = 0
        lat = []
        n = 0
        for req in (DATA / "attempts").glob(f"*/{model}/*/*/request.json"):
            n += 1
            p = req.with_name("result.json")
            if not p.exists():
                unknown += 1
                statuses["unknown_no_response"] += 1
                continue
            x = read(p)
            statuses[x["status"]] += 1
            if x.get("price_usd") is None:
                unknown += 1
            else:
                known += Decimal(str(x["price_usd"]))
            if x["status"] == "valid" and x.get("latency_seconds") is not None:
                lat.append(x["latency_seconds"])
        rows.append(
            [
                model.title(),
                n,
                f"{statistics.median(lat):.2f}" if lat else "—",
                f"${known}",
                unknown,
                "unknown" if unknown else f"${known}",
            ]
        )
    parts.append(
        "Latency is elapsed end-to-end request/response time for valid calls; concurrency was six combined workers. All attempts are counted, including retries. Unpriced failures are unknown rather than zero. Provider-parsed JSON and request/response hashes are preserved; exact original wire text is not retained."
    )
    parts.append(
        table(
            [
                "Model",
                "Attempts",
                "Median valid seconds",
                "Known cost subtotal",
                "Unpriced attempts",
                "Total cost",
            ],
            rows,
        )
    )
    parts.append("Statuses: " + ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())) + ".")
    parts.append("## Limitations and next step")
    parts.append(
        "Do not treat these deliberately balanced, partly synthetic assistant-labeled panels as an estimate of deployment error rates. Before promoting a prompt into automatic labeling, use an independent human-reviewed sample containing more diverse real screenshots, posters, cropped artwork, photographed packaging and ambiguous physical objects; examine both missed exclusions and false floors. Keep original scores and decision evidence available for review."
    )
    (HERE / "report.md").write_text("\n\n".join(parts) + "\n")
    print(json.dumps({"report": str(HERE / "report.md"), "statuses": dict(statuses)}))


if __name__ == "__main__":
    main()
