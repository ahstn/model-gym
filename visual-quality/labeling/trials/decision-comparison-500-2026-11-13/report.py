"""Compare fresh Decisions observations with the unchanged 10K baseline rows."""

import csv
import html
import json
import math
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

from prepare import DATA, ROOT, sha, write
from scipy.stats import spearmanr

NAMES = {
    "reference": "Reference",
    "luna": "Luna (high)",
    "deepseek": "DeepSeek (high)",
    "liquid": "Liquid d1.1",
    "jev": "Jev 1.14",
    "qrealign_lite": "Q-ReAlign Lite",
    "photo_ensemble": "Ensemble (37.5/37.5/25)",
    "perplexity": "Perplexity Decider v1.3 (free)",
    "gemini": "Gemini Decision 1",
}


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def metrics(cases, model):
    result = {}
    for dimension in ("technical", "aesthetic"):
        key = dimension + "_score"
        pairs = [(c["assessments"][model].get(key), c["assessments"]["reference"][key]) for c in cases]
        pairs = [(a, b) for a, b in pairs if numeric(a) and numeric(b)]
        result[dimension] = {
            "n": len(pairs),
            "mean": statistics.mean(a for a, _ in pairs) if pairs else None,
            "mae": statistics.mean(abs(a - b) for a, b in pairs) if pairs else None,
            "bias": statistics.mean(a - b for a, b in pairs) if pairs else None,
            "rho": float(spearmanr(*zip(*pairs)).statistic)
            if len({a for a, _ in pairs}) > 1 and len({b for _, b in pairs}) > 1
            else None,
        }
    latencies = [c["assessments"][model].get("latency_seconds") for c in cases]
    measured = [v for v in latencies if numeric(v)]
    result["latency_n"] = len(measured)
    result["latency_median_seconds"] = statistics.median(measured) if measured else None
    result["latency_mean_seconds"] = statistics.mean(measured) if measured else None
    result["numeric_pairs"] = sum(
        all(numeric(c["assessments"][model].get(d + "_score")) for d in ("technical", "aesthetic"))
        for c in cases
    )
    scope = Counter()
    for case in cases:
        truth = case["assessments"]["reference"].get("reference_eligibility")
        prediction = case["assessments"][model].get("scope_floor_applied")
        if (
            truth in ("eligible", "excluded")
            and type(prediction) is bool
            and model not in ("photo_ensemble", "qrealign_lite")
        ):
            scope[truth + "_n"] += 1
            scope["false_floor" if truth == "eligible" else "missed_exclusion"] += (
                prediction if truth == "eligible" else not prediction
            )
    result["scope"] = dict(scope)
    return result


def costs(cases, model):
    if model == "photo_ensemble":
        return {
            "known_usd": None,
            "unpriced_attempts": 0,
            "new_requests": 0,
            "basis": "derived; no independent charge",
        }
    if model in ("perplexity", "gemini"):
        requests = list((DATA / "results" / model).glob("attempts/*/*/request.json"))
        values = []
        for request in requests:
            result = request.with_name("result.json")
            values.append(json.loads(result.read_text()).get("price_usd") if result.exists() else None)
        return {
            "known_usd": str(sum((Decimal(str(v)) for v in values if v is not None), Decimal(0))),
            "unpriced_attempts": sum(v is None for v in values),
            "new_requests": len(requests),
            "basis": "new run; every attempt",
        }
    rows = [c["assessments"][model] for c in cases]
    values = [r.get("total_attempt_known_price_usd", r.get("price_usd")) for r in rows]
    return {
        "known_usd": str(sum((Decimal(str(v)) for v in values if v is not None), Decimal(0))),
        "unpriced_attempts": sum(
            r.get("unpriced_attempts", int(v is None)) for r, v in zip(rows, values, strict=True)
        ),
        "new_requests": 0,
        "basis": "retained historical receipts; no new spend",
    }


def number(value, digits=2):
    return "—" if value is None else f"{float(value):.{digits}f}"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def gallery(cases):
    labels = json.loads(
        (ROOT / "experiments/labeling-10000-2026-11-01/corpus-cluster-labels.json").read_text()
    )["clusters"]
    cards = []
    for c in cases:
        lines = []
        for model, name in NAMES.items():
            r = c["assessments"][model]
            cells = [
                name,
                number(r.get("technical_score")),
                number(r.get("aesthetic_score")),
                number(r.get("latency_seconds")),
                "—" if r.get("price_usd") is None else "$" + number(r["price_usd"], 6),
            ]
            lines.append("<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in cells) + "</tr>")
        ref = c["assessments"]["reference"].get("evidence", {})
        cards.append(
            f'<article id="{c["case_id"]}"><h2>{c["case_id"]} <small>{html.escape(c["selection_arm"])}</small></h2><a href="../images/{c["case_id"]}.png"><img loading="lazy" src="../images/{c["case_id"]}.png" alt="{c["case_id"]}"></a><table><tr><th>Assessor</th><th>Tech</th><th>Aest</th><th>Seconds</th><th>Price</th></tr>{"".join(lines)}</table><p>Cluster {c["content_cluster"]} · {html.escape(labels[str(c["content_cluster"])])} · {html.escape(c["category"])} · prior variance {c["selection_variance"]:.3f}</p><details><summary>Reference notes</summary>{html.escape(json.dumps(ref, ensure_ascii=False))}</details></article>'
        )
    style = "body{font:15px system-ui;background:#edf2f7;color:#183044;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:20px}article{background:white;border:1px solid #aac2d6;border-radius:12px;padding:16px}h2{margin:0 0 14px}small{float:right;font-size:12px;color:#657c90}img{width:100%;height:320px;object-fit:contain;background:#e2e8f0}table{width:100%;border-collapse:collapse}td,th{text-align:right;padding:6px;border-bottom:1px solid #dae4ef}td:first-child,th:first-child{text-align:left}tr:nth-child(2){background:#e5f0fb;font-weight:600}p,details{color:#56718a}"
    path = DATA / "gallery/index.html"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        f'<!doctype html><html><meta charset="utf-8"><title>500-image Decisions comparison</title><style>{style}</style><h1>Photo scoring comparison · 500 images</h1><p>Technical / aesthetic scores on 1–5. Old model observations are reused; only Perplexity and Gemini Decision 1 are new calls. Reference = assistant assessment. Ensemble = 37.5% Luna + 37.5% DeepSeek + 25% Liquid. Display rounded; calculations retain precision.</p><main>{"".join(cards)}</main></html>'
    )


def main():
    cases = json.loads((DATA / "subset.json").read_text())
    for case in cases:
        for model in ("perplexity", "gemini"):
            path = DATA / "results" / model / "observations" / (case["case_id"] + ".json")
            row = (
                json.loads(path.read_text())
                if path.exists()
                else {
                    "case_id": case["case_id"],
                    "status": "pending",
                    "technical_score": None,
                    "aesthetic_score": None,
                    "latency_seconds": None,
                    "price_usd": None,
                }
            )
            if path.exists():
                assert row["view_sha256"] == case["view_sha256"] and row["plan_sha256"] == sha(
                    DATA / "plan.json"
                )
                row["raw_response_sha256"] = sha(Path(row["raw_response_path"]))
            case["assessments"][model] = row
    rows = [
        {"selection_arm": c["selection_arm"], "content_cluster": c["content_cluster"], **r}
        for c in cases
        for r in c["assessments"].values()
    ]
    export = DATA / "exports"
    export.mkdir(exist_ok=True)
    (export / "labelled-images.jsonl").write_text(
        "".join(json.dumps(c, allow_nan=False) + "\n" for c in cases)
    )
    (export / "scores-long.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    fields = [
        "case_id",
        "model",
        "technical_score",
        "aesthetic_score",
        "latency_seconds",
        "price_usd",
        "status",
        "selection_arm",
        "content_cluster",
    ]
    with (export / "scores.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "selection": json.loads((DATA / "selection-summary.json").read_text()),
        "all": {m: metrics(cases, m) for m in NAMES},
        "costs": {m: costs(cases, m) for m in NAMES},
    }
    report["arms"] = {
        arm: {m: metrics([c for c in cases if c["selection_arm"] == arm], m) for m in NAMES}
        for arm in ("high_disagreement", "diversity_sample")
    }
    common = [
        c
        for c in cases
        if all(
            all(numeric(c["assessments"][m].get(d + "_score")) for d in ("technical", "aesthetic"))
            for m in NAMES
        )
    ]
    report["common_images"] = len(common)
    report["common"] = {m: metrics(common, m) for m in NAMES}
    write(DATA / "report.json", report)
    sections = [
        "# 500-image Decisions model comparison",
        "Frozen before new inference: 500 family-distinct images spanning all 64 EmbeddingGemma 2 clusters, comprising 384 Megalith, 97 VizWiz and 19 controls. Selection uses the first 1,000 reference-labelled examples: 375 favour high disagreement across the four existing API graders; 125 are sampled from the remaining images within each cluster. Reference values and new model outcomes were not used for selection. The final 500 confirmation references remain untouched.",
        "This is a disagreement-enriched diagnostic sample, not an unbiased population accuracy benchmark. MAE measures agreement with assistant references, not human ground truth. Both new models receive the frozen Jev questions unchanged, including independent technical/aesthetic outcomes and the 0.50 photographic-exclusion floor. No prompt tuning or ensemble weight change was made.",
    ]
    headers = [
        "Assessor",
        "Pairs",
        "Mean tech",
        "Mean aest",
        "Tech MAE",
        "Aest MAE",
        "Median seconds",
        "Recorded cost*",
    ]
    sections.append(
        table(
            headers,
            [
                [
                    name,
                    report["all"][m]["numeric_pairs"],
                    number(report["all"][m]["technical"]["mean"]),
                    number(report["all"][m]["aesthetic"]["mean"]),
                    number(report["all"][m]["technical"]["mae"], 3),
                    number(report["all"][m]["aesthetic"]["mae"], 3),
                    number(report["all"][m]["latency_median_seconds"]),
                    "—"
                    if report["costs"][m]["known_usd"] is None
                    else "$" + number(report["costs"][m]["known_usd"], 6),
                ]
                for m, name in NAMES.items()
            ],
        )
    )
    sections.append(
        "*Recorded costs include known retries. Only the two new models incur new spend. Historical observations retain their old request latency and cost; these are not simultaneous hardware/throughput comparisons. Lite and Reference rows are $0.00 by the accepted accounting convention. The ensemble has no independent request or charge."
    )
    sections.append(
        f"## Common coverage\n\nAll nine rows share {len(common)} images with both scores. Missing outcomes remain null; no model substitution or score-driven retry is used."
    )
    sections.append(
        table(
            ["Assessor", "T MAE", "T bias", "T rho", "A MAE", "A bias", "A rho"],
            [
                [
                    name,
                    *[
                        number(report["common"][m][d][v], 3)
                        for d in ("technical", "aesthetic")
                        for v in ("mae", "bias", "rho")
                    ],
                ]
                for m, name in NAMES.items()
            ],
        )
    )
    for arm in report["arms"]:
        sections.append("## " + arm.replace("_", " ").title())
        sections.append(
            table(
                ["Assessor", "T MAE", "A MAE"],
                [
                    [name, *[number(report["arms"][arm][m][d]["mae"], 3) for d in ("technical", "aesthetic")]]
                    for m, name in NAMES.items()
                ],
            )
        )
    sections.append("## Photography eligibility")
    sections.append(
        table(
            ["Assessor", "False floors / eligible", "Missed exclusions / excluded"],
            [
                [
                    name,
                    str(report["all"][m]["scope"].get("false_floor", 0))
                    + "/"
                    + str(report["all"][m]["scope"].get("eligible_n", 0)),
                    str(report["all"][m]["scope"].get("missed_exclusion", 0))
                    + "/"
                    + str(report["all"][m]["scope"].get("excluded_n", 0)),
                ]
                for m, name in NAMES.items()
                if m not in ("reference", "photo_ensemble", "qrealign_lite")
            ],
        )
    )
    sections.append(
        "[Per-image tables and exact images](gallery/index.html) · [Compact score CSV](exports/scores.csv) · [Image-level dataset](exports/labelled-images.jsonl). Raw responses and every request attempt remain under `results/`. Image statistics remain separate metadata and were not sent to the models."
    )
    sections.append("## Total costs per model")
    sections.append(
        table(
            ["Assessor", "Known USD", "Unpriced attempts", "New requests", "Basis"],
            [
                [
                    name,
                    number(report["costs"][m]["known_usd"], 6),
                    report["costs"][m]["unpriced_attempts"],
                    report["costs"][m]["new_requests"],
                    report["costs"][m]["basis"],
                ]
                for m, name in NAMES.items()
            ],
        )
    )
    (DATA / "report.md").write_text("\n\n".join(sections) + "\n")
    gallery(cases)
    print(
        json.dumps(
            {
                "common_images": len(common),
                "counts": {m: report["all"][m]["numeric_pairs"] for m in NAMES},
                "report": str(DATA / "report.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
