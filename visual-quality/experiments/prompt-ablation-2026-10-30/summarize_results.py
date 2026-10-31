"""Readable tables and complete regression inventory from verified frozen results."""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ("technical_quality", "aesthetics")
ROUTES = ("deepseek", "luna", "gemini", "haiku")
ARMS = ("baseline", "A", "B", "C", "D", "combined")


def number(x):
    return "—" if x is None else f"{x:.3f}"


def main():
    result = json.loads((HERE / "results.json").read_text())
    lines = [
        "# Complete prompt-ablation tables",
        "",
        "Scores use the 1–5 scale. MAE measures distance from frozen assistant assessments, not human accuracy. Negative deltas mean lower error. Individual additions are exploratory; the combined arm uses intervals adjusted for eight primary comparisons.",
        "",
        "## Scores and coverage",
        "",
        "| Model | Arm | Technical MAE | Aesthetic MAE | Valid T/A | Bias T/A |",
        "|---|---|---:|---:|---|---|",
    ]
    for route in ROUTES:
        for arm in ARMS:
            t, a = [result["tasks"][task][route]["arms"][arm] for task in TASKS]
            lines.append(
                f"| {route} | {arm} | {number(t['available']['mae'])} | {number(a['available']['mae'])} | {t['coverage']['valid_scores']}/{a['coverage']['valid_scores']} | {number(t['available']['bias'])} / {number(a['available']['bias'])} |"
            )
    lines += [
        "",
        "## Every addition versus its own baseline",
        "",
        "| Addition | Model | Task | MAE delta | Interval | Observed effect | Statistical support | Cases worsened ≥0.5 | New critical contradictions |",
        "|---|---|---|---:|---|---|---|---:|---:|",
    ]
    guardrails = []
    for arm in ARMS[1:]:
        for route in ROUTES:
            for task in TASKS:
                d = result["tasks"][task][route]["arm_minus_baseline"][arm]
                if not d["n"]:
                    continue
                effect = d["mae_difference"]
                ci = effect["ci99_375" if arm == "combined" else "ci95"]
                g = d["guardrails"]
                lines.append(
                    f"| {arm} | {route} | {task} | {effect['estimate']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {d['observed_impact']} | {d['support_status']} | {g['paired_error_worsened_at_least_0_5']['count']} | {g['new_critical_contradictions']['count']} |"
                )
                for name, sub in g["subgroups"].items():
                    if not sub["n"]:
                        continue
                    de = sub["mae_difference"]
                    guardrails.append(
                        {
                            "arm": arm,
                            "route": route,
                            "task": task,
                            "subgroup": name,
                            "n": sub["n"],
                            "families": sub["families"],
                            "delta_mae": de["estimate"],
                            "ci95_low": de["ci95"][0],
                            "ci95_high": de["ci95"][1],
                            "one_sided95_upper": de["one_sided95_upper"],
                            "observed_regression_over_0_10": sub["observed_mae_regression_over_0_10"],
                            "noninferiority_status": sub["noninferiority_status"],
                        }
                    )
    lines += [
        "",
        "## Targeted versus other cases",
        "",
        "A targets controlled noise/JPEG variants; B targets controlled blur variants. C uses VizWiz as an imperfect proxy for difficult composition. These metadata groups do not establish the visible issue in every image. D and combined apply generally.",
        "",
        "| Addition | Model | Task | Target n | Target ΔMAE | Other n | Other ΔMAE | Other noninferiority |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for arm in ("A", "B", "C"):
        for route in ROUTES:
            for task in TASKS:
                g = result["tasks"][task][route]["arm_minus_baseline"][arm]["guardrails"]["subgroups"]
                t, o = g["targeted_heuristic"], g["untargeted_complement"]
                te = t.get("mae_difference", {}).get("estimate")
                oe = o.get("mae_difference", {}).get("estimate")
                lines.append(
                    f"| {arm} | {route} | {task} | {t['n']} | {number(te)} | {o['n']} | {number(oe)} | {o['noninferiority_status']} |"
                )
    lines += [
        "",
        "## Combined-arm regression flags",
        "",
        "These overlapping groups are exploratory. A flag means observed MAE worsened by more than 0.10. Sparse groups and wide intervals prevent a general claim of no harm.",
        "",
    ]
    for route in ROUTES:
        for task in TASKS:
            d = result["tasks"][task][route]["arm_minus_baseline"]["combined"]
            g = d["guardrails"]
            lines.append(
                f"- **{route}, {task}:** groups: {', '.join(g['subgroup_regression_flags']) or 'none'}; cases worsened ≥0.5: {', '.join(g['paired_error_worsened_at_least_0_5']['case_ids']) or 'none'}; invalid increase: {g['invalid_increase']}."
            )
    lines += [
        "",
        "## Diagnostic averages",
        "",
        "| Ensemble | Arm | Technical MAE | Aesthetic MAE |",
        "|---|---|---:|---:|",
    ]
    for name, value in result["ensembles"].items():
        for arm in ARMS:
            t, a = [value["tasks"][task]["arms"][arm]["metrics"] for task in TASKS]
            lines.append(f"| {name} | {arm} | {number(t['mae'])} | {number(a['mae'])} |")
    lines += [
        "",
        "These averages are diagnostic, with no weight fitting. Technical and aesthetic dimensions remain separate.",
        "",
        "## Controlled technical severity diagnostics",
        "",
        "| Model | Arm | Nonincreasing families / complete | Mean parent-to-severe gap |",
        "|---|---|---:|---:|",
    ]
    for route in ROUTES:
        for arm in ARMS:
            d = result["tasks"]["technical_quality"][route]["arms"][arm]["controlled_diagnostics"]
            lines.append(
                f"| {route} | {arm} | {d['nonincreasing_families']} / {d['complete_families']} | {number(d['mean_parent_to_severe_gap'])} |"
            )
    lines += [
        "",
        "Source parents are not necessarily pristine. Severity strength is a diagnostic ordering, not perceptual ground truth; aesthetics need not decline monotonically.",
        "",
    ]
    (HERE / "analysis-tables.md").write_text("\n".join(lines))
    if guardrails:
        with (HERE / "guardrail-details.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(guardrails[0]))
            writer.writeheader()
            writer.writerows(guardrails)
    print(json.dumps({"tables": str(HERE / "analysis-tables.md"), "subgroup_rows": len(guardrails)}))


if __name__ == "__main__":
    main()
