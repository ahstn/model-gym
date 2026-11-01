"""Descriptive error slices at the existing frozen QC sharpness threshold."""

from __future__ import annotations

import json
from pathlib import Path

import analysis_metrics as metrics
import api_contract
from analyze import read_lines
from report import NAMES, number, table

HERE = Path(__file__).resolve().parent


def summarize(rows: list[dict], percentiles: dict) -> list[dict]:
    references = {row["case_id"]: row for row in rows if row["model"] == "reference"}
    lookup = {(row["case_id"], row["model"], row["arm"]): row for row in rows}
    groups = sorted({(row["model"], row["arm"]) for row in rows if row["model"] != "reference"})
    output = []
    for panel in metrics.PANELS:
        for scope in ("all_reference_cases", "reference_eligible_only"):
            for band in ("sharpness_percentile_at_most_25", "sharpness_percentile_above_25"):
                selected = [
                    identity for identity, ref in references.items()
                    if metrics.in_panel(ref, panel)
                    and (scope == "all_reference_cases" or ref["reference_eligibility"] == "eligible")
                    and (percentiles[identity] <= 25) == (band == "sharpness_percentile_at_most_25")
                ]
                for model, arm in groups:
                    item = {"panel": panel, "scope": scope, "band": band, "model": model,
                            "arm": arm, "planned_cases": len(selected), "case_ids": selected}
                    for task in metrics.TASKS:
                        pairs, deltas = [], []
                        for identity in selected:
                            ref, pred = references[identity], lookup.get((identity, model, arm), {})
                            if not (metrics.numeric(ref, task) and metrics.numeric(pred, task)):
                                continue
                            pairs.append((pred[task], ref[task]))
                            base = lookup.get((identity, model, "baseline"), {})
                            if arm != "baseline" and metrics.numeric(base, task):
                                deltas.append(abs(pred[task] - ref[task]) - abs(base[task] - ref[task]))
                        item[task] = metrics.agreement(pairs)
                        item[task]["paired_treatment_minus_baseline_mae"] = (
                            sum(deltas) / len(deltas) if deltas else None
                        )
                        item[task]["paired_treatment_cases"] = len(deltas)
                    output.append(item)
    return output


def main() -> None:
    score_path, stats_path = HERE / "scores-long.jsonl", HERE / "stats-percentiles.jsonl"
    rows, stats = read_lines(score_path), read_lines(stats_path)
    by_id = {row["case_id"]: row for row in stats}
    if len(by_id) != 4000 or len(stats) != 4000:
        raise ValueError("Expected all 4000 unique frozen percentile records")
    for row in rows:
        if row["view_sha256"] != by_id[row["case_id"]]["view_sha256"]:
            raise ValueError("Score/stat image identity mismatch")
    percentiles = {identity: api_contract.number(row["sharpness_peak_percentile"], 0, 100)
                   for identity, row in by_id.items()}
    results = summarize(rows, percentiles)
    record = {
        "status": "exploratory_descriptive_subgroup_analysis",
        "timing": "Added after inspecting completed Luna results; not an additional preregistered endpoint",
        "threshold": "Existing QC cutoff: sharpness_peak percentile <=25 against frozen background reference",
        "interpretation": "Low measured sharpness is not proof of unintended blur. No scores, primary endpoints or thresholds changed; no subgroup significance claim.",
        "inputs": {str(path): api_contract.file_sha(path) for path in
                   (score_path, stats_path, HERE / "qc-reference.json", Path(__file__), Path(metrics.__file__))},
        "rows": results,
    }
    (HERE / "sharpness-slices.json").write_text(json.dumps(record, indent=2) + "\n")
    chosen = {(row["model"], row["arm"], row["band"]): row for row in results
              if row["panel"] == "representative" and row["scope"] == "reference_eligible_only"}
    bands = ("sharpness_percentile_at_most_25", "sharpness_percentile_above_25")
    baseline, changes = [], []
    for model in (*metrics.BASE_MODELS, "luna_deepseek_mean"):
        cells = [NAMES[model]]
        for band in bands:
            task = chosen[(model, "baseline", band)]["technical_score"]
            cells.extend([task["n"], number(task["mae"], 3), number(task["bias"], 3)])
        baseline.append(cells)
    for model in ("luna", "deepseek", "luna_deepseek_mean"):
        for arm in ("own_stats", "swapped_stats"):
            cells = [NAMES[model], arm]
            for band in bands:
                task = chosen[(model, arm, band)]["technical_score"]
                cells.extend([task["paired_treatment_cases"], number(task["paired_treatment_minus_baseline_mae"], 3)])
            changes.append(cells)
    text = "\n\n".join([
        "## Exploratory sharpness slices",
        "This descriptive follow-up was added after inspecting Luna's completed results. It uses the existing frozen QC cutoff, without tuning a new threshold. The tables restrict the representative panel to reference-eligible photographs; all-case, challenge, development and aesthetic results remain in sharpness-slices.json. Low measured detail can reflect shallow focus, silhouettes or smooth subjects, so the lower band is not a verified blur label.",
        table(("Baseline", "Lower-band n", "Tech MAE", "Tech bias", "Other n", "Tech MAE", "Tech bias"), baseline),
        "Paired technical MAE changes below are treatment minus baseline; negative values are improvements. These subgroup point estimates have no multiplicity-adjusted inference and do not replace the primary 300-image comparison.",
        table(("Model", "Arm", "Lower-band paired n", "ΔMAE", "Other paired n", "ΔMAE"), changes),
    ])
    (HERE / "sharpness-report.md").write_text(text + "\n")
    print(json.dumps({"descriptive_groups": len(results), "output": str(HERE / "sharpness-slices.json")}))


if __name__ == "__main__":
    main()
