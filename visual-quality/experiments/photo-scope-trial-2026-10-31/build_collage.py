"""Build the complete 72-image photography scoring grid from original views and recorded scores.

Run after the four image routes are complete, then run collage/render.mjs.
This builder does not generate, edit, crop, or transcode any source image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

EXPERIMENT = Path(__file__).resolve().parent
DATA = EXPERIMENT.parents[1] / "data" / EXPERIMENT.name
TASKS = ("technical_quality", "aesthetics")
MODELS = (("deepseek", "DeepSeek"), ("luna", "Luna"), ("gemini", "Gemini"), ("haiku", "Haiku"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index_unique(rows: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in result:
            raise ValueError(f"Duplicate {label}: {case_id}")
        result[case_id] = row
    return result


def task_value(row: dict, task: str, reference: bool = False) -> dict:
    rating = row[task]
    error = row.get("validation_errors", {}).get(task)
    status = rating.get("status", "ok" if rating.get("score") is not None else "unscorable")
    score = rating.get("score")
    if error or status == "invalid":
        if score is not None:
            raise ValueError("An invalid task must not supply a displayed score")
        status, exact = "invalid", ""
    elif status == "unscorable":
        if score is not None:
            raise ValueError("An unscorable task has a score")
        exact = ""
    elif status == "ok" and type(score) in (int, float):
        value = Decimal(str(score))
        if not value.is_finite() or not Decimal(1) <= value <= Decimal(5) or value % Decimal("0.1"):
            raise ValueError(f"Invalid displayed score: {score}")
        exact = format(value, ".2f")
        if not reference and row.get("wire_scores", {}).get(task) != exact:
            raise ValueError("Recorded numeric and wire scores disagree")
    else:
        raise ValueError(f"Unexpected rating status/score: {status}/{score}")
    return {
        "score": exact,
        "status": status,
        "evidence": rating.get("evidence", ""),
        "limitations": rating.get("limitations", []),
        "validation_error": error or "",
        "interval": rating.get("interval"),
    }


def load_inputs(experiment: Path, data: Path) -> tuple[list[dict], dict]:
    manifest = read_json(experiment / "manifest.json")
    case_ids = [case["case_id"] for case in manifest["cases"]]
    if case_ids != [f"P{index:03d}" for index in range(1, 73)]:
        raise ValueError("Expected the complete 72-image corpus, P001 through P072")
    cases = index_unique(manifest["cases"], "manifest case")
    references = index_unique(read_rows(experiment / "references.jsonl"), "reference")
    if set(references) != set(case_ids):
        raise ValueError("References must cover exactly the same 72 cases")
    freeze = read_json(experiment / "references-freeze.json")
    if freeze["references_sha256"] != digest(experiment / "references.jsonl"):
        raise ValueError("Reference file differs from its frozen hash")
    if freeze["manifest_sha256"] != digest(experiment / "manifest.json"):
        raise ValueError("Manifest differs from the reference freeze")
    routes = read_json(experiment / "routes.json")
    pins = {
        str(path.resolve()): digest(path)
        for path in (
            experiment / "manifest.json",
            experiment / "references.jsonl",
            experiment / "references-freeze.json",
            experiment / "routes.json",
            experiment / "prompt.json",
            experiment / "photo-scope.txt",
            experiment / "build_collage.py",
            experiment / "collage/collage.css",
            experiment / "collage/render.mjs",
        )
    }
    observations, runs = {}, {}
    for route, _ in MODELS:
        run_path = data / route / "images" / "run.json"
        report_path = data / route / "images" / "report.json"
        runs[route] = read_json(run_path)
        report = read_json(report_path)
        if report["status"] != "complete" or report["attempted"] != 72:
            raise ValueError(f"Route is not complete: {route}")
        if {case["case_id"] for case in runs[route]["cases"]} != set(case_ids):
            raise ValueError(f"Frozen route case set differs: {route}")
        for source in (run_path, report_path):
            pins[str(source.resolve())] = digest(source)
        path = data / route / "images" / "observations.jsonl"
        observations[route] = index_unique(
            [row for row in read_rows(path) if row.get("arm_id") == "photo_scope"], route
        )
        missing = sorted(set(case_ids) - observations[route].keys())
        if len(observations[route]) != 72:
            raise ValueError(f"Expected 72 observations: {route}")
        if missing:
            raise ValueError(f"Missing photography-scope {route} observations: {', '.join(missing)}")
        pins[str(path.resolve())] = digest(path)
    selected = []
    for case_id in case_ids:
        case, reference = cases[case_id], references[case_id]
        if reference.get("photo_domain_eligible") not in (True, False, None):
            raise ValueError(f"Invalid reference eligibility: {case_id}")
        if reference.get("eligibility") not in {"eligible", "excluded", "uncertain"}:
            raise ValueError(f"Missing explicit reference eligibility: {case_id}")
        if (reference["eligibility"] == "excluded") != (reference["photo_domain_eligible"] is False):
            raise ValueError(f"Contradictory reference eligibility: {case_id}")
        path = Path(case["image_path"])
        if digest(path) != case["view_sha256"] or reference["view_sha256"] != case["view_sha256"]:
            raise ValueError(f"Original/reference image hash mismatch: {case_id}")
        assessments = [
            {
                "route": "reference",
                "label": "Ref",
                "model_id": "assistant reference with photography eligibility policy",
                "reasoning_effort": "",
                "prompt_sha256": "",
                "request_sha256": "",
                **{task: task_value(reference, task, reference=True) for task in TASKS},
            }
        ]
        for route, label in MODELS:
            row = observations[route][case_id]
            provenance = row["provenance"]
            if row["view_sha256"] != case["view_sha256"]:
                raise ValueError(f"Observed image hash mismatch: {route}/{case_id}")
            if provenance["route"] != route or provenance["arm_id"] != "photo_scope":
                raise ValueError(f"Observation provenance mismatch: {route}/{case_id}")
            if provenance["model_returned"] not in routes[route]["accepted_models"]:
                raise ValueError(f"Unexpected model: {route}/{case_id}")
            run = runs[route]
            frozen_case = next(case for case in run["cases"] if case["case_id"] == case_id)
            if provenance["run_id"] != run["run_id"] or provenance["request_sha256"] != frozen_case["request_sha256"]:
                raise ValueError(f"Recorded request identity mismatch: {route}/{case_id}")
            if provenance["prompt_sha256"] != run["prompts"]["photo_scope"]["sha256"]:
                raise ValueError(f"Recorded prompt identity mismatch: {route}/{case_id}")
            settings = routes[route]["settings"]
            assessments.append(
                {
                    "route": route,
                    "label": label,
                    "model_id": provenance["model_returned"],
                    "reasoning_effort": settings.get("reasoning", {}).get(
                        "effort", settings.get("reasoning_effort", "")
                    ),
                    "prompt_sha256": provenance["prompt_sha256"],
                    "request_sha256": provenance["request_sha256"],
                    **{task: task_value(row, task) for task in TASKS},
                }
            )
        if reference["photo_domain_eligible"] is False and any(
            assessments[0][task]["score"] != "1.00" for task in TASKS
        ):
            raise ValueError(f"Excluded reference must use the explicit 1.00 floor: {case_id}")
        selected.append({"case": case, "assessments": assessments,
                         "eligibility": reference["eligibility"],
                         "photo_domain_eligible": reference["photo_domain_eligible"]})
    selected.sort(
        key=lambda item: (
            Decimal(item["assessments"][0]["technical_quality"]["score"] or "Infinity"),
            item["case"]["case_id"],
        )
    )
    return selected, {
        "schema_version": "photography-collage-v1",
        "arm_id": "photo_scope",
        "manifest_id": manifest["manifest_id"],
        "input_files_sha256": pins,
        "selection_rule": "All 72 corpus images; no score-based omission",
        "human_gold": False,
        "display_order": "Frozen assistant technical score ascending, then case ID; unscorable last",
        "image_policy": "Original delivered PNG bytes; browser scales full frame with object-fit contain",
        "cases": [
            {
                "display_position": index,
                "case_id": item["case"]["case_id"],
                "family_id": item["case"]["family_id"],
                "image_path": item["case"]["image_path"],
                "view_sha256": item["case"]["view_sha256"],
                "width": item["case"]["width"],
                "height": item["case"]["height"],
                "eligibility": item["eligibility"],
                "photo_domain_eligible": item["photo_domain_eligible"],
                "new_image": int(item["case"]["case_id"][1:]) >= 51,
                "score_cells": [{
                    "model": assessment["route"], "task": task,
                    "score": assessment[task]["score"], "status": assessment[task]["status"],
                    "text": format(Decimal(assessment[task]["score"]), ".1f")
                    if assessment[task]["score"] else assessment[task]["status"],
                } for assessment in item["assessments"] for task in TASKS],
            }
            for index, item in enumerate(selected, 1)
        ],
    }


def write_csv(path: Path, selected: list[dict]) -> None:
    fields = ["display_position", "case_id", "arm_id", "assessor", "model_id", "reasoning_effort",
              "reference_eligibility", "new_image"]
    for task in TASKS:
        fields += [
            f"{task}_{name}"
            for name in ("score", "status", "evidence", "limitations", "validation_error", "interval")
        ]
    fields += ["view_sha256", "image_path", "prompt_sha256", "request_sha256"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, item in enumerate(selected, 1):
            case = item["case"]
            for assessment in item["assessments"]:
                row = {
                    "display_position": index,
                    "reference_eligibility": item["eligibility"],
                    "new_image": int(case["case_id"][1:]) >= 51,
                    "case_id": case["case_id"],
                    "arm_id": "reference" if assessment["route"] == "reference" else "photo_scope",
                    "assessor": assessment["label"],
                    "model_id": assessment["model_id"],
                    "reasoning_effort": assessment["reasoning_effort"],
                    "view_sha256": case["view_sha256"],
                    "image_path": case["image_path"],
                    "prompt_sha256": assessment["prompt_sha256"],
                    "request_sha256": assessment["request_sha256"],
                }
                for task in TASKS:
                    for key, value in assessment[task].items():
                        row[f"{task}_{key}"] = json.dumps(value) if isinstance(value, (list, dict)) else value
                writer.writerow(row)


def score_cell(assessment: dict, task: str) -> str:
    rating = assessment[task]
    title = rating["evidence"] or rating["validation_error"] or rating["status"]
    value = format(Decimal(rating["score"]), ".1f") if rating["score"] else rating["status"]
    unavailable = ' class="unavailable"' if not rating["score"] else ""
    return (
        f'<td data-model="{assessment["route"]}" data-task="{task}" '
        f'data-score="{rating["score"]}" data-status="{rating["status"]}" '
        f'title="{html.escape(title, quote=True)}"{unavailable}>{value}</td>'
    )


def render_html(selected: list[dict], output: Path, *, first_position: int = 1,
                page_number: int | None = None) -> str:
    cards = []
    for index, item in enumerate(selected, first_position):
        case = item["case"]
        case_id = case["case_id"]
        url = quote(os.path.relpath(case["image_path"], output), safe="/")
        rows = []
        for assessment in item["assessments"]:
            row_class = "reference" if assessment["route"] == "reference" else ""
            rows.append(
                f'<tr class="{row_class}"><th scope="row">{assessment["label"]}</th>'
                + "".join(score_cell(assessment, task) for task in TASKS)
                + "</tr>"
            )
        badges = []
        if int(case_id[1:]) >= 51:
            badges.append('<span class="badge new-image">New image</span>')
        if item["photo_domain_eligible"] is False:
            badges.append('<span class="badge excluded">Outside photo scope · Ref</span>')
        elif item["eligibility"] == "uncertain":
            badges.append('<span class="badge uncertain">Scope uncertain · Ref</span>')
        cards.append(f'''<article class="card" id="{case_id}" data-case-id="{case_id}" data-eligibility="{item["eligibility"]}">
  <h2><span>{case_id}</span><span class="position">{index:02d} / 72</span></h2>
  <div class="scope-meta">{"".join(badges)}</div>
  <a class="image-link" href="{url}" aria-label="Open original scored PNG for {case_id}">
    <img src="{url}" width="{case["width"]}" height="{case["height"]}" alt="Original scored image {case_id}" loading="eager" decoding="async">
  </a>
  <table aria-label="Scores for {case_id}"><thead><tr><th scope="col">Assessor</th><th scope="col">Tech</th><th scope="col">Aest</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  <p class="original-link"><a href="{url}">Open original ↗</a></p>
</article>''')
    title = "72 images. Two scores. Four models."
    body_class = ""
    if page_number is not None:
        title = f"Images {first_position}–{first_position + len(selected) - 1} of 72."
        body_class = ' class="split-page"'
    return (
        f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Photography scoring trial</title><link rel="stylesheet" href="collage.css"></head>
<body{body_class}><header>
  <p class="eyebrow">Photography scoring trial · 31 October 2026</p>
  <h1>{title}</h1>
  <p class="intro">Technical quality and aesthetics on a <strong>1–5 scale</strong>; higher is better. <strong>Photography-only policy: excluded non-photographs receive 1.00 for both scores.</strong><br>
  Ref is an assistant assessment, not human gold. The original 50 quality references are reused with the new eligibility rule; the additional 22 images were reviewed separately.<br>
  All four models receive the same photography-scope prompt and exact image bytes. Their actual validated scores are shown, including disagreements with the reference.</p>
  <div class="model-key"><span><b>DeepSeek</b> V4.1 Flash · no reasoning · RunInfra</span><span><b>Luna</b> GPT-6 Luna · high · OpenRouter</span><span><b>Gemini</b> 3.8 Flash · medium · OpenRouter</span><span><b>Haiku</b> Claude 5.5 · low · OpenRouter</span></div>
  <div class="legend"><span>Tech = technical quality</span><span>Aest = aesthetics</span><span>Order: Ref technical score, then case ID</span><span>Full frame · no crop</span><a href="scores.csv" download>Exact scores + evidence CSV</a><a href="selection.json">All 72 images + hashes</a><a href="originals-grid.html">Full grid</a><a href="originals-grid-page-1.html">Page 1</a><a href="originals-grid-page-2.html">Page 2</a></div>
</header><main class="grid" aria-label="{len(selected)} original images and recorded scores">
'''
        + "\n".join(cards)
        + """
</main><footer>
  All 72 corpus images are included: the original 50 plus 22 newly selected images. An “Outside photo scope” badge records the reference eligibility decision; model disagreements remain visible.
  Images are the exact delivered PNGs, scaled to fit without cropping. Select an image for full resolution; hover over a score for recorded evidence. Missing valid ratings are marked invalid or unscorable.
  Displayed scores have one decimal; CSV retains exact two-decimal scores, per-task status, evidence and validation errors. The 1.00 floor expresses suitability for a photography collection, not the artistic merit of a design or painting.
</footer></body></html>
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    experiment, data = args.experiment.resolve(), args.data.resolve()
    selected, selection = load_inputs(experiment, data)
    output = experiment / "collage"
    if not (output / "collage.css").is_file() or not (output / "render.mjs").is_file():
        raise ValueError("Collage stylesheet and renderer must be present")
    write_csv(output / "scores.csv", selected)
    (output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    (output / "originals-grid.html").write_text(render_html(selected, output))
    for page_number, start in enumerate((0, 36), 1):
        (output / f"originals-grid-page-{page_number}.html").write_text(
            render_html(selected[start:start + 36], output,
                        first_position=start + 1, page_number=page_number)
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "images": len(selected),
                "assessments": len(selected) * 5,
                "arm_id": "photo_scope",
                "next": f"node {output / 'render.mjs'}",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
