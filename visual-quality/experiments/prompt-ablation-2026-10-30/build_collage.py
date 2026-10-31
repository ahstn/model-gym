"""Build the fixed 36-image inspection grid from original views and recorded scores.

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
    selection = read_json(experiment / "collage-selection.json")
    case_ids = selection["case_ids"]
    if selection.get("arm_id") != "combined" or not selection.get("locked_before_outputs"):
        raise ValueError("Collage requires the frozen combined-arm selection")
    if len(case_ids) != 36 or len(set(case_ids)) != 36:
        raise ValueError("Selection must contain exactly 36 unique cases")
    manifest = read_json(experiment / "manifest.json")
    cases = index_unique(manifest["cases"], "manifest case")
    references = index_unique(read_rows(experiment / "references.jsonl"), "reference")
    freeze = read_json(experiment / "references-freeze.json")
    if freeze["references_sha256"] != digest(experiment / "references.jsonl"):
        raise ValueError("Reference file differs from its frozen hash")
    if freeze["manifest_sha256"] != digest(experiment / "manifest.json"):
        raise ValueError("Manifest differs from the reference freeze")
    routes = read_json(experiment / "routes.json")
    pins = {
        str(path.resolve()): digest(path)
        for path in (
            experiment / "collage-selection.json",
            experiment / "manifest.json",
            experiment / "references.jsonl",
            experiment / "references-freeze.json",
            experiment / "routes.json",
            experiment / "arms.json",
        )
    }
    observations = {}
    for route, _ in MODELS:
        path = data / route / "images" / "observations.jsonl"
        observations[route] = index_unique(
            [row for row in read_rows(path) if row.get("arm_id") == "combined"], route
        )
        missing = sorted(set(case_ids) - observations[route].keys())
        if missing:
            raise ValueError(f"Missing combined-arm {route} observations: {', '.join(missing)}")
        pins[str(path.resolve())] = digest(path)
    selected = []
    for case_id in case_ids:
        case, reference = cases[case_id], references[case_id]
        path = Path(case["image_path"])
        if digest(path) != case["view_sha256"] or reference["view_sha256"] != case["view_sha256"]:
            raise ValueError(f"Original/reference image hash mismatch: {case_id}")
        assessments = [
            {
                "route": "reference",
                "label": "Ref",
                "model_id": "blinded assistant assessment",
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
            if provenance["route"] != route or provenance["arm_id"] != "combined":
                raise ValueError(f"Observation provenance mismatch: {route}/{case_id}")
            if provenance["model_returned"] not in routes[route]["accepted_models"]:
                raise ValueError(f"Unexpected model: {route}/{case_id}")
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
        selected.append({"case": case, "assessments": assessments})
    selected.sort(
        key=lambda item: (
            Decimal(item["assessments"][0]["technical_quality"]["score"] or "Infinity"),
            item["case"]["case_id"],
        )
    )
    return selected, {
        "schema_version": "original-collage-selection-v1",
        "arm_id": "combined",
        "manifest_id": manifest["manifest_id"],
        "input_files_sha256": pins,
        "selection_rule": selection["selection_rule"],
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
            }
            for index, item in enumerate(selected, 1)
        ],
    }


def write_csv(path: Path, selected: list[dict]) -> None:
    fields = ["display_position", "case_id", "arm_id", "assessor", "model_id", "reasoning_effort"]
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
                    "case_id": case["case_id"],
                    "arm_id": "reference" if assessment["route"] == "reference" else "combined",
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


def render_html(selected: list[dict], output: Path) -> str:
    cards = []
    for index, item in enumerate(selected, 1):
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
        cards.append(f'''<article class="card" id="{case_id}" data-case-id="{case_id}">
  <h2><span>{case_id}</span><span class="position">{index:02d} / 36</span></h2>
  <a class="image-link" href="{url}" aria-label="Open original scored PNG for {case_id}">
    <img src="{url}" width="{case["width"]}" height="{case["height"]}" alt="Original scored image {case_id}" loading="eager" decoding="async">
  </a>
  <table aria-label="Scores for {case_id}"><thead><tr><th scope="col">Assessor</th><th scope="col">Tech</th><th scope="col">Aest</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  <p class="original-link"><a href="{url}">Open original ↗</a></p>
</article>''')
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>36 original images · Four-model prompt trial</title><link rel="stylesheet" href="collage.css"></head>
<body><header>
  <p class="eyebrow">Prompt trial · 30 October 2026 · Fixed combined prompt</p>
  <h1>36 images. Two scores. Four models.</h1>
  <p class="intro">Technical quality and aesthetics on a <strong>1–5 scale</strong>; higher is better. <strong>Ref is a blind assistant assessment, not human gold.</strong><br>
  All four models use the same combined prompt additions and original image bytes. This is an inspection sample from the 50-image trial.</p>
  <div class="model-key"><span><b>DeepSeek</b> V4.1 Flash · no reasoning · RunInfra</span><span><b>Luna</b> GPT-6 Luna · high · OpenRouter</span><span><b>Gemini</b> 3.8 Flash · medium · OpenRouter</span><span><b>Haiku</b> Claude 5.5 · low · OpenRouter</span></div>
  <div class="legend"><span>Tech = technical quality</span><span>Aest = aesthetics</span><span>Order: Ref technical score, then case ID</span><span>Full frame · no crop</span><a href="scores.csv" download>Exact scores + evidence CSV</a><a href="selection.json">Selection + hashes</a></div>
</header><main class="grid" aria-label="36 original images and recorded scores">
"""
        + "\n".join(cards)
        + """
</main><footer>
  The selection was fixed before model outputs: all 32 family representatives plus four severe controlled variants. The combined prompt is the preselected display arm, not a claim that it won the experiment.
  Images are original delivered PNGs, scaled to fit without cropping. Select an image for full resolution; hover over a score for its recorded evidence. A missing valid rating is marked invalid or unscorable.
  Displayed scores have one decimal; CSV retains the exact two-decimal model scores, per-task status and evidence. Ref scores were frozen before model image requests.
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
    print(
        json.dumps(
            {
                "output": str(output),
                "images": len(selected),
                "assessments": len(selected) * 5,
                "arm_id": "combined",
                "next": f"node {output / 'render.mjs'}",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
