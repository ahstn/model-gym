"""Create the preselected 72-image gallery from exact originals and saved scores."""

from __future__ import annotations

import csv
import html
import importlib.util
import json
import shutil
from decimal import Decimal
from pathlib import Path

import continuation

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / HERE.name
OUTPUT = HERE / "collage"
SPEC = importlib.util.spec_from_file_location(
    "photo_grid_helpers", HERE.parent / "photo-scope-trial-2026-10-31/build_collage.py"
)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
TASKS = helpers.TASKS


def main():
    selection = helpers.read_json(HERE / "collage-selection.json")
    refs = helpers.index_unique(helpers.read_rows(HERE / "references.jsonl"), "reference")
    freeze = helpers.read_json(HERE / "references-freeze.json")
    if freeze["references_sha256"] != helpers.digest(HERE / "references.jsonl"):
        raise ValueError("References changed")
    if freeze["manifest_sha256"] != helpers.digest(HERE / "manifest.json") or freeze[
        "selection_sha256"
    ] != helpers.digest(HERE / "collage-selection.json"):
        raise ValueError("Reference corpus or selection changed")
    for filename, expected in freeze["inputs"].items():
        if helpers.digest(Path(filename)) != expected:
            raise ValueError("Reference source changed: " + filename)
    if set(refs) != set(selection["case_ids"]) or len(refs) != 72:
        raise ValueError("Expected exactly the preselected 72 references")
    pins = {
        str(path): helpers.digest(path)
        for path in (
            HERE / "collage-selection.json",
            HERE / "references.jsonl",
            HERE / "references-freeze.json",
            HERE / "build_collage.py",
            Path(helpers.__file__),
            OUTPUT / "collage.css",
            OUTPUT / "render.mjs",
        )
    }
    observed, runs = {}, {}
    for route in ("luna", "deepseek"):
        run, report, observations, route_pins = continuation.load_verified_route(route)
        if report["status"] != "complete" or report["attempted"] != 2000:
            raise ValueError("Both 2000-image runs must finish")
        effort = run["settings"].get("reasoning_effort", run["settings"].get("reasoning", {}).get("effort"))
        if effort != "high":
            raise ValueError("Both models must request high reasoning")
        runs[route] = run
        observed[route] = helpers.index_unique(observations, route)
        pins.update(route_pins)
        for name, expected in run["inputs"].items():
            if helpers.digest(Path(name)) != expected:
                raise ValueError("Frozen input changed: " + name)
    images = OUTPUT / "images"
    images.mkdir(exist_ok=True)
    selected, csv_rows, render_cases = [], [], []
    for case in sorted(
        selection["cases"], key=lambda c: (refs[c["case_id"]]["technical_quality"]["score"], c["case_id"])
    ):
        case_id = case["case_id"]
        reference = refs[case_id]
        source = Path(case["image_path"])
        if helpers.digest(source) != case["view_sha256"] or reference["view_sha256"] != case["view_sha256"]:
            raise ValueError("Image identity mismatch")
        target = images / f"{case_id}.png"
        if not target.exists():
            shutil.copyfile(source, target)
        if helpers.digest(target) != case["view_sha256"]:
            raise ValueError("Published original changed")
        assessments = [
            {
                "route": "reference",
                "label": "Ref",
                **{task: helpers.task_value(reference, task, True) for task in TASKS},
            }
        ]
        for route, label in (("luna", "Luna"), ("deepseek", "DeepSeek")):
            row = observed[route][case_id]
            run = runs[route]
            frozen_case = next(c for c in run["cases"] if c["case_id"] == case_id)
            provenance = row["provenance"]
            if (
                row["view_sha256"] != case["view_sha256"]
                or provenance["run_id"] != run["run_id"]
                or provenance["route"] != route
                or row["arm_id"] != "photo_scope"
                or provenance["arm_id"] != "photo_scope"
                or provenance["request_sha256"] != frozen_case["request_sha256"]
                or provenance["prompt_sha256"] != run["prompts"]["photo_scope"]["sha256"]
                or provenance["model_returned"] not in run["route_config"]["accepted_models"]
            ):
                raise ValueError("Model observation provenance mismatch")
            assessments.append(
                {"route": route, "label": label, **{task: helpers.task_value(row, task) for task in TASKS}}
            )
        item = {
            "case": case,
            "reference": reference,
            "assessments": assessments,
            "url": f"images/{case_id}.png",
        }
        selected.append(item)
        cells = []
        for assessment in assessments:
            for task in TASKS:
                rating = assessment[task]
                cells.append(
                    {
                        "model": assessment["route"],
                        "task": task,
                        "score": rating["score"],
                        "status": rating["status"],
                        "text": format(Decimal(rating["score"]), ".1f")
                        if rating["score"]
                        else rating["status"],
                    }
                )
                csv_rows.append(
                    {
                        "display_position": len(selected),
                        "case_id": case_id,
                        "assessor": assessment["label"],
                        "task": task,
                        "score": rating["score"],
                        "status": rating["status"],
                        "evidence": rating["evidence"],
                        "limitations": json.dumps(rating["limitations"]),
                        "validation_error": rating["validation_error"],
                        "reference_eligibility": reference["eligibility"],
                        "known_anchor": case["known_anchor"],
                        "image_path": item["url"],
                        "view_sha256": case["view_sha256"],
                    }
                )
        render_cases.append(
            {
                "case_id": case_id,
                "image_path": str(target),
                "view_sha256": case["view_sha256"],
                "width": case["width"],
                "height": case["height"],
                "eligibility": reference["eligibility"],
                "score_cells": cells,
            }
        )
    with (OUTPUT / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    (OUTPUT / "selection.json").write_text(
        json.dumps({"arm_id": "dual_high_2000", "input_files_sha256": pins, "cases": render_cases}, indent=2)
        + "\n"
    )
    for stem, subset, start, split in (
        ("originals-grid", selected, 1, False),
        ("originals-grid-page-1", selected[:36], 1, True),
        ("originals-grid-page-2", selected[36:], 37, True),
    ):
        (OUTPUT / f"{stem}.html").write_text(render_html(subset, start, split))
    print(json.dumps({"images": len(selected), "score_cells": len(csv_rows), "output": str(OUTPUT)}))


def render_html(items, start, split):
    cards = []
    for position, item in enumerate(items, start):
        case, ref = item["case"], item["reference"]
        case_id, url = case["case_id"], item["url"]
        badges = '<span class="badge new-image">Known poster check</span>' if case["known_anchor"] else ""
        if ref["eligibility"] != "eligible":
            label = (
                "Outside photo scope · Ref" if ref["eligibility"] == "excluded" else "Scope uncertain · Ref"
            )
            badges += f'<span class="badge excluded">{label}</span>'
        rows = []
        for assessment in item["assessments"]:
            cls = "reference" if assessment["route"] == "reference" else ""
            rows.append(
                f'<tr class="{cls}"><th scope="row">{assessment["label"]}</th>'
                + "".join(helpers.score_cell(assessment, task) for task in TASKS)
                + "</tr>"
            )
        cards.append(f'''<article class="card" id="{case_id}" data-case-id="{case_id}" data-eligibility="{ref["eligibility"]}">
<h2><span>{case_id}</span><span class="position">{position:02d} / 72</span></h2><div class="scope-meta">{badges}</div>
<a class="image-link" href="{url}"><img src="{url}" width="{case["width"]}" height="{case["height"]}" alt="Original scored image {case_id}" loading="eager" decoding="async"></a>
<table aria-label="Scores for {case_id}"><thead><tr><th>Assessor</th><th>Tech</th><th>Aest</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p class="original-link"><a href="{url}">Open original ↗</a></p></article>''')
    title = (
        f"Images {start}–{start + len(items) - 1} of 72."
        if split
        else "72 images. Two models. High reasoning."
    )
    body = ' class="split-page"' if split else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><link rel="stylesheet" href="collage.css"></head><body{body}><header>
<p class="eyebrow">2,000-image photography trial · 31 October 2026</p><h1>{title}</h1>
<p class="intro">Technical quality and aesthetics on a <strong>1–5 scale</strong>; higher is better. Clearly non-photographic content receives <strong>1.00 for both scores</strong>.<br>
The 72-image subset was selected before inference. Ref contains fresh assistant judgments made without student scores; it is not human ground truth.<br>
Both models scored the same 2,000 images using the unchanged photography prompt and structured JSON. This grid shows the 72 reviewed cases.</p>
<div class="model-key"><span><b>Ref</b> Independent assistant assessment</span><span><b>Luna</b> GPT-6 Luna · high · OpenRouter</span><span><b>DeepSeek</b> V4.1 Flash 1017 · high · RunInfra</span></div>
<div class="legend"><span>Tech = technical quality</span><span>Aest = aesthetics</span><span>Order: Ref technical score, then case ID</span><span>Full frame · no crop</span>
<a href="scores.csv">Scores + evidence CSV</a><a href="originals-grid.html">Full grid</a><a href="originals-grid-page-1.html">Page 1</a><a href="originals-grid-page-2.html">Page 2</a></div>
</header><main class="grid">{"".join(cards)}</main><footer>
71 fresh cases plus one known poster regression check. The subset covers natural scenes, real capture problems and controlled defects; it is not a population-weighted sample.
Images are the exact delivered PNGs, scaled without cropping. Hover over a score for evidence. Model disagreements and invalid ratings remain visible.
Scores have one decimal here and two in the CSV. The policy floor measures suitability for a photography collection, not the artistic merit of a design or painting.
</footer></body></html>"""


if __name__ == "__main__":
    main()
