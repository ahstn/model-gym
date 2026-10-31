"""Build a 72-image comparison of DeepSeek none/high and the three existing routes.

Reuse the original photography-grid validation/formatting helpers without changing
its files. High-reasoning scores must come from completed, recorded model outputs.
"""
from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "photo-scope-trial-2026-10-31"
DATA = HERE.parents[1] / "data"
_spec = importlib.util.spec_from_file_location("original_photo_grid", PRIOR / "build_collage.py")
prior = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(prior)
TASKS = prior.TASKS


def reasoning(settings: dict) -> str:
    return settings.get("reasoning", {}).get("effort", settings.get("reasoning_effort", ""))


def load_inputs() -> tuple[list[dict], dict]:
    selected, selection = prior.load_inputs(PRIOR, DATA / PRIOR.name)
    directory = DATA / HERE.name / "deepseek/images"
    run_path, report_path = directory / "run.json", directory / "report.json"
    observations_path = directory / "observations.jsonl"
    run, report = prior.read_json(run_path), prior.read_json(report_path)
    if report["status"] != "complete" or report["attempted"] != 72:
        raise ValueError("DeepSeek high must finish all 72 image calls before building")
    if run["route"] != "deepseek" or reasoning(run["settings"]) != "high":
        raise ValueError("Expected the DeepSeek high-reasoning route")
    config = run["route_config"]
    if reasoning(config["settings"]) != "high":
        raise ValueError("High reasoning differs between run and route settings")
    cases = prior.index_unique(run["cases"], "high route case")
    case_ids = {item["case"]["case_id"] for item in selected}
    if set(cases) != case_ids:
        raise ValueError("High route must contain exactly the same 72 images")
    arm_ids = {case["arm_id"] for case in cases.values()}
    if arm_ids != {"photo_scope"}:
        raise ValueError("Expected the same photo_scope arm")
    observations = prior.index_unique(prior.read_rows(observations_path), "high observation")
    if set(observations) != case_ids:
        raise ValueError("High observations must cover all 72 cases exactly")
    pins = selection["input_files_sha256"]
    for path in (Path(__file__).resolve(), HERE / "collage/collage.css", HERE / "collage/render.mjs",
                 run_path, report_path, observations_path):
        pins[str(path)] = prior.digest(path)
    for path, expected in run["inputs"].items():
        if prior.digest(Path(path)) != expected:
            raise ValueError(f"High route frozen input changed: {path}")
        pins[path] = expected
    for item in selected:
        case_id = item["case"]["case_id"]
        case, row = cases[case_id], observations[case_id]
        provenance = row["provenance"]
        original_hash = item["case"]["view_sha256"]
        if case["view_sha256"] != original_hash or row["view_sha256"] != original_hash:
            raise ValueError(f"High route image differs: {case_id}")
        if row["arm_id"] != "photo_scope" or provenance["arm_id"] != "photo_scope":
            raise ValueError(f"High route arm differs: {case_id}")
        if provenance["route"] != "deepseek" or provenance["run_id"] != run["run_id"]:
            raise ValueError(f"High route provenance differs: {case_id}")
        if provenance["request_sha256"] != case["request_sha256"]:
            raise ValueError(f"High route request differs: {case_id}")
        if provenance["model_returned"] not in config["accepted_models"]:
            raise ValueError(f"Unexpected high route model: {case_id}")
        if provenance["prompt_sha256"] != run["prompts"]["photo_scope"]["sha256"]:
            raise ValueError(f"High prompt identity differs: {case_id}")
        original_deepseek = item["assessments"][1]
        if original_deepseek["route"] != "deepseek" or original_deepseek["reasoning_effort"] != "none":
            raise ValueError("Expected original DeepSeek no-reasoning assessment")
        if provenance["prompt_sha256"] != original_deepseek["prompt_sha256"]:
            raise ValueError("None/high comparison must use identical scoring prompts")
        original_deepseek.update(route="deepseek_none", label="DeepSeek none")
        item["assessments"].insert(2, {
            "route": "deepseek_high", "label": "DeepSeek high",
            "model_id": provenance["model_returned"], "reasoning_effort": "high",
            "prompt_sha256": provenance["prompt_sha256"],
            "request_sha256": provenance["request_sha256"],
            **{task: prior.task_value(row, task) for task in TASKS},
        })
    selection.update(
        schema_version="deepseek-reasoning-comparison-grid-v1",
        arm_id="deepseek_reasoning_comparison",
        source_arm_id="photo_scope",
        assessment_order=["reference", "deepseek_none", "deepseek_high", "luna", "gemini", "haiku"],
        comparison_policy="Same 72 image bytes, scoring prompt and references; DeepSeek reasoning changes none to high and output allowance changes 2048 to 16384",
    )
    for item, record in zip(selected, selection["cases"], strict=True):
        record["score_cells"] = [{
            "model": assessment["route"], "task": task,
            "score": assessment[task]["score"], "status": assessment[task]["status"],
            "text": format(Decimal(assessment[task]["score"]), ".1f")
            if assessment[task]["score"] else assessment[task]["status"],
        } for assessment in item["assessments"] for task in TASKS]
    return selected, selection


def render_html(selected: list[dict], output: Path, **kwargs) -> str:
    document = prior.render_html(selected, output, **kwargs)
    replacements = {
        "72 images. Two scores. Four models.": "72 images. DeepSeek none vs high.",
        "Photography scoring trial · 31 October 2026": "DeepSeek reasoning comparison · 31 October 2026",
        "All four models receive the same photography-scope prompt and exact image bytes. Their actual validated scores are shown, including disagreements with the reference.":
            "DeepSeek is compared at none (2,048-token cap) and high (16,384-token cap), using the identical photography-scope prompt and image bytes. Luna, Gemini and Haiku retain their previous scores. Actual validated outputs are shown.",
        "<span><b>DeepSeek</b> V4.1 Flash · no reasoning · RunInfra</span>":
            "<span><b>DeepSeek none</b> V4.1 Flash · none requested · RunInfra</span><span><b>DeepSeek high</b> V4.1 Flash · high requested · RunInfra</span>",
        '<tr class=""><th scope="row">DeepSeek high</th>':
            '<tr class="high-reasoning"><th scope="row">DeepSeek high</th>',
        "Exact scores + evidence CSV": "All six assessments + evidence CSV",
    }
    for old, new in replacements.items():
        if old not in document and old != "72 images. Two scores. Four models.":
            raise ValueError(f"Shared HTML template changed: {old}")
        document = document.replace(old, new)
    return document


def main() -> None:
    selected, selection = load_inputs()
    output = HERE / "collage"
    prior.write_csv(output / "scores.csv", selected)
    (output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    (output / "originals-grid.html").write_text(render_html(selected, output))
    for page_number, start in enumerate((0, 36), 1):
        (output / f"originals-grid-page-{page_number}.html").write_text(
            render_html(selected[start:start + 36], output,
                        first_position=start + 1, page_number=page_number)
        )
    print(json.dumps({"output": str(output), "images": 72, "assessments": 432,
                      "score_cells": 864, "next": f"node {output / 'render.mjs'}"}, indent=2))


if __name__ == "__main__":
    main()
