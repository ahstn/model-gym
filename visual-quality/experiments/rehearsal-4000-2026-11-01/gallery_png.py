"""Render the saved 72-case selection using exact delivered images and CSV rows."""

from __future__ import annotations

import json
from pathlib import Path

from gallery import GALLERY, HERE, MODELS, digest, money, numeric, read_rows, score_text
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def render(
    *,
    gallery: Path = GALLERY,
    cases: dict | None = None,
    rows: dict | None = None,
    references_path: Path = HERE / "references.jsonl",
    scores_path: Path = HERE / "scores-long.csv",
    title_text: str = "Photo scoring rehearsal",
    footer_text: str = "Exact views, evidence and stats arms in the linked gallery",
) -> None:
    selection = json.loads((gallery / "selection.json").read_text())
    if selection["references_sha256"] != digest(references_path):
        raise ValueError("Reference selection changed")
    ids = selection["case_ids"]
    if len(ids) != 72 or len(set(ids)) != 72:
        raise ValueError("Expected 72 distinct comparison cases")
    if cases is None:
        cases = {case["case_id"]: case for case in json.loads((HERE / "manifest.json").read_text())["cases"]}
    if rows is None:
        rows = read_rows(scores_path)
    title = ImageFont.truetype(BOLD, 30)
    body = ImageFont.truetype(FONT, 18)
    bold = ImageFont.truetype(BOLD, 18)
    small = ImageFont.truetype(FONT, 16)
    pages = []
    for page in range(4):
        width, height = 1964, 3455
        canvas = Image.new("RGB", (width, height), "#eef2f6")
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 18), f"{title_text}  |  Page {page + 1} / 4", font=title, fill="#172d40")
        draw.text(
            (26, 60),
            "Baseline technical / aesthetic scores: 1.00–5.00  |  Reference: independent assistant assessment",
            font=small,
            fill="#53697c",
        )
        draw.text(
            (26, 84),
            "Seconds and price: selected attempt  |  HTTP request to complete response  |  Lite / Reference: assigned $0.00  |  Ensemble: no new request",
            font=small,
            fill="#53697c",
        )
        draw.text(
            (26, 108),
            "Ensemble: 37.5% Luna + 37.5% DeepSeek + 25% Liquid per outcome. Unknown price: dash. Lite has no photography-scope gate.",
            font=small,
            fill="#53697c",
        )
        page_ids = ids[page * 18 : (page + 1) * 18]
        for index, case_id in enumerate(page_ids):
            case = cases[case_id]
            x, y = 24 + index % 3 * 644, 148 + index // 3 * 544
            card_width, card_height = 628, 528
            draw.rounded_rectangle(
                (x, y, x + card_width, y + card_height), radius=9, fill="white", outline="#bfd0de", width=2
            )
            draw.text((x + 14, y + 10), case_id, font=bold, fill="#172d40")
            badge = case["benchmark_partition"].upper()
            draw.text(
                (x + card_width - 14 - draw.textlength(badge, font=small), y + 11),
                badge,
                font=small,
                fill="#71889b",
            )
            top, photo_height = y + 41, 211
            draw.rectangle((x + 2, top, x + card_width - 2, top + photo_height), fill="#e2e8ef")
            source = Path(case["image_path"])
            if digest(source) != case["view_sha256"]:
                raise ValueError("Changed delivered image: " + case_id)
            with Image.open(source) as original:
                preview = original.convert("RGB")
                preview.thumbnail((card_width - 12, photo_height - 4), Image.Resampling.LANCZOS)
                canvas.paste(
                    preview,
                    (x + (card_width - preview.width) // 2, top + (photo_height - preview.height) // 2),
                )
            table_top = top + photo_height + 7
            columns = [
                ("Assessor", x + 14),
                ("Tech", x + 284),
                ("Aest", x + 350),
                ("Seconds", x + 424),
                ("Price", x + 609),
            ]
            for column_index, (text, right) in enumerate(columns):
                left = right if column_index == 0 else right - draw.textlength(text, font=small)
                draw.text((left, table_top), text, font=small, fill="#71889b")
            for model_index, (model, label) in enumerate(MODELS):
                arm = "reference" if model == "reference" else "baseline"
                row = rows[case_id].get(f"{model}:{arm}", {})
                row_top = table_top + 24 + model_index * 30
                if model in ("reference", "photo_ensemble"):
                    draw.rectangle(
                        (x + 8, row_top - 2, x + card_width - 8, row_top + 26),
                        fill="#e4f0fb" if model == "reference" else "#f3f6f9",
                    )
                else:
                    draw.line((x + 12, row_top + 26, x + card_width - 12, row_top + 26), fill="#e6edf3")
                draw.text(
                    (x + 14, row_top), label, font=bold if model == "reference" else body, fill="#172d40"
                )
                values = [
                    score_text(row, "technical_score"),
                    score_text(row, "aesthetic_score"),
                    numeric(row.get("latency_seconds")),
                    money(row.get("price_usd")),
                ]
                for column_index, value in enumerate(values, start=1):
                    face = small if column_index == 4 else bold
                    right = columns[column_index][1]
                    draw.text(
                        (right - draw.textlength(value, font=face), row_top), value, font=face, fill="#254d70"
                    )
            draw.text(
                (x + 14, y + card_height - 24),
                footer_text,
                font=small,
                fill="#71889b",
            )
        draw.text(
            (26, height - 30),
            "Selection spans reference technical scores; it is not a population-weighted sample. Previews preserve framing but reduce fine detail.",
            font=small,
            fill="#53697c",
        )
        target = gallery / f"comparison-page-{page + 1}.png"
        canvas.save(target, optimize=True)
        pages.append(
            {
                "path": target.name,
                "sha256": digest(target),
                "width": width,
                "height": height,
                "case_ids": page_ids,
            }
        )
        print(target.name, flush=True)
    validation = {
        "pages": pages,
        "selection_sha256": digest(gallery / "selection.json"),
        "scores_sha256": digest(scores_path),
        "images": 72,
        "models_per_image": len(MODELS),
        "score_cells": 72 * len(MODELS) * 2,
        "source": "exact delivered image bytes; resized only for page display",
    }
    (gallery / "png-validation.json").write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    render()
