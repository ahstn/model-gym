"""Select fresh candidate families without using any model outputs; render visual review sheets."""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
if (HERE / "candidate-pool.json").exists():
    raise RuntimeError("Existing candidate evidence is immutable; use a new experiment directory")
P = HERE.parent
old = json.loads((P / "eligibility-optimization-2026-11-01/reference-manifest.json").read_text())
rehearsal = json.loads((P / "rehearsal-4000-2026-11-01/manifest.json").read_text())
candidates = json.loads((P / "rehearsal-4000-2026-11-01/candidate-manifest.json").read_text())["cases"]
used = {c["family_id"] for c in [*old["cases"], *rehearsal["cases"]]}
rng = random.Random(202611011202)
selected = []
for component, n in [("vizwiz", 140), ("megalith", 140), ("plotqa", 40), ("clevr", 40)]:
    rows = [
        c
        for c in candidates
        if c["component"] == component
        and c.get("historically_fresh")
        and c["family_id"] not in used
        and c.get("operator") is None
    ]
    rng.shuffle(rows)
    rows = rows[:n]
    selected.extend(rows)
(HERE / "candidate-pool.json").write_text(
    json.dumps(
        {
            "seed": 202611011202,
            "cases": selected,
            "note": "Fresh source families absent from the prior 4000 and prior eligibility panels; sampled without model outputs. Candidates, not adjudicated labels.",
        },
        indent=2,
    )
    + "\n"
)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
for component in ("vizwiz", "megalith"):
    rows = [r for r in selected if r["component"] == component]
    for page in range((len(rows) + 24) // 25):
        canvas = Image.new("RGB", (1500, 1250), "white")
        draw = ImageDraw.Draw(canvas)
        for i, c in enumerate(rows[page * 25 : (page + 1) * 25]):
            x = i % 5 * 300
            y = i // 5 * 250
            with Image.open(c["image_path"]) as im:
                im.thumbnail((280, 205))
                canvas.paste(im, (x + (300 - im.width) // 2, y + 32 + (205 - im.height) // 2))
            draw.text((x + 5, y + 5), c["candidate_id"], font=font, fill="black")
        canvas.save(HERE / "review" / f"{component}-{page + 1:02d}.jpg", quality=87)
print(
    json.dumps(
        {k: sum(c["component"] == k for c in selected) for k in ("vizwiz", "megalith", "plotqa", "clevr")}
    )
)
