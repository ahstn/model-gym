"""Build a balanced, family-separated development and holdout set; labels reviewed on sheets."""

import hashlib
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
P = HERE.parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-expanded-120-2026-11-01")
rng = random.Random(202611011203)
VIZ = [
    "C4842",
    "C4637",
    "C4758",
    "C4869",
    "C4701",
    "C4936",
    "C4511",
    "C4979",
    "C4734",
    "C5006",
    "C4666",
    "C4670",
    "C5120",
    "C4541",
    "C5175",
    "C0823",
    "C5220",
    "C5244",
    "C4503",
    "C5303",
    "C0584",
    "C5037",
    "C5174",
    "C4971",
    "C0154",
    "C5017",
    "C5307",
    "C0883",
    "C5043",
    "C0609",
    "C0206",
    "C5191",
    "C4561",
    "C5032",
    "C5023",
    "C5103",
    "C4522",
    "C0118",
    "C5051",
    "C4574",
    "C5123",
    "C5000",
    "C4620",
    "C5149",
    "C4831",
    "C5181",
    "C5008",
    "C5186",
    "C5057",
    "C4729",
    "C4662",
    "C5050",
    "C4843",
    "C4699",
    "C4651",
    "C4770",
    "C0725",
    "C0443",
    "C4989",
    "C4507",
    "C5042",
    "C4602",
    "C4928",
    "C4649",
    "C4673",
    "C0035",
    "C4576",
    "C4886",
]
DEV_HIST = ["R3554", "R0580", "R1377", "R3373", "R2563", "R1393", "R1381", "R2037", "R2362", "R2455"]
HOLD_HIST = ["R0060", "R0134", "R0218", "R0226", "R0317", "R0331", "R0356", "R0437", "R0460", "R0477"]
MET_DEV = ["M787866", "M334196"]
MET_HOLD = ["M637486", "M916228"]


def read(p):
    return json.loads(Path(p).read_text())


pool = read(HERE / "candidate-pool.json")["cases"]
byid = {x["candidate_id"]: x for x in pool}
old = read(P / "rehearsal-4000-2026-11-01/manifest.json")["cases"]
oldmap = {x["case_id"]: x for x in old}
met = {x["case_id"]: x for x in read(HERE / "met-candidates.json")["records"]}
prior = read(P / "eligibility-optimization-2026-11-01/reference-manifest.json")["cases"]
prior_map = {x["case_id"]: x for x in prior}
ref = {
    x["case_id"]: x
    for x in map(json.loads, (P / "rehearsal-4000-2026-11-01/references.jsonl").read_text().splitlines())
}
assert all(id in byid for id in VIZ), set(VIZ) - set(byid)
rng.shuffle(VIZ)
v = VIZ[:60]
m = [x["candidate_id"] for x in pool if x["component"] == "megalith" and x["candidate_id"] != "C0528"]
rng.shuffle(m)
m = m[:60]
sets = {"development": [], "heldout": []}
for group, ids in [("physical_context", v), ("photo_scene", m)]:
    for part, subset in [("development", ids[:30]), ("heldout", ids[30:])]:
        for id in subset:
            sets[part].append(
                {
                    **byid[id],
                    "case_id": id,
                    "partition": part,
                    "eligibility": "eligible",
                    "group": group,
                    "evidence": "Visually reviewed photograph of "
                    + (
                        "a physical object or surroundings including text and packaging."
                        if group == "physical_context"
                        else "a physical scene or object."
                    ),
                }
            )
for component, group in [("plotqa", "chart"), ("clevr", "render")]:
    ids = [x["candidate_id"] for x in pool if x["component"] == component]
    rng.shuffle(ids)
    ids = ids[:20]
    for part, subset in [("development", ids[:10]), ("heldout", ids[10:])]:
        for id in subset:
            sets[part].append(
                {
                    **byid[id],
                    "case_id": id,
                    "partition": part,
                    "eligibility": "excluded",
                    "group": group,
                    "evidence": "Flat generated chart."
                    if group == "chart"
                    else "Obvious rendered geometric scene.",
                }
            )
for part, ids in [("development", DEV_HIST), ("heldout", HOLD_HIST)]:
    for id in ids:
        base = prior_map[id] if id in prior_map else oldmap[id]
        assert id in prior_map or ref[id]["eligibility"] == "excluded"
        sets[part].append(
            {
                **base,
                "partition": part,
                "eligibility": "excluded",
                "group": "historical_flat",
                "evidence": "Previously model-scored natural flat artwork, poster, page or illustration.",
                "previous_eligibility_trial": id in prior_map,
            }
        )
for part, ids in [("development", MET_DEV), ("heldout", MET_HOLD)]:
    for id in ids:
        sets[part].append(
            {
                **met[id],
                "partition": part,
                "eligibility": "excluded",
                "group": "museum_flat",
                "evidence": "Public-domain flat artwork, drawing or poster presented as the image itself.",
            }
        )


def screen(photo, num, part):
    # Two different families of UI per partition, four layouts total; screenshot paired with its source photo.
    w, h = 1200, 900
    style = (num % 2) + (0 if part == "development" else 2)
    colors = ["#232b3a", "#f3f4f7", "#251b30", "#f5f7fb"]
    canvas = Image.new("RGB", (w, h), colors[style])
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    if style == 0:
        d.rectangle((0, 0, w, 70), fill="#414d61")
        d.text((22, 20), "●  ●  ●    Browser / album / image", font=font, fill="white")
        d.rectangle((0, 70, 180, h), fill="#e5e9f2")
        d.text((18, 110), "Albums\n\nRecent\n\nShared", font=font, fill="#35455d")
        area = (200, 90, 1180, 850)
    elif style == 1:
        d.rectangle((0, 0, w, 100), fill="#fff")
        d.text((30, 25), "PHOTO EDITOR       File  Edit  Image  Filters", font=font, fill="#303642")
        d.rectangle((0, 100, 120, h), fill="#cbd5e1")
        d.text((20, 130), "Tools\n\nCrop\n\nLight", font=font, fill="#303642")
        area = (130, 120, 1180, 850)
    elif style == 2:
        d.rectangle((0, 0, w, 90), fill="#111827")
        d.text((30, 25), "VIDEO PLAYER                         Library   Search", font=font, fill="white")
        d.rectangle((0, 810, w, h), fill="#111827")
        d.text((30, 835), "▶       ━━━━━━━━━━━━━━━━━━━     00:24 / 01:12     ▣", font=font, fill="white")
        area = (50, 105, 1150, 800)
    else:
        d.rectangle((130, 15, 1070, 885), fill="white")
        d.text((160, 40), "Social feed                         Follow", font=font, fill="#253244")
        d.text((160, 805), "♡    Comment       Share", font=font, fill="#253244")
        area = (160, 90, 1040, 790)
    with Image.open(photo["image_path"]) as im:
        im = im.convert("RGB")
        im.thumbnail((area[2] - area[0], area[3] - area[1]), Image.Resampling.LANCZOS)
        canvas.paste(
            im,
            (area[0] + (area[2] - area[0] - im.width) // 2, area[1] + (area[3] - area[1] - im.height) // 2),
        )
    b = io.BytesIO()
    canvas.save(b, format="PNG", optimize=True)
    raw = b.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    path = DATA / "views" / f"{sha}.png"
    path.write_bytes(raw)
    return {
        "case_id": f"X{num:04d}",
        "family_id": photo["family_id"],
        "image_path": str(path),
        "view_sha256": sha,
        "width": w,
        "height": h,
        "partition": part,
        "eligibility": "excluded",
        "group": "synthetic_screenshot",
        "source": "Synthetic interface screenshot control",
        "source_photo_case_id": photo["case_id"],
        "template": style,
        "evidence": "Synthetic interface surrounding a photograph; not a photograph of a physical screen.",
    }


num = 0
for part in ("development", "heldout"):
    parents = [x for x in sets[part] if x["eligibility"] == "eligible"]
    rng.shuffle(parents)
    for photo in parents[:28]:
        num += 1
        sets[part].append(screen(photo, num, part))
for part, rows in sets.items():
    assert len(rows) == 120, (part, len(rows))
    assert sum(r["eligibility"] == "excluded" for r in rows) == 60
    assert sum(r["eligibility"] == "eligible" for r in rows) == 60
    rng.shuffle(rows)
assert not ({c["family_id"] for c in sets["development"]} & {c["family_id"] for c in sets["heldout"]})
for c in sets["development"] + sets["heldout"]:
    p = Path(c["image_path"])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == c["view_sha256"]
    with Image.open(p) as im:
        if min(im.size) < 280:
            original = c["view_sha256"]
            im = im.convert("RGB")
            ratio = 280 / min(im.size)
            im = im.resize((round(im.width * ratio), round(im.height * ratio)), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            im.save(out, format="PNG")
            raw = out.getvalue()
            sha = hashlib.sha256(raw).hexdigest()
            path = DATA / "views" / f"{sha}.png"
            path.write_bytes(raw)
            c.update(
                {
                    "original_view_sha256": original,
                    "image_path": str(path),
                    "view_sha256": sha,
                    "width": im.width,
                    "height": im.height,
                    "size_normalization": "Upscaled without crop; shortest side 280 pixels for provider compatibility.",
                }
            )
manifest = {
    "schema_version": 1,
    "seed": 202611011203,
    "label_method": "Assistant visual review, not human ground truth.",
    "selection": "120 per partition, each 60 eligible and 60 excluded. 10 historical flat artwork cases per split, four new public-domain museum images split 2 per panel, charts/renders 10 each per split, 28 synthetic screenshot controls per split paired with eligible photos.",
    "limitations": "Historic cases were model-scored in earlier trials and development ones are known prior failures. Holdout historical cases were selected on prior reference classifications, not model response. Screenshots are synthesized from four UI templates and share parents in their own split. Families separated across partitions, but no claim that all near-duplicates are removed.",
    "cases": sets["development"] + sets["heldout"],
}
with (HERE / "reference-manifest-rebuilt.json").open("x") as output:
    output.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(
    json.dumps(
        {
            k: {"n": len(v), "groups": {g: sum(x["group"] == g for x in v) for g in {x["group"] for x in v}}}
            for k, v in sets.items()
        }
    )
)
