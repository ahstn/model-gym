"""Freeze balanced assistant-reviewed eligibility cases and paired screenshot controls."""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-optimization-2026-11-01")
OLD = Path("/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/published")
GROUPS = {
    "plots": "C5339 C5488 C5405 C5470 C5480 C5345 C5448 C0450",
    "renders": "C5322 C5370 C5439 C5410 C0659 C5490 C5331 C5328",
    "context": "C4567 C5027 C4613 C4762 C3265 C1363 C5021 C5310 C5207 C5206 C4889 C4591 C5088 C0646 C2295 C4550 C4864 C5265 C2549 C0973",
    "scene": "C1708 C3244 C3631 C2181 C1232 C2742 C4296 C0198 C1196 C1004 C0879 C2415 C0654 C2608 C0069 C0743 C2143 C2958 C2196 C3171",
    "art": "C1546 C4794",
}


def screenshot(photo, number, partition):
    im = Image.new("RGB", (1280, 960), "#d5dce4")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
    d.rectangle((0, 0, 1279, 50), fill="#283341")
    d.text((23, 10), "●  ●  ●     Image library", font=f, fill="white")
    d.rectangle((0, 51, 1279, 105), fill="#eff2f6")
    d.text((28, 65), "←    →    ⟳        images.example / album / photo", font=f, fill="#3c4b5e")
    d.rectangle((0, 106, 199, 959), fill="#f8fafc")
    d.text((25, 133), "Library", font=b, fill="#243a50")
    d.text((25, 195), "Photos\n\nAlbums\n\nFavorites\n\nRecent", font=f, fill="#55677a")
    d.rectangle((200, 106, 1279, 170), fill="white")
    d.text((230, 124), "Photo viewer                    Share     Download", font=f, fill="#293d50")
    with Image.open(photo["image_path"]) as source:
        image = source.convert("RGB")
        image.thumbnail((1020, 740), Image.Resampling.LANCZOS)
        im.paste(image, (200 + (1080 - image.width) // 2, 180 + (760 - image.height) // 2))
    b = io.BytesIO()
    im.save(b, format="PNG", optimize=True)
    raw = b.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    p = DATA / "views" / f"{sha}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    return {
        "case_id": f"S{number:04d}",
        "family_id": photo["family_id"],
        "image_path": str(p),
        "view_sha256": sha,
        "delivered_bytes": len(raw),
        "source": "synthetic screenshot control",
        "source_photo_case_id": photo["case_id"],
        "source_photo_sha256": photo["view_sha256"],
        "width": im.width,
        "height": im.height,
        "partition": partition,
        "eligibility": "excluded",
        "group": "screenshot",
        "evidence": "Visible browser navigation, address bar, sidebar and photo-viewer interface surround an image.",
    }


def main():
    fresh = json.loads((HERE / "candidate-selection-fresh.json").read_text())
    by_id = {c["candidate_id"]: c for c in fresh["cases"]}
    oldfamilies = {c["family_id"] for c in json.loads((OLD / "manifest.json").read_text())["cases"]}
    rng = random.Random(2026110104)
    parts = {"development": [], "heldout": []}
    for group, values in GROUPS.items():
        ids = values.split()
        rng.shuffle(ids)
        half = len(ids) // 2
        for partition, subset in [("development", ids[:half]), ("heldout", ids[half:])]:
            for id in subset:
                source = by_id[id]
                assert source["family_id"] not in oldfamilies and source["historically_fresh"]
                with Path(source["image_path"]).open("rb") as image_file:
                    assert hashlib.file_digest(image_file, "sha256").hexdigest() == source["view_sha256"]
                eligible = group in ("context", "scene")
                parts[partition].append(
                    {
                        **source,
                        "case_id": id,
                        "partition": partition,
                        "eligibility": "eligible" if eligible else "excluded",
                        "group": group,
                        "evidence": {
                            "context": "Photograph of a physical object, text, label or artwork with depth or surrounding context.",
                            "scene": "Photograph of a physical scene, object or subject.",
                            "plots": "Flat computer-generated chart or diagram.",
                            "renders": "Obvious computer-rendered geometric scene.",
                            "art": "Image itself is a cropped flat drawing or document reproduction.",
                        }[group],
                    }
                )
    for partition, indices in [("development", [1, 3]), ("heldout", [2])]:
        for i in indices:
            c = json.loads((HERE / "review" / f"commons-{i:04d}.json").read_text())
            parts[partition].append(
                {
                    **c,
                    "partition": partition,
                    "eligibility": "excluded",
                    "group": "poster",
                    "evidence": "Flat public-domain poster or document reproduction presented as the image itself.",
                }
            )
    # The paired screenshot and its photographic parent must remain in the same partition.
    number = 0
    for partition, n in [("development", 9), ("heldout", 10)]:
        parents = [c for c in parts[partition] if c["eligibility"] == "eligible"]
        rng.shuffle(parents)
        for photo in parents[:n]:
            number += 1
            parts[partition].append(screenshot(photo, number, partition))
    for partition, rows in parts.items():
        assert (
            len(rows) == 40
            and sum(r["eligibility"] == "excluded" for r in rows) == 20
            and sum(r["eligibility"] == "eligible" for r in rows) == 20
        ), (partition, len(rows))
        rng.shuffle(rows)
    assert not ({x["family_id"] for x in parts["development"]} & {x["family_id"] for x in parts["heldout"]})
    result = {
        "schema_version": 1,
        "reference_type": "Assistant eligibility assessments of visually inspected candidate sheets, not human ground truth",
        "seed": 2026110104,
        "partition_rule": "Family-grouped, balanced 20 excluded / 20 eligible per partition. 19 synthetic UI screenshots are paired with fresh eligible photographs in the same partition.",
        "freshness": "Original candidate families are absent from the prior 4,000-image run and marked historically fresh; three public-domain Commons posters were downloaded separately. Exact-hash/family checks do not prove absence of all visual near-duplicates.",
        "limitations": "Only five real flat artwork/document cases, three public-domain posters; other exclusion controls include synthetic UI screenshots, simple graphs and renders. Deliberately balanced and not a deployment prevalence sample.",
        "cases": parts["development"] + parts["heldout"],
    }
    # The frozen manifest also contains the later, explicitly historical challenge.
    # Never overwrite it when reconstructing this initial fresh-only selection.
    with (HERE / "reference-manifest-initial.json").open("x") as output:
        output.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "cases": len(result["cases"]),
                "family_count": len({c["family_id"] for c in result["cases"]}),
                "partitions": {k: len(v) for k, v in parts.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
