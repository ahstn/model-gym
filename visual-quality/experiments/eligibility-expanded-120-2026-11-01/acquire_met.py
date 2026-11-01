"""Acquire public-domain Met collection records and representative image views."""

import concurrent.futures
import hashlib
import io
import json
import random
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-expanded-120-2026-11-01")
BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
if (HERE / "met-candidates.json").exists():
    raise RuntimeError("Existing acquisition evidence is immutable; use a new experiment directory")
terms = ("poster", "painting", "drawing", "print")
rng = random.Random(202611011201)
ids = {}
for term in terms:
    r = requests.get(
        "https://collectionapi.metmuseum.org/public/collection/v1.1/search",
        params={"q": term, "hasImages": "true", "limit": 140},
        timeout=25,
    )
    r.raise_for_status()
    ids[term] = r.json().get("objectIDs", [])
    rng.shuffle(ids[term])
    ids[term] = ids[term][:85]
(HERE / "met-search.json").write_text(
    json.dumps(
        {
            "terms": ids,
            "seed": 202611011201,
            "note": "Met v1.1 search; candidate labels adjudicated visually before use",
        },
        indent=2,
    )
    + "\n"
)
tasks = []
seen = set()
for term in terms:
    for ident in ids[term]:
        if ident not in seen:
            seen.add(ident)
            tasks.append((term, ident))


def get(task):
    term, ident = task
    try:
        r = requests.get(f"{BASE}/objects/{ident}", timeout=25)
        r.raise_for_status()
        x = r.json()
        if not x.get("isPublicDomain") or not x.get("primaryImageSmall"):
            return None
        response = requests.get(x["primaryImageSmall"], timeout=35)
        response.raise_for_status()
        raw = response.content
        source_sha = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as im:
            image = im.convert("RGB")
            image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            b = io.BytesIO()
            image.save(b, format="PNG", optimize=True)
            content = b.getvalue()
            sha = hashlib.sha256(content).hexdigest()
            target = DATA / "views" / f"{sha}.png"
            target.write_bytes(content)
            return {
                "case_id": f"M{ident}",
                "family_id": f"met:{ident}",
                "image_path": str(target),
                "view_sha256": sha,
                "width": image.width,
                "height": image.height,
                "source": "The Metropolitan Museum of Art Open Access",
                "source_page": x.get("objectURL"),
                "source_url": x["primaryImageSmall"],
                "source_sha256": source_sha,
                "source_license": "public domain (Met isPublicDomain)",
                "source_title": x.get("title"),
                "source_object_name": x.get("objectName"),
                "source_medium": x.get("medium"),
                "source_artist": x.get("artistDisplayName"),
                "query": term,
            }
    except (requests.RequestException, ValueError, OSError) as e:
        return {"error": type(e).__name__, "object_id": ident, "query": term}


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    rows = list(pool.map(get, tasks))
good = [x for x in rows if x and "error" not in x]
errors = [x for x in rows if x and "error" in x]
(HERE / "met-candidates.json").write_text(
    json.dumps({"records": good, "errors": errors, "requested": len(tasks)}, indent=2) + "\n"
)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
for page in range((len(good) + 29) // 30):
    canvas = Image.new("RGB", (1500, 1500), "white")
    d = ImageDraw.Draw(canvas)
    for j, c in enumerate(good[page * 30 : (page + 1) * 30]):
        x = (j % 5) * 300
        y = (j // 5) * 250
        with Image.open(c["image_path"]) as im:
            im.thumbnail((280, 205))
            canvas.paste(im, (x + (300 - im.width) // 2, y + 31 + (205 - im.height) // 2))
        d.text((x + 5, y + 5), f"{c['case_id']} {c['query']}", font=font, fill="black")
    canvas.save(HERE / "review" / f"met-{page + 1:02d}.jpg", quality=87)
print(
    json.dumps(
        {"requested": len(tasks), "usable": len(good), "errors": len(errors), "pages": (len(good) + 29) // 30}
    )
)
