"""Create a truthful 240-view gallery backed by the scored request records."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import analyze_expanded as analysis
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-expanded-120-2026-11-01")
OLD = Path("/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01")
PREVIOUS = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-optimization-2026-11-01")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest = analysis.read(HERE / "reference-manifest.json")
    results = analysis.read(HERE / "results.json")
    winner = analysis.read(HERE / "winner.json")
    assert results["winner_sha256"] == sha(HERE / "winner.json") and results["reference_sha256"] == sha(
        HERE / "reference-manifest.json"
    )
    by = {c["case_id"]: c for c in manifest["cases"]}
    joined = {id: {} for id in by}
    rows = list(results["case_rows"])
    dev = [c for c in manifest["cases"] if c["partition"] == "development"]
    for model in analysis.MODELS:
        selected = winner["models"][model]["round"]
        for arm, rid in [("original", "original"), ("previous", "previous"), ("selected", selected)]:
            for row in analysis.data(rid, model, dev):
                row["display_arm"] = arm
                rows.append(row)
    for row in rows:
        c = joined[row["case_id"]].setdefault(row["model"], {})
        assert row["display_arm"] not in c
        c[row["display_arm"]] = {
            k: row.get(k)
            for k in (
                "status",
                "technical_score",
                "aesthetic_score",
                "technical_native",
                "aesthetic_native",
                "exclusion_probability",
                "scope_floor_applied",
                "latency_seconds",
                "price_usd",
            )
        }
    assert (
        len(by) == 240
        and len(rows) == 1440
        and all(
            set(ms) == set(analysis.MODELS)
            and all(set(arms) == {"original", "previous", "selected"} for arms in ms.values())
            for ms in joined.values()
        )
    )
    gallery = HERE / "gallery"
    previews = gallery / "expanded-previews"
    previews.mkdir(exist_ok=True)
    cases = []
    for c in manifest["cases"]:
        p = Path(c["image_path"])
        assert sha(p) == c["view_sha256"]
        target = previews / (c["case_id"] + ".jpg")
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((600, 400), Image.Resampling.LANCZOS)
            im.save(target, quality=90, optimize=True)
        root = (
            "/expanded-originals/"
            if p.resolve().is_relative_to((DATA / "views").resolve())
            else "/eligibility-originals/"
            if p.resolve().is_relative_to((PREVIOUS / "views").resolve())
            else "/originals/"
        )
        cases.append(
            {
                k: c.get(k)
                for k in (
                    "case_id",
                    "partition",
                    "eligibility",
                    "group",
                    "evidence",
                    "source",
                    "source_page",
                    "source_photo_case_id",
                )
            }
            | {
                "view_sha256": c["view_sha256"],
                "original": root + c["view_sha256"] + ".png",
                "results": joined[c["case_id"]],
            }
        )
    d = {
        "reference_sha256": sha(HERE / "reference-manifest.json"),
        "results_sha256": sha(HERE / "results.json"),
        "winners": {m: winner["models"][m]["round"] for m in analysis.MODELS},
        "cases": cases,
    }
    (gallery / "expanded-eligibility.js").write_text(
        "const TRIAL=" + json.dumps(d, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ";\n"
    )
    shutil.copy2(HERE.parent / "rehearsal-4000-2026-11-01/gallery/style.css", gallery / "style.css")
    target = OLD / "published/gallery"
    (target / "expanded-previews").mkdir(exist_ok=True)
    for name in ("expanded-eligibility.html", "expanded-eligibility.js"):
        shutil.copy2(gallery / name, target / name)
    for path in previews.glob("*.jpg"):
        shutil.copy2(path, target / "expanded-previews" / path.name)
    shutil.copy2(HERE / "report.md", target / "expanded-eligibility-report.md")
    (gallery / "manifest.json").write_text(
        json.dumps(
            {
                "views": len(cases),
                "display_rows": len(rows),
                "results_sha256": d["results_sha256"],
                "gallery_js_sha256": sha(gallery / "expanded-eligibility.js"),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"views": len(cases), "display_rows": len(rows)}))


if __name__ == "__main__":
    main()
