"""Publish the scored comparison without changing the previous 4,000-view dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "rehearsal-4000-2026-11-01"
PUBLISHED = Path("/home/ahstn/git/model-gym/visual-quality/data/rehearsal-4000-2026-11-01/published")
OWN = Path("/home/ahstn/git/model-gym/visual-quality/data/eligibility-optimization-2026-11-01")


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    manifest = json.loads((HERE / "reference-manifest.json").read_text())
    results = json.loads((HERE / "results.json").read_text())
    winners = json.loads((HERE / "winners.json").read_text())
    assert results["reference_sha256"] == sha(HERE / "reference-manifest.json") and results[
        "winners_sha256"
    ] == sha(HERE / "winners.json")
    by_case = {c["case_id"]: c for c in manifest["cases"]}
    joined = {id: {} for id in by_case}
    for row in results["case_rows"]:
        assert row["case_id"] in joined
        model = row["model"]
        arm = row["display_arm"]
        m = joined[row["case_id"]].setdefault(model, {})
        assert arm not in m
        m[arm] = {
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
    assert len(by_case) == 100 and all(
        set(joined[id]) == {"liquid", "jev"}
        and all(set(v) == {"baseline", "candidate"} for v in joined[id].values())
        for id in joined
    )
    supplemental = json.loads((HERE / "resolution-retest-results.json").read_text())
    assert supplemental["freeze_sha256"] == sha(HERE / "resolution-retest-freeze.json")
    extra = supplemental["case"]
    joined[extra["case_id"]] = {}
    for row in supplemental["rows"]:
        model = row["model"]
        arm = row["display_arm"]
        assert arm not in joined[extra["case_id"]].setdefault(model, {})
        joined[extra["case_id"]][model][arm] = {
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
    assert len(supplemental["rows"]) == 4 and all(
        set(joined[extra["case_id"]][m]) == {"baseline", "candidate"} for m in ("liquid", "jev")
    )
    gallery = HERE / "gallery"
    previews = gallery / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    cases = []
    for c in [*manifest["cases"], extra]:
        source = Path(c["image_path"])
        assert sha(source) == c["view_sha256"]
        target = previews / (c["case_id"] + ".jpg")
        with Image.open(source) as im:
            image = im.convert("RGB")
            image.thumbnail((600, 400), Image.Resampling.LANCZOS)
            image.save(target, quality=90, optimize=True)
        own = source.resolve().is_relative_to((OWN / "views").resolve())
        original = ("/eligibility-originals/" if own else "/originals/") + c["view_sha256"] + ".png"
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
            | {"original": original, "view_sha256": c["view_sha256"], "results": joined[c["case_id"]]}
        )
    data = {
        "reference_sha256": sha(HERE / "reference-manifest.json"),
        "results_sha256": sha(HERE / "results.json"),
        "winners": {m: winners["models"][m]["candidate_round"] for m in ("liquid", "jev")},
        "cases": cases,
    }
    (gallery / "eligibility-trial.js").write_text(
        "const TRIAL=" + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + ";\n"
    )
    shutil.copy2(OLD / "gallery" / "style.css", gallery / "style.css")
    # Add only new gallery files to the existing published gallery; never rewrite frozen data.js or prior pages.
    destination = PUBLISHED / "gallery"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("eligibility-trial.html", "eligibility-trial.js"):
        shutil.copy2(gallery / name, destination / name)
    for p in previews.glob("*.jpg"):
        existing = destination / "previews" / p.name
        if existing.exists():
            # Reuse old R-prefixed previews without changing the frozen gallery.
            if sha(existing) != sha(p):
                shutil.copy2(existing, p)
        else:
            shutil.copy2(p, existing)
    shutil.copy2(HERE / "report.md", destination / "eligibility-report.md")
    proof = {
        "cases": len(cases),
        "display_rows": len(results["case_rows"]) + len(supplemental["rows"]),
        "supplemental_resolution_retest": 1,
        "fresh_holdout": 40,
        "historical_challenge": 20,
        "development": 40,
        "source_reference_sha256": data["reference_sha256"],
        "source_results_sha256": data["results_sha256"],
        "gallery_js_sha256": sha(gallery / "eligibility-trial.js"),
        "preview_hashes": {p.name: sha(p) for p in sorted(previews.glob("*.jpg"))},
    }
    (gallery / "gallery-manifest.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "display_rows": len(results["case_rows"]) + len(supplemental["rows"]),
                "published": str(destination),
            }
        )
    )


if __name__ == "__main__":
    main()
