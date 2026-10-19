"""Build source-disjoint photographic panels for extended Decider optimisation."""

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw

TRACK = Path(__file__).resolve().parent
ROOT = TRACK.parents[1]
DATA = ROOT / "data/tracks/ai_detection_v2"
CACHE = DATA / "source-assets"
MANIFEST = TRACK / "dataset.json"
SEED = "20261011-ai-expanded-v2"
SOURCES = {
    "clic": ("marco-willi/synthclic", "feff43b543929916bac097e30217cc3fdc09c32d", "clic2020"),
    "buster": ("marco-willi/synthbuster-plus", "dbfb72f1ee96e953ee5cff80c58832fe89e1d2b5", "raise1k"),
}
spec = importlib.util.spec_from_file_location("prior_builder", ROOT / "tracks/ai_detection/build_dataset.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
sha = old.sha


@cache
def clic_subset(image_id):
    document = json.loads((ROOT / "evidence/ai-expansion/clic-native-filelists.json").read_text())
    matches = {
        key.split(":")[0]
        for key, archive in document.items()
        for name in archive["names"]
        if Path(name).stem == image_id
    }
    if len(matches) != 1:
        raise ValueError(f"Unknown or conflicting native CLIC subset: {image_id}")
    return matches.pop()


def previous_ids():
    ids = set()
    for path in [
        ROOT / "tracks/ai_detection/dataset.json",
        ROOT / "tracks/ai_detection/eligibility.json",
        ROOT / "tracks/ai_detection/evidence/acquisition-v1-not-used.json",
    ]:

        def walk(x):
            if isinstance(x, dict):
                if "image_id" in x:
                    ids.add(x["image_id"])
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(json.loads(path.read_text()))
    return ids


def catalogue():
    import native

    return native.catalogue(sys.modules[__name__])


def selections(groups, decisions):
    excluded = previous_ids()
    jobs = [
        ("train", "clic", "train", 80, ("FLUX.1-dev", "FLUX.1-schnell", "SD3-medium")),
        ("dev", "clic", "train", 40, ("FLUX.1-dev", "FLUX.1-schnell", "SD3-medium")),
        ("test", "clic", "test", 40, ("imagen3",)),
        ("external", "buster", "test", 40, ("stable-diffusion-xl",)),
    ]
    cohort = set()
    for kind, native_split in [("clic", "train"), ("clic", "test"), ("buster", "test")]:
        real_kind = SOURCES[kind][2]
        real_items = sorted(
            [
                (key, family[real_kind])
                for key, family in groups.items()
                if key[:2] == (kind, native_split) and real_kind in family
            ],
            key=lambda pair: (pair[1]["native_file"], pair[1]["native_group"], pair[1]["native_local_index"]),
        )
        if kind == "clic" and native_split == "train":
            for subset in ("professional", "mobile"):
                cohort.update(
                    key
                    for key, _ in [(key, item) for key, item in real_items if clic_subset(key[2]) == subset][
                        :100
                    ]
                )
        else:
            cohort.update(key for key, _ in (real_items[:200] if kind == "buster" else real_items[:125]))
    used, result = set(), []
    for split, kind, native, count, generators in jobs:
        ranked = sorted(
            [key for key in groups if key[:2] == (kind, native) and key[2] not in excluded and key in cohort],
            key=lambda key: sha(f"{SEED}:{key}".encode()),
        )
        for n in range(count):
            generator = generators[n % len(generators)]
            for key in ranked:
                if key in used or SOURCES[kind][2] not in groups[key] or generator not in groups[key]:
                    continue
                decision_key = f"{kind}:{key[2]}:{generator}"
                used.add(key)
                result.append(
                    {
                        "key": decision_key,
                        "group": key,
                        "generator": generator,
                        "split": split,
                        "items": [groups[key][SOURCES[kind][2]], groups[key][generator]],
                    }
                )
                break
            else:
                raise ValueError("Insufficient complete eligible groups")
    for pair in result:
        if decisions.get(pair["key"]) is not False:
            continue
        kind, native_split, _ = pair["group"]
        ranked = sorted(
            [
                key
                for key in groups
                if key[:2] == (kind, native_split) and key in cohort and key[2] not in excluded
            ],
            key=lambda key: sha(f"{SEED}:{key}".encode()),
        )
        for key in ranked:
            decision_key = f"{kind}:{key[2]}:{pair['generator']}"
            if (
                key in used
                or SOURCES[kind][2] not in groups[key]
                or pair["generator"] not in groups[key]
                or decisions.get(decision_key) is False
            ):
                continue
            used.add(key)
            pair.update(
                key=decision_key,
                group=key,
                items=[groups[key][SOURCES[kind][2]], groups[key][pair["generator"]]],
            )
            break
        else:
            raise ValueError("Insufficient replacement families")
    return result


def verify(document):
    eligibility_file = TRACK / "eligibility.json"
    if document["eligibility_sha256"] != sha(eligibility_file.read_bytes()):
        raise ValueError("Frozen eligibility ledger changed")
    decisions = json.loads(eligibility_file.read_text())["decisions"]
    excluded_ids = previous_ids()
    prior_hashes, prior_pixels, _ = old.prior_references()
    for x in (
        json.loads((ROOT / "tracks/quality/dataset.json").read_text())["samples"]
        + json.loads((ROOT / "tracks/ai_detection/dataset.json").read_text())["samples"]
    ):
        prior_hashes.update(x[k] for k in ("original_sha256", "delivered_sha256"))
        prior_pixels.add(x["decoded_rgb_sha256"])
    groups, hashes, pixels = defaultdict(list), set(), set()
    for x in document["samples"]:
        if x["source"]["image_id"] in excluded_ids:
            raise ValueError("Previously reviewed source family reused")
        original = (
            CACHE
            / f"{('clic' if 'synthclic' in x['source']['dataset'] else 'buster')}-{x['source']['source_kind']}-{x['source']['image_id']}.bin"
        )
        if sha(original.read_bytes()) != x["original_sha256"]:
            raise ValueError("Native original hash mismatch")
        key = f"{('clic' if 'synthclic' in x['source']['dataset'] else 'buster')}:{x['source']['image_id']}:{x['source']['paired_generator']}"
        if decisions.get(key) is not True:
            raise ValueError("Pair lacks positive medium review")
        if x["labels"]["ai_generated"]:
            allowed = {
                "train": {"SD3-medium", "FLUX.1-dev", "FLUX.1-schnell"},
                "dev": {"SD3-medium", "FLUX.1-dev", "FLUX.1-schnell"},
                "test": {"imagen3"},
                "external": {"stable-diffusion-xl"},
            }
            if x["source"]["generator"] not in allowed[x["split"]]:
                raise ValueError("Generator holdout changed")
        raw = (ROOT / x["image"]).read_bytes()
        h, p = sha(raw), old.pixel_hash(raw)
        if h != x["delivered_sha256"] or p != x["decoded_rgb_sha256"]:
            raise ValueError("Delivered hash mismatch")
        if h in prior_hashes | hashes or p in prior_pixels | pixels or x["original_sha256"] in prior_hashes:
            raise ValueError("Prior or internal exact duplicate")
        hashes.add(h)
        pixels.add(p)
        groups[x["group_id"]].append(x)
    for pair in groups.values():
        if (
            len(pair) != 2
            or {x["labels"]["ai_generated"] for x in pair} != {True, False}
            or len({x["split"] for x in pair}) != 1
        ):
            raise ValueError("Invalid group or split leakage")
    if Counter(x["split"] for x in document["samples"]) != {
        "train": 160,
        "dev": 80,
        "test": 80,
        "external": 80,
    }:
        raise ValueError("Split balance changed")
    print("Verified 400 images / 200 groups, source-disjoint splits and no exact prior-panel overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-candidates", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(json.loads(MANIFEST.read_text()))
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    (DATA / "images").mkdir(exist_ok=True)
    decisions_file = TRACK / "eligibility.json"
    decisions = json.loads(decisions_file.read_text())["decisions"] if decisions_file.exists() else {}
    selected = selections(catalogue(), decisions)
    items = [item for pair in selected for item in pair["items"]]
    import native

    raw = native.acquire_many(sys.modules[__name__], items)
    if args.prepare_candidates:
        (DATA / "pending.json").write_text(
            json.dumps([{k: v for k, v in pair.items() if k != "items"} for pair in selected], indent=2)
            + "\n"
        )
        for start in range(0, len(selected), 20):
            sheet = Image.new("RGB", (1000, 20 * 140), "white")
            draw = ImageDraw.Draw(sheet)
            for n, pair in enumerate(selected[start : start + 20]):
                for m in range(2):
                    from io import BytesIO

                    im = Image.open(BytesIO(raw[(start + n) * 2 + m])).convert("RGB")
                    im.thumbnail((440, 115))
                    sheet.paste(im, (m * 500, n * 140 + 20))
                draw.text((0, n * 140), f"{start + n}: {pair['key']}", fill="black")
            sheet.save(DATA / f"contact-{start:03d}.jpg")
        print("Prepared 200 pairs for medium eligibility review")
        return
    if any(decisions.get(p["key"]) is not True for p in selected):
        raise ValueError("Medium review incomplete")
    samples = []
    deliveries = []
    for i, (item, original) in enumerate(zip(items, raw)):
        pair = selected[i // 2]
        row = item["row"]
        kind = item["kind"]
        repo, revision, real = SOURCES[kind]
        generated = row["source"] != real
        if row["label"] != int(generated):
            raise ValueError("Source label conflict")
        encoded, original_size, delivered_size = old.deliver(original)
        path = f"data/tracks/ai_detection_v2/images/ai4-{i:03d}.jpg"
        deliveries.append((ROOT / path, encoded))
        samples.append(
            {
                "id": f"ai4-{i:03d}",
                "image": path,
                "group_id": f"{real}:{row['image_id']}",
                "split": pair["split"],
                "original_sha256": sha(original),
                "delivered_sha256": sha(encoded),
                "decoded_rgb_sha256": old.pixel_hash(encoded),
                "original_size": original_size,
                "delivered_size": delivered_size,
                "delivered_bytes": len(encoded),
                "source": {
                    "dataset": repo,
                    "revision": revision,
                    "config": "default",
                    "split": item["native_split"],
                    "row_index": item["row_idx"],
                    "native_image": f"data/tracks/ai_detection_v2/source-assets/{kind}-{row['source']}-{row['image_id']}.bin",
                    "native_parquet": item["native_url"],
                    "native_row_group": item["native_group"],
                    "native_group_row": item["native_local_index"],
                    "image_id": row["image_id"],
                    "clic_subset": clic_subset(row["image_id"]) if kind == "clic" else None,
                    "source_kind": row["source"],
                    "generator": row["source"] if generated else None,
                    "paired_generator": pair["generator"],
                    "asset_url": item["native_url"],
                    "source_url": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
                    "representation": "Native embedded parquet image bytes; source EXIF is preserved in the audit cache",
                    "matching_basis": "Published shared source image ID; caption matching, not pixel alignment",
                    "license": {
                        "id": ("CC0-1.0" if clic_subset(row["image_id"]) == "mobile" else "Unsplash CLIC2020")
                        + (
                            "; generated author release subject to original photograph terms"
                            if generated
                            else ""
                        )
                        if kind == "clic"
                        else (
                            "CC-BY-NC-SA-4.0" if generated else "RAISE non-commercial research and education"
                        ),
                        "url": f"https://data.vision.ee.ethz.ch/cvl/clic/LICENSE_{clic_subset(row['image_id'])}_2020.txt"
                        if kind == "clic"
                        else "https://zenodo.org/records/10066460",
                        "use": "Non-commercial research; binaries excluded from version control",
                    },
                },
                "labels": {
                    "ai_generated": generated,
                    "source_label": row["label"],
                    "basis": "Published provenance, not detector output",
                },
            }
        )
    doc = {
        "version": 2,
        "track": "ai_detection_v2",
        "task": "ai",
        "created": "2026-10-11",
        "selection_seed": SEED,
        "eligibility_sha256": sha(decisions_file.read_bytes()),
        "delivery": "Same EXIF transpose, RGB512x512 centre crop and JPEG92 policy for both classes; inference excludes source metadata",
        "selection": "Hash-ranked whole caption families within bounded native cohorts: first100 professional and100 mobile CLIC native train families; first125 CLIC native test families; all200 Synthbuster native test families minus111 previously reviewed IDs. Imagen3 final generator holdout; SDXL-only unseen RAISE families external",
        "cohort_restriction": "Native row-group locality constrains source coverage; the cohort is not a uniform sample of the full release. Families remain disjoint; photographer and capture-session isolation is not established.",
        "samples": samples,
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != doc:
        raise ValueError("Frozen manifest changed")
    for path, encoded in deliveries:
        path.write_bytes(encoded)
    verify(doc)
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    main()
