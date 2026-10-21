"""Select a fixed diagnostic panel; keep labels out of inference requests."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
AGI_REV = "bf9b7bb8d633f44832d6ba51ff496a25d8e8e2ae"
AGI_BASE = f"https://huggingface.co/datasets/strawhat/agiqa-3k/resolve/{AGI_REV}"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def deliver(raw):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    original = image.size
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92, subsampling=0, optimize=True)
    if len(output.getvalue()) > 180_000:
        raise ValueError("Image exceeds the conservative Clef payload limit")
    return output.getvalue(), original, image.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True, help="visual-quality/data directory")
    args = parser.parse_args()
    data = ROOT / "data"
    (data / "images").mkdir(parents=True, exist_ok=True)
    manifest = args.pilot_root / "pilot-10k-v1/train.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    by_id = {r["id"]: r for r in rows}
    selected = []
    for operator in ("gaussian_blur", "gaussian_noise", "jpeg_compression"):
        variants = sorted(
            (r for r in rows if r.get("transform", {}).get("operator") == operator),
            key=lambda r: (r["parent_id"], r["transform"]["severity_index"]),
        )
        parent = variants[0]["parent_id"]
        selected += [by_id[parent]] + [r for r in variants if r["parent_id"] == parent]
    viz = sorted((r for r in rows if r["component"] == "vizwiz"), key=lambda r: r["id"])
    for unrecognizable in (False, True):
        selected += [
            r for r in viz if (r["source_labels"]["human_unrecognizable_votes"] >= 3) == unrecognizable
        ][:3]
    for component in ("clevr", "plotqa", "owned_controls"):
        selected += sorted((r for r in rows if r["component"] == component), key=lambda r: r["id"])[:2]
    samples = []

    def add(raw, source):
        delivered, original, size = deliver(raw)
        sample_id = f"sample-{len(samples):03d}"
        path = f"data/images/{sample_id}.jpg"
        (ROOT / path).write_bytes(delivered)
        samples.append(
            {
                "id": sample_id,
                "image": path,
                "source": source,
                "original_sha256": sha(raw),
                "delivered_sha256": sha(delivered),
                "original_size": original,
                "delivered_size": size,
                "delivered_bytes": len(delivered),
            }
        )

    for row in selected:
        raw = (args.pilot_root / row["image_path"]).read_bytes()
        if sha(raw) != row["sha256"]:
            raise ValueError(f"Source hash mismatch for {row['id']}")
        keep = (
            "id",
            "component",
            "source_url",
            "download_url",
            "source_revision",
            "license",
            "source_labels",
            "parent_id",
            "transform",
            "content_kind",
            "split_group",
        )
        source = {k: row[k] for k in keep if k in row}
        source["pilot_split"] = "train"
        add(raw, source)
    # Same subject index across six generators: paired content, known AI provenance.
    metadata = fetch(f"{AGI_BASE}/data.csv")
    (data / "agiqa-data.csv").write_bytes(metadata)
    agi_rows = list(csv.DictReader(io.StringIO(metadata.decode("utf-8-sig"))))
    for generator in ("AttnGAN", "DALLE2", "glide", "midjourney", "sd1.5", "xl2.2"):
        name = f"{generator}_normal_000.jpg"
        label = next(r for r in agi_rows if r["name"] == name)
        url = f"{AGI_BASE}/images/{name}"
        add(
            fetch(url),
            {
                "id": name,
                "component": "agiqa",
                "source_revision": AGI_REV,
                "source_url": "https://github.com/lcysyzxdxc/AGIQA-3k-Database",
                "download_url": url,
                "license": {"id": "MIT", "note": "Upstream database license; HF mirror"},
                "source_labels": label,
                "known_ai_generated": True,
                "generator": generator,
            },
        )
    output = {
        "version": 1,
        "pilot_manifest_sha256": sha(manifest.read_bytes()),
        "selection": "First sorted parent per distortion operator; 3 VizWiz rows per recognizability stratum; 2 per control source; AGIQA normal_000 from each generator.",
        "delivery": "EXIF transpose, RGB, full frame, max edge 512, no upscaling, JPEG quality 92 subsampling 0; no metadata sent to models",
        "samples": samples,
    }
    (ROOT / "evidence/samples.json").write_text(json.dumps(output, indent=2) + "\n")
    sheet = Image.new("RGB", (5 * 220, 6 * 190), "white")
    draw = ImageDraw.Draw(sheet)
    for i, sample in enumerate(samples):
        im = Image.open(ROOT / sample["image"])
        im.thumbnail((215, 160))
        x, y = (i % 5) * 220, (i // 5) * 190
        sheet.paste(im, (x, y))
        draw.text((x, y + 162), f"{sample['id']} {sample['source']['component']}", fill="black")
    sheet.save(data / "contact-sheet.jpg")
    print(f"Prepared {len(samples)} samples")


if __name__ == "__main__":
    main()
