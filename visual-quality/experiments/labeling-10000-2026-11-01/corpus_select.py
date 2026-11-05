"""Freeze a fresh content-stratified benchmark and produce duplicate-review evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from corpus_common import DATA, PREVIOUS, SEED, digest, stable_key, write_json
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

from visual_quality.validation import _HashIndex


def vectors(cases):
    return {
        r["candidate_id"]: np.load(DATA / "embedding-cache" / f"{r['view_sha256']}.npy", allow_pickle=False)
        for r in cases
        if (DATA / "embedding-cache" / f"{r['view_sha256']}.npy").exists()
    }


def sample(pool, count, clusters, tag, sqrt=False):
    if len(pool) < count:
        raise ValueError(f"{tag}: insufficient candidates {len(pool)} < {count}")
    groups = defaultdict(list)
    for row in pool:
        groups[clusters[row["candidate_id"]]].append(row)
    quotas = {key: 0 for key in groups}
    exponent = 0.5 if sqrt else 1.0
    for _ in range(count):
        key = max(
            (k for k in groups if quotas[k] < len(groups[k])),
            key=lambda k: (len(groups[k]) ** exponent / (quotas[k] + 1), -k),
        )
        quotas[key] += 1
    return [
        row
        for key in sorted(groups)
        for row in sorted(groups[key], key=lambda r: stable_key(tag, r["image_id"]))[: quotas[key]]
    ]


def review_pages(pairs, lookup, prefix):
    outputs = []
    for start in range(0, len(pairs), 12):
        page = Image.new("RGB", (1500, 1440), "#edf1f5")
        draw = ImageDraw.Draw(page)
        for i, pair in enumerate(pairs[start : start + 12]):
            x, y = (i % 2) * 750, (i // 2) * 240
            for offset, key in [(0, "left"), (370, "right")]:
                row = lookup[pair[key]]
                with Image.open(row["image_path"]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((355, 195))
                    page.paste(image, (x + offset + (355 - image.width) // 2, y + 24))
                label = row.get("case_id", row["candidate_id"])
                draw.text((x + offset + 5, y + 5), label, fill="#182938")
            draw.text(
                (x + 5, y + 220), f"cos={pair['cosine']:.4f}; dhash={pair['dhash_distance']}", fill="#182938"
            )
        target = DATA / "review" / f"{prefix}-{start // 12 + 1:02d}.jpg"
        target.parent.mkdir(exist_ok=True)
        page.save(target, quality=90)
        outputs.append(str(target))
    return outputs


def overview_pages(rows, prefix):
    outputs = []
    for start in range(0, len(rows), 25):
        page = Image.new("RGB", (1600, 1750), "#edf1f5")
        draw = ImageDraw.Draw(page)
        for i, row in enumerate(rows[start : start + 25]):
            x, y = (i % 5) * 320, (i // 5) * 350
            with Image.open(row["image_path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((310, 310))
                page.paste(image, (x + (320 - image.width) // 2, y + 30 + (310 - image.height) // 2))
            draw.text(
                (x + 5, y + 6),
                f"{row['case_id']} | cluster{row['benchmark_content_cluster']}",
                fill="#182938",
            )
        target = DATA / "review" / f"{prefix}-{start // 25 + 1:02d}.jpg"
        target.parent.mkdir(exist_ok=True)
        page.save(target, quality=90)
        outputs.append(str(target))
    return outputs


def benchmark(phase="all"):
    candidates = json.loads((DATA / "candidate-manifest.json").read_text())["cases"]
    vec = vectors(candidates)
    old = json.loads((PREVIOUS / "manifest.json").read_text())["cases"]
    old_ids = {r["image_id"] for r in old}
    previous_examples = None
    prior_frame_ids = set()
    reviewed_exclusions = set()
    if phase == "confirmation":
        previous_examples = json.loads((DATA / "benchmark-examples-manifest.json").read_text())
        assert previous_examples["status"] == "frozen_after_duplicate_review"
        prior_frame_ids = {r["image_id"] for r in previous_examples["frame"]["cases"]}
        reviewed_exclusions = set(previous_examples.get("excluded_candidate_ids", []))
    index = _HashIndex()
    seen_hashes, seen_pixels, seen_views = set(), set(), set()
    for row in candidates:
        if not row.get("fresh_for_expansion_benchmark") or row["image_id"] in prior_frame_ids:
            index.add(int(row["dhash"], 16), row["image_id"])
            seen_hashes.add(row["source_sha256"])
            seen_pixels.add(row["decoded_sha256"])
            seen_views.add(row["view_sha256"])
    pool, excluded = [], []
    available = [
        r
        for r in candidates
        if r.get("fresh_for_expansion_benchmark")
        and r["image_id"] not in prior_frame_ids
        and r["candidate_id"] not in reviewed_exclusions
    ]
    for row in sorted(available, key=lambda r: stable_key("fresh-benchmark", r["image_id"])):
        duplicate = (
            row["image_id"] in old_ids
            or row["source_sha256"] in seen_hashes
            or row["decoded_sha256"] in seen_pixels
            or row["view_sha256"] in seen_views
        )
        near = index.nearby(int(row["dhash"], 16), 2)
        if duplicate or near:
            excluded.append(
                {"image_id": row["image_id"], "reason": "exact/perceptual family guard", "matches": near}
            )
            continue
        pool.append(row)
        index.add(int(row["dhash"], 16), row["image_id"])
        seen_hashes.add(row["source_sha256"])
        seen_pixels.add(row["decoded_sha256"])
        seen_views.add(row["view_sha256"])
    frame = []
    frame_quotas = (
        [("megalith", 1200), ("vizwiz", 300)] if phase == "all" else [("megalith", 600), ("vizwiz", 150)]
    )
    for category, count in frame_quotas:
        values = [r for r in pool if r["category"] == category]
        if len(values) < count:
            raise ValueError(f"Need {count} fresh {category}, have {len(values)}")
        frame.extend(values[:count])
    stem = "benchmark" if phase == "all" else f"benchmark-{phase}"
    frame_path = DATA / f"{stem}-frame.json"
    if frame_path.exists():
        frozen_frame = json.loads(frame_path.read_text())
        frame = frozen_frame["cases"]
    else:
        write_json(
            frame_path,
            {
                "cases": frame,
                "candidate_manifest_sha256": digest(DATA / "candidate-manifest.json"),
                "selection": "Fixed source quotas and seeded image-ID order before reference/model labels or embedding completion",
                "freshness_exclusions": excluded,
            },
        )
    missing = [r["candidate_id"] for r in frame if r["candidate_id"] not in vec]
    if missing:
        raise ValueError(f"Fixed {phase} frame awaits {len(missing)} embeddings; see {frame_path}")
    matrix = np.stack([vec[r["candidate_id"]] for r in frame])
    cluster_seed = SEED % 2**32
    cluster_count = 32 if phase == "all" else 16
    labels = KMeans(n_clusters=cluster_count, n_init=20, random_state=cluster_seed).fit_predict(matrix)
    clusters = {r["candidate_id"]: int(label) for r, label in zip(frame, labels, strict=True)}
    examples = []
    for category, count in [("megalith", 400), ("vizwiz", 100)]:
        examples.extend(sample([r for r in frame if r["category"] == category], count, clusters, "example"))
    used = {r["candidate_id"] for r in examples}
    confirmation = []
    if phase == "all":
        for category, count in [("megalith", 400), ("vizwiz", 100)]:
            confirmation.extend(
                sample(
                    [r for r in frame if r["category"] == category and r["candidate_id"] not in used],
                    count,
                    clusters,
                    "confirmation",
                )
            )
    elif phase == "confirmation":
        confirmation, examples = examples, []
    selected = []
    for partition, rows, start in [
        ("expansion_examples", examples, 4001),
        ("expansion_confirmation", confirmation, 4501),
    ]:
        for number, row in enumerate(sorted(rows, key=lambda r: stable_key(partition, r["image_id"])), start):
            selected.append(
                {
                    **row,
                    "case_id": f"R{number:04d}",
                    "benchmark_partition": partition,
                    "benchmark_order": number - 3500,
                    "publication_order": number - 3500,
                    "benchmark_content_cluster": clusters[row["candidate_id"]],
                    "historical_exposure": False,
                }
            )
    # Inspect strongest embedding neighbours, including the entire currently embedded background.
    all_embedded = [r for r in candidates if r["candidate_id"] in vec]
    all_matrix = np.stack([vec[r["candidate_id"]] for r in all_embedded])
    lookup = {r["candidate_id"]: r for r in all_embedded}
    lookup.update({r["candidate_id"]: r for r in selected})
    pairs = []
    for start in range(0, len(selected), 100):
        scores = np.stack([vec[r["candidate_id"]] for r in selected[start : start + 100]]) @ all_matrix.T
        for row, similarities in zip(selected[start : start + 100], scores, strict=True):
            for j in np.argsort(similarities)[-7:][::-1]:
                other = all_embedded[j]
                if other["image_id"] == row["image_id"]:
                    continue
                cosine = float(similarities[j])
                if cosine >= 0.94:
                    pairs.append(
                        {
                            "left": row["candidate_id"],
                            "right": other["candidate_id"],
                            "cosine": cosine,
                            "dhash_distance": (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count(),
                            "review_status": "pending",
                            "cross_boundary": other["candidate_id"] not in used,
                        }
                    )
    unique = {}
    for pair in sorted(pairs, key=lambda p: -p["cosine"]):
        unique.setdefault(tuple(sorted((pair["left"], pair["right"]))), pair)
    pairs = list(unique.values())
    pages = review_pages(pairs[:96], lookup, f"{stem}-neighbours")
    overviews = overview_pages(selected, f"{stem}-overview")
    write_json(
        DATA / f"{stem}-duplicate-review.json",
        {
            "status": "pending",
            "threshold": 0.94,
            "pairs": pairs,
            "contact_sheet_pairs": min(len(pairs), 96),
            "contact_sheets": pages,
            "note": "Embedding similarity is a review cue, not proof of duplicate identity.",
        },
    )
    previous_bench = [
        {**r, "benchmark_order": i, "publication_order": i, "historical_exposure": True}
        for i, r in enumerate(old[:500], 1)
    ]
    if previous_examples:
        previous_bench = previous_examples["cases"]
    proposal = {
        "schema_version": 1,
        "status": "awaiting_duplicate_review",
        "cases": previous_bench + selected,
        "counts": {
            "old_references": 500,
            "new_examples": 500,
            "new_confirmation": 500 if phase != "examples" else 0,
        },
        "frame": {
            "cases": frame,
            "clusters": clusters,
            "k": cluster_count,
            "n_init": 20,
            "seed": cluster_seed,
        },
        "excluded": excluded,
        "candidate_manifest_sha256": digest(DATA / "candidate-manifest.json"),
        "selection_uses_model_scores": False,
        "overview_pages": overviews,
        "publication_rule": "benchmark_order <= 1000",
        "freshness": "New acquired families; exact/pixel/view identity and dHash<=2 guards against previous candidate pool",
    }
    if previous_examples:
        proposal["example_manifest_sha256"] = digest(DATA / "benchmark-examples-manifest.json")
        proposal["excluded_candidate_ids"] = sorted(reviewed_exclusions)
        proposal["prior_example_review"] = {
            "duplicate_replacements": previous_examples.get("duplicate_replacements", []),
            "notes": previous_examples.get("duplicate_review_notes"),
        }
    write_json(DATA / f"{stem}-proposal.json", proposal)
    write_json(DATA / f"{stem}-review-image-index.json", lookup)
    print(
        json.dumps({"selected": len(proposal["cases"]), "review_pairs": len(pairs), "pages": pages}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["benchmark"])
    parser.add_argument("--phase", choices=["all", "examples", "confirmation"], default="all")
    benchmark(parser.parse_args().phase)
