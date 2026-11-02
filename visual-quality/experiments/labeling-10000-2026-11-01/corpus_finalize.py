"""Extend the frozen 4,000 views using content-balanced selection and new families."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from corpus_common import DATA, PREVIOUS, SEED, digest, stable_key, write_json
from corpus_select import sample, vectors
from sklearn.cluster import KMeans

from visual_quality.generate import _parent_view, make_variants
from visual_quality.validation import _HashIndex, inspect_image
from visual_quality.views import _render


def family(parent):
    source = Path(parent["source_path"])
    assert digest(source) == parent["source_sha256"]
    original, canonical_preprocessing = _parent_view(source)
    stream = io.BytesIO()
    original.save(stream, format="PNG", compress_level=6)
    canonical_bytes = stream.getvalue()
    canonical_sha = hashlib.sha256(canonical_bytes).hexdigest()
    canonical_path = DATA / "family-canonical" / f"{canonical_sha}.png"
    canonical_path.parent.mkdir(exist_ok=True)
    canonical_path.write_bytes(canonical_bytes)
    input_parent = {
        **parent,
        "id": parent["image_id"],
        "image_path": parent["source_path"],
        "source_group": parent["family_id"],
    }
    variants = make_variants(DATA, [input_parent], SEED)
    payloads = [canonical_bytes] + [Path(r["image_path"]).read_bytes() for r in variants]
    edge = 1536
    while True:
        rendered = [_render(value, edge) for value in payloads]
        if all(len(payload) <= 2650000 for payload, _ in rendered):
            break
        edge = edge * 9 // 10
        if edge < 256:
            raise ValueError("Controlled family cannot satisfy PNG byte budget")
    if any(
        min(prep["output_dimensions"]) < 256 or np.prod(prep["output_dimensions"]) < 78400
        for _, prep in rendered
    ):
        raise ValueError("Controlled family dimensions below common contract")
    if len({tuple(prep["output_dimensions"]) for _, prep in rendered}) != 1:
        raise ValueError("Controlled family members have unequal viewing scale")
    records = []
    for i, (payload, prep) in enumerate(rendered):
        view_sha = hashlib.sha256(payload).hexdigest()
        path = DATA / "views" / f"{view_sha}.png"
        path.write_bytes(payload)
        row = {
            **parent,
            "image_path": str(path),
            "view_sha256": view_sha,
            "preprocessing": prep,
            "width": prep["output_dimensions"][0],
            "height": prep["output_dimensions"][1],
            "delivered_bytes": len(payload),
            "target_edge": edge,
            "benchmark_partition": "rehearsal",
            "benchmark_order": None,
            "publication_order": None,
            "stratum": "controlled",
            "role": "source_parent" if i == 0 else "variant",
            "controlled_parent": i == 0,
            "operator": variants[0]["transform"]["operator"],
            "severity": None if i == 0 else i,
            "embedding_view_sha256": parent["view_sha256"],
            "family_preprocessing": {
                "canonical_parent_path": str(canonical_path),
                "canonical_parent_sha256": canonical_sha,
                "canonical_parent": canonical_preprocessing,
                "shared_final_edge": edge,
                "final_parent_view_sha256": hashlib.sha256(rendered[0][0]).hexdigest(),
            },
        }
        if i:
            variant = variants[i - 1]
            checked = inspect_image({**variant, "component": "distortions"}, DATA)
            row.update(
                image_id=variant["id"],
                source_id=variant["source_id"],
                source=variant["source"],
                source_revision=variant["source_revision"],
                source_path=variant["image_path"],
                source_sha256=variant["sha256"],
                decoded_sha256=checked["decoded_sha256"],
                dhash=checked["dhash"],
                component="distortions",
                category="distortions",
                candidate_id=None,
                parent_candidate_id=parent["candidate_id"],
                parent_image_id=parent["image_id"],
                transform=variant["transform"],
            )
        records.append(row)
    assert len({r["view_sha256"] for r in records}) == 4
    return records


def main():
    benchmark = json.loads((DATA / "benchmark-manifest.json").read_text())
    if benchmark["status"] != "frozen_after_duplicate_review":
        raise ValueError("Benchmark needs a completed independent duplicate review")
    previous = json.loads((PREVIOUS / "manifest.json").read_text())
    old = previous["cases"]
    candidates = json.loads((DATA / "candidate-manifest.json").read_text())["cases"]
    vec = vectors(candidates)
    if len(vec) < len(candidates):
        raise ValueError(f"Candidate embeddings incomplete: {len(vec)}/{len(candidates)}")
    matrix = np.stack([vec[r["candidate_id"]] for r in candidates])
    labels = KMeans(n_clusters=64, n_init=20, random_state=SEED % 2**32).fit_predict(matrix)
    clusters = {r["candidate_id"]: int(label) for r, label in zip(candidates, labels, strict=True)}
    old_ids = {r["image_id"] for r in old}
    benchmark_new = [r for r in benchmark["cases"] if r["image_id"] not in old_ids]
    kept = [
        {
            **r,
            "benchmark_order": i if i <= 500 else None,
            "publication_order": i if i <= 500 else None,
            "historical_exposure": True,
        }
        for i, r in enumerate(old, 1)
    ] + benchmark_new
    index = _HashIndex()
    sources, pixels, views, families = set(), set(), set(), set()
    for row in kept:
        sources.add(row["source_sha256"])
        pixels.add(row["decoded_sha256"])
        views.add(row["view_sha256"])
        families.add(row["family_id"])
        index.add(int(row["dhash"], 16), row["image_id"])
    rejected = []
    pool = []
    excluded = set(benchmark.get("excluded_candidate_ids", []))
    for row in sorted(candidates, key=lambda r: stable_key("corpus-candidate", r["image_id"])):
        if row["candidate_id"] in excluded or row["family_id"] in families:
            continue
        exact = (
            row["source_sha256"] in sources or row["decoded_sha256"] in pixels or row["view_sha256"] in views
        )
        near = index.nearby(int(row["dhash"], 16), 2) if row["category"] != "controls" else []
        if exact or near:
            rejected.append(
                {"candidate_id": row["candidate_id"], "reason": "exact/perceptual guard", "near": near}
            )
            continue
        pool.append(row)
        sources.add(row["source_sha256"])
        pixels.add(row["decoded_sha256"])
        views.add(row["view_sha256"])
        families.add(row["family_id"])
        if row["category"] != "controls":
            index.add(int(row["dhash"], 16), row["image_id"])
    controls_pool = [
        r
        for r in pool
        if r["category"] == "megalith"
        and not r["source_metadata"]["icc"]
        and not r["source_metadata"]["alpha"]
    ]
    parents = sample(controls_pool, 180, clusters, "controlled-parent", sqrt=True)
    controlled = []
    used = set()
    family_failures = []
    for parent in parents:
        try:
            members = family(parent)
        except (ValueError, OSError) as error:
            family_failures.append({"image_id": parent["image_id"], "error": str(error)})
            continue
        controlled.extend(members)
        used.add(parent["image_id"])
        if len(used) == 150:
            break
    if len(used) != 150:
        raise ValueError("Could not construct150 additional controlled families")
    additions = controlled
    for category, total in [("megalith", 7500), ("vizwiz", 1500), ("controls", 250)]:
        already = sum(r["category"] == category for r in kept + additions)
        available = [r for r in pool if r["category"] == category and r["image_id"] not in used]
        selected = sample(available, total - already, clusters, f"final-{category}", sqrt=True)
        additions.extend(
            {
                **r,
                "benchmark_partition": "rehearsal",
                "benchmark_order": None,
                "publication_order": None,
                "role": "standalone",
                "controlled_parent": False,
                "operator": None,
                "severity": None,
            }
            for r in selected
        )
        used.update(r["image_id"] for r in selected)
    for i, row in enumerate(sorted(additions, key=lambda r: stable_key("final-order", r["image_id"])), 5001):
        row["case_id"] = f"R{i:04d}"
        kept.append(row)
    assert len(kept) == 10000 and len({r["case_id"] for r in kept}) == 10000
    assert len({r["view_sha256"] for r in kept}) == 10000
    assert len({r["family_id"] for r in kept}) == 9250
    assert sum(r.get("benchmark_order") is not None for r in kept) == 1500
    for row in kept:
        candidate = next(
            (c for c in candidates if c["image_id"] == row.get("parent_image_id", row["image_id"])), None
        )
        if candidate:
            row["content_cluster"] = clusters[candidate["candidate_id"]]
        assert digest(row["image_path"]) == row["view_sha256"]
    result = {
        "schema_version": 1,
        "status": "selected_pending_final_duplicate_audit",
        "seed": SEED,
        "cases": kept,
        "counts": {
            "images": 10000,
            "families": 9250,
            "benchmark_images": 1500,
            "categories": dict(Counter(r["category"] for r in kept)),
            "partitions": dict(Counter(r["benchmark_partition"] for r in kept)),
        },
        "preprocessing_policy": previous["preprocessing_policy"],
        "inputs": {
            "previous_manifest_sha256": digest(PREVIOUS / "manifest.json"),
            "candidate_manifest_sha256": digest(DATA / "candidate-manifest.json"),
            "benchmark_manifest_sha256": digest(DATA / "benchmark-manifest.json"),
            "finalize_script_sha256": digest(__file__),
        },
        "selection_policy": {
            "clusters": 64,
            "n_init": 20,
            "method": "Capacity-capped sqrt(cluster-size) quotas with seeded within-cluster selection",
            "labels_or_scores_used": False,
            "preserved_previous_views": 4000,
        },
        "training_status": "All task masks false; observations need review",
    }
    write_json(DATA / "manifest.json", result)
    write_json(
        DATA / "corpus-selection-audit.json",
        {
            "rejected": rejected,
            "family_failures": family_failures,
            "candidate_cluster_sizes": dict(Counter(int(v) for v in labels)),
            "selected_cluster_sizes": dict(Counter(r.get("content_cluster") for r in kept)),
            "candidate_clusters": clusters,
            "manifest_sha256": digest(DATA / "manifest.json"),
        },
    )
    print(json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()
