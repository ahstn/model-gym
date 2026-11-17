# ruff: noqa: I001
# Imports follow the explicit legacy run-context bootstrap order.
"""Select an incremental content-balanced corpus without consulting quality scores."""

from __future__ import annotations
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
import numpy as np
from PIL import Image
import corpus
from corpus import DATA, OLD, SEED, digest, stable_key, write_json
import corpus_finalize as legacy
from corpus_select import sample


def load_vectors(rows):
    values = np.stack(
        [np.load(DATA / "embedding-cache" / (r["view_sha256"] + ".npy"), allow_pickle=False) for r in rows]
    )
    assert values.shape == (len(rows), 768) and np.isfinite(values).all()
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)
    return values


def quotas(pool, count, old, clusters, tag):
    groups = defaultdict(list)
    for row in pool:
        groups[clusters[row["candidate_id"]]].append(row)
    prior = Counter(r["content_cluster"] for r in old)
    chosen = Counter()
    if len(pool) < count:
        raise ValueError(f"{tag}: need {count}, available {len(pool)}")
    for _ in range(count):
        k = max(
            (k for k in groups if chosen[k] < len(groups[k])),
            key=lambda k: ((len(groups[k]) + prior[k]) ** 0.5 / (prior[k] + chosen[k] + 1), -k),
        )
        chosen[k] += 1
    return [
        r
        for k in sorted(groups)
        for r in sorted(groups[k], key=lambda r: stable_key(tag, r["image_id"]))[: chosen[k]]
    ]


def main():
    if (DATA / "manifest.json").exists():
        raise ValueError("Selection already frozen; do not replace a scored manifest")
    old = corpus.old_cases()
    pool = json.loads((DATA / "candidate-manifest.json").read_text())["cases"]
    z = np.load(OLD / "corpus-embeddings.npz", allow_pickle=False)
    assert z["case_ids"].tolist() == [r["case_id"] for r in old]
    ov = z["embeddings"]
    nv = load_vectors(pool)
    # Freeze the initial 64 content groups: normalized mean of unique-family
    # exemplars. This retains the reviewed theme IDs without fitting on labels.
    centers = []
    for k in range(64):
        seen = set()
        indices = []
        for i, row in enumerate(old):
            if row["content_cluster"] == k and row["family_id"] not in seen:
                indices.append(i)
                seen.add(row["family_id"])
        c = ov[indices].mean(axis=0)
        centers.append(c / np.linalg.norm(c))
    centers = np.stack(centers)
    labels = (nv @ centers.T).argmax(axis=1)
    clusters = {r["candidate_id"]: int(k) for r, k in zip(pool, labels, strict=True)}
    candidates, excluded, related = [], [], []
    allv = np.concatenate([ov, nv])
    allrows = old + pool
    for start in range(0, len(pool), 128):
        sims = nv[start : start + 128] @ allv.T
        for offset, similarities in enumerate(sims):
            index = start + offset
            row = pool[index]
            similarities[len(old) + index] = -1
            neighbors = np.argsort(similarities)[-6:][::-1]
            matches = []
            block = False
            for j in neighbors:
                other = allrows[j]
                cosine = float(similarities[j])
                distance = (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count()
                if cosine >= 0.97:
                    matches.append(
                        {
                            "left": row["candidate_id"],
                            "right": other.get("case_id", other["candidate_id"]),
                            "cosine": cosine,
                            "dhash_distance": distance,
                        }
                    )
                if cosine >= 0.985 and distance <= 6 and j < len(old) + index:
                    block = True
            related.extend(matches)
            if block:
                excluded.append(row["candidate_id"])
            else:
                candidates.append(row)
    available = Counter(r["category"] for r in candidates)
    control_count = min(250, available["controls"])
    targets = {"megalith": 7750 - control_count, "vizwiz": 1500, "controls": control_count}
    chosen = []
    for category, count in targets.items():
        chosen += quotas(
            [r for r in candidates if r["category"] == category],
            count,
            [r for r in old if r["category"] == category],
            clusters,
            "select-" + category,
        )
    parents = sample(
        [r for r in chosen if r["category"] == "megalith"], 250, clusters, "variant-parents", sqrt=True
    )
    parent_ids = {r["image_id"] for r in parents}
    final = [dict(r) for r in chosen if r["image_id"] not in parent_ids]
    with ThreadPoolExecutor(8) as pool:
        for i, members in enumerate(pool.map(legacy.family, parents), 1):
            final.extend(members)
            if i % 25 == 0:
                print(f"Generated controlled families {i}/250", flush=True)
    assert len(final) == 10000
    final.sort(key=lambda r: stable_key("case-order", r["image_id"]))
    for i, row in enumerate(final, 10001):
        row.update(
            case_id=f"R{i:05d}",
            benchmark_partition="rehearsal",
            benchmark_order=None,
            publication_order=None,
            content_cluster=clusters[row.get("parent_candidate_id") or row["candidate_id"]],
            expansion_run="2026-11-16",
        )
        if row["category"] == "distortions":
            with Image.open(row["source_path"]) as image:
                row["source_format"] = image.format
                row["source_metadata"] = {
                    "format": image.format,
                    "frames": getattr(image, "n_frames", 1),
                    "icc": bool(image.info.get("icc_profile")),
                    "alpha": "A" in image.getbands() or "transparency" in image.info,
                }
        row["embedding_view_sha256"] = row.get("embedding_view_sha256", row["view_sha256"])
        row["embedding_mode"] = (
            "exact_view" if row["embedding_view_sha256"] == row["view_sha256"] else "family_representative"
        )
    assert len({r["image_id"] for r in final}) == len(final) == len({r["view_sha256"] for r in final})
    assert not ({r["view_sha256"] for r in final} & {r["view_sha256"] for r in old})
    assert not ({r["family_id"] for r in final} & {r["family_id"] for r in old})
    manifest = {
        "schema_version": "labeling-20000-increment-v1",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "old_manifest_sha256": digest(OLD / "manifest.json"),
        "candidate_manifest_sha256": digest(DATA / "candidate-manifest.json"),
        "selection_policy": "Frozen 64 initial-corpus content prototypes; square-root pooled-content quotas offset by initial counts, per source; score-blind seeded selection; conservative identity/perceptual exclusions. No new reference benchmark.",
        "counts": dict(Counter(r["category"] for r in final)),
        "cases": final,
        "training_ready": False,
    }
    write_json(DATA / "manifest.json", manifest)
    write_json(
        DATA / "combined-manifest.json",
        {
            "schema_version": "labeling-20000-combined-v1",
            "initial_manifest_sha256": digest(OLD / "manifest.json"),
            "increment_manifest_sha256": digest(DATA / "manifest.json"),
            "cases": old + final,
        },
    )
    write_json(
        DATA / "candidate-related-review.json",
        {
            "threshold": 0.97,
            "automatic_exclusion": "cosine >= .985 and dhash <= 6; conservative sample exclusion, not a duplicate assertion",
            "excluded_candidates": excluded,
            "pairs": related,
        },
    )
    themes = {
        r["cluster"]: r["theme"] for r in json.loads((OLD / "content-coverage.json").read_text())["clusters"]
    }
    coverage = []
    for k in range(64):
        coverage.append(
            {
                "cluster": k,
                "theme": themes[k],
                "theme_provenance": "initial10K reviewed theme, assigned via frozen prototype",
                "candidate_count": sum(int(l) == k for l in labels),
                "initial_count": sum(r["content_cluster"] == k for r in old),
                "new_count": sum(r["content_cluster"] == k for r in final),
            }
        )
    write_json(
        DATA / "content-coverage.json",
        {
            "clusters": coverage,
            "categories": manifest["counts"],
            "embedding_model": "google/embeddinggemma-2",
            "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
            "embedding_modes": dict(Counter(r["embedding_mode"] for r in final)),
        },
    )
    np.savez_compressed(DATA / "cluster-prototypes.npz", centroids=centers)
    vectors = np.stack(
        [
            np.load(DATA / "embedding-cache" / (r["embedding_view_sha256"] + ".npy"), allow_pickle=False)
            for r in final
        ]
    )
    np.savez_compressed(
        DATA / "corpus-embeddings.npz",
        case_ids=np.asarray([r["case_id"] for r in old + final]),
        embeddings=np.concatenate([ov, vectors]),
        embedding_view_sha256=np.concatenate(
            [z["embedding_view_sha256"], np.asarray([r["embedding_view_sha256"] for r in final])]
        ),
    )
    print(manifest["counts"], flush=True)


if __name__ == "__main__":
    main()
