# ruff: noqa: I001
# Imports follow the explicit legacy run-context bootstrap order.
"""Select an incremental content-balanced corpus without consulting quality scores."""

from __future__ import annotations
import json
import hashlib
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
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


def load_prototypes(path):
    with np.load(path, allow_pickle=False) as archive:
        centers = archive["centroids"].copy()
    if centers.shape != (64, 768) or not np.isfinite(centers).all():
        raise ValueError("Expected the frozen 64 original content prototypes")
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1, atol=1e-5)
    return centers


def exclude_semantic_duplicates(old, pool, old_vectors, new_vectors):
    """Compare all strong neighbors, not just six, before conservative exclusion."""
    candidates, excluded, related = [], [], []
    all_vectors = np.concatenate([old_vectors, new_vectors])
    all_rows = old + pool
    with threadpool_limits(limits=8):
        for start in range(0, len(pool), 128):
            similarities = new_vectors[start : start + 128] @ all_vectors.T
            for offset, values in enumerate(similarities):
                index = start + offset
                row = pool[index]
                values[len(old) + index] = -1
                strong = np.flatnonzero(values >= 0.985)
                blocked = any(
                    j < len(old) + index
                    and (int(row["dhash"], 16) ^ int(all_rows[j]["dhash"], 16)).bit_count() <= 6
                    for j in strong
                )
                n = min(6, len(values))
                nearest = np.argpartition(values, -n)[-n:]
                for j in nearest[np.argsort(values[nearest])[::-1]]:
                    if values[j] < 0.97:
                        continue
                    other = all_rows[j]
                    related.append(
                        {
                            "left": row["candidate_id"],
                            "right": other.get("case_id") or other["candidate_id"],
                            "cosine": float(values[j]),
                            "dhash_distance": (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count(),
                        }
                    )
                if blocked:
                    excluded.append(row["candidate_id"])
                else:
                    candidates.append(row)
    return candidates, excluded, related


def historical_matcher(old):
    """Build the same identity/perceptual boundary for generated family members."""
    protected = corpus.read_lines(corpus.DATA_ROOT / "pilot-10k-v1/development.jsonl") + corpus.read_lines(
        corpus.DATA_ROOT / "pilot-10k-v1/test.jsonl"
    )
    fields = {
        "image_id": "id",
        "family_id": "split_group",
        "source_id": "source_id",
        "source_sha256": "sha256",
        "decoded_sha256": "decoded_sha256",
    }
    historical = {
        field: {row[field] for row in old} | {row[other] for row in protected}
        for field, other in fields.items()
    }
    historical["view_sha256"] = {row["view_sha256"] for row in old}
    index = corpus._HashIndex()
    for row in old + protected:
        index.add(int(row["dhash"], 16), row.get("image_id", row.get("id")))

    def match(row):
        for field, values in historical.items():
            if row[field] in values:
                return f"historical/protected {field}"
        nearby = index.nearby(int(row["dhash"], 16), 2)
        if nearby:
            return "historical/protected perceptual match: " + ", ".join(nearby)
        return None

    return match


def generate_families(chosen, clusters, old, count=500):
    """Replace failed/overlapping variants with deterministic alternative parents."""
    # The legacy generator strips metadata before RGB conversion. ICC/alpha
    # sources remain eligible base images, but cannot be controlled parents.
    available = [
        r
        for r in chosen
        if r["category"] == "megalith"
        and r.get("source_metadata", {}).get("icc") is False
        and r.get("source_metadata", {}).get("alpha") is False
    ]
    accepted, rejects = {}, []
    matches = historical_matcher(old)
    view_families = {r["view_sha256"]: r["family_id"] for r in chosen}
    pixel_families = {r["decoded_sha256"]: r["family_id"] for r in chosen}

    def render(parent):
        try:
            return legacy.family(parent), None
        except (OSError, ValueError, AssertionError) as error:
            return [], f"{type(error).__name__}: {error}"

    while len(accepted) < count:
        parents = sample(available, count - len(accepted), clusters, "variant-parents", sqrt=True)
        attempted = {r["image_id"] for r in parents}
        available = [r for r in available if r["image_id"] not in attempted]
        with ThreadPoolExecutor(8) as executor:
            for parent, (members, reason) in zip(parents, executor.map(render, parents), strict=True):
                for member in members:
                    reason = reason or matches(member)
                    for field, owners in (("view_sha256", view_families), ("decoded_sha256", pixel_families)):
                        owner = owners.get(member[field])
                        if owner is not None and owner != member["family_id"]:
                            reason = reason or f"new cross-family duplicate {field}"
                if reason:
                    rejects.append({"image_id": parent["image_id"], "reason": reason})
                    continue
                accepted[parent["image_id"]] = members
                for member in members:
                    view_families[member["view_sha256"]] = member["family_id"]
                    pixel_families[member["decoded_sha256"]] = member["family_id"]
                if len(accepted) % 25 == 0:
                    print(f"Generated controlled families {len(accepted)}/{count}", flush=True)
        corpus.write_lines(DATA / "family-generation-rejections.jsonl", rejects)
    final = [dict(r) for r in chosen if r["image_id"] not in accepted]
    final.extend(member for members in accepted.values() for member in members)
    return final


def validate_final(rows, old):
    matches = historical_matcher(old)
    collisions = [{"image_id": row["image_id"], "reason": reason} for row in rows if (reason := matches(row))]
    if collisions:
        write_json(DATA / "final-historical-rejections.json", collisions)
        raise ValueError(f"Final corpus has {len(collisions)} historical/protected matches")


def publish_selection(manifest, old, coverage, review, old_vectors, old_embedding_views):
    """Stage every dependent artifact, then publish the frozen manifest last."""
    with tempfile.TemporaryDirectory(prefix=".selection-", dir=DATA) as temporary:
        stage = corpus.Path(temporary)
        write_json(stage / "manifest.json", manifest)
        write_json(
            stage / "combined-manifest.json",
            {
                "schema_version": "labeling-40000-combined-v1",
                "initial_manifest_sha256": manifest["old_manifest_sha256"],
                "previous_combined_manifest_sha256": manifest["old_manifest_sha256"],
                "increment_manifest_sha256": digest(stage / "manifest.json"),
                "cases": old + manifest["cases"],
            },
        )
        write_json(stage / "content-coverage.json", coverage)
        write_json(stage / "candidate-related-review.json", review)
        shutil.copyfile(OLD / "cluster-prototypes.npz", stage / "cluster-prototypes.npz")
        vectors = np.stack(
            [
                np.load(
                    DATA / "embedding-cache" / (row["embedding_view_sha256"] + ".npy"), allow_pickle=False
                )
                for row in manifest["cases"]
            ]
        )
        np.savez_compressed(
            stage / "corpus-embeddings.npz",
            case_ids=np.asarray([row["case_id"] for row in old + manifest["cases"]]),
            embeddings=np.concatenate([old_vectors, vectors]),
            embedding_view_sha256=np.concatenate(
                [old_embedding_views, np.asarray([row["embedding_view_sha256"] for row in manifest["cases"]])]
            ),
        )
        with corpus.publication_lock():
            if (DATA / "manifest.json").exists():
                raise ValueError("Selection was frozen by another process; staged outputs discarded")
            if digest(DATA / "candidate-manifest.json") != manifest["candidate_manifest_sha256"]:
                raise ValueError("Candidate frame changed during selection; staged outputs discarded")
            corpus.pin_initial_evidence()
            for path in sorted(stage.iterdir()):
                if path.name != "manifest.json":
                    path.replace(DATA / path.name)
            # This is the completion marker. A failure before this rename leaves
            # the run unpublished and safely rerunnable, even after partial moves.
            (stage / "manifest.json").replace(DATA / "manifest.json")


def main():
    if (DATA / "manifest.json").exists():
        raise ValueError("Selection already frozen; do not replace a scored manifest")
    corpus.pin_initial_evidence()
    old = corpus.old_cases()
    candidate_bytes = (DATA / "candidate-manifest.json").read_bytes()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    pool = json.loads(candidate_bytes)["cases"]
    embedding_receipt = json.loads((DATA / "embedding-operations/provenance.json").read_text())
    if embedding_receipt["candidate_manifest_sha256"] != candidate_sha:
        raise ValueError("Run embedding pass on final candidate manifest before selection")
    z = np.load(OLD / "corpus-embeddings.npz", allow_pickle=False)
    assert z["case_ids"].tolist() == [r["case_id"] for r in old]
    ov = z["embeddings"]
    nv = load_vectors(pool)
    # The saved prototypes were derived from the initial 10K. Recomputing means
    # on the larger corpus would drift both the content IDs and reviewed themes.
    centers = load_prototypes(OLD / "cluster-prototypes.npz")
    labels = (nv @ centers.T).argmax(axis=1)
    clusters = {r["candidate_id"]: int(k) for r, k in zip(pool, labels, strict=True)}
    candidates, excluded, related = exclude_semantic_duplicates(old, pool, ov, nv)
    available = Counter(r["category"] for r in candidates)
    control_count = min(500, available["controls"])
    targets = {"megalith": 15500 - control_count, "vizwiz": 3000, "controls": control_count}
    chosen = []
    for category, count in targets.items():
        chosen += quotas(
            [r for r in candidates if r["category"] == category],
            count,
            [r for r in old if r["category"] == category],
            clusters,
            "select-" + category,
        )
    final = generate_families(chosen, clusters, old)
    assert len(final) == 20000
    final.sort(key=lambda r: stable_key("case-order", r["image_id"]))
    for i, row in enumerate(final, 20001):
        row.update(
            case_id=f"R{i:05d}",
            benchmark_partition="rehearsal",
            benchmark_order=None,
            publication_order=None,
            content_cluster=clusters[row.get("parent_candidate_id") or row["candidate_id"]],
            expansion_run="2026-11-17",
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
    assert len({r["case_id"] for r in old + final}) == 40000
    validate_final(final, old)
    manifest = {
        "schema_version": "labeling-40000-increment-v1",
        "status": "frozen",
        "frozen_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "old_manifest_sha256": digest(OLD / "combined-manifest.json"),
        "candidate_manifest_sha256": candidate_sha,
        "prototype_source_sha256": digest(OLD / "cluster-prototypes.npz"),
        "requested_counts": {"megalith": 15000, "vizwiz": 3000, "controls": 500, "distortions": 1500},
        "control_shortfall_replaced_by_megalith": 500 - control_count,
        "selection_policy": "Frozen 64 initial-corpus content prototypes; square-root pooled-content quotas offset by previous 20K counts, per source; score-blind seeded selection; conservative identity/perceptual exclusions. No new reference benchmark.",
        "counts": dict(Counter(r["category"] for r in final)),
        "cases": final,
        "training_ready": False,
    }
    review = {
        "threshold": 0.97,
        "automatic_exclusion": "cosine >= .985 and dhash <= 6; conservative sample exclusion, not a duplicate assertion",
        "excluded_candidates": excluded,
        "pairs": related,
    }
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
                "combined_share": sum(r["content_cluster"] == k for r in old + final) / 40000,
            }
        )
    publish_selection(
        manifest,
        old,
        {
            "clusters": coverage,
            "categories": manifest["counts"],
            "embedding_model": "google/embeddinggemma-2",
            "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
            "embedding_modes": dict(Counter(r["embedding_mode"] for r in final)),
        },
        review,
        ov,
        z["embedding_view_sha256"],
    )
    print(manifest["counts"], flush=True)


if __name__ == "__main__":
    main()
