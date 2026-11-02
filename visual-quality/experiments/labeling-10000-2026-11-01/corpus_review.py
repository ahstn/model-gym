"""Surface content coverage, related images and cross-family duplicate candidates."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from corpus_common import DATA, digest, write_json
from corpus_select import review_pages
from PIL import Image, ImageDraw


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    cases = manifest["cases"]
    source_candidates = json.loads((DATA / "candidate-manifest.json").read_text())["cases"]
    candidates = {r["image_id"]: r for r in source_candidates}
    cluster_labels = json.loads(Path(__file__).with_name("corpus-cluster-labels.json").read_text())
    selection = json.loads((DATA / "corpus-selection-audit.json").read_text())
    prior_path = DATA / "final-duplicate-review.json"
    prior_review = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    prior_pairs = {
        (row["left"], row["right"]): row
        for row in prior_review.get("pairs", [])
        if prior_review.get("manifest_sha256") == digest(DATA / "manifest.json")
    }
    values = []
    provenance = []
    for row in cases:
        view = row["view_sha256"]
        path = DATA / "embedding-cache" / f"{view}.npy"
        mode = "exact_delivered_view"
        if not path.exists():
            ancestor = candidates[row.get("parent_image_id", row["image_id"])]
            view = ancestor["view_sha256"]
            path = DATA / "embedding-cache" / f"{view}.npy"
            mode = "canonical_family_content_representative"
        values.append(np.load(path, allow_pickle=False))
        provenance.append({"case_id": row["case_id"], "embedding_view_sha256": view, "mode": mode})
    matrix = np.stack(values)
    family_members = defaultdict(list)
    for row in cases:
        family_members[row["family_id"]].append(row["case_id"])
    np.savez_compressed(
        DATA / "corpus-embeddings.npz",
        case_ids=np.asarray([r["case_id"] for r in cases]),
        embeddings=matrix,
        embedding_view_sha256=np.asarray([r["embedding_view_sha256"] for r in provenance]),
        embedding_modes=np.asarray([r["mode"] for r in provenance]),
    )
    related = {}
    duplicates = []
    for start in range(0, len(cases), 128):
        similarities = matrix[start : start + 128] @ matrix.T
        for i, scores in enumerate(similarities, start):
            row = cases[i]
            choices = []
            for j in np.argsort(scores)[-40:][::-1]:
                other = cases[j]
                if i == j or row["family_id"] == other["family_id"]:
                    continue
                choices.append({"case_id": other["case_id"], "cosine": float(scores[j])})
                if len(choices) == 5:
                    break
            related[row["case_id"]] = choices
            for other in choices:
                if other["cosine"] >= 0.985 and row["case_id"] < other["case_id"]:
                    target = next(r for r in cases if r["case_id"] == other["case_id"])
                    duplicates.append(
                        {
                            "left": row["case_id"],
                            "right": other["case_id"],
                            "cosine": other["cosine"],
                            "dhash_distance": (int(row["dhash"], 16) ^ int(target["dhash"], 16)).bit_count(),
                            "cross_partition": row["benchmark_partition"] != target["benchmark_partition"],
                            "review_status": "pending",
                            **{
                                key: value
                                for key, value in prior_pairs.get(
                                    (row["case_id"], other["case_id"]), {}
                                ).items()
                                if key in {"review_status", "evidence"}
                            },
                        }
                    )
    duplicates.sort(key=lambda r: (not r["cross_partition"], -r["cosine"]))
    lookup = {r["case_id"]: r for r in cases}
    pages = review_pages(duplicates[:144], lookup, "final-duplicate-candidates")
    groups = defaultdict(list)
    for i, row in enumerate(cases):
        groups[row["content_cluster"]].append(i)
    exemplars = []
    for cluster, indices in sorted(groups.items()):
        centroid = matrix[indices].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        order = np.argsort(matrix[indices] @ centroid)[::-1]
        chosen = []
        families = set()
        for pos in order:
            i = indices[pos]
            if cases[i]["family_id"] not in families:
                chosen.append(i)
                families.add(cases[i]["family_id"])
            if len(chosen) == 6:
                break
        exemplars.append(
            {
                "cluster": cluster,
                "theme": cluster_labels["clusters"][str(cluster)],
                "image_count": len(indices),
                "image_fraction": len(indices) / len(cases),
                "family_count": len({cases[i]["family_id"] for i in indices}),
                "candidate_count": selection["candidate_cluster_sizes"][str(cluster)],
                "candidate_fraction": selection["candidate_cluster_sizes"][str(cluster)]
                / len(source_candidates),
                "case_ids": [cases[i]["case_id"] for i in chosen],
            }
        )
    for start in range(0, len(exemplars), 8):
        page = Image.new("RGB", (1440, 8 * 210), "#edf1f5")
        draw = ImageDraw.Draw(page)
        for line, group in enumerate(exemplars[start : start + 8]):
            y = line * 210
            draw.text(
                (5, y + 3),
                f"Cluster {group['cluster']} | n={group['image_count']} | {group['family_count']} families",
                fill="#182938",
            )
            for col, case_id in enumerate(group["case_ids"]):
                row = lookup[case_id]
                with Image.open(row["image_path"]) as image:
                    image = image.convert("RGB")
                    image.thumbnail((230, 165))
                    page.paste(image, (col * 240 + (230 - image.width) // 2, y + 23))
                draw.text((col * 240 + 4, y + 190), case_id, fill="#182938")
        page.save(DATA / "review" / f"cluster-exemplars-{start // 8 + 1:02d}.jpg", quality=90)
    write_json(
        DATA / "related-images.json",
        {
            "manifest_sha256": digest(DATA / "manifest.json"),
            "neighbours": related,
            "embedding_provenance": provenance,
            "family_members": {key: value for key, value in family_members.items() if len(value) > 1},
            "note": "Content similarity does not imply equal image quality; variants use a family representative when recorded",
        },
    )
    write_json(
        DATA / "final-duplicate-review.json",
        {
            "status": "pending"
            if any(row["review_status"] == "pending" for row in duplicates)
            else "reviewed_with_documented_uncertainties",
            "manifest_sha256": digest(DATA / "manifest.json"),
            "pairs": duplicates,
            "contact_sheets": pages,
            "review_limit": 144,
            "all_pairs_recorded": len(duplicates),
            "scope": "Cross-family nearest neighbours; same-family controlled variants excluded",
            "removed_confirmed_capture_pairs": manifest.get("final_review", {}).get(
                "replaced_incoming_rehearsal_views", 0
            ),
        },
    )
    write_json(
        DATA / "content-coverage.json",
        {
            "clusters": exemplars,
            "largest_clusters": sorted(exemplars, key=lambda r: -r["image_count"])[:10],
            "embedding_modes": dict(Counter(r["mode"] for r in provenance)),
            "categories": dict(Counter(r["category"] for r in cases)),
            "cluster_theme_method": cluster_labels["method"],
            "coverage_note": "Largest clusters flag common themes for review. Candidate counts describe canonical source candidates, whereas selected counts also include controlled derivatives; these denominators differ.",
        },
    )
    print(
        json.dumps(
            {
                "related_images": len(related),
                "duplicate_candidates": len(duplicates),
                "clusters": len(exemplars),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
