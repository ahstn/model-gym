"""Surface cross-family neighbors and content coverage for local corpus review."""

# ruff: noqa: I001 -- bootstrap the frozen adapter context first.
import html
import json
import os
import numpy as np
from threadpoolctl import threadpool_limits
from corpus import DATA, digest, write_json


def main():
    rows = json.loads((DATA / "combined-manifest.json").read_text())["cases"]
    z = np.load(DATA / "corpus-embeddings.npz", allow_pickle=False)
    assert z["case_ids"].tolist() == [r["case_id"] for r in rows]
    vectors = z["embeddings"]
    neighbors = {}
    pairs = []
    with threadpool_limits(limits=8):
        for start in range(0, len(rows), 128):
            sims = vectors[start : start + 128] @ vectors.T
            for offset, values in enumerate(sims):
                i = start + offset
                row = rows[i]
                # More than six candidates are retrieved so controlled siblings
                # can be removed while retaining six other-family neighbors.
                best = np.argpartition(values, -24)[-24:]
                best = best[np.argsort(values[best])[::-1]]
                related = []
                for j in best:
                    other = rows[j]
                    if other["family_id"] == row["family_id"]:
                        continue
                    cosine = float(values[j])
                    distance = (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count()
                    related.append(
                        {"case_id": other["case_id"], "cosine": cosine, "dhash_distance": distance}
                    )
                    if j > i and cosine >= 0.97:
                        pairs.append(
                            {
                                "left": row["case_id"],
                                "right": other["case_id"],
                                "cosine": cosine,
                                "dhash_distance": distance,
                                "status": "candidate_for_review_not_confirmed_duplicate",
                            }
                        )
                    if len(related) == 6:
                        break
                neighbors[row["case_id"]] = related
    write_json(
        DATA / "related-images.json",
        {
            "schema_version": "related-20000-v1",
            "manifest_sha256": digest(DATA / "combined-manifest.json"),
            "embeddings_sha256": digest(DATA / "corpus-embeddings.npz"),
            "scope": "six cross-family content neighbors; semantic proximity is not a quality label or confirmed duplication",
            "neighbors": neighbors,
        },
    )
    pairs.sort(key=lambda p: (-p["cosine"], p["dhash_distance"], p["left"]))
    write_json(
        DATA / "possible-duplicates.json",
        {"threshold": 0.97, "pairs": pairs, "status": "review_candidates", "training_ready": False},
    )
    directory = DATA / "review"
    directory.mkdir(exist_ok=True)
    lookup = {r["case_id"]: r for r in rows}
    coverage = json.loads((DATA / "content-coverage.json").read_text())
    lines = [
        '<!doctype html><meta charset="utf-8"><title>20K corpus review</title><style>body{font:16px system-ui;max-width:1250px;margin:auto;padding:24px;background:#f1f5f9}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}.pairs{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}article{background:white;padding:12px}img{width:47%;height:230px;object-fit:contain}a{color:#174a70}</style><h1>20K corpus review</h1><p>Semantic neighbors support content review; they do not imply equal quality or prove duplication. Training masks remain disabled. Original reference and publication membership are unchanged.</p><h2>Content coverage</h2><table><tr><th>Cluster</th><th>Initial theme</th><th>Initial 10K</th><th>New 10K</th><th>Combined</th></tr>'
    ]
    for c in sorted(coverage["clusters"], key=lambda c: -(c["initial_count"] + c["new_count"])):
        lines.append(
            f"<tr><td>{c['cluster']}</td><td>{html.escape(c['theme'])}</td><td>{c['initial_count']}</td><td>{c['new_count']}</td><td>{c['initial_count'] + c['new_count']}</td></tr>"
        )
    lines += [
        "</table>",
        f'<h2>Highest-similarity review candidates</h2><p>Showing at most 200 of {len(pairs):,} candidate pairs. Complete neighbors and candidates are in the adjacent JSON files. No pair below is asserted to be a duplicate.</p><div class="pairs">',
    ]
    for pair in pairs[:200]:
        lines.append(
            f"<article><b>{pair['left']} / {pair['right']}</b><p>Cosine {pair['cosine']:.4f}; dHash distance {pair['dhash_distance']}</p>"
        )
        for key in ("left", "right"):
            row = lookup[pair[key]]
            url = html.escape(os.path.relpath(row["image_path"], directory), quote=True)
            lines.append(f'<a href="{url}"><img loading="lazy" src="{url}" alt="{row["case_id"]}"></a>')
        lines.append("</article>")
    lines.append("</div>")
    (directory / "index.html").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "neighbor_records": len(neighbors),
                "possible_duplicate_pairs": len(pairs),
                "represented_clusters": len(coverage["clusters"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
