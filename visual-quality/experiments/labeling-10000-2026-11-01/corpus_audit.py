"""Validate frozen corpus identities, delivered files and publication boundaries."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from corpus_common import DATA, PREVIOUS, digest, write_json


def main():
    path = DATA / "manifest.json"
    manifest = json.loads(path.read_text())
    cases = manifest["cases"]
    old = json.loads((PREVIOUS / "manifest.json").read_text())["cases"]
    benchmark = json.loads((DATA / "benchmark-manifest.json").read_text())["cases"]
    by_id = {row["case_id"]: row for row in cases}
    assert len(cases) == len(by_id) == 10000
    assert len({row["image_id"] for row in cases}) == 10000
    assert len({row["view_sha256"] for row in cases}) == 10000
    assert Counter(row["category"] for row in cases) == {
        "megalith": 7500,
        "vizwiz": 1500,
        "distortions": 750,
        "controls": 250,
    }
    for row in old:
        current = by_id[row["case_id"]]
        for field in ("image_id", "image_path", "view_sha256", "source_sha256", "family_id"):
            assert current[field] == row[field], (row["case_id"], field)
    for row in benchmark:
        current = by_id[row["case_id"]]
        for field in ("image_id", "view_sha256", "benchmark_order", "publication_order"):
            assert current[field] == row[field], (row["case_id"], field)
    references = [row for row in cases if row.get("benchmark_order") is not None]
    assert sorted(row["benchmark_order"] for row in references) == list(range(1, 1501))
    allowlist = [row for row in references if row["publication_order"] <= 1000]
    assert len(allowlist) == 1000
    family_partitions = defaultdict(set)
    for row in cases:
        assert row["license"] and row["source_revision"]
        assert all(value is False for value in row["task_masks"].values())
        family_partitions[row["family_id"]].add(row["benchmark_partition"])
    assert all(len(partitions) == 1 for partitions in family_partitions.values())
    with ThreadPoolExecutor(4) as executor:
        hashes = list(executor.map(lambda row: digest(row["image_path"]), cases))
    assert hashes == [row["view_sha256"] for row in cases]
    embeddings = np.load(DATA / "corpus-embeddings.npz", allow_pickle=False)
    assert embeddings["case_ids"].tolist() == [row["case_id"] for row in cases]
    assert embeddings["embeddings"].shape == (10000, 768)
    assert np.isfinite(embeddings["embeddings"]).all()
    np.testing.assert_allclose(np.linalg.norm(embeddings["embeddings"], axis=1), 1, atol=1e-5)
    result = {
        "status": "passed",
        "manifest_sha256": digest(path),
        "views": len(cases),
        "families": len(family_partitions),
        "benchmark": len(references),
        "publication_allowlist_images": len(allowlist),
        "preserved_previous_views": len(old),
        "all_delivered_hashes_verified": True,
        "all_training_masks_false": True,
        "explicit_family_partition_crossings": 0,
        "embedding_shape": list(embeddings["embeddings"].shape),
        "note": "Identity checks do not establish exhaustive visual deduplication; consult the separate reviewed candidate ledger.",
    }
    write_json(DATA / "corpus-integrity-audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
