"""Check source-group isolation when medium screening replaces whole pairs."""

from collections import Counter, defaultdict

from tracks.ai_detection.build_dataset import GENERATORS, SEED, choose_pairs, sha


def source_families():
    return {
        f"source-{i:03d}": {
            kind: {"row": {"image_id": f"source-{i:03d}", "source": kind}}
            for kind in ("raise1k", *GENERATORS)
        }
        for i in range(200)
    }


def test_replacement_retains_pair_split_and_generator_without_reusing_a_source():
    families = source_families()
    order = sorted(families, key=lambda key: sha(f"{SEED}:{key}".encode()))
    decisions = [
        {"image_id": key, "generator": generator, "eligible": key not in {order[0], order[60]}}
        for generator in GENERATORS
        for key in families
    ]
    selected, pending = choose_pairs(families, decisions)
    assert not pending
    assert selected[0][0]["row"]["image_id"] == order[61]
    assert selected[0][1:] == ("train", GENERATORS[0])
    groups = defaultdict(list)
    for item, split, generator in selected:
        groups[item["row"]["image_id"]].append((item["row"]["source"], split, generator))
    assert len(groups) == 60
    assert all(len(pair) == 2 and pair[0][0] == "raise1k" for pair in groups.values())
    assert all(len({entry[1] for entry in pair}) == 1 for pair in groups.values())
    assert Counter(split for _, split, _ in selected) == {"train": 30, "dev": 30, "test": 60}


def test_unreviewed_pairs_are_pending_not_silently_accepted():
    selected, pending = choose_pairs(source_families(), [])
    assert len(selected) == 120
    assert len(pending) == 60
    assert len({entry["image_id"] for entry in pending}) == 60
