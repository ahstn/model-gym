"""Apply explicitly reviewed duplicate decisions and freeze benchmark identities."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
from corpus_common import DATA, digest, stable_key, write_json
from corpus_select import review_pages, vectors


def replace_examples():
    proposal_path = DATA / "benchmark-examples-proposal.json"
    proposal = json.loads(proposal_path.read_text())
    if proposal.get("duplicate_replacements"):
        raise ValueError("Replacements already applied; do not mutate reviewed identities twice")
    archive = DATA / "review" / "examples-before-replacements.json"
    write_json(archive, proposal)
    conflicts = {
        "R4191": ["C6c35f07c1fe24dbb"],
        "R4114": ["Cfa732efa292741e7"],
        "R4343": ["C7c599e715b456707", "C19592536bb5eb69b"],
        "R4146": ["Cc66f1af6e1adf2bc"],
    }
    excluded = {"Ce2a5db208f825763", "Cde8521cd6cdfde4a", "C0c8b50b7f220dcf6", "C19592536bb5eb69b"}
    rows = proposal["cases"]
    used = {r.get("candidate_id") for r in rows}
    for row in rows:
        if row["case_id"] in conflicts:
            excluded.add(row["candidate_id"])
    replacements = []
    clusters = proposal["frame"]["clusters"]
    for index, row in enumerate(rows):
        if row["case_id"] not in conflicts:
            continue
        options = [
            r
            for r in proposal["frame"]["cases"]
            if r["candidate_id"] not in used | excluded
            and r["category"] == row["category"]
            and clusters[r["candidate_id"]] == row["benchmark_content_cluster"]
        ]
        if not options:
            raise ValueError(f"No same-source/same-cluster replacement for {row['case_id']}")
        replacement = min(options, key=lambda r: stable_key("example", r["image_id"]))
        rows[index] = {
            **replacement,
            **{
                key: row[key]
                for key in (
                    "case_id",
                    "benchmark_partition",
                    "benchmark_order",
                    "publication_order",
                    "benchmark_content_cluster",
                    "historical_exposure",
                )
            },
        }
        used.add(replacement["candidate_id"])
        replacements.append(
            {
                "case_id": row["case_id"],
                "rejected_candidate_id": row["candidate_id"],
                "replacement_candidate_id": replacement["candidate_id"],
                "confirmed_related_candidates": conflicts[row["case_id"]],
                "reason": "Parent visual review confirmed same-capture family with retained old corpus; replace before reference scoring",
            }
        )
    proposal["duplicate_replacements"] = replacements
    proposal["excluded_candidate_ids"] = sorted(excluded)
    proposal["duplicate_review_notes"] = (
        "Parent independently inspected all four first-round contact sheets. Three exact duplicate aliases "
        "were excluded while retaining the selected views. Four same-capture-family conflicts with the "
        "retained 4,000 were replaced before references. Red-panda, butterfly and contrail similarities "
        "were thematic only. Near-black low-information similarities remain unresolved review candidates, "
        "not asserted duplicate identities; exact source/pixel/view and dHash guards remain in force."
    )
    candidates = json.loads((DATA / "candidate-manifest.json").read_text())["cases"]
    vec = vectors(candidates)
    embedded = [r for r in candidates if r["candidate_id"] in vec]
    matrix = np.stack([vec[r["candidate_id"]] for r in embedded])
    lookup = {r["candidate_id"]: r for r in embedded}
    lookup.update({r["candidate_id"]: r for r in rows if r.get("candidate_id")})
    pairs = []
    for item in replacements:
        row = next(r for r in rows if r["case_id"] == item["case_id"])
        similarity = vec[row["candidate_id"]] @ matrix.T
        for j in np.argsort(similarity)[-7:][::-1]:
            other = embedded[j]
            if other["candidate_id"] == row["candidate_id"]:
                continue
            pairs.append(
                {
                    "left": row["candidate_id"],
                    "right": other["candidate_id"],
                    "cosine": float(similarity[j]),
                    "dhash_distance": (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count(),
                    "review_status": "pending",
                }
            )
    pages = review_pages(pairs, lookup, "benchmark-examples-replacements")
    write_json(DATA / "benchmark-examples-replacement-review.json", {"pairs": pairs, "contact_sheets": pages})
    write_json(DATA / "benchmark-examples-review-image-index.json", lookup)
    write_json(proposal_path, proposal)
    print(json.dumps({"replacements": replacements, "pages": pages}), flush=True)


def freeze(phase):
    proposal = DATA / f"benchmark-{phase}-proposal.json"
    record = json.loads(proposal.read_text())
    record["status"] = "frozen_after_duplicate_review"
    record["frozen_at_utc"] = datetime.now(UTC).isoformat()
    record["proposal_sha256"] = digest(proposal)
    record.setdefault(
        "review_authority", "Parent assistant visual review before independent reference scoring"
    )
    target = DATA / f"benchmark-{phase}-manifest.json"
    if target.exists():
        raise ValueError(f"Frozen manifest already exists: {target}")
    review_path = DATA / f"benchmark-{phase}-duplicate-review.json"
    review = json.loads(review_path.read_text())
    exclusions = set(record.get("excluded_candidate_ids", []))
    for index, pair in enumerate(review["pairs"]):
        if index >= review["contact_sheet_pairs"]:
            pair["review_status"] = "unreviewed_candidate_retained_in_queue"
            continue
        pair["review_status"] = (
            "reviewed_family_conflict_resolved_by_exclusion_or_replacement"
            if exclusions.intersection((pair["left"], pair["right"]))
            else "visually_reviewed_no_confirmed_duplicate_identity"
        )
    review["status"] = "reviewed_with_recorded_limitations"
    review["review_notes"] = record.get("duplicate_review_notes")
    review["reviewed_at_utc"] = record["frozen_at_utc"]
    review["limitation"] = (
        "This review does not establish exhaustive deduplication. Low-information frames and "
        "unverified event associations remain uncertain; semantic similarity alone is not identity."
    )
    write_json(review_path, review)
    record["duplicate_review_sha256"] = digest(review_path)
    write_json(target, record)
    if phase == "confirmation":
        write_json(DATA / "benchmark-manifest.json", record)
    print(target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["replace-examples", "freeze-examples", "freeze-confirmation"])
    action = parser.parse_args().action
    if action == "replace-examples":
        replace_examples()
    else:
        freeze(action.removeprefix("freeze-"))
