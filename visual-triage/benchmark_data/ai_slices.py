"""Derive fixed AI-origin slices; never fit a threshold or change reference labels."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_analyze import auc, valid_prediction


def summarize(rows):
    counts = Counter()
    bands = {
        name: {"real": 0, "ai": 0} for name in ("low_p_le_0.2", "review_0.2_lt_p_lt_0.8", "high_p_ge_0.8")
    }
    for row in rows:
        probability, label = row["p"], row["y"]
        counts[("t" if (probability >= 0.5) == label else "f") + ("p" if probability >= 0.5 else "n")] += 1
        band = (
            "low_p_le_0.2"
            if probability <= 0.2
            else "high_p_ge_0.8"
            if probability >= 0.8
            else "review_0.2_lt_p_lt_0.8"
        )
        bands[band]["ai" if label else "real"] += 1
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    real, generated = tn + fp, tp + fn
    recall = tp / generated if generated else None
    specificity = tn / real if real else None
    return {
        "n": len(rows),
        "real": real,
        "ai": generated,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": (tp + tn) / len(rows),
        "ai_recall": recall,
        "specificity": specificity,
        "false_positive_rate": fp / real if real else None,
        "balanced_accuracy": (recall + specificity) / 2 if real and generated else None,
        "brier": mean((row["p"] - row["y"]) ** 2 for row in rows),
        "auroc": auc([row["p"] for row in rows], [int(row["y"]) for row in rows]),
        "bands": bands,
        "review_count": sum(bands["review_0.2_lt_p_lt_0.8"].values()),
        "high_band_precision": bands["high_p_ge_0.8"]["ai"] / sum(bands["high_p_ge_0.8"].values())
        if sum(bands["high_p_ge_0.8"].values())
        else None,
    }


def main():
    paths = {
        "manifest": ROOT / "evidence/benchmarks/ai.json",
        "assistant_reference": ROOT / "evidence/assessments/ai-60.json",
        "final_ledger": ROOT / "evidence/final/ai.jsonl",
    }
    source = {row["id"]: row for row in json.loads(paths["manifest"].read_text())["samples"]}
    reference = {row["id"]: row for row in json.loads(paths["assistant_reference"].read_text())["samples"]}
    ledger = [json.loads(line) for line in paths["final_ledger"].read_text().splitlines()]
    assert len(ledger) == 180
    grouped = {}
    for arm in sorted({row["arm"] for row in ledger}):
        records = [row for row in ledger if row["arm"] == arm]
        assert len(records) == 60 and len({row["sample"] for row in records}) == 60
        assert all(valid_prediction(row) for row in records)
        grouped[arm] = []
        for record in records:
            sample = source[record["sample"]]
            own = reference[record["sample"]]
            assert record["delivered_sha256"] == sample["delivered_sha256"] == own["delivered_sha256"]
            grouped[arm].append(
                {
                    "id": sample["id"],
                    "p": record["normalized"]["ai_probability"],
                    "y": sample["labels"]["ai_generated"],
                    "medium": own["medium"],
                    "source": sample["source"]["source_kind"],
                }
            )
    grouped["assistant_reference"] = [
        {
            "id": sample_id,
            "p": own["ai_probability"],
            "y": source[sample_id]["labels"]["ai_generated"],
            "medium": own["medium"],
            "source": source[sample_id]["source"]["source_kind"],
        }
        for sample_id, own in reference.items()
    ]
    result = {
        "inputs_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        },
        "threshold": 0.5,
        "review_band_open": [0.2, 0.8],
        "slice_basis": "Frozen assistant medium tags from direct visual inspection before final API requests; photo contains 30 real plus 18 AI, graphic 7 AI, render 5 AI.",
        "limitations": [
            "Assistant medium tags are model judgments, not human annotations.",
            "All real images are DOCCI camera photos; generated media and content are unmatched. A style-only rule scores70% here without establishing origin detection.",
            "These fixed post-evaluation descriptive slices do not tune prompts or thresholds. No population or generator-generalization claim follows from five images per generator.",
            "Small samples, downsampled JPEG views, and source/camera confounds remain. Probabilities are uncalibrated; high-band precision here is not a production guarantee.",
        ],
        "arms": {},
    }
    for arm, rows in grouped.items():
        records = [row for row in ledger if row["arm"] == arm]
        result["arms"][arm] = {
            "requested_models": sorted({row["requested_model"] for row in records}),
            "resolved_models": sorted({row["resolved_model"] for row in records}),
            "providers": sorted({row["provider"] for row in records}),
            "requested_reasoning": [
                json.loads(value)
                for value in sorted(
                    {json.dumps(row.get("requested_reasoning"), sort_keys=True) for row in records}
                )
            ],
            "all": summarize(rows),
            "assistant_medium_photo": summarize([row for row in rows if row["medium"] == "photo"]),
            "by_visible_medium": {
                medium: summarize([row for row in rows if row["medium"] == medium])
                for medium in sorted({row["medium"] for row in rows})
            },
            "by_source": {
                name: summarize([row for row in rows if row["source"] == name])
                for name in sorted({row["source"] for row in rows})
            },
        }
    output = ROOT / "evidence/benchmarks/ai-slices.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output.relative_to(ROOT))
    for arm, data in result["arms"].items():
        print(
            arm,
            json.dumps(
                {
                    "all": data["all"],
                    "photo": data["assistant_medium_photo"],
                    "generator_tp": {
                        k: v["confusion"]["tp"] for k, v in data["by_source"].items() if k != "docci"
                    },
                }
            ),
        )


if __name__ == "__main__":
    main()
