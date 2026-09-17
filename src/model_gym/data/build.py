"""Corpus assembly: load labelled JSONL, dedupe, split by group, write Arrow-free JSONL."""

from __future__ import annotations

import json
import random

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from model_gym.categories import CATEGORIES
from model_gym.data import synthetic
from model_gym.data.schema import (
    RiskRecord,
    category_counts,
    dedupe,
    iter_paths,
    level_counts,
    read_jsonl,
    write_jsonl,
)
from model_gym.labels import NUM_LABELS

SPLITS: tuple[str, ...] = ("train", "validation", "test")

# Below this many rows in a split, evaluation numbers are not worth reading.
MIN_ROWS_PER_SPLIT: dict[str, int] = {"train": 40, "validation": 8, "test": 8}
# Below this many rows for a level in train, the level is effectively unlearnable.
MIN_ROWS_PER_LEVEL: int = 5
# Below this many training rows for a tool family, its per-category score is noise.
MIN_ROWS_PER_CATEGORY: int = 5


@dataclass(frozen=True, slots=True)
class CorpusSummary:
    """What ``build-data`` produced, for logging and the dataset card."""

    output_dir: Path
    split_counts: Mapping[str, int]
    split_levels: Mapping[str, Mapping[int, int]]
    split_categories: Mapping[str, Mapping[str, int]]
    group_counts: Mapping[str, int]
    duplicates_dropped: int
    synthetic_rows: int
    source_files: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def total(self) -> int:
        return sum(self.split_counts.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "totals": {
                "records": self.total,
                "groups": sum(self.group_counts.values()),
                "duplicates_dropped": self.duplicates_dropped,
                "synthetic_rows": self.synthetic_rows,
            },
            "splits": {split: dict(levels) for split, levels in self.split_levels.items()},
            "split_totals": dict(self.split_counts),
            "categories": {category: _category_row(category, self.split_categories) for category in CATEGORIES},
            "source_files": list(self.source_files),
            "warnings": list(self.warnings),
        }

    def format(self) -> str:
        header = ["level", *SPLITS, "total"]
        lines = ["  ".join(f"{cell:>10}" for cell in header)]
        for level in range(1, NUM_LABELS + 1):
            row = [f"L{level}"] + [str(self.split_levels[split].get(level, 0)) for split in SPLITS]
            row.append(str(sum(int(cell) for cell in row[1:])))
            lines.append("  ".join(f"{cell:>10}" for cell in row))
        totals = ["total"] + [str(self.split_counts[split]) for split in SPLITS] + [str(self.total)]
        lines.append("  ".join(f"{cell:>10}" for cell in totals))
        lines.append("")
        lines.append("  ".join(f"{cell:>10}" for cell in ["category", *SPLITS, "total"]))
        for category in CATEGORIES:
            row = [category] + [str(self.split_categories[split].get(category, 0)) for split in SPLITS]
            row.append(str(sum(int(cell) for cell in row[1:])))
            lines.append("  ".join(f"{cell:>10}" for cell in row))
        return "\n".join(lines)


def _category_row(category: str, split_categories: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    """Per-split row counts for one category, always with a total."""
    row = {split: split_categories[split].get(category, 0) for split in SPLITS}
    row["total"] = sum(row.values())
    return row


def load_records(
    sources: Sequence[Path | str],
    *,
    include_synthetic: bool,
    seed: int = 0,
) -> tuple[list[RiskRecord], list[RiskRecord], tuple[str, ...]]:
    """Load curated records, append generated ones, and dedupe.

    Curated rows are read first so they win when a generated command collides.
    Returns ``(records, dropped_duplicates, source_files)``.
    """
    del seed  # generation is deterministic; kept for signature symmetry
    files = tuple(str(path) for path in iter_paths(sources))
    if not files:
        raise ValueError(f"no *.jsonl corpus files found under {list(sources)}")
    records: list[RiskRecord] = []
    for path in files:
        records.extend(read_jsonl(path))
    rows = len(records)
    if include_synthetic:
        records.extend(synthetic.generate())
    unique, dropped = dedupe(records)
    return (
        unique,
        dropped,
        files + ((f"<generated:{synthetic.SOURCE}>",) if include_synthetic and len(records) > rows else ()),
    )


def split_records(
    records: Iterable[RiskRecord],
    *,
    ratios: Sequence[float],
    seed: int,
) -> dict[str, list[RiskRecord]]:
    """Split records by group so near-duplicates never straddle splits.

    Groups are assigned greedily to whichever split is furthest below its target
    share, which keeps small corpora from collapsing into a single split.
    """
    if len(ratios) != len(SPLITS):
        raise ValueError(f"expected {len(SPLITS)} split ratios, got {len(ratios)}")
    if any(ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
        raise ValueError(f"split ratios must be non-negative and sum to a positive value, got {list(ratios)}")

    grouped: dict[str, list[RiskRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group].append(record)
    if not grouped:
        raise ValueError("no records to split")

    group_names = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(group_names)

    total = sum(len(rows) for rows in grouped.values())
    share = sum(ratios)
    targets = {split: ratios[index] / share * total for index, split in enumerate(SPLITS)}
    assigned: dict[str, list[RiskRecord]] = {split: [] for split in SPLITS}
    for name in group_names:
        rows = grouped[name]
        split = max(
            SPLITS, key=lambda candidate: (targets[candidate] - len(assigned[candidate]), -SPLITS.index(candidate))
        )
        assigned[split].extend(rows)

    _backfill_empty_splits(assigned, group_names)
    for rows in assigned.values():
        rng.shuffle(rows)
    return assigned


def _backfill_empty_splits(assigned: dict[str, list[RiskRecord]], group_names: Sequence[str]) -> None:
    """Move the smallest group out of the largest split into an empty one."""
    if len(group_names) < len(SPLITS):
        return
    for split in SPLITS:
        if assigned[split]:
            continue
        donor = max(SPLITS, key=lambda candidate: len(assigned[candidate]))
        donor_groups: dict[str, list[RiskRecord]] = defaultdict(list)
        for record in assigned[donor]:
            donor_groups[record.group].append(record)
        if len(donor_groups) < 2:
            return
        smallest = min(sorted(donor_groups), key=lambda name: (len(donor_groups[name]), name))
        moved = donor_groups[smallest]
        assigned[donor] = [record for record in assigned[donor] if record.group != smallest]
        assigned[split].extend(moved)


def build_corpus(
    *,
    sources: Sequence[Path | str],
    output_dir: Path | str,
    synthetic_output: Path | str | None = None,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 13,
    include_synthetic: bool = True,
) -> CorpusSummary:
    """Build the train/validation/test JSONL files and the dataset card."""
    output_dir = Path(output_dir)
    if include_synthetic and synthetic_output is not None:
        write_jsonl(synthetic_output, synthetic.generate())
    records, dropped, source_files = load_records(sources, include_synthetic=include_synthetic, seed=seed)
    if not records:
        raise ValueError(f"no records loaded from {list(sources)}")

    splits = split_records(records, ratios=ratios, seed=seed)
    split_levels = {split: level_counts(rows) for split, rows in splits.items()}
    split_categories = {split: category_counts(rows) for split, rows in splits.items()}
    group_counts = {split: len({record.group for record in rows}) for split, rows in splits.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {split: write_jsonl(output_dir / f"{split}.jsonl", rows) for split, rows in splits.items()}
    summary = CorpusSummary(
        output_dir=output_dir,
        split_counts=split_counts,
        split_levels=split_levels,
        split_categories=split_categories,
        group_counts=group_counts,
        duplicates_dropped=len(dropped),
        synthetic_rows=sum(1 for record in records if record.source == synthetic.SOURCE),
        source_files=source_files,
        warnings=_warnings(splits, split_levels, split_categories),
    )
    card = output_dir / "dataset_card.json"
    card.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _warnings(
    splits: Mapping[str, Sequence[RiskRecord]],
    split_levels: Mapping[str, Mapping[int, int]],
    split_categories: Mapping[str, Mapping[str, int]],
) -> tuple[str, ...]:
    messages: list[str] = []
    for split in SPLITS:
        count = len(splits[split])
        floor = MIN_ROWS_PER_SPLIT[split]
        if count < floor:
            messages.append(f"{split} has {count} rows; {floor}+ recommended before trusting its numbers")
    train_levels = split_levels["train"]
    for level in range(1, NUM_LABELS + 1):
        count = train_levels.get(level, 0)
        if count < MIN_ROWS_PER_LEVEL:
            messages.append(f"level {level} has {count} training rows; the model cannot learn it reliably")
    for split in ("validation", "test"):
        missing = [level for level in range(1, NUM_LABELS + 1) if not split_levels[split].get(level)]
        if missing:
            messages.append(f"{split} has no examples for level(s) {missing}; per-level metrics will be 0")
    present = {category for category in CATEGORIES if sum(row.get(category, 0) for row in split_categories.values())}
    for category in sorted(present):
        count = split_categories["train"].get(category, 0)
        if count < MIN_ROWS_PER_CATEGORY:
            messages.append(
                f"category {category} has {count} training rows; its per-category score is not readable yet"
            )
    absent = [category for category in CATEGORIES if category not in present]
    if absent:
        messages.append(f"no rows for {len(absent)} categories: {', '.join(absent)}")
    return tuple(messages)
