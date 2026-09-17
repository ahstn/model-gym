"""Tool categories: the corpus layout is one file per category."""

from __future__ import annotations

import re

from pathlib import Path

import pytest

from model_gym.categories import (
    CATEGORIES,
    CATEGORY_DESCRIPTIONS,
    coerce_category,
    is_category,
)
from model_gym.data import synthetic
from model_gym.data.schema import read_jsonl

SEED_DIR = Path("data/seed")


def seed_files() -> list[Path]:
    return sorted(SEED_DIR.glob("*.jsonl"))


def test_categories_are_unique_snake_case_and_described() -> None:
    assert len(set(CATEGORIES)) == len(CATEGORIES)
    assert set(CATEGORY_DESCRIPTIONS) == set(CATEGORIES)
    for category in CATEGORIES:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", category), category
        assert CATEGORY_DESCRIPTIONS[category]


def test_coerce_category_normalizes_and_validates() -> None:
    assert coerce_category("  AWS  ") == "aws"
    assert coerce_category("Elasticsearch") == "elasticsearch"
    assert is_category("sql")
    assert not is_category("seed_commands")
    with pytest.raises(ValueError, match="unknown category"):
        coerce_category("kubectl")
    with pytest.raises(TypeError, match="category must be a string"):
        coerce_category(3)


def test_every_generator_family_declares_a_known_category() -> None:
    for family in synthetic.FAMILIES:
        assert is_category(family.category), f"{family.name} has category {family.category!r}"
        for variant in family.variants:
            assert variant.category is None or is_category(variant.category), (
                f"{family.name}:{variant.name} has category {variant.category!r}"
            )


@pytest.mark.parametrize("path", seed_files(), ids=lambda path: path.stem)
def test_seed_files_are_named_after_their_category(path: Path) -> None:
    # The file name must be a category, and every row must carry that same category.
    assert is_category(path.stem), f"{path} is not named after a category"
    records = read_jsonl(path)
    assert records, f"{path} is empty"
    assert {record.category for record in records} == {path.stem}


def test_seed_directory_holds_the_category_files() -> None:
    names = {path.stem for path in seed_files()}
    assert names, f"no corpus files in {SEED_DIR}"
    assert names <= set(CATEGORIES), f"unexpected corpus files: {sorted(names - set(CATEGORIES))}"
    assert {"shell", "system", "git", "containers", "kubernetes", "sql"} <= names
