"""Tool-family categories.

The reviewed corpus is stored one file per category, ``data/seed/<category>.jsonl``.
A record's ``category`` is the tool family a command belongs to; the separate
``shell`` field stays the syntax dialect (``bash``, ``sql``, ``redis``, ``http``,
``powershell``), because one dialect spans many categories.

``read_jsonl`` checks that a file named after a category only contains records of
that category, so a misplaced row cannot slip into the corpus unnoticed.
"""

from __future__ import annotations

from typing import Any, Final

CATEGORIES: Final[tuple[str, ...]] = (
    "shell",
    "system",
    "git",
    "containers",
    "kubernetes",
    "terraform",
    "aws",
    "gcp",
    "azure",
    "sql",
    "redis",
    "elasticsearch",
    "onepassword",
    "sensitive_files",
)

CATEGORY_DESCRIPTIONS: Final[dict[str, str]] = {
    "shell": "POSIX shell, coreutils, and block devices",
    "system": "systemd, firewalls, users, kernel knobs, scheduled jobs",
    "git": "git history, working tree, and remotes",
    "containers": "docker, docker compose, podman",
    "kubernetes": "kubectl and helm",
    "terraform": "terraform, opentofu, terragrunt",
    "aws": "AWS CLI",
    "gcp": "Google Cloud SDK, gsutil, bq",
    "azure": "Azure CLI",
    "sql": "SQL statements and their client invocations",
    "redis": "redis-cli and Redis administration",
    "elasticsearch": "Elasticsearch and OpenSearch HTTP APIs",
    "onepassword": "1Password CLI and secret-store administration",
    "sensitive_files": "reading, copying, or publishing files that carry secrets",
}

DEFAULT_CATEGORY: Final[str] = "shell"


def coerce_category(value: Any) -> str:
    """Validate and normalise a category name."""
    if not isinstance(value, str):
        raise TypeError(f"category must be a string, got {type(value).__name__}")
    text = value.strip().lower()
    if text not in CATEGORY_DESCRIPTIONS:
        raise ValueError(f"unknown category {value!r}; expected one of {list(CATEGORIES)}")
    return text


def is_category(value: str) -> bool:
    """True when ``value`` names a category, used to read meaning from file names."""
    return value in CATEGORY_DESCRIPTIONS


def category_filename(category: str) -> str:
    """The corpus file name for a category."""
    return f"{coerce_category(category)}.jsonl"
