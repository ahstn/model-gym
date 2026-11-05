# Current evaluation

The [10,000-image run](labeling-10000-2026-11-01/README.md) is the retained experiment: final report, 1,500-reference benchmark, validated 72-image collage and reproduction tools. Active prompts, routes and first 1,000 example records are in [photo labeling](../labeling/README.md).

Shared response validation, reference review, Lite requests and gallery rendering live in [the scoring package](../src/visual_quality/scoring/README.md). They no longer import code from historical experiment folders. The pinned Lite GPU server is in [labeling/runtime](../labeling/runtime/README.md).

Thirteen earlier experiment trees were archived after a complete restore and hash check. Their findings, exact revision and recovery commands are in the [historical experiment catalogue](../archives/experiments.md). Restore historical inputs into ignored data when rebuilding the corpus; do not repopulate this directory with archived runs.

An existing uncommitted EmbeddingGemma clustering directory is local work and was preserved separately from this tracked experiment selection.
