# Published 72-image scoring trials

This bundle makes the two completed 31 October trials inspectable after cloning the repository. It contains the exact 72 delivered PNGs once, portable gallery copies, and the 360 saved inference attempts with model outputs and usage. Original frozen experiment files remain unchanged.

| Trial | Portable gallery | Report | Full grid PNG |
|---|---|---|---|
| Four models, original reasoning settings | [Open gallery](photo-scope/originals-grid.html) | [Read report](../photo-scope-trial-2026-10-31/README.md) | [View grid](../photo-scope-trial-2026-10-31/collage/originals-grid.png) |
| DeepSeek none versus high reasoning | [Open gallery](deepseek-high/originals-grid.html) | [Read report](../deepseek-high-trial-2026-10-31/README.md) | [View grid](../deepseek-high-trial-2026-10-31/collage/originals-grid.png) |

Open either HTML file in a browser from the checkout. The galleries have working relative image links, two smaller pages, and downloadable CSVs with exact scores and evidence. GitHub's source viewer does not execute HTML; the linked PNG grids are directly viewable there.

The [publication manifest](publication-manifest.json) records every original image's byte hash, source path, and five completed inference runs. All five runs contain 72 original attempts. In each `results/<trial>/<route>/` directory:

- `run.json`, `report.json` and `observations.jsonl` are byte-identical copies of the locally saved files.
- `attempts.jsonl` stores each exact parsed request metadata record and result record, including the unchanged raw response body string and usage. Original request/result file hashes preserve provenance across the JSONL serialization.

Request metadata contains hashes and byte counts, not API credentials or embedded image payloads. The export checks common credential patterns, embedded image payloads, image hashes and gallery link resolution. Frozen run metadata retains its original local provenance paths; those paths are evidence rather than portable dependencies. Portable gallery copies change only image locations, leaving scores and evidence unchanged.

The shared images total 127,103,535 bytes; no individual image exceeds GitHub's file limit. Only the 72 published trial views are included. Bulk source corpora, caches and local inference directories remain ignored.

To regenerate from the existing local run records:

```sh
visual-quality/.venv/bin/python visual-quality/experiments/published-trials-2026-10-31/export.py
```
