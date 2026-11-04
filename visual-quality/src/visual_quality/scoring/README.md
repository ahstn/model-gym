# Shared scoring support

The current labeling workflow imports these modules directly. They were extracted from the completed 4,000-image and dual-score experiments during cleanup; scoring rules, model mapping and rendering behavior are unchanged.

| Module | Responsibility |
|---|---|
| `api_contract.py`, `wire.py` | Response parsing, two-task validation, score transport and billing validation |
| `lite_runner.py` | Measured Lite requests, score validation, concurrency and durable attempts |
| `reference_review.py` | Review validation, timing and frozen-reference checks |
| `gallery.py`, `gallery_png.py`, `validate_gallery.py` | HTML/PNG presentation and output/browser checks |
| `history.py` | Resolve restored historical evidence independently of executable helpers |

The package imports without archived data. Rebuilding the corpus, reusing old observations or reproducing historical default commands requires the [restored experiment inputs](../../../archives/experiments.md). By default these live under `visual-quality/data/experiment-history-2026-11-04/tree/visual-quality/experiments/`; `VISUAL_QUALITY_HISTORY` can point to an alternative directory containing the restored experiment folders. The current 10,000-image gallery and report accept their existing saved exports directly.

New executions record relocated source hashes. Old request plans, receipts and frozen results retain their original source identities and must not be relabelled or resumed under changed code. For byte-exact historical replay, use the pinned old checkout described in the archive guide. No models were rescored during extraction.
