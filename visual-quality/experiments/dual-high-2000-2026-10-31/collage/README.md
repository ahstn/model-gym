# Viewing and reproducing the gallery

Open `originals-grid.html` in a browser from the checkout. It uses the 72 exact delivered PNGs in `images/`, and all image and CSV links are relative. The two page HTML files show 36 cases each. Each card has six score cells: technical and aesthetic scores for Ref, Luna and DeepSeek. The gallery CSV preserves exact scores and evidence; the parent experiment's `per-image.csv` also contains the two-model arithmetic mean.

The renderer requires Node, Chromium and the project's Python environment with Pillow. Set `GRID_CHROMIUM_PATH` to a local Chromium executable when the recorded default location is unavailable.

From the repository root, the default mode verifies the original local source pins, including ignored run journals, before rendering:

```sh
node visual-quality/experiments/dual-high-2000-2026-10-31/collage/render.mjs
```

After the completed results and first render have been published, render from a checkout without the ignored inference cache:

```sh
node visual-quality/experiments/dual-high-2000-2026-10-31/collage/render.mjs --published
```

Published mode verifies the local `publication-manifest.json` and all exported evidence hashes, relocates original run pins to their `results/<route>/source-evidence/` copies, and verifies gallery sources against the saved `render-validation.json`. It uses only the checkout's `images/` originals. Existing PNG outputs may be absent; their recorded source files and previous validation record must remain present. Both modes compare each displayed image URL with its case's hashed original, check all 432 score cells, wait for every image to decode and paint, and validate raster content before saving the three PNGs. The resulting `render-validation.json` records either `live_source_pins` or `published_checkout` audit mode.

This reproduces the published gallery from its saved sources. Browser/font differences may change PNG bytes or layout; a layout validation failure must be investigated rather than ignored. Rebuilding the original corpus or rerunning inference still requires the ignored source data and credentials. `build_collage.py` regenerates gallery source files from that original local cache; use the renderer's published mode to reproduce the saved gallery without it.
