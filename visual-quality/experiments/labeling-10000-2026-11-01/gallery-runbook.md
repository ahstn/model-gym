# Local gallery and reference collage

`gallery_run.py` reads a saved export snapshot and reuses the rehearsal HTML and Pillow renderers. It does not call models, change observations, publish files or switch the running gallery service. The full gallery, thumbnails, four PNG pages and proof records stay under ignored run data.

From the repository root, build the current preview:

```sh
RUN_DATA=/home/ahstn/git/model-gym/visual-quality/data/labeling-10000-2026-11-01
python visual-quality/experiments/labeling-10000-2026-11-01/gallery_run.py \
  --exports "$RUN_DATA/exports-preview" \
  --output "$RUN_DATA/gallery-preview" \
  --browser-check
```

After the final export is ready, run the same command with `--exports "$RUN_DATA/exports" --output "$RUN_DATA/gallery"`. Keep the default selection file. The source export must remain stable while rendering; the generator checks its hashes before and after the build. The browser check uses the installed local Chromium and requires the environment to permit its local socket operations.

The first run freezes `gallery-collage-selection.json` under run data. It selects 72 equally spaced ranks in reference technical score, breaking ties by case ID, from the first 1,000 explicit publication orders. It reads no model scores for selection. It pins the complete reference and manifest hashes and the publication identity list. Later runs fail if the selection or these inputs differ. The last 500 benchmark references never enter the collage selection.

The interactive gallery shows all 10,000 images, including missing assessments, with filters for all six original partitions and the current content clusters. Related images and family links come from the frozen related-image index. Technical and aesthetic scores remain separate. The weighted ensemble uses the saved canonical 37.5% Luna, 37.5% DeepSeek and 25% Liquid observations; the gallery preserves their exported display precision and native values.

Image statistics appear in a separate review section, with units, analysis dimensions and the relative-focus caveat. They do not enter assessment records or change scores. Reference review windows retain their measured durations and timing basis, including preparation and tool overhead. They are distinct from HTTP latency. Missing prices remain dashes; only the recorded Reference and Lite accounting overrides display $0.00. The ensemble has no independent HTTP latency or charge.

Every successful build writes `generation-proof.json`. It pins all input hashes and output data, verifies every score/metadata join and preview hash, and verifies the four HTML/PNG pages against their source scores and selection. With `--browser-check`, it also records filter, pagination, sorting, deep-link, metadata and image-loading checks. `png-validation.json` records the four page hashes; `preview-index.json` allows unchanged thumbnails to be reused safely.

The 72-image collage is a reference-range illustration, not a population-weighted sample. All outputs remain experimental and training masks stay disabled. These local artifacts are separate from the bounded structured publication export; do not add the full gallery or its embedded records to git.
