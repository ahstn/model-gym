# Gallery validation

**Final publication validation passed.** The full gallery contains 4,000 cases, all 40 content clusters and the frozen 72-image comparison selection. The final run passed 205 browser checks, loaded all 4,000 previews, matched the rendered data to all 27,500 CSV rows and verified the four static pages' 1,008 score cells. The hash-bound receipt is [gallery/publication-validation.json](gallery/publication-validation.json). All four final PNG pages were opened and inspected without observed clipping; [gallery/visual-inspection.json](gallery/visual-inspection.json) records their hashes.

An earlier successful run used the 72-image layout fixture at `/tmp/rehearsal-real-layout-check/gallery`, whose model scores were incomplete. That fixture remains a layout check, not a final result.

`validate_gallery.py` independently joins gallery records to the score CSV, checks corpus identities, source-view links, partitions, cluster labels, related-image results and family links, decodes every preview, and checks the four static HTML tables against the CSV. It also verifies the rendered PNG hashes, dimensions, saved reference-only selection and CSV provenance. Across the four pages it checks 72 distinct images, seven assessor rows per image and 1,008 technical/aesthetic score cells, alongside latency and price cells.

A temporary browser harness tests the actual gallery script in headless Chromium: default benchmark selection, partitions, pagination, every cluster filter, case search, all three scoring arms, exact displayed values, empty search, reference sorting and related-image deep links. Every preview is loaded by the browser. The source gallery is unchanged by the harness. The fixture passed 193 browser checks and loaded all 72 previews. Pages 2 and 3 were also inspected visually; all card rows and footers were visible with no clipping. The root reviewer inspected pages 1 and 4 separately.

After the final `gallery.py` and `gallery_png.py` runs, execute from this experiment directory:

```sh
/home/ahstn/git/model-gym/visual-quality/.venv/bin/python validate_gallery.py
```

The final mode requires all 4,000 cases, the 500-case benchmark split into 100 development / 300 representative / 100 challenge images, 3,500 other rehearsal cases and all 40 content clusters. It writes `gallery/publication-validation.json` only after all checks pass. Its receipt pins the checker, CSV, gallery data, HTML, selection, previews and PNG files. Artifact consistency does not assert that every provider returned a valid score; missing scoring observations remain visible and are accounted in the evaluation report.

Chromium's crash handler is blocked by this host's execution sandbox, so the browser invocation needs host execution through the normal approval mechanism. It reads local trial files and writes only a temporary browser profile. The successful fixture receipt is `/tmp/rehearsal-real-layout-check/gallery/interim-layout-validation.json`.
