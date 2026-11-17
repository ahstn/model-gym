# 40K corpus content coverage

**Status at 2026-11-17 17:49 UTC:** corpus preparation, selection, clustering and retrieval validation are complete; model scoring is still running. These are aggregate corpus facts, not completed labeling or accuracy results. Training masks remain disabled.

The expansion adds 20,000 views to the preserved 20,000, giving **40,000 views from 37,000 explicit source families**. There are 36,000 standalone families and 1,000 controlled families containing a parent and three derivatives. Known families remain together; unknown photographer, session and event relationships are not resolved by these counts.

## Source mix

| Acquisition category | Prior 20K | New 20K | Combined 40K | Combined share |
|---|---:|---:|---:|---:|
| Megalith source images | 15,000 | 15,201 | 30,201 | 75.50% |
| VizWiz capture-quality images | 3,000 | 3,000 | 6,000 | 15.00% |
| Controlled distortion views | 1,500 | 1,500 | 3,000 | 7.50% |
| Non-photographic controls | 500 | 299 | 799 | 2.00% |
| **Total** | **20,000** | **20,000** | **40,000** | **100.00%** |

Only 299 eligible fresh controls remained; the declared fallback replaced the 201-control shortfall with Megalith images. Combined controls comprise 399 PlotQA and 400 CLEVR views. The 3,000 controlled derivatives comprise 963 blur, 1,017 noise and 1,020 JPEG views. Acquisition categories and configured distortion severity are not ground-truth quality labels.

## Content distribution

Actual EmbeddingGemma 2 image vectors assign content to the frozen 64 original prototypes. Selection uses seeded, score-blind, capacity-limited quotas that account for previous source-specific cluster counts. All 64 groups gained images. The ten largest groups occupy **21.475%** of the combined corpus, compared with **22.305%** for the ten largest groups in the prior 20K; no combined group exceeds 2.37%.

| Largest theme groups | Prior 20K | New 20K | Combined 40K | Combined share |
|---|---:|---:|---:|---:|
| Books, documents and printed material | 451 | 497 | 948 | 2.37% |
| Workshops and conference groups | 473 | 446 | 919 | 2.30% |
| Street festivals and public gatherings | 494 | 421 | 915 | 2.29% |
| Residential architecture | 468 | 433 | 901 | 2.25% |
| Social gatherings and crowds | 443 | 413 | 856 | 2.14% |
| Urban streets | 432 | 400 | 832 | 2.08% |
| Youth outdoor groups | 439 | 383 | 822 | 2.06% |
| Plants, leaves and ground details | 420 | 395 | 815 | 2.04% |
| Team sports | 416 | 389 | 805 | 2.01% |
| Buildings and street furniture | 406 | 371 | 777 | 1.94% |

Other groups cover food, household objects, landscapes, animals, aircraft, portraits, obstructed views, charts and synthetic scenes. People, gatherings and built environments still feature prominently across multiple groups, and 75.50% of the corpus comes from Megalith. Reduced concentration at the top does not establish uniformly improved balance: scatter plots gained only 9 views (103 total), mixed charts 22 (124 total), nearly black frames 20 (151 total), and zoo mammals 58 (154 total).

All theme names match the committed [original theme-label file](../../../experiments/labeling-10000-2026-11-01/corpus-cluster-labels.json). Those descriptions came from six reviewed, distinct-family exemplars per original cluster; they are approximate summaries, not verified per-image subjects. Frozen prototypes enable comparable counts but do not establish coverage of unfamiliar subjects or automatically discover new groups.

## Related-image review limits

Retrieval provides six cross-family neighbors per view: **240,000 directed links**. The similarity threshold surfaces **1,190 unconfirmed candidate pairs involving 680 distinct images**: 327 prior/prior, 519 prior/new and 344 new/new. Of these, 700 pair two controls, illustrating that similar content does not establish duplication. Fourteen candidates were conservatively excluded during selection using cosine similarity at least 0.985 together with dHash distance at most six. Six final review pairs meet both thresholds; all belong to the unchanged prior 20K.

These are review candidates, not confirmed duplicates or completed visual review. The final output retains selected nearest neighbors rather than every pair above the threshold. Neighbors exclude the query's family but need not represent six distinct other families. Unknown related captures can remain. In addition, 3,509 retrieval rows use explicitly recorded family representatives; 36,491 use exact-view embeddings. Representatives organize content, not image quality. Graders and image statistics use the exact delivered views.

## Aggregate evidence

Counts were checked against the frozen corpus and aggregate artifacts. Retrieval validation checked every stored similarity and family join, the review links, and unchanged manifest, embedding and prototype hashes. The following aggregate evidence is retained under the ignored run data directory; full per-image artifacts remain outside Git.

| Evidence | SHA256 |
|---|---|
| `content-coverage.json` | `238852015b8a87ebd1b74afd93c78ca649491301a2617e221f450a24a1ab793b` |
| `operations/related-validation.json` | `7b136dc23a273d230cfb9616f004b6a16d8b27ac0423d123f7ec84354dbc6b3c` |
| Committed `corpus-cluster-labels.json` | `3768e2660ddab252b60463d91c52490bcab158fcb61136fd86411273c2c84c56` |
