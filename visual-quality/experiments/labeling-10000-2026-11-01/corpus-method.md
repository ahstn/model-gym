# Corpus expansion method

This expansion preserves the 4,000 delivered views and their case identifiers from the previous rehearsal. The completed mix is 7,500 broad-source Megalith images, 1,500 authentic-fault VizWiz images, 750 controlled derivatives and 250 non-photographic controls: 10,000 views from 9,250 explicit source families. These are acquisition/sampling categories, not inferred quality labels or definitive photographic eligibility decisions.

The additional benchmark contains 1,000 newly acquired source identities: 800 Megalith and 200 VizWiz. Its first 500 examples and subsequent 500 confirmation cases each contain 400/100 from those sources. The original 500 references retain their historical exposure and partition provenance. The first 1,000 benchmark images by `benchmark_order` are the publication allowlist; the final 500 reference cases and all other scores stay outside Git. Unknown photographer/session groups prevent a claim of exhaustive family independence, even after the checks below.

## Acquisition and views

Additional Megalith acquisition uses the existing pinned CC0 metadata revision and seeded source-pool candidate sample. Additional VizWiz acquisition uses the pinned official quality-issue training annotations and original training ZIP members, including member size/CRC and archive ETag validation. Existing acquisition manifests and source bytes are preserved; new acquisition receipts have separate paths. Every selected image retains its source license, attribution, source revision and hashes. Source model ratings and captions are not used for selection or embedding.

Megalith downloading allows two attempts with 10-second connect / 20-second read timeouts, a 60-second transfer resource bound and the adapter's existing 48 MiB/32-million-pixel limits. Failures are recorded and replaced in deterministic candidate order. Download availability and resource limits therefore affect the sampling frame. The implementation reuses verified complete cached files.

The completed acquisition added 5,000 Megalith and 1,300 VizWiz source images, with 544 failed Megalith acquisitions recorded separately. After preparation and identity/dimension guards, the combined canonical candidate pool contained 11,605 images: 8,601 Megalith, 2,204 VizWiz and 800 controls. Existing unused pilot training sources could supply rehearsal candidates; pilot development/test families were protected. Preparation recorded 1,495 rejected input records, including repeated cached inputs and protected/dimension cases; that count is not a count of failed network downloads.

Views use the existing full-frame sRGB PNG pipeline: EXIF orientation, embedded ICC conversion, white alpha compositing, no crop/padding/upscale, initial long-edge cap 1,536 pixels. The cap decreases by 10% if necessary to fit the common 2.65 MB request budget. Images below the common model dimension/area limits are recorded as rejections. Images are never substituted with placeholders.

Additional controlled families contain a canonical parent and three increasing configured levels of one blur, noise or JPEG operation. All four members use the same final size/budget. The original source remains unchanged. Configuration severity is not proof that a human will perceive every consecutive variant as worse; the transformed family remains entirely in rehearsal.

## Actual EmbeddingGemma 2 inference

- Model: `google/embeddinggemma-2`.
- Revision: `f9972ac22e696a105c482202646979e33e4b5a1e`.
- Checkpoint SHA256: `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`.
- Text-plus-vision parameters: 438,760,448, with `audio_config=None`.
- Image processing: maximum 280 soft tokens; output 768 dimensions, normalized to unit L2 length.
- FP32 inference. GPU TF32 is disabled.
- Pinned versions: PyTorch 2.14.0, Torchvision 0.29.2, Transformers 5.22.1, Sentence Transformers 6.2.0, NumPy 2.5.4 and Pillow 13.0.0. Prior/local CPU builds carry the `+cpu` suffix.

Prior embeddings are reused only for identical delivered PNG hashes. Sixty-four overlapping views were independently embedded on the new GPU worker before combining CPU/GPU vectors. Minimum cosine was 0.9999998808 and maximum coordinate difference was 0.0000039889, passing the predeclared minimum cosine 0.9999 and maximum coordinate difference 0.001 checks. This establishes numerical compatibility for this overlap set, not a new quality-rating claim.

The GPU worker is reached through an authenticated localhost SSH tunnel. To avoid uploading gigabytes of PNGs, public source images can be retrieved on the worker and rendered with Pillow 12.3.0 and the exact existing `_render` function. Both original source SHA256 and final delivered PNG SHA256 must match local artifacts before inference. Other views use their actual bytes. The remote model checkpoint was fetched anonymously with `hf download` at the pinned revision and checked against the local checkpoint hash. Account credentials are not needed or transmitted for model downloading.

Two local CPU fallback workers use the previous pinned environment and processing settings while GPU preparation runs. Their view-keyed caches and operation records remain outside Git. Transport changes do not change preprocessing, batch size or model precision. GPU calls serialize inference even when public-source retrieval is concurrent.

All 11,605 canonical candidate images have actual image embeddings. In the final 10,000-row retrieval matrix, 9,130 rows use an exact delivered-view vector. The other 870 use their explicitly recorded canonical family content representative: 750 controlled derivatives and 120 parents resized to satisfy the common family request budget. This is content grouping, not a claim that a blurred derivative has the same technical quality as its original. Graders and image statistics always receive the exact delivered view. `corpus-embeddings.npz` and `related-images.json` record the embedding-view hash and mode per row; the latter also lists controlled-family members separately.

The owned RTX 4090 Runpod worker was stopped through the plugin with status `EXITED` after 2,699 seconds. At the quoted $0.89/hour GPU rate, estimated compute was $0.6673. The later posted billing bucket reports $0.671291 GPU plus $0.004167 disk, or **$0.675457** for that interval. Retained-storage charges can post separately after compute stops. Operation receipts, billing snapshots and the executed worker-source snapshot are retained under `embedding-operations/`.

## Selection and duplicate review

Each new 500-case benchmark block has a fixed source frame of 600 Megalith and 150 VizWiz candidates. Candidate identity and frame membership are pinned before reference/model labels and before waiting for every embedding. Sixteen KMeans clusters (`n_init=20`, seed derived from 2026110110000) organize each frame. Source quotas are respected and selection is proportional to content-cluster capacities with seeded within-cluster ordering. The second block excludes all first-block frame families, including unselected frame members.

The complete candidate pool is clustered into 64 content groups. Capacity-capped square-root cluster quotas temper overrepresented groups for rehearsal sampling. Exact source, decoded-pixel and delivered-view hashes prevent duplicate selections. Conservative dHash candidate guards preserve family separation from the original pilot development/test sets and prior benchmark families. Such perceptual guards can also remove distinct low-detail images; recorded exclusions are candidates/guards, not claims of proven identity.

Embedding neighbours at high similarity are surfaced with original-image contact sheets for review. Semantic similarity alone is not a duplicate decision or a quality label. Confirmed duplicate/related families must not cross evaluation partitions. When later candidates overlap a frozen reference family, incoming non-benchmark members are rejected or grouped; reference labels are not changed in response to model outputs.

Both new benchmark blocks received full 500-image overview inspection, nearest-pair review and replacement-neighbour inspection before reference scoring. Confirmed capture-family conflicts included a photographed book/hand/table, a bowl, stage performers, a ceremony, a robotics event and a lake capture sequence; replacements preserved source quotas. Some replacements deliberately moved across content clusters to avoid selecting another photograph from the same event. All decisions and excluded identities are in the off-Git review ledgers.

The final corpus review inspected all 47 surfaced pairs with cosine at least 0.985. Four incoming rehearsal views were replaced, including a near-duplicate of an original benchmark seascape. Their 24 replacement neighbours were inspected before release. The remaining 43 pairs comprise 42 indeterminate nearly black pairs and one thematic red-panda pair with different capture dates. Distinct hashes plus weak visual information do not prove that dark images are unrelated; these remain available for review. A separate audit of 15 late-arriving benchmark neighbours found no additional supported identity. These reviews are bounded candidate checks, not exhaustive inspection of every pair or recovery of all event/photographer groups.

The 64 content clusters have approximate assistant theme names from six unique-family centroid exemplars per cluster. Largest realized groups are workshops/conferences (270 images, 2.70%), street/public gatherings (260, 2.60%), printed material (231, 2.31%) and residential architecture (230, 2.30%). Smaller specific groups include horses/riders (40) and zoo mammals (65). Repeated venues, organized events, school activities and kayaking remain source concentrations to inspect before a training split. Theme labels are descriptive cluster summaries, not per-image ground truth.

## Completed artifacts and validation

The full candidate manifest, embeddings, image paths, contact sheets, selected manifests and operation receipts are under ignored `visual-quality/data/labeling-10000-2026-11-01/`.

- `manifest.json`: frozen 10,000-image corpus; SHA256 `65f4df78e3e9d7aea62a0695f386462e3363957773eace00848c61bb5a75e859`.
- `benchmark-manifest.json`: frozen 1,500 references; original 500, 500 additional examples and 500 additional confirmation cases. `references/confirmation-selection.json` derives the confirmation review packets without example cases or model scores.
- `inference-rehearsal-manifest.json`: all 8,500 nonbenchmark images. `new-rehearsal-manifest.json`: new nonbenchmark cases R5001–R10000.
- `content-coverage.json`, `related-images.json`, `corpus-embeddings.npz`: cluster coverage, five nearest cross-family neighbours, controlled-family membership and vector provenance.
- `final-duplicate-review.json`, `final-replacements-proposal.json`, `review/`: final decisions, replacement evidence, contact sheets and archived prior identities.
- `corpus-integrity-audit.json`: all 10,000 file hashes match, 4,000 historical identities remain unchanged, all 1,500 benchmark IDs/views/orders match the frozen benchmark, exactly 1,000 images are on the publication allowlist, no explicit family crosses a partition, all training task masks remain false, and the 10,000 × 768 retrieval matrix is finite and L2 normalized.

The focused tests cover capacity-bounded content quotas, byte-for-byte view renderer parity, and controlled-family size/hash/source-preservation invariants: **3 passed**. The corpus Python files pass Ruff. Final artifact validation additionally checks the real corpus; small tests alone are not used as evidence that data acquisition or inference completed.

The scripts use the main visual-quality Python environment for Pillow 12.3 rendering, with the pinned embedding environment's site-packages appended for scikit-learn. Prepending the embedding environment would select its Pillow 13 renderer and change PNG hashes. `corpus_acquire.py` and `corpus_prepare.py` are resumable; `corpus_select.py` creates review proposals; `corpus_freeze.py` freezes reviewed benchmark proposals; `corpus_finalize.py` builds the initial complete corpus; `corpus_complete.py` applies an explicitly approved, hash-bound final replacement plan; `corpus_review.py` refreshes retrieval/coverage artifacts while retaining decisions for the same manifest; `corpus_audit.py` validates the resulting artifacts. Frozen benchmark views must not be rebuilt or replaced after reference scoring starts.
