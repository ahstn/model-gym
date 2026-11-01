# Corpus preparation and resume

All paths below refer to the current machine. Original data and environments remain under `/home/ahstn/git/model-gym/visual-quality`; scripts and frozen experiment artifacts are in this worktree. Do not rerun a frozen stage over published manifests without preserving its superseded provenance.

The source preparation and embedding scripts, candidate manifest, checkpoint revision and inference environment are pinned while embedding workers run. Each worker resumes only when its manifest, script, configuration, software and completed-vector hashes match. Two workers use six CPU threads each; their logs and resumable vectors are in the bulk data directory.

```sh
/home/ahstn/git/model-gym/visual-quality/data/embeddinggemma2-clustering-2026-10-31/.venv/bin/python \
  visual-quality/experiments/rehearsal-4000-2026-11-01/corpus_embed.py --shard 0 --threads 6
```

Run shard `1` in a second process. The existing 384-image vector artifact may supply identical views only when its pinned inference configuration and software match; every reuse is recorded in the shard metadata.

Clustering uses the original view environment, with the existing embedding environment appended to the import path for scikit-learn and SciPy. This preserves the Pillow and NumPy versions used to render delivered views. This example runs the review stage after the first 1,000 candidate vectors exist:

```sh
PYTHONPATH=/tmp/model-gym-rehearsal-20261101/visual-quality/src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/ahstn/git/model-gym/visual-quality/.venv/bin/python - <<'PY'
import runpy
import sys
from pathlib import Path

experiment = Path('/tmp/model-gym-rehearsal-20261101/visual-quality/experiments/rehearsal-4000-2026-11-01')
sys.path.append('/home/ahstn/git/model-gym/visual-quality/data/embeddinggemma2-clustering-2026-10-31/.venv/lib/python3.13/site-packages')
sys.path.insert(0, str(experiment))
sys.argv = ['corpus_select.py', 'preview']
runpy.run_path(str(experiment / 'corpus_select.py'), run_name='__main__')
PY
```

Review the candidate and historical-neighbour contact sheets, record `benchmark-candidate-review.json`, then change the stage to `benchmark` to select the early 500. The root agent can begin blind reference review from `benchmark-manifest.json`. Six manually approved scope-boundary candidates are reserved by `benchmark-scope-selection.json`; its judgments must remain hidden from the blind reviewers.

After all 5,500 candidate vectors finish, run `corpus_neighbours.py` with the same environment, inspect its contact sheets and complete `cross-boundary-review.json`. Confirmed historical exposure is the sole permitted reason to replace an early benchmark case; keep superseded reference provenance. Then run `corpus_finalize.py` to construct the final 4,000 manifest, including canonical-parent reconstruction for the 100 controlled families. `corpus_cluster_review.py` produces cluster exemplars, coverage counts and cross-family related-image links.

The completed full-pool review confirmed one historical near-duplicate: original R0045/C0725 and C5253. `benchmark_revision.py` proposed C0004 by deterministic hash ordering within the original fresh VizWiz pool, preferring the same content cluster. After root visual acceptance of four neighbour sheets, it published `benchmark-manifest-v2.json` and `benchmark-revisions.json`. The original benchmark manifest and its reviews remain intact. Both members of the confirmed historical pair are excluded from the final corpus.

The active finalization commands therefore use explicit versioned inputs:

```text
corpus_neighbours.py --benchmark-manifest /absolute/path/to/benchmark-manifest-v2.json --output-stem cross-boundary-v2
corpus_finalize.py --benchmark-manifest /absolute/path/to/benchmark-manifest-v2.json --cross-boundary-stem cross-boundary-v2
```

The first command has already completed; do not overwrite its artifacts. The v2 review carries forward the original reviewed pairs, records the resolved replacement and documents visual checks for the newly surfaced preview pairs. Finalization pins the active benchmark, revision receipt and versioned neighbour/review paths. `corpus_validate.py` follows those declared paths by default. The original neighbour and proposal generator sources are retained in `pipeline-snapshots/` with their original hashes; pinned prepare/embed/select scripts were not altered.

Image statistics use the original view environment plus the existing OpenCV dependency directory:

```sh
PYTHONPATH=/tmp/model-gym-image-stats-deps-20261101:/tmp/model-gym-rehearsal-20261101/image-stats/src \
/home/ahstn/git/model-gym/visual-quality/.venv/bin/python \
  visual-quality/experiments/rehearsal-4000-2026-11-01/corpus_stats.py \
  --manifest visual-quality/experiments/rehearsal-4000-2026-11-01/manifest.json --workers 4
```

Add `/tmp/model-gym-rehearsal-20261101/visual-quality/src` to that `PYTHONPATH` for `stats_prompts.py` and `corpus_validate.py`. The former freezes the 3,200-family StatsReference, exact tied-percentile QC records, 4,000 text blocks and 500 matched background donors. The latter checks identities, protected partitions, family isolation, benchmark preservation, calibration exclusions, donor uniqueness and false training masks before model calls.

`test_corpus.py` tests selection capacity, zero-quota behavior, duplicate-vector behavior, the actual KMeans seed and protected candidate exclusions. `test_corpus_stats.py` tests cache separation for different original files that share a view and detects source changes before cached results are reused.
