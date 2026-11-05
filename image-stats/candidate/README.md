# Smaug + AI declarations + image-stats

One retained candidate: **review if Smaug ≥ 0.5 OR the exact file explicitly
declares AI generation**. Image-stats supplies diagnostic metrics and file
facts; its quality flags do not classify AI origin. The combined flag is not a
probability or proof of authenticity. Missing metadata means unknown origin.

Smaug is pinned to `abacusai/Smaug-AI-Image-Detector` revision
`c7c95951e04a87b884a4ad479401b1128c3903bd`, with three 528px crops and BF16 on CUDA.
We retain the released 0.5 threshold; native-only fitted thresholds and the
three-detector ensemble are not adopted. See [the concise trial record](TRIALS.md).

## Retained data

- [images.json](images.json): index of 100 images, 50 real-source and 50 generated.
- `records/<id>.json`: one final JSON per image, containing all five delivery views.
  Each view contains **`provenance`** (fields, declarations, full validation results
  and policy), **`image_stats`** (`analysis` metrics and `file_facts`), the frozen
  **`smaug`** observation, and the combined **`decision`**. For example,
  [fake-012](records/fake-012.json) and [real-001](records/real-001.json).
- [generation.json](generation.json): exact prompt, model, provider, parameters,
  prompt/request hashes and native hash for every generated image. Reference
  images were not uploaded. Provider checkpoint revisions were not disclosed.
- `assets/<view>/<id>.<extension>`: all 100 originals and 400 delivery copies,
  **500 files / 1,251,542,471 bytes**, retained byte-for-byte through Git LFS.
- [config.json](config.json): the candidate's pinned configuration and SDK policy.
- [capture.json](capture.json): frozen observation/source hashes and aggregate results.

Views are native, losslessly metadata-stripped PNG, JPEG Q75 within 1024px,
WebP Q75 within 768px, and a 1280×960 raster-copy simulation. The last is not a
live desktop screenshot. Metadata records are joined by exact file SHA-256;
copies never inherit their original's credentials. Training remains disabled.
Real-source licenses and the publication-allowlist order are retained in each
record. The real label is a camera-origin source assumption plus visual review.

Native credentials comprise **0 verified, 6 invalid, 34 untrusted and 60 missing**
under c2pa-python 0.38.0 / Rust SDK 0.91.0, offline with default trust anchors.
Ten unsigned XMP declarations overlap the 40 C2PA AI declarations. Parse errors
and two 32 MiB validator-limit errors remain explicit. An invalid/untrusted
AI declaration contributes a review flag and never becomes verified provenance.

## Use

```sh
git lfs install
git lfs pull
uv sync --project image-stats --extra candidate
uv run --project image-stats image-stats-authenticity photo.jpg --output inspection.json
uv run --project image-stats python image-stats/candidate/validate.py
```

For offline inference, first cache the pinned checkpoint, then pass `--offline`
and optionally `--cache-dir`. Smaug runs locally; C2PA validation does not fetch
remote manifests or certificate revocation data. The base `image-stats`
installation remains model-free; the optional `candidate` extra adds the GPU
model and pinned validator dependencies.

Frozen records reuse the measured trial observations with their original
latency/reuse provenance; exporting JSON is not new inference. A new CLI run
measures its own Smaug inference and computes provenance and statistics again.
Detailed retired experiments remain locally archived under ignored
`visual-quality/data/authenticity-consolidation-2026-11-05/`; they are not active
code dependencies. The generated files and final records are portable in Git/LFS.
