# image-stats

Deterministic, model-free image statistics for three jobs:

1. **Labelling.** Store the statistics next to every image so you can stratify samples, audit teacher scores (for example a 4.0 on a frame with almost no detail), and check that synthetic blur, noise, and JPEG variants move the metrics in the expected order.
2. **Teaching.** Turn the statistics into a short plain-language block for a teacher prompt, ranked against a corpus. Attach optional evidence images. Gate frames that have no content before paying for a model call.
3. **Inference.** `analyse()` is a pure function of the pixels. It uses only `numpy`, `opencv-python-headless`, and `pillow`, needs no network or GPU, and has no non-commercial weights. It takes about 20 ms at 512×512 and about 0.2 s at 1536×1024 on an Apple Silicon laptop.

## Run

Python 3.11+ and `uv` are required. Run commands from this directory.

```sh
uv sync
uv run image-stats metrics                       # metric definitions
uv run image-stats analyse photos/ --output out/stats.jsonl --workers 8
uv run image-stats calibrate out/stats.jsonl --output out/reference.json --name pilot-photos
uv run image-stats describe out/stats.jsonl --reference out/reference.json --output out/prompts.jsonl
uv run image-stats evidence photo.jpg --output-dir out/evidence
uv run pytest
```

For frozen visual-quality views, read the manifest directly:

```sh
uv run image-stats analyse --manifest ../visual-quality/data/label-pilot-12/views/views.jsonl \
  --id-key view_id --path-key image_path --output out/views-stats.jsonl
```

`analyse` writes one JSON line per image, plus a `.run.json` summary with flag counts and software versions. An unreadable image produces an `error` record, and the command exits with status 1. Other images are still written.

## Python API

```python
from image_stats import Reference, analyse, evidence_maps, read_metadata, render

analysis = analyse(image_bytes)          # bytes, path, PIL image, or uint8/float array
if analysis.gate:                        # "uniform_frame": apply the policy floor and skip the model
    ...
reference = Reference.load("reference.json")
prompt_block = render(analysis, reference)
maps = evidence_maps(image_bytes)        # {"sharpness", "noise_residual", "clipping", "spectrum"} PIL images
metadata = read_metadata(original_bytes) # EXIF camera fields, JPEG quality estimate, bytes per megapixel
```

An example prompt block for a blurred photo:

```text
Automatic pixel measurements, ranked against 20 images in 'skimage-sample'. They can be wrong for unusual
scenes, deliberate blur, or artistic intent; use them to check what you see, not to replace it.
- Detail in the sharpest region: very low (0th percentile)
- Frame area close to peak detail: 29%
- Luma noise in flat areas: low (5th percentile)
- Pixels crushed to black: 10%
...
```

## What it measures

`uv run image-stats metrics` prints the full registry from [`metrics.py`](src/image_stats/metrics.py). There are 33 metrics in seven groups:

| Group | Metrics |
|---|---|
| Exposure | luma mean, median, p01, p99; shadow, highlight, and per-channel clipping; dark-channel haze prior |
| Contrast | RMS, dynamic range, Michelson (p01/p99), median local tile contrast, luma entropy |
| Sharpness | Laplacian variance (global, sharpest tile, median tile), focus area, Tenengrad, high-frequency spectral ratio, Canny edge density, gradient anisotropy (motion smear) |
| Noise | Immerkaer sigma for luma and chroma, measured in the flattest unclipped tiles |
| Colour | Hasler-Suesstrunk colourfulness, mean saturation, Lab chroma p99, mean a\*/b\* cast |
| Content | flat-tile area, palette size, top-colour share |
| Compression | 8×8 blockiness at native resolution |

`read_metadata` reads the original file. It returns format, dimensions, frames, bytes per megapixel, alpha, ICC, camera make and model, software, exposure time, f-number, ISO, focal length, JPEG quality estimate, and chroma subsampling. It never reads capture time, GPS, or serial numbers.

Flags are scale-free rules in [`flags.py`](src/image_stats/flags.py): `uniform_frame` (the only gating flag), `near_black`, `near_white`, `shadow_clipping`, `highlight_clipping`, `graphic_like`, `monochrome`, and `blocking`. Sharpness and noise have no absolute flags. Their raw values depend on content and resolution, so judge them as percentiles against a reference built from the same corpus and `max_edge`.

## Rules that keep the numbers comparable

- **Analyse the pixels the model sees.** Decoding follows the visual-quality view policy: EXIF transpose, embedded ICC to sRGB with perceptual intent, alpha composited on white, then a Pillow LANCZOS thumbnail to `max_edge` (1536 by default, never upscaled). An original analysed at 1536 therefore goes through the same steps as its frozen view.
- **Read file facts from the original.** Views are re-encoded PNGs with no EXIF, and resizing destroys the 8×8 JPEG grid. Pass `source_path` (or `--source-key`) to take metadata and `source_blockiness` from the original. `views.jsonl` does not store the original path, so join it from the dataset manifest first.
- **Version everything.** Every record carries `ALGORITHM_VERSION`. A `Reference` records the version and `max_edge` it was built with, and `describe` refuses a mismatch. Bump the version whenever a formula or default changes.
- **Give the model ranks, not raw numbers.** A teacher cannot interpret `laplacian_var=12.4`, and numbers invite anchoring. Measure the effect before relying on it: score the reviewed cases with and without the block and compare error against the reference judgments per quality band.

## Limitations

- Validation so far uses synthetic images with known blur, noise, clipping, and JPEG quality, plus scikit-image sample photos. Flag thresholds are initial guesses. Recalibrate them against reviewed labels before using them for anything except the uniform-frame gate.
- `noise_sigma` reads about 5% low on flat synthetic noise and lower on busy photos, because it takes the 10th percentile of tiles. Use it as a rank.
- `gradient_anisotropy` also rises for scenes with strong parallel lines, such as buildings and fences.
- The JPEG quality estimate reads high below about quality 20, because 8-bit tables clip.
- Documents photographed on textured paper are not flagged `graphic_like`. Detecting text, screenshots, and non-photographic scope needs OCR or a CLIP/SigLIP zero-shot score. Add those as separate optional extractors so the core stays dependency-light and licence-clean.
