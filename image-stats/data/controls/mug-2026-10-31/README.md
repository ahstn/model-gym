# Mug control set, 31 October 2026

Generated mug photographs from fal and OpenRouter, made from the prompt in [the protocol](../../../../docs/mug-provenance-protocol-2026-10-31.md), with the reference image they edit. [manifest.jsonl](manifest.jsonl) records route, endpoint, request ID, arm, origin label, prompt and reference SHA-256 and file SHA-256. [requests.jsonl](requests.jsonl) holds the fal request records and the OpenRouter responses, with image data elided.

The set has three parts:

- **Web UI saves (6).** Saved from each service's web UI. The fal files are byte-identical to the fal API outputs: each matches the CDN file in its fal request record. The OpenRouter files' requests and prompt are not recorded.
- **OpenRouter API outputs (4).** Generated through the API on the same day to separate provider behaviour from the web UI. Each output is stored as decoded from the response. The prompt is the protocol prompt without its trailing newline, which is exactly what the fal requests sent.
- **Reference (1).** The input image from OpenRouter's [product-placement benchmark](https://openrouter.ai/benchmarks/media/images/product-placement), downloaded from `model-assets.openrouter.ai`.

| ID | Route and endpoint | Arm | Format as delivered |
| --- | --- | --- | --- |
| `fal-flux-2` | fal `fal-ai/flux-2-pro` | text only | JPEG 1024², q95 4:4:4 |
| `fal-krea-2` | fal `krea/v2/large/text-to-image` | text only | PNG 1024², no ancillary chunks |
| `fal-seedream` | fal `bytedance/seedream/v5/pro/text-to-image` | text only | JPEG 2048², q75 4:2:0 |
| `openrouter-krea-2-medium` | OpenRouter web UI | text only | WebP VP8 1024² (simple format) |
| `openrouter-nano-banana-pro` | OpenRouter web UI | reference edit | WebP VP8 1024² (simple format) |
| `openrouter-riverflow-2.6-pro` | OpenRouter web UI | reference edit | WebP VP8 1024² (simple format) |
| `api-openrouter-nano-banana-pro-chat` | OpenRouter chat completions, `google/gemini-3-pro-image` | reference edit | PNG 1024² |
| `api-openrouter-nano-banana-pro-images` | OpenRouter Image API, `google/gemini-3-pro-image` | reference edit | PNG 1024² |
| `api-openrouter-riverflow-2.6-pro` | OpenRouter Image API, `sourceful/riverflow-v2.6-pro` | reference edit | WebP VP8 1024² (simple format) |
| `api-openrouter-krea-2-medium` | OpenRouter Image API, `krea/krea-2-medium` | text only | PNG 1024², no ancillary chunks |
| `reference-product-placement` | OpenRouter benchmark asset | reference | PNG 1024² |

The four API calls cost $0.52. Riverflow and Krea only accept the Image API (`POST /api/v1/images`); chat completions returns 404 for them.

### The reference is AI-generated

The benchmark's "reference mug photo" carries a valid C2PA manifest signed by OpenAI OpCo, LLC ("OpenAI Media Service API", 21 August 2026). It records three actions:

- `c2pa.created`, with `trainedAlgorithmicMedia` and software agent `gpt-image` 2.0
- `c2pa.converted`
- `c2pa.watermarked.unbound`

Its checkerboard "transparency" background is painted into the RGB pixels, which is why image-stats flags 40% highlight clipping.

The reference-edit outputs therefore contain no camera pixels. Their `origin_label` is `fully_generated`, not `ai_edit`. Reference fidelity can still be measured against this image, but the set still has no `real_control`.

## Provenance

[scan_provenance.py](scan_provenance.py) walks the JPEG segments, PNG chunks and WebP RIFF chunks. It reads:

- EXIF and XMP
- IPTC-IIM, from a JPEG APP13 segment or a PNG `Raw profile type iptc` text chunk
- PNG text chunks
- byte signatures, such as JUMBF boxes and source-type strings

It then calls `extract_c2pa` from [visual-triage's provenance.py](../../../../visual-triage/tracks/metadata_v3/provenance.py) with the frozen offline policy (`c2pa-python` 0.38.0, Rust SDK 0.91.0, default trust anchors, no network). Results are in [provenance.jsonl](provenance.jsonl).

| ID | C2PA | Signer | Declared origin | XMP | IPTC |
| --- | --- | --- | --- | --- | --- |
| `fal-flux-2` | valid, untrusted signer | Black Forest Labs Inc. | `c2pa.created`, `trainedAlgorithmicMedia`, model `FLUX.2` | none | none |
| `fal-seedream` | valid, untrusted signer | Byteplus Pte. Ltd. | `c2pa.created`, `trainedAlgorithmicMedia`, model `dola-seedream-5-0-pro` | none | none |
| `api-openrouter-nano-banana-pro-chat` | valid, untrusted signer | Google LLC | `c2pa.created`, `trainedAlgorithmicMedia`; `c2pa.edited` "Applied imperceptible SynthID watermark" | `trainedAlgorithmicMedia`, Credit "Made with Google AI" | 2:110 Credit "Made with Google AI" |
| `api-openrouter-nano-banana-pro-images` | valid, untrusted signer | Google LLC | same as chat | same | same |
| `reference-product-placement` | valid, untrusted signer | OpenAI OpCo, LLC | `c2pa.created`, `trainedAlgorithmicMedia`, `gpt-image` 2.0; `c2pa.watermarked.unbound` | none | none |
| `fal-krea-2` | no manifest | | | none | none |
| `api-openrouter-krea-2-medium` | no manifest | | | none | none |
| `api-openrouter-riverflow-2.6-pro` | no manifest | | | none | none |
| `openrouter-krea-2-medium` | no manifest | | | none | none |
| `openrouter-nano-banana-pro` | no manifest | | | none | none |
| `openrouter-riverflow-2.6-pro` | no manifest | | | none | none |

None of the eleven has EXIF. "Valid" means the signature and the hard binding (the hash of the image bytes) both pass.

### By model and route

- **Google signs Nano Banana Pro, and OpenRouter's API keeps it.** Both OpenRouter routes return a PNG with three AI declarations: C2PA, XMP `DigitalSourceType` and an IPTC credit. All three survive the route.
- **The OpenRouter web UI strips it.** The web UI files are simple-format WebP, a single `VP8` chunk with no `VP8X` header, so they cannot hold metadata. At 27–58 KB per megapixel they were recompressed heavily. The benchmark page serves its images through Next.js image optimisation at quality 75, which produces the same format. The saved files do not match the benchmark gallery thumbnails byte for byte, so the exact page is unknown.
- **Krea does not sign.** Krea 2 Large through fal and Krea 2 Medium through OpenRouter's API are both bare PNGs. The OpenRouter web UI then converted Krea's PNG to WebP.
- **Riverflow 2.6 Pro does not sign its default output.** The API returns simple-format WebP, the endpoint's default `output_format`. A PNG request was not tested.
- **fal keeps credentials intact.** The FLUX 2 and Seedream manifests pass their hash checks, and the web UI files are the API bytes.
- **"Untrusted" comes from the frozen policy.** No signer here, including OpenAI and Google, chains to the pinned SDK's default anchors. Checking against the C2PA conformance trust list would need a separate, versioned trust policy.

### What the manifests say about edits

- **Google labels a reference edit as created, not composite.** The Nano Banana Pro manifest uses `c2pa.created` with `trainedAlgorithmicMedia`. It also lists one ingredient with relationship `inputTo`, which has no title, no format, no hash and no manifest of its own. The protocol expected `c2pa.opened` or `c2pa.edited` with `compositeWithTrainedAlgorithmicMedia`; Google did not use either. So the parser's `ai_generation_claim` is `True`, and the reference's own OpenAI credential is not carried forward.
- **Watermarks are declared, not detected.** Google's manifest says a SynthID watermark was applied. OpenAI's says the reference was watermarked (`c2pa.watermarked.unbound`); since May 2026 that watermark is SynthID. Neither watermark was checked with a detector. The ASCII string "SynthID" in the Nano Banana Pro bytes comes from the manifest text, not the watermark.

## Pixel statistics

`image-stats analyse` ran at the default `max_edge` of 1536 ([stats.jsonl](stats.jsonl), [stats.run.json](stats.run.json)) and at 1024 ([stats-1024.jsonl](stats-1024.jsonl)). Seedream is the only image above 1024², so the 1024 run is the one to compare. No image errored or was gated. There is no image-stats reference corpus in the repo, so values are raw, not percentiles.

| ID (1024 run) | `sharpness_peak` | `sharpness_median` | `focus_area` | `high_frequency_ratio` | `noise_sigma` | `chroma_noise_sigma` | `blockiness` | `highlight_clip` | `colourfulness` | `cast_b` | bytes/MP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fal-flux-2` | 3290 | 4.0 | 0.10 | 0.168 | 0.17 | 0.020 | 1.06 | **0.088** | 29.2 | 9.7 | 270k |
| `fal-krea-2` | 762 | 7.7 | 0.13 | 0.069 | 0.34 | 0.219 | 1.03 | 0.005 | 35.3 | 9.9 | 1.28M |
| `fal-seedream` | 1961 | 7.1 | 0.04 | 0.044 | 0.18 | 0.033 | **1.86** | 0.012 | 34.3 | 8.0 | 78k |
| `openrouter-krea-2-medium` | 451 | 5.2 | 0.08 | 0.033 | 0.05 | 0.030 | 1.10 | 0.026 | 22.5 | 6.1 | 27k |
| `openrouter-nano-banana-pro` | 3285 | 11.2 | 0.09 | 0.064 | 0.10 | 0.079 | 1.09 | 0.011 | 45.9 | 15.4 | 50k |
| `openrouter-riverflow-2.6-pro` | 3487 | 17.6 | 0.12 | 0.067 | 0.16 | 0.089 | 1.09 | 0.007 | 40.9 | 10.2 | 58k |
| `api-openrouter-nano-banana-pro-chat` | 3318 | 22.7 | 0.08 | 0.072 | 0.76 | 0.287 | 1.07 | 0.025 | 48.0 | 14.5 | 1.26M |
| `api-openrouter-nano-banana-pro-images` | 3371 | 20.5 | 0.07 | 0.070 | 0.77 | 0.309 | 1.07 | 0.034 | 51.0 | 15.7 | 1.25M |
| `api-openrouter-riverflow-2.6-pro` | 3149 | 6.1 | 0.06 | 0.062 | 0.17 | 0.091 | 1.19 | 0.005 | 50.7 | 16.0 | 143k |
| `api-openrouter-krea-2-medium` | 5478 | 5.4 | 0.03 | 0.094 | 0.26 | 0.174 | 1.04 | 0.001 | 32.5 | 9.8 | 1.10M |
| `reference-product-placement` | 3019 | 7.3 | 0.06 | 0.067 | 0.21 | 0.113 | 1.06 | **0.405** | 29.3 | 2.5 | 835k |

Two images raise flags. Seedream gets `blocking` from its q75 4:2:0 JPEG; the C2PA hash binding proves BytePlus encoded those bytes. The reference gets `highlight_clipping` from its painted checkerboard.

### Delivery encoding changes the noise metrics

The two Nano Banana Pro API PNGs have luma noise of 0.76–0.77, while the web UI WebP has 0.10. To test whether the re-encode causes this, the API PNGs were re-encoded as WebP at quality 75 with Pillow (libwebp 1.6.0) and analysed again ([derived/webp-q75](derived/webp-q75)):

| Source PNG | `noise_sigma` PNG → WebP q75 | `chroma_noise_sigma` | bytes/MP after |
| --- | --- | --- | ---: |
| `api-openrouter-nano-banana-pro-chat` | 0.76 → 0.03 | 0.29 → 0.03 | 40k |
| `api-openrouter-nano-banana-pro-images` | 0.77 → 0.08 | 0.31 → 0.05 | 39k |
| `api-openrouter-krea-2-medium` | 0.26 → 0.02 | 0.17 → 0.02 | 25k |
| `fal-krea-2` | 0.34 → 0.05 | 0.22 → 0.03 | 31k |

One lossy WebP pass cuts estimated noise by 7–24×. That lands each image in the range of the OpenRouter web UI files (0.05–0.16). Noise is therefore a measure of the delivery path as much as of the generator. A detector trained on raw API PNGs and tested on web saves would learn the encoder, not the model. Compare noise only between images with the same encoding history.

### Other findings

- **The shallow depth of field reads as intended.** Every output has 3–13% of the frame near peak detail, with the median tile far below the peak. In a camera corpus these values would rank as soft. This is the "deliberate blur" case from the image-stats README: rank sharpness against a corpus of the same kind of scene before treating it as a defect.
- **Compare images at the same size.** At the default 1536, Seedream's `sharpness_peak` is 886; downscaled to 1024 it is 1961.
- **FLUX 2 stands out on two measures.** Its `high_frequency_ratio` is 0.168, against 0.03–0.09 for the rest. It also has 8.8% of pixels clipped to white, where the window is blown out.
- **Nano Banana Pro's colour is consistent across runs.** Three independent generations give `cast_b` 14.5–15.7 and colourfulness 46–51. The warm shift is the model's, not chance. These are whole-frame values that include the café, so mug colour fidelity needs a mask (see [the protocol](../../../../docs/mug-provenance-protocol-2026-10-31.md#reference-fidelity)).

## Open items

- Check SynthID with a detector (SynthID Detector or the Gemini app) on the Nano Banana Pro PNGs and the reference. Use Krea as the negative control.
- Check the reference with OpenAI's Content Provenance API. This needs an OpenAI API key.
- Request Riverflow with `output_format: png` to see whether Sourceful signs any format.
- Measure fidelity for the reference edits against `reference-product-placement`, on a masked mug crop.
- Photograph a real mug as `real_control`. The benchmark reference cannot serve as one.
- Run the survival ladder on the signed files. The WebP q75 variant above already shows that one lossy pass removes every C2PA, XMP and IPTC signal.

## Reproduce

```sh
# Provenance: needs the pinned SDK in its own environment
uv venv /tmp/c2pa-env && uv pip install --python /tmp/c2pa-env/bin/python "c2pa-python==0.38.0" pillow
/tmp/c2pa-env/bin/python scan_provenance.py images/* > provenance.jsonl

# OpenRouter API outputs (billed); keys from 1Password
op run --env-file <(echo 'OPENROUTER_API_KEY=op://development/OpenRouter API Key/credential') -- \
  python fetch_openrouter.py OUT images sourceful/riverflow-v2.6-pro riverflow \
  --reference images/reference-openrouter-product-placement.png

# Pixel statistics, from image-stats/
M=data/controls/mug-2026-10-31
uv run image-stats analyse --manifest $M/manifest.jsonl --id-key id --path-key image_path \
  --source-key image_path --output $M/stats.jsonl
uv run image-stats analyse --manifest $M/manifest.jsonl --id-key id --path-key image_path \
  --source-key image_path --max-edge 1024 --output $M/stats-1024.jsonl
uv run image-stats analyse --manifest $M/derived/webp-q75/manifest.jsonl --id-key id --path-key image_path \
  --source-key image_path --max-edge 1024 --output $M/derived/webp-q75/stats-1024.jsonl
```

The fal request records come from `GET https://api.fal.ai/v1/models/requests/by-endpoint?endpoint_id=…&expand=payloads`, using `FAL_KEY` from the same vault.
