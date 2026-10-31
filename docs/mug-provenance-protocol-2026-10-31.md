# Mug provenance and image-stats protocol

Checked on 31 October 2026. This is a design, not a result: no images have been generated and no paid calls were made. Provider policies change often, so re-check the vendor pages before freezing a run.

The study has two goals:

1. Find which image providers attach C2PA credentials and invisible watermarks that survive delivery, so AI content can be classified from provenance signals.
2. Collect pixel statistics, metadata and labels from the same images, for AI detection and for testing [image-stats](../image-stats/README.md).

## Prompt

The exact prompt is in [mug-provenance-prompt-2026-10-31.txt](mug-provenance-prompt-2026-10-31.txt). Send those bytes unchanged to every provider and record their SHA-256 with each request.

| Arm | Input | Why |
| --- | --- | --- |
| `reference_edit` | Prompt plus the reference mug photo | Main product-photo task |
| `text_only` | Prompt with the first paragraph replaced by a written description of the mug, no reference | Gives a fully generated label; see [Origin labels](#origin-labels) |
| `real_control` | The real mug photographed in a café with a C2PA-signing phone, such as a Pixel (`computationalCapture`) | Matched real image with object and scene held fixed |

Request 3:2 (1536×1024) where the provider supports it and record what comes back. Native output size is a known provider shortcut ([Fake or JPEG?](https://arxiv.org/html/2403.17608)). Film grain stays out of the base prompt because it would contaminate the noise metrics; it is a variant below.

## Provider matrix

| Provider and endpoint | C2PA | Invisible watermark | Verification |
| --- | --- | --- | --- |
| OpenAI `gpt-image-2` (API, ChatGPT) | Yes, issuer "OpenAI OpCo, LLC" | SynthID since May 2026 | [Content Provenance API](https://developers.openai.com/api/docs/guides/content-provenance) via [tracks/openai_provenance](../visual-triage/tracks/openai_provenance/README.md), plus offline c2pa |
| Google Nano Banana Pro / 2 | Yes from Nano Banana Pro (November 2025) | SynthID | SynthID Detector or the Gemini app, plus c2pa. App vs API coverage is unconfirmed. |
| Adobe Firefly | Yes, for fully generated pixels; may also be stored in Adobe's cloud | None stated | c2pa, plus a Content Credentials cloud lookup |
| BFL FLUX via api.bfl.ai | Yes, on all API image outputs ([BFL](https://help.bfl.ai/articles/3705384267-c2pa-in-flux-generated-images)) | Optional soft-binding watermark | c2pa, including the soft-binding assertion |
| Microsoft MAI-Image (Foundry, Copilot) | Yes ([Microsoft](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/content-safety/provenance-disclosure)) | None stated | c2pa |
| Amazon Nova Canvas (Bedrock) | Yes ([AWS](https://docs.aws.amazon.com/ai/responsible-ai/nova-canvas/overview.html)) | Amazon watermark | Bedrock watermark-detection API, plus c2pa |
| Meta AI | IPTC metadata | Yes, plus a visible label | XMP digital source type; no public API |
| Midjourney, xAI Grok, Seedream, local FLUX or SD | Not documented in the sources checked | Not documented | Expected-negative controls |

- `gpt-image-1` and `gpt-image-1.5` are excluded from OpenAI's provenance coverage because they are due for deprecation ([OpenAI help](https://help.openai.com/en/articles/8912793-provenance-signals-in-openai-generated-content)). Use `gpt-image-2`.
- Resellers such as fal, Replicate and OpenRouter are a separate arm. They may re-encode the bytes and drop the manifest. Record the endpoint for every request.
- Keep the raw API response bytes. Web previews may not keep the signed bytes ([c2pa-sources.md](../visual-triage/research/metadata-2026-10-12/c2pa-sources.md)).

## Origin labels

A `reference_edit` output is an AI edit of a real photograph, not a fully generated image. Providers may record `c2pa.opened` and `c2pa.edited`, list the reference as an ingredient, and use `compositeWithTrainedAlgorithmicMedia` as the source type. The current parser in [provenance.py](../visual-triage/tracks/metadata_v3/provenance.py) only accepts `c2pa.created` with `trainedAlgorithmicMedia`, so those outputs return `ai_generation_claim: None`. Label the arms separately: `text_only` as fully generated, `reference_edit` as AI edit, `real_control` as capture.

## Provenance record per image

Read these from the native bytes:

- SHA-256 of the bytes and of the decoded RGB; MIME type from the header, not the extension.
- Provider, endpoint, model ID, seed, request date, prompt SHA-256 and arm.
- C2PA, with each result kept separate:
  - manifest present, embedded or remote
  - signature valid
  - hard binding valid
  - trust status under the frozen offline policy
  - list of actions and `digitalSourceType`
  - ingredients, and whether the reference's hash is recorded
  - soft-binding algorithm
  - timestamp authority and certificate expiry
- XMP `Iptc4xmpExt:DigitalSourceType`, PNG `tEXt`/`iTXt` chunks and the EXIF `Software` tag.
- Every watermark detector run on every output, not only its own provider's, to measure cross-provider false positives.

A missing signal does not establish a real photograph.

## Survival ladder

Apply each step to a copy of every output and record which signals survive: C2PA presence, hard binding, soft binding, SynthID and Amazon's watermark.

1. Strip all metadata.
2. Re-encode PNG→JPEG at q95, q75 and q50.
3. Resize to 50%.
4. Crop to 80%.
5. Simulate a screenshot.

Report these controlled changes separately from native results.

## image-stats measurements

Run `analyse` at the default `max_edge` of 1536 and pass the native file as `source_path` so metadata and `source_blockiness` come from the original.

| What the set tests | Metrics | Expectation |
| --- | --- | --- |
| Deliberate shallow depth of field | `sharpness_peak`, `sharpness_median`, `focus_area` | Peak high, median and focus area low. Checks whether artistic blur reads as a defect. |
| Clean generator noise | `noise_sigma` (luma and chroma) | Lower than `real_control` in flat areas |
| Encoding fingerprints | `jpeg_quality_estimate`, `jpeg_subsampling`, `source_blockiness`, `bytes_per_megapixel`, native dimensions | Differ by provider. Audit for shortcuts; do not use them as detector features. |
| Upsampler traces | High-frequency spectral ratio, `spectrum` evidence map | Possible grid peaks |
| Colour fidelity | `cast_a`, `cast_b`, colourfulness on the mug | Close to the reference |

`read_metadata` currently reads no XMP, no PNG text chunks and has no C2PA or JUMBF check. Two candidate fields would cover this set without breaking its privacy rules: `c2pa_marker_present` (JPEG APP11 or PNG `caBX`) and `xmp_digital_source_type`. Keep them in the audit record and out of detector context, as in [metadata-methods.md](../visual-triage/research/expansion-2026-10-11/metadata-methods.md).

### Controlled prompt variants

Each variant changes one clause of the base prompt and has a known expected direction:

| Variant | Change | Expected |
| --- | --- | --- |
| `deep_focus` | f/8 | `focus_area` up |
| `dim_evening` | Dim evening light | `luma_p50` down, noise up |
| `backlit_noon` | Harsh backlit noon sun | Highlight clipping |
| `film_grain` | Add "fine film grain" | `noise_sigma` up |

These are the generator-side version of the synthetic blur, noise and JPEG ordering checks in the image-stats README.

## Reference fidelity

These measurements sit outside image-stats. Mask the mug (SAM or by hand), then measure:

- silhouette IoU after alignment
- CIEDE2000 colour difference on the painted regions
- LPIPS or DINOv2 similarity on the mug crop
- whether the handle is on the right, which catches mirroring
- focus contrast: the sharpness map inside the mask divided by the sharpness map outside it

## Sources

- [OpenAI content provenance guide](https://developers.openai.com/api/docs/guides/content-provenance)
- [OpenAI provenance help article](https://help.openai.com/en/articles/8912793-provenance-signals-in-openai-generated-content)
- [OpenAI: advancing content provenance](https://openai.com/index/advancing-content-provenance/)
- [BFL C2PA documentation](https://help.bfl.ai/articles/3705384267-c2pa-in-flux-generated-images)
- [Amazon Nova Canvas service card](https://docs.aws.amazon.com/ai/responsible-ai/nova-canvas/overview.html)
- [Microsoft Foundry content provenance](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/content-safety/provenance-disclosure)
- [Watermark.ws provider overview](https://watermark.ws/learn/how-ai-image-generators-mark-their-pictures), a secondary source for Google, Adobe and Meta rows
