# C2PA feasibility and offline verification

Checked on 12 October 2026. The signed AI samples and camera controls support parser tests and controlled changes to metadata. They do not form a balanced native benchmark with independently verified camera and generator coverage. No paid model calls were made during this research.

## What each check means

[The C2PA specification](https://spec.c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification) distinguishes assertions, signed claims, content bindings and signer trust. A manifest can be present while its signature fails. A valid signature can accompany a binding failure if someone copies the manifest onto another asset. A valid signature and matching hard binding can still have an untrusted signer. Signed assertions describe provenance claims; they do not establish the truth of a depicted event. Missing credentials do not establish AI origin.

The code keeps these results separate. It does not infer trust from a producer name or add a certificate from an image to the trust store. An AI declaration must come from an explicit `c2pa.created` source-type assertion. The code extracts it only after overall validation passes. Expired, mismatched or invalid credentials cannot supply an origin claim to the model. A camera capture declaration remains a claim, even when validation passes.

## Verified samples and rights

| Source | Acquired material and offline result | Role and rights limit |
| --- | --- | --- |
| Native Firefly files acquired by the AI source workstream | 44 local files passed signature and asset-hash checks. Each has an active signed `trainedAlgorithmicMedia` creation assertion. All signers are untrusted under the frozen SDK defaults. | Native generated-source candidates. The source workstream retains the dataset licence and origin evidence. These are not trusted-producer certificates under this policy. |
| [Leica vendor demonstration](https://leica-camera.com/en/photography/content-credentials) | One 17.2 MB vendor-linked JPEG has an Adobe-edited active manifest and a Leica capture ingredient. Active signature and hard binding pass; signing certificates are untrusted under this policy. | Camera provenance derivative control, not an untouched native capture. No open reuse licence was established. Keep the file local and excluded from a redistributable native benchmark. |
| [Pixel Camera developer gallery](https://sherifhanna-google.github.io/) at repository commit `aafd4eba624d7a536fe7f286cbf7b8ed5bf5f1fc` | Three native production capture test vectors, 7.8 MB total, declare `computationalCapture`. Their signatures and hard bindings pass. Overall validation is invalid because of certificate and timestamp trust failures under the frozen SDK defaults. | Credible camera-origin technical controls. The pinned repository has no LICENSE or README terms. Local verification only; redistribution and dataset reuse remain unresolved. The gallery's published trust results use another trust profile and cannot replace local checks. |
| [C2PA public test files](https://github.com/c2pa-org/public-testfiles) at commit `22beccc075707475b038d8789d0136c009e43143` | One Nikon legacy test JPEG passes signature and binding checks but fails overall credential validation. | Conformance control, not independent real-world gold. Repository licence is CC BY-SA 4.0. Preserve attribution and share-alike terms for redistribution. |
| [C2PA Python SDK fixture](https://github.com/contentauth/c2pa-python) at tag `v0.38.0` | One small CA fixture passes signature and binding checks with an untrusted signer. | SDK technical control. Repository licence is Apache 2.0; no claim of camera-origin benchmark evidence is made. |
| [TrustMyContent HF collection](https://huggingface.co/datasets/TrustMyContent/C2PA_Certified_Image_Authenticity) | The repository card states that its signed `ai_detection_result` comes from the UncovAI detector. | Reject as independent origin gold. A signature authenticates the detector assertion, not the class label. Card declares CC BY 4.0. No image corpus was downloaded. |

Primary [OpenAI guidance](https://help.openai.com/en/articles/8912793-c2pa-in-chatgpt-images) states that ChatGPT and DALL·E 3 API images include C2PA metadata. Common processing can remove it. These sources could supply files, but web previews may not keep the original signed bytes. This pass did not establish a bounded, licensed set of native OpenAI or Bing files.

## Frozen SDK and exact API

The verified package is `c2pa-python==0.38.0`, with `c2pa.sdk_version() == "0.91.0"`. It was installed in a separate temporary environment. The parent workstream manages runtime dependencies. The code rejects other versions to keep the validation policy fixed.

[The pinned context guide](https://github.com/contentauth/c2pa-python/blob/v0.38.0/docs/context-settings.md) documents offline settings. The exact policy is:

```json
{
  "core": {"allowed_network_hosts": []},
  "verify": {
    "remote_manifest_fetch": false,
    "ocsp_fetch": false,
    "verify_after_reading": true,
    "verify_trust": true,
    "verify_timestamp_trust": true
  }
}
```

Trust overrides are `null`: no `trust`, `allowed_list` or `user_anchors` settings are supplied. The pinned SDK default anchors apply. Without online OCSP, these checks cannot confirm the current revocation state. Credentials stored only at remote URLs remain unavailable. The parser does not fetch those URLs.

[The Reader API](https://github.com/contentauth/c2pa-python/blob/v0.38.0/docs/usage.md) is `Reader(mime_type, binary_stream, context=context)`, with `Context.from_dict(settings)`. Read status through `get_active_manifest()`, `get_validation_results()` and `get_validation_state()`. `is_embedded()` is a method. `is_valid` is a reader-handle property and must not be used as an authenticity result. MIME type comes from the byte header, since the initial Firefly files have PNG names but JPEG bytes.

[provenance.py](../../tracks/metadata_v3/provenance.py) returns only `status`, `manifest_present`, `signature_valid`, `hard_binding_valid`, `trust_status` and `ai_generation_claim`. All values use booleans, null or fixed enums. Producer names, file paths, titles, IDs, GPS and raw assertions stay outside the model context. Input files are limited to 32 MiB.

## Controlled byte tests

Three local Firefly controls keep the decoded RGB pixels unchanged. Removing JPEG APP11 credentials returns `no_manifest`. Copying valid credentials onto another image keeps the signature valid but breaks its hard binding. Changing one COSE signature byte fails the signature check. Its hard binding stays valid. Both invalid controls suppress the AI declaration. Report these controlled changes separately from native source results.

Raw reports, download records, exact source revisions and hashes are in `data/metadata-v3/research/c2pa/`. Local validation passed 26 provenance tests and 13 reusable control-builder tests, including byte controls. Ruff checks also passed. The [control builder](../../tracks/metadata_v3/build_controls.py) returns a separate `cases` manifest. Context controls remove camera fields, native resolution or all context, or perform balanced camera swaps. They keep native bytes and delivered measurements fixed. Swaps use each test donor once, exclude the recipient's source group and balance donor classes within each recipient class. C2PA byte controls keep decoded RGB pixels fixed. Before use, each must pass its stated signature or binding failure check. Controls use test cases only and keep each base source group.

Exa ran five distinct queries with five requested results each: 25 requested search results. The HF repository-details tool worked. HF dataset search returned a tool-list error. The official HF API found repositories and checked revisions instead. Primary vendor pages, repository files and local checks were used to screen search results. Native signed AI files were found. Camera coverage is still too limited to form a balanced, licensed set with currently valid credentials.
