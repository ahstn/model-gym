# Native real photographs for the metadata trial

VISION is the strongest source found for this trial. Its original camera JPEGs retain the four requested exposure fields and have clear research-use rights. The final acquisition contains 160 untouched JPEGs from 20 devices with 20 distinct model names. It contains 40 train, 40 development and 80 test images, with no device or model name shared across partitions. These files support a native-camera cohort. They do not establish balanced EXIF coverage across real and AI images.

| Source | Evidence and access | Decision |
| --- | --- | --- |
| [VISION](https://lesc.dinfo.unifi.it/VISION/) | The official manifest has 7,565 natural-camera JPEG URLs across 35 devices. The [README](https://lesc.dinfo.unifi.it/VISION/README.txt) grants CC BY-SA 4.0 rights. The [paper](https://link.springer.com/article/10.1186/s13635-017-0067-2) documents native captures and separate social-media versions. | Acquired 160 original files from `/images/nat/`. Device and model names are separate between optimisation and final evaluation. |
| [FlickrExif](https://huggingface.co/datasets/CTU-OU/FlickrExif) | CC BY 4.0; broad camera and exposure labels. The current Hub record requires manual access approval. Its authors warn that image content correlates with acquisition labels. | Excluded from acquisition because access is gated. Useful warning against metadata shortcuts. |
| [PairCams](https://huggingface.co/datasets/CTU-OU/PairCams) | The current Hub record requires manual access approval. | Excluded; no account or agreement submitted. |
| [RAISE](https://loki.disi.unitn.it/RAISE/) | Guaranteed camera-native RAW images, three camera models and four photographers; [non-commercial research use](https://loki.disi.unitn.it/RAISE/guide.html). The official download requires a form and distributes NEF/TIFF rather than native JPEG. | Reserve source. VISION gives broader device coverage at lower download cost. |
| [Dresden](https://www.kaggle.com/datasets/micscodes/dresden-image-database) | Strong camera-origin labels, but the mirror combines a CC0 badge with restrictive copyright text and reports the original host as unavailable. | Excluded from this acquisition pass. Mirror rights are not a clear basis for reuse. |
| [MIRFLICKR](https://press.liacs.nl/mirflickr/) | Creative Commons photographs with camera metadata available for about 85% of images. The project states that its image files lack embedded EXIF; metadata is supplied separately. | Reserve for metadata-missing real photographs, not a source of untouched embedded EXIF. |
| [TrustMyContent](https://huggingface.co/datasets/TrustMyContent/C2PA_Certified_Image_Authenticity) | Ungated C2PA-bearing files. The card states CC BY 4.0, but the Hub licence metadata is empty. Real/AI labels come from the UncovAI detector. | Excluded as label ground truth. A signed detector prediction is not verified capture provenance. |

## Acquisition checks

The reviewed cohort uses 481,906,498 bytes. Its 160 original files are saved without conversion in `data/metadata-v3/candidates/real/`. The final manifest, `data/metadata-v3/research/real/vision-native-160-v2.json`, records each URL, SHA-256, decoded RGB hashes, pHash, dimensions, split, device group, model name, camera values, licence and credit. Its SHA-256 is `fe6f592ab20b3e8b1c0fe6f8397bf7856d86754c39097dd9187a146e25433cf2`. The original selection, excluded files, review sheets, official README and source list are preserved locally for audit.

| Native field | Present / 160 | Observed range |
| --- | ---: | --- |
| ISO | 159 | 25–2,000 |
| Exposure time | 159 | 0.000411–0.12 seconds |
| FNumber | 152 | f/2–f/2.6 |
| Focal length | 160 | 2.03–4.9 mm |
| All four direct fields | 151 | Train 40/40; development 40/40; test 71/80 |

Eight Wiko photos have native `ApertureValue` 2.65 in APEX units, but no `FNumber` tag. This can be converted to an f-number with `2 ** (ApertureValue / 2)`; the direct-field counts above do not make that conversion. One LG photograph lacks ISO and exposure time. These cases remain in the cohort as natural missingness, rather than being removed to make coverage look complete. Native resolution spans 0.69–20.66 megapixels. Phones and tablets give a narrow aperture and focal-length range; this is not broad camera-setting validation.

The script first ranks unique device models and their URLs with SHA-256 using seed `metadata-v3:real:20261012`. It assigns five devices to train, five to development and ten to test, with eight photos per device. Two reviewers then checked 267 unique candidates for photographic content and recognisable shared locations; all 160 retained images were reviewed. They used no detector outputs. Natural blur, noise, low quality and physical objects with visible photographic context remain eligible. Full-frame flat designs without physical context do not.

Visual review found three location clusters across splits: a modern campus block, a residential/shopping complex and an older institutional courtyard. The original 40 training images remain unchanged. Development and test views of those locations were replaced with the next eligible ranked candidates. Two location-bound groups were replaced in full: development iPhone 4 (`D09`) with Lumia 640 LTE (`D17`), and test OnePlus A3000 (`D25`) with Huawei Ascend (`D33`). [Eligibility decisions](real-eligibility.json) preserve 46 image flags; [device replacements](real-device-replacements.json) preserve the group rules and reviewed alternatives.

Exact hashes and 64-bit DCT pHash were compared with 1,580 prior processed and native assets from all four frozen tracks, the initial 30-image panel and both 60-image benchmarks. All prior assets were available. No final image had an exact overlap or pHash distance of six bits or fewer against prior images or another new image. All final native-byte hashes were rechecked. Visual scene checks were necessary because these hashes missed different views of the same locations.

The final splits isolate device and model names and remove recognised shared locations. They do not prove geographic or scene independence: generic interiors, walls and vegetation can hide a common location. Within-split scene repeats and the shared VISION collection remain limitations. Report accuracy as a test of this cohort, not broad camera or scene generalisation.

To repeat the source acquisition from `visual-triage`, with prior benchmark images present:

```sh
uv run python tracks/metadata_v3/acquire_real.py \
  --exclusions research/metadata-2026-10-12/real-eligibility.json \
  --device-replacements research/metadata-2026-10-12/real-device-replacements.json \
  --output data/metadata-v3/research/real/vision-native-160-v2.json
```

Preserve original bytes for local extraction, then send the same metadata-free processed image to every model arm. Omit filenames, source IDs, device identities, timestamps, GPS and free-text fields from model context. Separate native observations from controlled metadata removal or swapping. Do not treat missing EXIF as AI evidence or an EXIF camera claim as authenticated capture provenance.

## Search record

On 12 October 2026, six Exa searches requested 40 results in total. Searches covered RAISE, Dresden, open camera metadata, VISION, FlickrExif and signed camera captures. This count is the sum of requested `numResults`, not 40 unique validated sources. Exa fetched the VISION README and FlickrExif card. Three calls to the Hugging Face dataset-search connector failed because that endpoint was unavailable; public Hub HTTP APIs were then used to inspect records and gating without authentication. No gated files were fetched.

Cite VISION as Shullani, D., Fontani, M., Iuliani, M., Al Shaya, O., and Piva, A. (2017), “VISION: a video and image dataset for source identification”, *EURASIP Journal on Information Security*, DOI: [10.1186/s13635-017-0067-2](https://doi.org/10.1186/s13635-017-0067-2).
