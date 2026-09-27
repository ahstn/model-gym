# Pilot assembly record

Built and verified locally on 2026-09-27 with `configs/pilot-10k.json`, seed `20260927`, and the [pilot builder](../../README.md). Acquisition and assembly are complete. Quality scores and aesthetics scores have not been generated.

## Assembled data

| Component | Train | Development | Final test |
|---|---:|---:|---:|
| Megalith-CC0 photos | 5,000 | 200 | 200 |
| VizWiz photos with human-voted defects | 1,000 | 100 | 100 |
| PlotQA charts | 400 | 40 | 40 |
| CLEVR renders | 400 | 40 | 40 |
| Project-owned diagram and interface controls | 200 | 20 | 20 |
| Generated photo variants | 3,000 | 0 | 0 |
| **Total** | **10,000** | **400** | **400** |

The manifests contain 10,800 distinct decoded images in 7,533 source families. The listed image files total 28,623,831,476 bytes (28.62 GB); source metadata, unused spares, and caches require extra space. All media and local build outputs are ignored by Git.

The 3,000 variants use 1,000 parent photos that are already in train: 951 blur variants, 1,092 noise variants, and 957 JPEG variants. Each parent has three variants of one operator. Parent and variant records share the same split group. Each split has equal numbers of the five owned-control types: charts, flowcharts, circuit diagrams, mock interfaces, and isometric renders.

## Source snapshot

| Source | Acquired candidates | Pinned identity |
|---|---:|---|
| `Spawning/megalith-cc0` | 5,500 | HF revision `3aa3a5760d944af63939db9eaf94fdd6c68af888` |
| VizWiz Quality Issues | 1,260 | Annotation revision `736265e64805685d8595bc43c01364e8e51efae2`; official image archive and member checksums retained |
| `achang/plot_qa` | 500 | HF revision `e9da2cdb45c671f3faf63449e6c329f0c3281aa0` |
| CLEVR 1.0 | 500 | Official archive, S3 version `B2HsawCh6GoabZXQDbLoboNXB4FrlnvC` |
| Project-owned controls | 240 | Generator and dependency hashes in `software.json` |

The builder selected 7,800 source and control records from 8,000 candidates, then added 3,000 variants. Upstream training partitions supply both local holdout splits. Source-specific acquisition removed failed requests and duplicate candidates before this assembly; all 8,000 registered candidates passed image validation.

Source limitations remain relevant. Megalith has no photographer IDs, and its metadata hash differs from the delivered bytes for 347 of the 5,500 acquired files. The builder retains both identities and uses the downloaded file's SHA-256 for replay. PlotQA uses pinned HF Viewer PNGs; equivalence to the upstream original files is unverified, and inaccessible Viewer windows affected sampling. Full provenance and license evidence remain with each source cache. Retain these caches for exact replay.

## Verification evidence

`visual-pilot verify --root data` returned no errors after reading every listed image and checking its SHA-256. It also checked frozen metadata, manifest hashes, exact component quotas, and family and decoded-pixel separation across splits. All variants refer to training parents, and none copies a parent quality label. The saved software hashes match the source and dependency lock used for this build.

The focused test suite passed 51 tests and four subtests. Ruff checks and formatting checks passed. Visual spot checks covered all three distortion operators and all control types; they are not a full quality audit.

| Manifest | SHA-256 |
|---|---|
| `train.jsonl` | `c8bc9fa7b796239739b2f7a00cd99e2051342c382ec179867658d1b702246a53` |
| `development.jsonl` | `131329cc408e05bca16c67a20a21ab3e67807cd060c1032be6bbf0cd1582809b` |
| `test.jsonl` | `fd49c03efc574c19be6c413885225a83223bd5ab3e175fb97f8dc7e6f7159c2c` |

Local records are under `data/pilot-10k-v1/`: `report.json` records assembly results, and `verification.json` records the subsequent file check. Preview contact sheets are under `data/previews/`. Resolve manifest image paths from the `data/` directory, as specified by `dataset.json`.

## Remaining label work

All technical-quality and aesthetics values are null, with their task masks disabled. The 1,200 chart, render, and owned-control records have `photo_domain_eligible=false`; this is a domain label, not a zero quality or aesthetics score. Other records still need a photo-eligibility audit.

The broad-photo sample needs an independent quality-band audit. Human development review, teacher scoring, and calibration remain pending. No external evaluation exclusion manifest was supplied, so overlap with external benchmarks is not cleared. The build report therefore keeps `training_ready=false`.
