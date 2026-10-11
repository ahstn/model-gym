# Separate prompt validation tracks

The v2 pass completed 20 GEPA proposals per track. Both selected prompts improved their final error metrics; detection still misses too many positive cases for automatic rejection. Read the [extended validation](../EXTENDED-VALIDATION.md). Both tracks use Decider v1.1 with separate frozen datasets, prompts, protocols and ledgers. The completed v1 results remain in the [validation report](../VALIDATION.md).

| Track | Source and target | Train / development / final images | External images |
| --- | --- | ---: | ---: |
| [Quality v2](quality_v2/SOURCES.md) | KonIQ-10k authentic photographs; native human MOS on the 1–5 scale | 150 / 100 / 100 | None |
| [AI detection v2](ai_detection_v2/SOURCES.md) | SynthCLIC matched photographs; source provenance labels | 160 / 80 / 80 | 80, from Synthbuster |
| [Quality v1 evidence](quality/README.md) | KADID-10K controlled defects; Decider | 40 / 40 / 80 | None |
| [AI detection v1 evidence](ai_detection/README.md) | Synthbuster and RAISE matched photographs; Luna | 30 / 30 / 60 | None |

Quality preserves raw MOS: higher means better quality. AI labels describe generated or real origin. Each photo group or matched pair stays in one split. Earlier final images remain excluded from the new panels. The source pages document acquisition, licence limits and duplicate checks.

## Selection and cost

Each v2 run attempted 20 GEPA proposals. Luna reflection receives training feedback and at most four training images. Quality can change context and technical-quality rubric descriptions. The aesthetics question, output schema and ascending score anchors stay fixed. AI can change context only.

- Quality maximises mean group reward `1 - absolute_error / 4`. Human aesthetics is not scored. Low quality remains strictly below 2.6.
- AI maximises mean group reward `1 - Brier_error`. AI probability means generated origin. Classification uses `>= 0.5`; the open review band is `(0.2, 0.8)`.
- The full development panel stays fixed. Keep the baseline unless the best reward improves by at least 0.005. Freeze selection before final or external calls. A retained baseline needs one final evaluation.
- An `accepted` proposal passed validation; it does not establish an improvement. Rejected text keeps the prior candidate for that attempt. Every paid attempt remains in the ledger.

The combined limit is $5 per track, including search, reflection, final inference, external inference and the pixel-statistics diagnostic. Unknown costs stop further calls unless one exact failed attempt has an explicit, reviewed budget reservation. Such a reservation does not establish its actual charge; reports keep that cost unknown. Freeze the diagnostic before testing. It uses the selected prompt with fixed context and permits no retuning. There is no camera or EXIF arm.

Repeated development comparisons can overfit. Source coverage, unknown pretraining exposure and incomplete photographer isolation limit transfer claims. Pixel judgements do not prove provenance. Quality has no external panel or human aesthetics target.

## Run and inspect

The [optimisation guide](OPTIMISATION.md) gives reproduction, resume, selection and final diagnostic commands. See the final [quality results](quality_v2/report/README.md), [AI results](ai_detection_v2/report/README.md), and pixel-statistics tests for [quality](quality_v2/report/metadata-report.md) and [AI](ai_detection_v2/report/metadata-report.md). Each run uses its own directory; v1 protocols and ledgers keep their original settings.

From `visual-triage`, restore the [archived evidence](../ARCHIVE.md) and rebuild saved v1 reports without API calls:

```sh
python3 evidence_archive.py restore
uv run --locked python tracks/report.py --track quality
uv run --locked python tracks/report.py --track ai_detection
```
