# Dataset and method expansion

The search found two suitable sources for new Decider trials: **KonIQ-10k for technical quality** and **SynthCLIC for AI origin**. A separate Synthbuster panel tests AI transfer. The acquired sets contain 750 new images. We selected images using human scores and records of their source. Detector results did not guide selection.

| Candidate | Use in this pass | Next useful test |
| --- | --- | --- |
| KonIQ-10k | 350 authentic photographs with raw human quality scores | External transfer on another human-rated source |
| SynthCLIC | 320 matched photos; Imagen 3 held out from search | More unseen generators and camera sources |
| Synthbuster / RAISE | 80 external images, with SDXL as the generated class | Broader generator coverage |
| Community Forensics Eval | Research lead: 51,836 rows from 21 generators | Larger external test; verify pair links and source terms first |
| KonIQ++ | Research lead; shares KonIQ images | Human defect labels to test which errors Decider sees |
| LIVE Challenge / SPAQ | Deferred pending access or clearer rights | Authentic defects, phone attributes and camera context |
| AGIQA-3K / Q-Bench | Deferred; their targets differ from this technical MOS objective | Generated-image quality ranks and low-level perception questions |

GPT-6.1 Sol agents at medium effort read the requested [Awesome IQA catalogue](iqa-catalog.md), then checked author papers, data releases and terms. Exa supplied several search rounds. The Hugging Face plugin was unavailable; official Hub APIs and pinned native files supplied the dataset checks and downloads. Research-only terms were accepted where the source permitted this use. Detailed rights limits remain with each dataset.

Read the [quality source review](quality-sources.md), [AI source review](ai-sources.md) and [metadata evidence](metadata-methods.md) for alternatives and limits. The [experiment plan](experiment-plan.md) fixes 20 GEPA proposals per track, training-only reflection, development selection and fresh final tests. The [cost recovery note](cost-recovery.md) records one failed request without assigning it an invented charge.

Visible defect checks, five-level rubrics and human calibration are useful research directions. Q-Align, PromptIQA and PACE use training or human calibration. This trial changes prompts only. Camera metadata, pairwise comparisons and a fitted blind-IQA baseline remain separate experiments.

The [completed validation](../../EXTENDED-VALIDATION.md) reports both 20-proposal searches, final errors, pixel-statistics tests and per-image evidence.
