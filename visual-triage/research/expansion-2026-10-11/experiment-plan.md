# Extended Decider experiment

This pass tests whether longer prompt search improves Decider on human quality ratings and source-labelled AI origin. It preserves the previous tests and uses new images. The model is `perplexity/pplx-decider-v1.1-27b` in both tracks. Luna supplies GEPA's prompt revisions; it is not the decision model.

## Data decisions

| Track | Selected source | Reason | Limit |
| --- | --- | --- | --- |
| Quality | KonIQ-10k, raw 1–5 human MOS, unchanged 1024×768 study images | Authentic defects and no reference image shown to human raters | Incomplete photographer metadata; no human aesthetics target or independent external source |
| AI origin | SynthCLIC matched source families; original Synthbuster as an external source | Provenance labels, matched scene descriptions and an unseen-generator test | Semantic matching is imperfect; sampling and generator coverage limit generalisation |

The source agents freeze counts, licence records, image hashes and group splits before inference. Non-commercial terms are accepted. A research purpose does not waive source access conditions or rights notices. Image files remain in the ignored data directory.

KonIQ++ can add human defect targets but reuses KonIQ images. LIVE Challenge requires an access request. SPAQ needs clearer rights and repeated-scene groups. AGIQA's mixed perception score has no defensible direct conversion into this technical 1–5 target. These sources remain research leads, not additional completed evaluations. See the [catalogue](iqa-catalog.md), [quality search](quality-sources.md) and [AI search](ai-sources.md).

## Search and selection

Each track runs **20 GEPA proposals**, within the requested 10–40 range. Quality can change context and technical rubric descriptions; the five ordered levels, output fields and aesthetics question stay fixed. AI search changes context only. The baseline is an unchanged copy of the previous task prompt.

Whole training groups supply feedback. Luna sees at most four training images per proposal, with their labels and errors. It must generalise the evidence and must not copy image IDs, dataset names or example-specific rules. Development groups select the prompt. No final or external image is available to reflection or selection.

Quality maximises group mean `1 - absolute_error / 4`. AI detection maximises group mean `1 - Brier_error`. Keep the baseline unless development reward improves by at least 0.005. This margin is a selection rule, not a significance test. Freeze the selected prompt before final calls.

Twenty proposals can still overfit development data. Final results therefore compare baseline and selected prompts on fresh groups, with paired group intervals. The AI test also holds out a generator and an external source. Results cannot establish that overfitting is absent.

## Additional context

Test the selected prompt with and without frozen pixel statistics on the same final images. The added fields describe the delivered view: dimensions, tonal quantiles and spread, near-black/white fractions, channel distributions and local intensity differences. These are measurements, not fitted quality scores. The fixed guidance warns against using resolution or simple statistics as origin proof.

Native EXIF availability, camera identities, software, filenames, source IDs and labels are excluded from model context. Optional numeric camera extraction is implemented for later research, but this pass does not test camera context. Its uneven availability could reveal source labels. See the [metadata evidence and limits](metadata-methods.md).

The context plan and measurements are frozen before final calls. This is a paired diagnostic; final outcomes cannot select a new prompt or context variant on the same images.

## Evidence and limits

Record every API attempt, valid-output rate, provider-reported cost, client latency and per-image errors. Include reflection and upstream BYOK costs. Unknown costs stop the run; recorded attempts are not retried on resume. Known-cost malformed proposals count as rejected rounds and retain the previous candidate.

Each track has a combined $5 guard for search, final comparison and metadata tests. The reports separate those costs. Quality reports MAE, rank correlation and low-quality detection below 2.6. AI reports Brier error, accuracy at 0.5, recall, false positives and AUROC. Aesthetic predictions remain unvalidated, and NSFW is outside this pass.
