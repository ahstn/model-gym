# Expansion to 20,000 labeled images

Decision accepted 2026-11-16. Retain the current ensemble: **37.5% Luna + 37.5% DeepSeek + 25% Liquid**, independently for technical and aesthetic scores. The [500-image comparison](../../trials/decision-comparison-500-2026-11-13/report.md) is a disagreement-enriched diagnostic against assistant references, not an unbiased population benchmark. Neither new Decisions model demonstrated grounds for replacement: Perplexity overestimated technical quality, while Gemini underestimated aesthetics.

Add 10,000 previously unlabeled images with EmbeddingGemma 2 content clustering, content-balanced sampling and related/duplicate review. Score the exact delivered views with Luna high, DeepSeek high, Liquid d1.1 and Jev 1.14 using the unchanged versioned v2 profile. Retain all initial 10K observations, including Q-ReAlign Lite, for historical comparisons; make no additional Q-ReAlign calls. Keep the existing 1,500 assistant references and the first-1,000 publication allowlist unchanged.

Store image-stats measurements separately from model responses; do not feed them to graders or use them to change labels. No training masks are enabled. All new per-image data stays under ignored `visual-quality/data/labeling-20000-2026-11-16/` in the primary checkout. Source code and aggregate reports are tracked here.

Status: preparation in progress; this document does not claim completion.
