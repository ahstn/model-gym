Assess whether each anonymous explanation is grounded in its supplied image. Inspect every original image individually at full detail. You receive case IDs, task, explanation text and limitations, but no scores, reference labels, model names or prompt arms. Do not access other files, source metadata, mappings or provider outputs.

For each label return label_id, case_id, judgment and a brief rationale tied to visible evidence. Allowed judgments:
- supported: the decisive visible claims are reasonably grounded; subjective aesthetic preferences are permitted.
- unsupported_specific_claim: a concrete claim (e.g. a named artifact, location or sharp subject detail) is more specific than the image supports.
- contradicted: an important visible claim conflicts with the image, e.g. calling a visibly blurred primary subject crisp.
- uncertain: the supplied scale or ambiguous scene prevents a responsible factual assessment.

Do not penalize concise evidence for omitting other issues. Do not assess numeric score accuracy, infer hidden artistic intent or treat an aesthetic opinion as an objective factual error. Report claim support, not whether you personally prefer the composition. Every listed label must receive a judgment. JSONL output, one row per label.
