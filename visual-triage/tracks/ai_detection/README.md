# AI-detection prompt track

**Validation complete: Luna's revised prompt improved accuracy from 85.0% to 88.3%, but cost 51% more.** Keep the original prompt as the cheaper default for further review tests. The revised prompt remains a candidate; its measured gain is too uncertain to justify replacement.

Both prompts saw the same 60 photos: 30 real and 30 generated, matched by source caption. At the fixed 0.5 threshold, the original detected 21 generated photos and falsely flagged no real photos. The revision detected 24 and falsely flagged one. Neither result establishes a safe automatic rejection rule.

| Measure | Original prompt | GEPA prompt |
| --- | ---: | ---: |
| Brier error, lower is better | 0.1241 | 0.1194 |
| AUROC | 0.961 | 0.975 |
| Both images classified correctly | 21/30 pairs | 23/30 pairs |
| Images in review band, `0.2 < p < 0.8` | 19/60 | 39/60 |
| Final cost | $0.009520 | $0.014369 |
| Median / P95 latency | 3.289 / 4.730 s | 3.847 / 6.471 s |

The paired 95% interval for the Brier improvement is −0.0326 to +0.0432. It includes no gain. The revision's higher recall comes with more uncertain predictions and one false flag. The source panel is small and uses older generators, so it cannot establish performance on current production images.

GEPA made three proposals using training feedback. Proposal two passed the training check and reduced development Brier error from 0.1351 to 0.1236. That choice was frozen before final calls. Total track cost was **$0.046161** for 231 calls, including search and reflection. Costs include Luna's reported upstream BYOK usage.

- [Full metrics and uncertainty](report/README.md)
- [Per-image predictions, correctness, cost and latency](../../summaries/ai_detection.csv)
- [Source labels, licence, matching and photo screen](SOURCES.md)
- [Frozen protocol](protocol.json), [selection](prompt-selection.json) and [selected prompt](prompts/selected.json)
- [GEPA trial summary](../../summaries/ai_detection.json)

From `visual-triage`, rebuild the report without API calls:

```sh
uv run --locked python tracks/report.py --track ai_detection
```

Use new source pairs for any later confirmation test. These final results must not guide prompt selection on the same images.

The compact exports are derived results. Restore the [full evidence archive](../../ARCHIVE.md) to validate raw responses or rebuild reports. CSV phase and panel fields identify each comparison.
