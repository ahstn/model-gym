# Quality prompt track

**Validation complete: Decider preserves much of the quality ordering, but misses most low-quality cases at the fixed threshold.** On 80 new KADID images, accuracy was 57.5%, low-quality recall was 6/40, and human-score MAE was 0.970 on the 1–5 scale. Spearman correlation was 0.781. The final calls cost $0.001662, with median latency of 0.911 seconds.

GEPA proposed three context revisions. None passed its training-batch improvement check, so the original prompt was retained and tested once. Total cost, including prompt trials and reflection, was $0.005600. This pass supports further calibration work, not automatic rejection or unverified teacher labels.

- [Results and limits](report/README.md)
- [Per-image scores, correctness, cost and latency](../../summaries/quality.csv)
- [Source labels, licence and sampling](SOURCES.md)
- [Frozen protocol](protocol.json) and [prompt selection](prompt-selection.json)
- [GEPA trial summary](../../summaries/quality.json)

The split contains 40 train, 40 development and 80 final images, grouped by pristine parent. Half of each split has human DMOS below 2.6. Human ratings used a reference image; Decider received only the distorted image. There is no human aesthetic target.

## Errors that matter

Mean signed error was **+0.878 points**. Scores exceeded human DMOS for 66/80 images. Post-test inspection of the three largest errors found visible damage despite recognisable subjects:

| Image | Human / Decider score | Observed defect |
| --- | ---: | --- |
| [kadid-095](../../data/tracks/quality/kadid-095.png) | 1.73 / 3.849 | Washed-out butterfly photo with clipped highlights |
| [kadid-116](../../data/tracks/quality/kadid-116.png) | 1.83 / 3.930 | Six flat coloured blocks cover parts of a forest photo |
| [kadid-107](../../data/tracks/quality/kadid-107.png) | 1.27 / 3.345 | Strong colour spread across a church photo |

One possible cause is the score anchors. The prompt reserves low scores for major information loss; KADID measures annoyance relative to a clean reference. This mismatch may combine with missed defects. Decider returns rubric choices, so these records do not establish which defects it noticed. The inspection took place after testing and did not guide a prompt change.

## Reproduce

From `visual-triage`, rebuild the report without API calls:

```sh
uv run --locked python tracks/report.py --track quality
```

To inspect or extend the optimiser, install its pinned dependency with `uv sync --locked --group dev --group optimisation`, then use `tracks/optimise.py --help`. Keep a fresh output directory for each search. Once these final results have informed a change, use new source groups for the next final test. Preserve this run's protocol and evidence.

The compact exports are derived results. Restore the [full evidence archive](../../ARCHIVE.md) to validate raw responses or rebuild reports. CSV phase and panel fields identify each comparison.
