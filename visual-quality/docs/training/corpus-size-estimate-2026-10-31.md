# Corpus size estimate for dual-score training

31 October 2026. **Plan for 50,000 independent source-image families total: 40,000 training, 5,000 validation and 5,000 locked test.** This is an engineering estimate for adapting the pretrained 9–12B vision-language models in the [current backbone plan](../models/backbone-refresh-2026-10-30.md). It is not a measured minimum, a promise of accuracy, or an estimate for training a model from scratch. No new acquisition, labeling or training run is started by this note.

Count one representative image per independent family toward these targets. Crops, resized copies, burst near duplicates and controlled degradations belong to the same family; additional views can be useful training examples but do not increase the independent-image count. Two scores per image and two teacher calls also do not increase that count. Preserve existing protected test assignments and check overlap before incorporating any earlier trial images.

## Start smaller and measure the benefit

Reserve the same 5,000 validation and 5,000 test families before training. Use nested, stratified training subsets:

| Stage | Training families | Validation families | Locked test families | Total families |
|---|---:|---:|---:|---:|
| Initial learning-curve point | 5,000 | 5,000 | 5,000 | 15,000 |
| First substantial baseline | 10,000 | 5,000 | 5,000 | 20,000 |
| Intermediate | 20,000 | 5,000 | 5,000 | 30,000 |
| Main target | 40,000 | 5,000 | 5,000 | 50,000 |

Use validation to compare dataset sizes, model choices, calibration and prompts. Keep the test sealed until those choices are fixed. Report technical and aesthetic MAE, bias, rank correlation, scope errors and controlled-defect ordering separately; combined average error can hide regressions in one dimension. Compare under a declared training/compute policy and repeat close comparisons across seeds so extra data and extra optimizer steps are not silently conflated.

Expand toward 100,000 total only if the validation learning curves and uncovered domains justify it. Preserve the original holdouts; growth would primarily add training families. A useful smaller baseline can be built before reaching 50,000, but the current 2,000-image teacher trial does not establish how much student-training data is sufficient.

## Why this scale is a reasonable starting point

The original [Q-Align paper, Tables 3 and 6](https://arxiv.org/html/2312.17090) reports a pretrained LMM fine-tuned with about 2K versus 7K KonIQ images for technical quality, and 26K versus 236K AVA images for aesthetics. Its 26K aesthetic setting reaches SRCC 0.776, compared with 0.822 using 236K. This supports testing a useful model in the tens of thousands while leaving room for further gains. Different labels, architectures and image distributions mean those results do not establish our required count. The larger aesthetic example is one reason to target broad coverage beyond a small defect-only corpus.

The [KonIQ-10k authors](https://database.mmsp-kn.de/koniq-10k-database.html) describe 10,073 images with 1.2 million human quality ratings. Image count alone misses the strength of that supervision; 10,000 single model-generated labels are not equivalent evidence.

Our [existing pilot](../datasets/pilot-10k-assembly.md) contains 10,800 rows but 7,533 source groups. Its primary scalar targets remain unset. The [inventory review](../datasets/inventory-and-adoption-review-2026-10-29.md) also found no acquired broad human-rated aesthetics corpus. MSC and TAD66K remain candidate additions with source-specific rights and overlap checks, not available training labels that can simply be counted toward this target.

## Label reliability and coverage

Continue evaluating the separate technical and aesthetic means of Luna and DeepSeek as versioned training targets. Preserve both component scores, disagreement, eligibility and provenance. Resolve scope disagreements before averaging: a 1.00 exclusion from one model and a normal score from the other must not become an apparently valid middle score without review. Clearly non-photographic controls keep the agreed 1.00/1.00 policy; ambiguous provenance alone is not grounds for exclusion.

The [2,000-image trial](../../experiments/photo-scoring-2026-10-31/README.md) has 72 assistant assessments, of which 71 are fresh reference cases. The average's technical bias is +0.304 on that subset. Its combined advantage over Luna is uncertain, and Luna has better controlled-defect technical consistency. These findings support human calibration and targeted checks; they do not justify applying a universal -0.30 correction.

Before large-scale labeling, independently human-rate about 1,000 representative development images for both dimensions, with multiple raters and disagreement retained. Count these families within the 40,000 training allocation and keep them out of validation/test; they may join training after the rubric is frozen. Use this development panel to refine the rubric and evaluate calibration. Separately build human-reviewed validation/test evidence under the final rubric, preferably across the full holdouts. If only a subset receives independent ratings, restrict human-accuracy claims to that subset and report teacher agreement separately. Neither the teacher mean nor assistant reference scores become human ground truth. Existing experimental labels remain unapproved; this estimate changes no training masks.

Cover ordinary photos and score extremes across people, landscapes, wildlife, architecture, products, food, indoor and low-light scenes. Include technically clean but aesthetically weak images, aesthetically strong images with defects, intentional blur/grain, authentic capture failures, subtle compression/noise and physical signs or artwork photographed in context. Retain clearly non-photographic controls. Balance these dimensions deliberately without forcing technical and aesthetic scores to match or filling most rows with easy middle-quality images.

Keep representative validation/test sampling independent of teacher disagreements. Use a separately identified challenge panel for targeted failures, and report its metrics separately from the representative sample. Reserve source collections or capture conditions absent from training to measure transfer. Group families across all sources before splitting; keep all derived views in their parent's partition. Human source-dataset MOS and the project rubric also need explicit mapping, rather than blindly pooling scales.

## Teacher-label cost estimate

The completed trial reports known combined API cost of **$2.591454105 for 2,000 images**, or about **$0.001296 per image for both teachers**. Straight-line projections at the same routes, delivered views and response behavior are:

| Images scored by both teachers | Estimated combined API cost |
|---|---:|
| 20,000 | $25.91 |
| 50,000 | $64.79 |
| 100,000 | $129.57 |

These are annotation-only projections, not quotes or authorized runs. Extra derivative images add calls. Human rating, image acquisition, storage and GPU training are additional costs. The source trial has five unpriced pre-serve capacity refusals; its known total is not a reconciled invoice. Pricing, reasoning lengths and retry behavior may change. Human label quality and dataset coverage are currently more consequential planning constraints than this measured teacher-call cost.
